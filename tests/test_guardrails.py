"""Tests for the goal-driven self-correction loop and its safety guardrails."""
from __future__ import annotations

from unittest import mock

from src.agents.coder import Patch
from src.agents.diagnostic import Diagnosis
from src.agents.reporter import PatchReport
from src.orchestrator import (
    STAGNATION_HALT_AFTER,
    STAGNATION_STRATEGY_AFTER,
    _error_signature,
    run_patchcraft_loop,
)
from src.sandbox.runner import TestResult


def _write(root, relpath: str, content: str):
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _diagnosis() -> Diagnosis:
    return Diagnosis(
        summary="offset bug",
        root_cause="off-by-one",
        affected_files=["src/app.py"],
        confidence=0.9,
    )


def _patch(content: str) -> Patch:
    return Patch(files=[{"file_path": "src/app.py", "new_content": content}])


def _report() -> PatchReport:
    return PatchReport(title="T", summary="S", diff="d", pr_markdown="m")


def _fail(stdout: str = "1 failed", stderr: str = "AssertionError") -> TestResult:
    return TestResult(success=False, stdout=stdout, stderr=stderr, exit_code=1)


def _ok() -> TestResult:
    return TestResult(success=True, stdout="1 passed", exit_code=0)


class TestGoalDrivenLoop:
    def test_loops_until_green_without_retry_cap(self, tmp_path) -> None:
        """No max_retries: the loop iterates until tests turn green."""
        _write(tmp_path, "src/app.py", "x = 1\n")
        results = [_fail("fail A"), _fail("fail B"), _fail("fail C"), _ok()]

        with (
            mock.patch("src.orchestrator.diagnose", return_value=_diagnosis()),
            mock.patch("src.orchestrator.generate_patch", return_value=_patch("x = 2\n")),
            mock.patch(
                "src.orchestrator.correct_patch",
                side_effect=lambda **kw: _patch(f"x = 2\n# fix {len(kw['test_feedback'])}\n"),
            ),
            mock.patch("src.orchestrator.SandboxRunner.run_tests", side_effect=results),
            mock.patch("src.orchestrator.generate_report", return_value=_report()),
        ):
            result = run_patchcraft_loop(str(tmp_path), "Fix", "mock")

        assert result.success is True
        assert result.iterations == 4  # kept going past any arbitrary limit of 3
        assert result.halt_reason is None

    def test_explicit_limit_still_respected(self, tmp_path) -> None:
        """max_retries remains available as an opt-in hard cap."""
        _write(tmp_path, "src/app.py", "x = 1\n")
        counter = {"n": 0}

        def _always_fail():
            counter["n"] += 1
            return _fail(f"unique-failure-{counter['n']}")

        with (
            mock.patch("src.orchestrator.diagnose", return_value=_diagnosis()),
            mock.patch("src.orchestrator.generate_patch", return_value=_patch("x = 2\n")),
            mock.patch(
                "src.orchestrator.correct_patch",
                side_effect=lambda **kw: _patch(f"x = 2\n# {counter['n']}\n"),
            ),
            mock.patch("src.orchestrator.SandboxRunner.run_tests", side_effect=_always_fail),
        ):
            result = run_patchcraft_loop(str(tmp_path), "Fix", "mock", max_retries=2)

        assert result.success is False
        assert result.iterations == 2
        assert result.halt_reason == "Iteration limit reached (2)."
        # rollback restored the original file
        assert (tmp_path / "src/app.py").read_text(encoding="utf-8") == "x = 1\n"


class TestLoopDetection:
    def test_error_signature_ignores_volatile_numbers(self) -> None:
        """Timings/addresses must not defeat repetition detection."""
        a = _fail(stdout="FAILED test_add - assert 3 == 2", stderr="took 0.12s at 0x7f2a1")
        b = _fail(stdout="FAILED test_add - assert 3 == 2", stderr="took 0.98s at 0xff00b")
        assert _error_signature(a) == _error_signature(b)

    def test_error_signature_keeps_meaningful_changes(self) -> None:
        """Different assertion outcomes are real progress, not repetition."""
        a = _fail(stdout="FAILED test_add - assert 3 == 2")
        b = _fail(stdout="FAILED test_add - assert 4 == 4")
        assert _error_signature(a) != _error_signature(b)

    def test_stagnation_forces_strategy_change_then_halts(self, tmp_path) -> None:
        """Same failure repeats: directive injected first, graceful halt later."""
        _write(tmp_path, "src/app.py", "x = 1\n")
        captured_feedback: list[str] = []

        def _correct(**kwargs) -> Patch:
            captured_feedback.append(kwargs["test_feedback"])
            return _patch(f"x = {len(captured_feedback)}\n")

        with (
            mock.patch("src.orchestrator.diagnose", return_value=_diagnosis()),
            mock.patch("src.orchestrator.generate_patch", return_value=_patch("x = 0\n")),
            mock.patch("src.orchestrator.correct_patch", side_effect=_correct),
            mock.patch(
                "src.orchestrator.SandboxRunner.run_tests",
                side_effect=lambda: _fail("FAILED test_x", "AssertionError"),
            ),
        ):
            result = run_patchcraft_loop(str(tmp_path), "Fix", "mock")

        assert result.success is False
        assert result.iterations == STAGNATION_HALT_AFTER
        assert result.halt_reason is not None
        assert "Stagnation detected" in result.halt_reason
        # strategy-change directive reached the coder after STAGNATION_STRATEGY_AFTER
        assert len(captured_feedback) == STAGNATION_HALT_AFTER - 1
        assert "STAGNATION WARNING" in captured_feedback[STAGNATION_STRATEGY_AFTER - 1]
        assert "STAGNATION WARNING" not in captured_feedback[0]
        # rolled back safely
        assert (tmp_path / "src/app.py").read_text(encoding="utf-8") == "x = 1\n"

    def test_different_errors_do_not_count_as_stagnation(self, tmp_path) -> None:
        """Changing failures mean progress: no strategy directive injected."""
        _write(tmp_path, "src/app.py", "x = 1\n")
        captured: list[str] = []

        def _correct(**kwargs) -> Patch:
            captured.append(kwargs["test_feedback"])
            return _patch(f"x = {len(captured)}\n")

        fails = [
            TestResult(success=False, stdout=f"unique-failure-{i}", stderr="", exit_code=1)
            for i in range(STAGNATION_STRATEGY_AFTER + 1)
        ]

        with (
            mock.patch("src.orchestrator.diagnose", return_value=_diagnosis()),
            mock.patch("src.orchestrator.generate_patch", return_value=_patch("x = 0\n")),
            mock.patch("src.orchestrator.correct_patch", side_effect=_correct),
            mock.patch("src.orchestrator.SandboxRunner.run_tests", side_effect=[*fails]),
        ):
            result = run_patchcraft_loop(str(tmp_path), "Fix", "mock", max_retries=len(fails))

        assert all("STAGNATION WARNING" not in fb for fb in captured)


class TestBudgetLimits:
    def test_token_budget_halts(self, tmp_path) -> None:
        """Per-task token accounting stops the loop before runaway spend."""
        _write(tmp_path, "src/app.py", "x = 1\n")

        def _generate(*args, **kwargs) -> Patch:
            sink = kwargs["usage_sink"]
            sink(100, 100)  # simulate a 200-token completion
            return _patch("x = 2\n")

        with (
            mock.patch("src.orchestrator.diagnose", return_value=_diagnosis()),
            mock.patch("src.orchestrator.generate_patch", side_effect=_generate),
            mock.patch(
                "src.orchestrator.SandboxRunner.run_tests",
                side_effect=lambda: _fail("boom"),
            ),
        ):
            result = run_patchcraft_loop(str(tmp_path), "Fix", "mock", token_budget=150)

        assert result.success is False
        assert result.halt_reason is not None
        assert "Token budget exhausted" in result.halt_reason
        assert result.iterations == 1  # halted at the top of iteration 2

    def test_time_budget_halts_before_first_iteration(self, tmp_path) -> None:
        _write(tmp_path, "src/app.py", "x = 1\n")
        with (
            mock.patch("src.orchestrator.diagnose", return_value=_diagnosis()),
            mock.patch("time.monotonic", side_effect=[0.0, 999.0]),
        ):
            result = run_patchcraft_loop(str(tmp_path), "Fix", "mock", time_budget_seconds=60)

        assert result.success is False
        assert result.halt_reason is not None
        assert "Time budget exhausted" in result.halt_reason
        assert result.iterations == 0

    def test_credit_floor_halts(self, tmp_path) -> None:
        _write(tmp_path, "src/app.py", "x = 1\n")
        with (
            mock.patch("src.orchestrator.diagnose", return_value=_diagnosis()),
            mock.patch("src.orchestrator._remaining_credits", return_value=0.25),
        ):
            result = run_patchcraft_loop(str(tmp_path), "Fix", "mock", min_remaining_credits=1.0)

        assert result.success is False
        assert result.halt_reason is not None
        assert "credit" in result.halt_reason.lower()
