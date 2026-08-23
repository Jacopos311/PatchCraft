"""Tests for targeted test selection (Roadmap Step 2.1)."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

from src.agents.coder import Patch
from src.core.repo_index import RepoIndex
from src.core.targeted_tests import (
    MAX_TARGETS,
    select_targeted_tests,
)
from src.sandbox.runner import SandboxRunner, TestResult


def _write(root: Path, rel: str, content: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _diagnosis():
    from src.agents.diagnostic import Diagnosis

    return Diagnosis(
        summary="offset bug",
        root_cause="off-by-one",
        affected_files=["src/app.py"],
        confidence=0.9,
    )


def _report():
    from src.agents.reporter import PatchReport

    return PatchReport(title="t", summary="s", diff="d", pr_markdown="m")


class TestSelection:
    def test_companion_test_file_found(self, tmp_path: Path):
        _write(tmp_path, "src/app.py", "def add(a, b):\n    return a + b\n")
        _write(tmp_path, "tests/test_app.py", "def test_add():\n    assert True\n")
        index = RepoIndex.build(tmp_path)

        result = select_targeted_tests(tmp_path, ["src/app.py"], index=index)

        assert result.has_targets is True
        assert "tests/test_app.py" in result.node_ids

    def test_import_dependent_test_found(self, tmp_path: Path):
        """A test file that imports the affected module is selected."""
        _write(tmp_path, "src/calc.py", "def multiply(a, b):\n    return a * b\n")
        _write(
            tmp_path,
            "tests/test_calc_extra.py",
            "from src.calc import multiply\n\ndef test_mul():\n    assert multiply(2, 3) == 6\n",
        )
        index = RepoIndex.build(tmp_path)

        result = select_targeted_tests(tmp_path, ["src/calc.py"], index=index)

        assert result.has_targets is True
        assert "tests/test_calc_extra.py" in result.node_ids

    def test_no_test_files_means_no_targets(self, tmp_path: Path):
        _write(tmp_path, "src/app.py", "x = 1\n")
        index = RepoIndex.build(tmp_path)

        result = select_targeted_tests(tmp_path, ["src/app.py"], index=index)

        assert result.has_targets is False
        assert result.node_ids == []
        assert any("falling back" in n.lower() for n in result.notes)

    def test_max_targets_fallback(self, tmp_path: Path):
        """More than MAX_TARGETS companion tests -> full-suite fallback."""
        for i in range(MAX_TARGETS + 5):
            _write(
                tmp_path,
                f"tests/test_big_{i}.py",
                f"import src.big\n\ndef test_{i}():\n    pass\n",
            )
        _write(tmp_path, "src/big.py", "x = 1\n")
        index = RepoIndex.build(tmp_path, force=True)

        result = select_targeted_tests(tmp_path, ["src/big.py"], index=index)

        assert result.has_targets is False
        assert any("exceed" in n.lower() or "limit" in n.lower() for n in result.notes)


class TestRunnerTargets:
    def test_with_pytest_targets_appends_to_command(self):
        cmd = ["/usr/bin/python3", "-m", "pytest"]
        result = SandboxRunner._with_pytest_targets(cmd, ["test_a.py::test_1"])
        assert result == [*cmd, "--", "test_a.py::test_1"]

    def test_non_pytest_command_ignores_targets(self):
        cmd = ["npm", "test"]
        result = SandboxRunner._with_pytest_targets(cmd, ["test_a.py"])
        assert result == cmd


class TestLoopIntegration:
    def test_loop_uses_targeted_then_full_gate(self, tmp_path: Path):
        """Two-phase: targeted green then full suite gate then success."""
        _write(tmp_path, "src/app.py", "def add(a, b):\n    return a - b\n")
        _write(
            tmp_path,
            "tests/test_app.py",
            "from src.app import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        )

        surgical_patch = Patch(files=[{
            "file_path": "src/app.py",
            "edits": [{"find": "    return a - b\n", "replace": "    return a + b\n"}],
        }])

        calls = {"run_tests": 0}

        def _tracking_run_tests(targets=None):
            calls["run_tests"] += 1
            return TestResult(success=True, stdout=f"call-{calls['run_tests']}", exit_code=0)

        with (
            mock.patch("src.orchestrator.diagnose", return_value=_diagnosis()),
            mock.patch("src.orchestrator.generate_patch", return_value=surgical_patch),
            mock.patch("src.sandbox.runner.SandboxRunner.run_tests", side_effect=_tracking_run_tests),
            mock.patch("src.orchestrator.generate_report", return_value=_report()),
        ):
            from src.orchestrator import run_patchcraft_loop

            result = run_patchcraft_loop(str(tmp_path), "Fix sign", model="mock")

        assert result.success is True
        # Two-phase: targeted (1) + full suite gate (2) -- at minimum.
        assert calls["run_tests"] >= 2, (
            f"expected at least 2 run_tests calls (targeted + full), got {calls['run_tests']}"
        )