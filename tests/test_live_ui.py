"""Tests for the streaming/live iteration UI (Roadmap Step 3.2).

Covers:
* the per-run token registry (src.core.runstats);
* RunState parsing (stage, iteration n/∞, last verdict, budget);
* pipe-friendly mode (plain lines, zero ANSI escapes);
* interactive mode wiring (rich Live updates mocked);
* end-to-end: orchestrator milestones flow through LiveRunView;
* the TUI status footer mirrors the same state.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

from src.agents.coder import Patch
from src.core import runstats
from src.gui.live_panel import ENV_TOKEN_BUDGET, LiveRunView, RunState


@pytest.fixture(autouse=True)
def _clean_runstats():
    runstats.reset()
    yield
    runstats.reset()


def _write(root: Path, rel: str, content: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestRunStats:
    def test_begin_current_add(self) -> None:
        stats = runstats.begin_run()
        assert runstats.current_run() is stats
        assert stats.total == 0
        stats.add(100, 50)
        stats.add(10, 5)
        assert stats.prompt_tokens == 110
        assert stats.completion_tokens == 55
        assert stats.total == 165

    def test_begin_run_resets_window(self) -> None:
        first = runstats.begin_run()
        first.add(999, 999)
        second = runstats.begin_run()
        assert second is not first
        assert second.total == 0
        assert runstats.current_run() is second

    def test_reset_clears_current(self) -> None:
        runstats.begin_run().add(1, 2)
        runstats.reset()
        assert runstats.current_run() is None


class TestRunStateParsing:
    def test_iteration_numeric_total(self) -> None:
        state = RunState()
        state.observe("iteration", "Iteration 2/5")
        assert state.iteration == 2
        assert state.iteration_total == 5
        assert state.iteration_label == "2/5"

    def test_iteration_unbounded(self) -> None:
        state = RunState()
        state.observe("iteration", "Iteration 1/∞")
        assert state.iteration == 1
        assert state.iteration_total is None
        assert state.iteration_label == "1/∞"

    def test_test_verdict_pass_and_fail(self) -> None:
        state = RunState()
        state.observe("test", "exit_code=0 success=True")
        assert state.last_verdict is True
        state.observe("test", "exit_code=1 success=False")
        assert state.last_verdict is False

    def test_running_message_keeps_verdict_unchanged(self) -> None:
        state = RunState()
        state.observe("test", "Running 2 targeted test file(s): tests/test_a.py")
        assert state.last_verdict is None
        assert state.stage == "test"

    def test_stage_transitions(self) -> None:
        state = RunState()
        state.observe("context", "Documentation and sources collected.")
        assert state.stage == "context"
        state.observe("diagnosis", "offset bug")
        assert state.stage == "diagnosis"
        state.observe("patch", "Applied changes:\nsrc/app.py")
        assert state.stage == "patch"

    def test_summary_line_contents(self) -> None:
        runstats.begin_run().add(3412, 0)
        state = RunState(token_budget=10_000)
        state.observe("iteration", "Iteration 1/∞")
        state.observe("test", "exit_code=1 success=False")
        line = state.summary_line()
        assert "iter 1/∞" in line
        assert "3,412/10,000" in line
        assert "FAIL" in line
        assert "⏱" in line

    def test_summary_without_budget_or_stats(self) -> None:
        state = RunState()
        line = state.summary_line()
        assert "tokens —" in line


@pytest.fixture()
def piped_mode(monkeypatch):
    """Force the 'not attached to a TTY' code path."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    monkeypatch.delenv(ENV_TOKEN_BUDGET, raising=False)
    return None


class TestPipeMode:
    def test_plain_lines_without_ansi(self, piped_mode, capsys) -> None:
        view = LiveRunView(token_budget=5_000)
        view.sink("iteration", "Iteration 1/∞")
        view.sink("test", "exit_code=1 success=False\nFAILED tests/test_app.py")
        out = capsys.readouterr().out
        assert "[iteration] Iteration 1/∞" in out
        assert "[test] exit_code=1 success=False" in out  # first line only
        assert "\x1b[" not in out  # no ANSI escapes: CI logs stay clean

    def test_state_still_tracked_in_pipe_mode(self, piped_mode) -> None:
        view = LiveRunView(token_budget=5_000)
        view.sink("iteration", "Iteration 3/7")
        view.sink("test", "exit_code=0 success=True")
        assert view.state.iteration_label == "3/7"
        assert view.state.last_verdict is True

    def test_env_token_budget_fallback(self, piped_mode, monkeypatch) -> None:
        monkeypatch.setenv(ENV_TOKEN_BUDGET, "12345")
        view = LiveRunView()
        assert view.state.token_budget == 12345

    def test_invalid_env_budget_is_ignored(self, piped_mode, monkeypatch) -> None:
        monkeypatch.setenv(ENV_TOKEN_BUDGET, "not-a-number")
        view = LiveRunView()
        assert view.state.token_budget is None

    def test_sink_never_raises_on_bad_events(self, piped_mode, capsys) -> None:
        view = LiveRunView()
        view.sink("", "")           # empty stage/message must not crash
        view.sink("unknown", None)  # type: ignore[arg-type]  # unusable payload
        # Unusable payloads never propagate and never spam stderr.
        captured = capsys.readouterr()
        assert "[unknown]" in captured.out
        assert "[live-ui] render error" not in captured.err


class TestInteractiveMode:
    @pytest.fixture()
    def tty_mode(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        return None

    def test_live_panel_receives_updates(self, tty_mode) -> None:
        with mock.patch("rich.live.Live") as live_cls:
            instance = live_cls.return_value
            view = LiveRunView(token_budget=1_000)
            view.start()
            view.sink("iteration", "Iteration 2/4")
            view.sink("test", "exit_code=0 success=True")
            view.finish()

        live_cls.assert_called_once()               # panel created once
        instance.start.assert_called_once()
        assert instance.update.call_count == 2      # refreshed per milestone
        instance.stop.assert_called_once()

    def test_start_is_idempotent_and_finish_safe(self, tty_mode) -> None:
        with mock.patch("rich.live.Live") as live_cls:
            view = LiveRunView()
            view.start()
            view.start()  # second call must not create a second panel
            view.finish()
            view.finish()  # double finish must not raise
            assert live_cls.call_count == 1

    def test_no_panel_when_piped_even_after_start(self, piped_mode) -> None:
        with mock.patch("rich.live.Live") as live_cls:
            view = LiveRunView()
            view.start()
            view.sink("context", "Documentation and sources collected.")
            view.finish()
            live_cls.assert_not_called()


# ---------------------------------------------------------------------------
# End-to-end + TUI
# ---------------------------------------------------------------------------
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


class TestLoopIntegration:
    def test_milestones_stream_through_view(self, tmp_path: Path, capsys) -> None:
        """The orchestrator's event_sink feeds LiveRunView end-to-end."""
        _write(tmp_path, "src/app.py", "def add(a, b):\n    return a - b\n")
        patch = Patch(files=[{
            "file_path": "src/app.py",
            "edits": [{"find": "    return a - b\n", "replace": "    return a + b\n"}],
        }])

        def _ok_run_tests(targets=None):
            from src.sandbox.runner import TestResult

            return TestResult(success=True, stdout="exit_code=0 success=True", exit_code=0)

        with (
            mock.patch("sys.stdout.isatty", return_value=False),
            mock.patch("src.orchestrator.diagnose", return_value=_diagnosis()),
            mock.patch("src.orchestrator.generate_patch", return_value=patch),
            mock.patch("src.sandbox.runner.SandboxRunner.run_tests", side_effect=_ok_run_tests),
            mock.patch("src.orchestrator.generate_report", return_value=_report()),
        ):
            from src.orchestrator import run_patchcraft_loop

            view = LiveRunView(token_budget=10_000)
            result = run_patchcraft_loop(
                str(tmp_path), "Fix sign", model="mock", event_sink=view.sink,
            )

        assert result.success is True
        out = capsys.readouterr().out
        for expected in ("[start]", "[context]", "[diagnosis]", "[iteration]", "[done]"):
            assert expected in out
        assert "\x1b[" not in out
        view.finish()

    def test_runstats_registry_active_during_loop(self, tmp_path: Path) -> None:
        """begin_run() creates the token window consumed by live views."""
        _write(tmp_path, "src/app.py", "x = 1\n")

        class _NoopDiagnosis:
            summary = "s"
            root_cause = "r"
            affected_files: list[str] = []
            confidence = 1.0

            def model_dump_json(self, indent=None) -> str:
                return "{}"

        with (
            mock.patch("src.orchestrator.diagnose", return_value=_NoopDiagnosis()),
            mock.patch("src.orchestrator.generate_patch",
                       return_value=Patch(files=[])),
            mock.patch(
                "src.sandbox.runner.SandboxRunner.run_tests",
                return_value=mock.Mock(success=False, exit_code=1, stdout="", stderr="",
                                       missing_dependency=None, subset="full"),
            ),
        ):
            from src.orchestrator import run_patchcraft_loop

            run_patchcraft_loop(str(tmp_path), "Fix", model="mock", max_retries=1)

        stats = runstats.current_run()
        assert stats is not None  # window opened by the loop
        assert stats.total == 0   # mocked agents never report usage


class TestTuiFooter:
    def test_footer_mirrors_pipeline_state(self) -> None:
        pytest.importorskip("textual")
        import asyncio

        from textual.widgets import Static

        from src.gui.app import PatchCraftApp
        from src.gui.pipeline import PipelineEvent

        async def scenario() -> None:
            app = PatchCraftApp(load_credits_on_mount=False)
            async with app.run_test(size=(110, 42)) as pilot:
                with mock.patch.object(Static, "update") as update_mock:
                    app._handle_event(PipelineEvent(stage="iteration",
                                                    message="Iteration 2/3"))
                    app._handle_event(PipelineEvent(stage="test",
                                                    message="exit_code=1 success=False"))
                    await pilot.pause()

                rendered = [str(c.args[0]) for c in update_mock.call_args_list]
                assert any("iter 2/3" in text for text in rendered)
                assert any("FAIL" in text for text in rendered)

        asyncio.run(scenario())
