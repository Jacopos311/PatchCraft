"""Tests for the PatchCraft Textual GUI (headless smoke + pipeline bridge)."""
from __future__ import annotations

import asyncio
from unittest import mock

import pytest

pytest.importorskip("textual")

from rich.console import Console  # noqa: E402

from src.core.credits import CreditsError  # noqa: E402
from src.gui import pipeline  # noqa: E402
from src.gui.app import PatchCraftApp  # noqa: E402


class TestCreditsBridge:
    def test_format_line_with_limit(self) -> None:
        line = pipeline.format_credits_line({"usage": 30.0, "limit": 100.0})
        assert "$30" in line
        assert "of $100" in line
        assert "%" in line

    def test_format_line_free_tier(self) -> None:
        line = pipeline.format_credits_line(
            {"usage": 7.25, "limit": None, "is_free_tier": True}
        )
        assert "$7.25" in line
        assert "Free tier" in line

    def test_format_line_no_key(self) -> None:
        assert "no API key" in pipeline.format_credits_line(None)

    def test_snapshot_swallows_errors(self) -> None:
        with mock.patch.object(pipeline, "fetch_credits", side_effect=CreditsError("boom")):
            assert pipeline.credits_snapshot() is None


class TestHeadlessApp:
    @staticmethod
    def _run(coro):
        return asyncio.run(coro)

    def test_app_mounts_headless(self) -> None:
        """The TUI mounts headlessly with all key widgets present."""

        async def scenario() -> None:
            app = PatchCraftApp(load_credits_on_mount=False)
            async with app.run_test(size=(110, 42)) as pilot:
                await pilot.pause()
                assert app.query_one("#credits") is not None
                assert app.query_one("#repo-path") is not None
                assert app.query_one("#issue-select") is not None
                assert app.query_one("#diff") is not None
                assert app.query_one("#report") is not None

        self._run(scenario())

    def test_app_streams_mocked_pipeline(self) -> None:
        """A mocked pipeline streams logs, diff, report and green status."""
        from src.agents.reporter import PatchReport
        from src.orchestrator import RunResult
        from textual.widgets import Static, TextArea

        fake_result = RunResult(
            success=True,
            iterations=1,
            report=PatchReport(title="T", summary="S", diff="D", pr_markdown="M"),
            files_changed=["a.py"],
        )

        def fake_run(*args, **kwargs):
            on_event = kwargs["on_event"]
            on_event(pipeline.PipelineEvent(stage="diagnosis", message="found bug"))
            on_event(pipeline.PipelineEvent(stage="patch", message="applied a.py"))
            on_event(pipeline.PipelineEvent(stage="test", message="exit_code=0 success=True"))
            on_event(pipeline.PipelineEvent(stage="diff", message="+ new line"))
            on_event(pipeline.PipelineEvent(stage="report", message="# PR markdown"))
            on_event(pipeline.PipelineEvent(stage="done", message="ok"))
            return fake_result

        async def scenario() -> None:
            app = PatchCraftApp(load_credits_on_mount=False)
            async with app.run_test(size=(120, 46)) as pilot:
                with mock.patch("src.gui.app.run_pipeline", side_effect=fake_run):
                    app._run_pipeline_worker("./repo", "Title: x\nBody: y", "m/x", 1)
                    for _ in range(200):
                        await pilot.pause()
                        if "Success" in app.status_text:
                            break
                    assert "✅ Success in 1 iteration(s)" in app.status_text
                    diff_box = app.query_one("#diff", TextArea)
                    report_box = app.query_one("#report", TextArea)
                    assert "+ new line" in diff_box.text
                    assert "# PR markdown" in report_box.text
                    assert app.query_one("#start").disabled is False

        self._run(scenario())