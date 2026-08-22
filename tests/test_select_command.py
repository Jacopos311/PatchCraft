"""Test del comando Click `select` di main.py (con mock)."""
from __future__ import annotations

from unittest import mock

from click.testing import CliRunner

from main import cli
from src.agents.reporter import PatchReport
from src.orchestrator import RunResult


def _issue(number: int, title: str = "bug") -> dict:
    return {"number": number, "title": title, "body": f"corpo {number}", "html_url": f"/{number}"}


def test_select_success_pipeline() -> None:
    """Selezione 2 → run_patchcraft_loop riceve titolo+body e repo locale."""
    issues = [_issue(12), _issue(27, "secondo")]

    with mock.patch(
        "src.orchestrator.run_patchcraft_loop",
        return_value=RunResult(
            success=True,
            iterations=1,
            report=PatchReport(title="t", summary="s", diff="d", pr_markdown="m"),
        ),
    ) as loop_mock, mock.patch(
        "src.github.issue_fetcher.get_open_issues", return_value=issues
    ), mock.patch("rich.prompt.IntPrompt.ask", return_value=2):
        result = CliRunner().invoke(cli, ["select", "owner/repo", "./local"])

    assert result.exit_code == 0
    assert loop_mock.call_count == 1
    _, kwargs = loop_mock.call_args
    assert kwargs["repo_path"] == "./local"
    # default is now the goal-driven loop: no arbitrary retry cut-off
    assert kwargs["max_retries"] is None
    assert "Title: secondo" in kwargs["issue_description"]
    assert "corpo 27" in kwargs["issue_description"]


def test_select_empty_issues_exits() -> None:
    """Lista vuota → panel di errore e SystemExit(1)."""
    with mock.patch("src.github.issue_fetcher.get_open_issues", return_value=[]):
        result = CliRunner().invoke(cli, ["select", "owner/repo", "./local"])

    assert result.exit_code == 1
    assert "No issues found" in result.output


def test_select_github_error_exits() -> None:
    """GitHubAPIError → panel di errore e SystemExit(1)."""
    from src.github.issue_fetcher import GitHubAPIError

    with mock.patch(
        "src.github.issue_fetcher.get_open_issues",
        side_effect=GitHubAPIError("Repository 'x' not found (404)."),
    ):
        result = CliRunner().invoke(cli, ["select", "owner/repo", "./local"])

    assert result.exit_code == 1
    assert "not found" in result.output