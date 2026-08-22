"""PatchCraft — CLI entry point (click + rich)."""
from __future__ import annotations

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt
from rich.table import Table

from config.settings import Settings
from src.core.credits import render_credits_panel
from src.core.llm import MaxRetriesExceeded, call_llm

CONSOLE = Console()


@click.group()
@click.option(
    "--model", "-m",
    default=None,
    help="Primary LLM model (default: PATCHCRAFT_PRIMARY_MODEL or deepseek/deepseek-chat).",
)
@click.pass_context
def cli(ctx: click.Context, model: str | None) -> None:
    """PatchCraft: AI agents that diagnose bugs and generate patches."""
    ctx.ensure_object(dict)
    settings = Settings.from_env()
    ctx.obj["settings"] = settings
    ctx.obj["model"] = model or settings.primary_model


@cli.command("ask")
@click.argument("prompt")
@click.option("--system", default="You are a helpful assistant.", show_default=True)
@click.pass_obj
def ask(obj: dict, prompt: str, system: str) -> None:
    """Query the primary LLM with `PROMPT` (automatic fallback included)."""
    render_credits_panel(CONSOLE)  # non-blocking widget at CLI startup
    model = obj["model"]
    try:
        with CONSOLE.status(f"Querying {model} ..."):
            answer = call_llm(provider_model=model, system_prompt=system, user_prompt=prompt)
    except MaxRetriesExceeded as exc:
        CONSOLE.print(Panel(str(exc), title="[bold red]LLM unavailable[/]", border_style="red"))
        raise SystemExit(1) from exc
    CONSOLE.print(Panel(answer, title=f"[bold green]{model}[/]"))


@cli.command("run")
@click.argument("repo_path")
@click.argument("issue")
@click.option("--max-retries", "max_retries", default=-1, show_default=True,
              help="Hard cap on patch+test iterations. -1 = run until tests "
                   "pass (loop detection and budgets still apply).")
@click.option("--token-budget", default=None, type=int,
              help="Max total LLM tokens per task (env PATCHCRAFT_TOKEN_BUDGET).")
@click.option("--time-budget", "time_budget", default=None, type=float,
              help="Wall-clock budget in seconds per task (env PATCHCRAFT_TIME_BUDGET).")
@click.option("--min-credits", "min_credits", default=None, type=float,
              help="Halt if OpenRouter remaining credits drop below this value "
                   "(env PATCHCRAFT_MIN_CREDITS).")
@click.pass_obj
def run(
    obj: dict,
    repo_path: str,
    issue: str,
    max_retries: int,
    token_budget: int | None,
    time_budget: float | None,
    min_credits: float | None,
) -> None:
    """Run the PatchCraft pipeline on REPO_PATH with the given ISSUE.

    REPO_PATH is the target project directory; ISSUE is the bug
    description to fix (wrap it in quotes).
    """
    from src.orchestrator import run_patchcraft_loop

    render_credits_panel(CONSOLE)  # non-blocking widget at pipeline start
    try:
        result = run_patchcraft_loop(
            repo_path=repo_path,
            issue_description=issue,
            model=obj["model"],
            max_retries=max_retries if max_retries > 0 else None,
            token_budget=token_budget,
            time_budget_seconds=time_budget,
            min_remaining_credits=min_credits,
        )
    except (NotADirectoryError, ValueError) as exc:
        CONSOLE.print(Panel(str(exc), title="[bold red]Configuration error[/]", border_style="red"))
        raise SystemExit(1) from exc

    if not result.success:
        if result.halt_reason:
            CONSOLE.print(f"[red]Halted:[/] {result.halt_reason}")
        raise SystemExit(1)


@cli.command("select")
@click.argument("github_repo")
@click.argument("local_repo_path")
@click.option("--label", default="bug", show_default=True,
              help="GitHub label used to filter open issues.")
@click.option("--limit", default=10, show_default=True,
              help="Maximum number of issues to display (1-100).")
@click.option("--max-retries", "max_retries", default=-1, show_default=True,
              help="Hard cap on patch+test iterations. -1 = run until tests "
                   "pass (loop detection and budgets still apply).")
@click.option("--token-budget", default=None, type=int,
              help="Max total LLM tokens per task (env PATCHCRAFT_TOKEN_BUDGET).")
@click.option("--time-budget", "time_budget", default=None, type=float,
              help="Wall-clock budget in seconds per task (env PATCHCRAFT_TIME_BUDGET).")
@click.option("--min-credits", "min_credits", default=None, type=float,
              help="Halt if OpenRouter remaining credits drop below this value "
                   "(env PATCHCRAFT_MIN_CREDITS).")
@click.pass_obj
def select(
    obj: dict,
    github_repo: str,
    local_repo_path: str,
    label: str,
    limit: int,
    max_retries: int,
    token_budget: int | None,
    time_budget: float | None,
    min_credits: float | None,
) -> None:
    """Pick a GitHub issue (GITHUB_REPO) and solve it in a local repo.

    GITHUB_REPO is in owner/repo format (e.g. langchain-ai/langgraph);
    LOCAL_REPO_PATH is the local project directory where the patch applies.
    """
    from src.github.issue_fetcher import GitHubAPIError, get_open_issues
    from src.orchestrator import run_patchcraft_loop

    render_credits_panel(CONSOLE)  # non-blocking widget at CLI startup
    try:
        with CONSOLE.status(f"Fetching issues for [bold]{github_repo}[/] ..."):
            issues = get_open_issues(github_repo, label=label, limit=limit)
    except ValueError as exc:
        CONSOLE.print(Panel(str(exc), title="[bold red]Invalid parameters[/]", border_style="red"))
        raise SystemExit(1) from exc
    except GitHubAPIError as exc:
        CONSOLE.print(Panel(str(exc), title="[bold red]GitHub error[/]", border_style="red"))
        raise SystemExit(1) from exc

    if not issues:
        CONSOLE.print(Panel(
            f"No open issues with label '[bold yellow]{label}[/]' for "
            f"[bold]{github_repo}[/].",
            title="[bold yellow]No issues found[/]",
            border_style="yellow",
        ))
        raise SystemExit(1)

    table = Table(
        title=f"Open issues with label '{label}' — {github_repo}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("GitHub ID", justify="right")
    table.add_column("Title")
    for i, issue in enumerate(issues, start=1):
        table.add_row(str(i), str(issue.get("number", "?")), issue.get("title", ""))
    CONSOLE.print(table)

    choice = IntPrompt.ask(
        "[bold]Select the issue number to solve[/]",
        choices=[str(i) for i in range(1, len(issues) + 1)],
    )
    selected = issues[choice - 1]
    title = selected.get("title", "")
    body = selected.get("body") or ""
    issue_description = f"Title: {title}\n\nBody:\n{body}"
    CONSOLE.print(f"[green]Selected:[/] #{selected.get('number', '?')} — {title}")

    try:
        run_patchcraft_loop(
            repo_path=local_repo_path,
            issue_description=issue_description,
            model=obj["model"],
            max_retries=max_retries if max_retries > 0 else None,
            token_budget=token_budget,
            time_budget_seconds=time_budget,
            min_remaining_credits=min_credits,
        )
    except (NotADirectoryError, ValueError) as exc:
        CONSOLE.print(Panel(str(exc), title="[bold red]Configuration error[/]", border_style="red"))
        raise SystemExit(1) from exc


@cli.command("gui")
@click.option("--max-retries", "max_retries", default=-1, show_default=True,
              help="Hard cap on iterations in the GUI pipeline. -1 = run until "
                   "tests pass (loop detection and budgets still apply).")
@click.pass_obj
def gui(obj: dict, max_retries: int) -> None:
    """Launch the PatchCraft interactive TUI (Textual)."""
    from src.gui import launch_gui

    launch_gui(model=obj["model"], max_retries=max_retries if max_retries > 0 else None)


if __name__ == "__main__":
    cli(prog_name="patchcraft")