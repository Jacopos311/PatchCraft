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

# ---------------------------------------------------------------------------
# Exit codes documented for CI use (README > "Exit codes").
# ---------------------------------------------------------------------------
EXIT_OK = 0              # tests green, pipeline converged
EXIT_NO_CONVERGENCE = 1  # pipeline ran but did not solve the issue
EXIT_CONFIG_ERROR = 2    # invalid configuration / bad input
EXIT_BUDGET_HALT = 3     # halted by a guardrail (token/time/credit/iteration)

_BUDGET_HALT_MARKERS = ("budget", "iteration limit", "credits below")


def _halt_exit_code(result) -> int:
    """Map a finished pipeline run to its documented exit code."""
    if result.success:
        return EXIT_OK
    reason = (getattr(result, "halt_reason", None) or "").lower()
    if any(marker in reason for marker in _BUDGET_HALT_MARKERS):
        return EXIT_BUDGET_HALT
    return EXIT_NO_CONVERGENCE


def _load_repo_config(repo_path: str):
    """Load ``<repo>/.patchcraft.yml``; exit 2 with a clear message on error."""
    from src.core.config import ConfigError, load_config_with_warnings

    try:
        config, warnings = load_config_with_warnings(repo_path)
    except ConfigError as exc:
        CONSOLE.print(Panel(str(exc), title="[bold red]Configuration error[/]", border_style="red"))
        raise SystemExit(EXIT_CONFIG_ERROR) from exc
    for warning in warnings:
        CONSOLE.print(f"[yellow]⚠ config:[/] {warning}")
    return config


def _resolve_model(obj: dict, config) -> str:
    """CLI -m flag wins over ``model:`` in .patchcraft.yml over env/default."""
    if obj.get("model_explicit"):
        return obj["model"]
    return config.model or obj["model"]


def _apply_fallback_models(config) -> None:
    """Apply ``fallback_models`` from the config file (Step 3.3)."""
    from src.core.llm import set_default_fallback_chain

    set_default_fallback_chain(config.fallback_models)


def _print_git_summary(result) -> None:
    """Show the deliverable branch/commit when the git workflow ran (Step 4.1)."""
    branch = getattr(result, "git_branch", None)
    if not branch:
        return
    sha = getattr(result, "commit_sha", None)
    CONSOLE.print(
        f"[bold green]Branch:[/] {branch}" + (f" · commit {sha[:10]}" if sha else "")
    )


@click.group()
@click.option(
    "--model", "-m",
    default=None,
    help="Primary LLM model (default: PATCHCRAFT_PRIMARY_MODEL or deepseek/deepseek-chat).",
)
@click.option(
    "--yes", "-y", "yes",
    is_flag=True, default=False, show_default=True,
    help="Skip interactive confirmations (headless/CI runs).",
)
@click.pass_context
def cli(ctx: click.Context, model: str | None, yes: bool) -> None:
    """PatchCraft: AI agents that diagnose bugs and generate patches."""
    ctx.ensure_object(dict)
    settings = Settings.from_env()
    ctx.obj["settings"] = settings
    ctx.obj["model"] = model or settings.primary_model
    ctx.obj["model_explicit"] = model is not None
    ctx.obj["yes"] = yes



@cli.command("ask")
@click.argument("prompt")
@click.option("--system", default="You are a helpful assistant.", show_default=True)
@click.option("--no-cache", "no_cache", is_flag=True, default=False,
              show_default=True,
              help="Disable the LLM memo cache (env PATCHCRAFT_NO_CACHE).")
@click.pass_obj
def ask(obj: dict, prompt: str, system: str, no_cache: bool) -> None:
    """Query the primary LLM with `PROMPT` (automatic fallback included)."""
    from src.core.cache import configure_memo_cache, get_memo_cache

    render_credits_panel(CONSOLE)  # non-blocking widget at CLI startup
    # Scope the memo-cache configuration to THIS command only (Step 3.1):
    # the previous process-wide state is restored afterwards.
    memo = get_memo_cache()
    saved_enabled, saved_base_dir = memo.enabled, memo.base_dir
    configure_memo_cache(enabled=not no_cache)
    model = obj["model"]
    try:
        with CONSOLE.status(f"Querying {model} ..."):
            answer = call_llm(provider_model=model, system_prompt=system, user_prompt=prompt)
    except MaxRetriesExceeded as exc:
        CONSOLE.print(Panel(str(exc), title="[bold red]LLM unavailable[/]", border_style="red"))
        raise SystemExit(1) from exc
    finally:
        memo.enabled, memo.base_dir = saved_enabled, saved_base_dir
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
@click.option("--auto-install", "auto_install", is_flag=True, default=False,
              show_default=True,
              help="Install missing dependencies once and retry tests when a "
                   "dependency error is detected (default off).")
@click.option("--no-cache", "no_cache", is_flag=True, default=False,
              show_default=True,
              help="Disable the caching layer: LLM memo cache and "
                   "targeted-test verdict cache (env PATCHCRAFT_NO_CACHE).")
@click.option("--allow-dirty", "allow_dirty", is_flag=True, default=False,
              show_default=True,
              help="Allow running even if the git working tree is not clean.")
@click.option("--issue-number", "issue_number", default=None, type=int,
              help="GitHub issue number being solved (used for the branch name "
                   "and the 'Fixes #N' commit trailer).")
@click.option("--issue-title", "issue_title", default=None,
              help="Issue title used for the patchcraft/* branch name.")
@click.pass_obj
def run(
    obj: dict,
    repo_path: str,
    issue: str,
    max_retries: int,
    token_budget: int | None,
    time_budget: float | None,
    min_credits: float | None,
    auto_install: bool,
    no_cache: bool,
    allow_dirty: bool,
    issue_number: int | None,
    issue_title: str | None,
) -> None:
    """Run the PatchCraft pipeline on REPO_PATH with the given ISSUE.

    REPO_PATH is the target project directory; ISSUE is the bug
    description to fix (wrap it in quotes).
    """
    from src.core.gitflow import GitSafetyError
    from src.gui.live_panel import LiveRunView
    from src.orchestrator import run_patchcraft_loop

    # Step 3.3: configuration file provides the defaults; CLI flags win.
    cfg = _load_repo_config(repo_path)
    effective_model = _resolve_model(obj, cfg)
    _apply_fallback_models(cfg)
    effective_token_budget = token_budget if token_budget is not None else cfg.token_budget
    effective_time_budget = time_budget if time_budget is not None else cfg.time_budget
    effective_min_credits = min_credits if min_credits is not None else cfg.min_credits
    effective_max_retries = (
        max_retries if max_retries > 0 else (cfg.max_retries or None)
    )

    render_credits_panel(CONSOLE)  # non-blocking widget at pipeline start
    # Step 3.2: live milestone panel (rich Live on TTY, plain lines when piped).
    live_view = LiveRunView(token_budget=effective_token_budget)
    live_view.start()
    try:
        result = run_patchcraft_loop(
            repo_path=repo_path,
            issue_description=issue,
            model=effective_model,
            max_retries=effective_max_retries,
            token_budget=effective_token_budget,
            time_budget_seconds=effective_time_budget,
            min_remaining_credits=effective_min_credits,
            auto_install=auto_install,
            use_cache=not no_cache,
            event_sink=live_view.sink,
            issue_number=issue_number,
            issue_title=issue_title,
            allow_dirty=allow_dirty,
        )
    except (NotADirectoryError, ValueError) as exc:
        CONSOLE.print(Panel(str(exc), title="[bold red]Configuration error[/]", border_style="red"))
        raise SystemExit(EXIT_CONFIG_ERROR) from exc
    except GitSafetyError as exc:
        CONSOLE.print(Panel(str(exc), title="[bold red]Git safety stop[/]", border_style="red"))
        raise SystemExit(EXIT_CONFIG_ERROR) from exc
    finally:
        live_view.finish()

    _print_git_summary(result)

    if not result.success:
        if result.halt_reason:
            CONSOLE.print(f"[red]Halted:[/] {result.halt_reason}")
        raise SystemExit(_halt_exit_code(result))


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
@click.option("--auto-install", "auto_install", is_flag=True, default=False,
              show_default=True,
              help="Install missing dependencies once and retry tests when a "
                   "dependency error is detected (default off).")
@click.option("--no-cache", "no_cache", is_flag=True, default=False,
              show_default=True,
              help="Disable the caching layer: LLM memo cache and "
                   "targeted-test verdict cache (env PATCHCRAFT_NO_CACHE).")
@click.option("--allow-dirty", "allow_dirty", is_flag=True, default=False,
              show_default=True,
              help="Allow running even if the git working tree is not clean.")
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
    auto_install: bool,
    no_cache: bool,
    allow_dirty: bool,
) -> None:
    """Pick a GitHub issue (GITHUB_REPO) and solve it in a local repo.

    GITHUB_REPO is in owner/repo format (e.g. langchain-ai/langgraph);
    LOCAL_REPO_PATH is the local project directory where the patch applies.
    """
    from src.core.gitflow import GitSafetyError
    from src.github.issue_fetcher import GitHubAPIError, get_open_issues
    from src.orchestrator import run_patchcraft_loop

    # Step 3.3: configuration file provides the defaults; CLI flags win.
    cfg = _load_repo_config(local_repo_path)
    effective_model = _resolve_model(obj, cfg)
    _apply_fallback_models(cfg)
    effective_token_budget = token_budget if token_budget is not None else cfg.token_budget
    effective_time_budget = time_budget if time_budget is not None else cfg.time_budget
    effective_min_credits = min_credits if min_credits is not None else cfg.min_credits
    effective_max_retries = (
        max_retries if max_retries > 0 else (cfg.max_retries or None)
    )

    render_credits_panel(CONSOLE)  # non-blocking widget at CLI startup
    try:
        with CONSOLE.status(f"Fetching issues for [bold]{github_repo}[/] ..."):
            issues = get_open_issues(github_repo, label=label, limit=limit)
    except ValueError as exc:
        CONSOLE.print(Panel(str(exc), title="[bold red]Invalid parameters[/]", border_style="red"))
        raise SystemExit(EXIT_CONFIG_ERROR) from exc
    except GitHubAPIError as exc:
        CONSOLE.print(Panel(str(exc), title="[bold red]GitHub error[/]", border_style="red"))
        raise SystemExit(EXIT_CONFIG_ERROR) from exc

    if not issues:
        CONSOLE.print(Panel(
            f"No open issues with label '[bold yellow]{label}[/]' for "
            f"[bold]{github_repo}[/].",
            title="[bold yellow]No issues found[/]",
            border_style="yellow",
        ))
        raise SystemExit(EXIT_NO_CONVERGENCE)

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

    if obj.get("yes"):
        # --yes (Step 3.3): headless run on the first listed issue.
        choice_index = 0
        CONSOLE.print("[green]--yes:[/] solving the first listed issue.")
    else:
        choice = IntPrompt.ask(
            "[bold]Select the issue number to solve[/]",
            choices=[str(i) for i in range(1, len(issues) + 1)],
        )
        choice_index = choice - 1
    selected = issues[choice_index]
    title = selected.get("title", "")
    body = selected.get("body") or ""
    issue_description = f"Title: {title}\n\nBody:\n{body}"
    CONSOLE.print(f"[green]Selected:[/] #{selected.get('number', '?')} — {title}")

    try:
        from src.gui.live_panel import LiveRunView

        live_view = LiveRunView(token_budget=effective_token_budget)
        live_view.start()
    except Exception:  # noqa: BLE001 - the view is optional sugar
        live_view = None
    try:
        result = run_patchcraft_loop(
            repo_path=local_repo_path,
            issue_description=issue_description,
            model=effective_model,
            max_retries=effective_max_retries,
            token_budget=effective_token_budget,
            time_budget_seconds=effective_time_budget,
            min_remaining_credits=effective_min_credits,
            auto_install=auto_install,
            use_cache=not no_cache,
            event_sink=live_view.sink if live_view else None,
            issue_number=selected.get("number"),
            issue_title=title or None,
            allow_dirty=allow_dirty,
            github_repo=github_repo,
        )
    except (NotADirectoryError, ValueError) as exc:
        CONSOLE.print(Panel(str(exc), title="[bold red]Configuration error[/]", border_style="red"))
        raise SystemExit(EXIT_CONFIG_ERROR) from exc
    except GitSafetyError as exc:
        CONSOLE.print(Panel(str(exc), title="[bold red]Git safety stop[/]", border_style="red"))
        raise SystemExit(EXIT_CONFIG_ERROR) from exc
    finally:
        if live_view is not None:
            live_view.finish()

    _print_git_summary(result)
    if not result.success:
        if result.halt_reason:
            CONSOLE.print(f"[red]Halted:[/] {result.halt_reason}")
        raise SystemExit(_halt_exit_code(result))


@cli.command("fix")
@click.option("--repo", "-r", "github_repo", default=None,
              help="owner/repo of the GitHub project. Optional when ISSUE_REF "
                   "is a full URL (the repository is derived from it); "
                   "REQUIRED for bare issue numbers.")
@click.argument("issue_ref")
@click.argument("local_repo_path")
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
@click.option("--auto-install", "auto_install", is_flag=True, default=False,
              show_default=True,
              help="Install missing dependencies once and retry tests when a "
                   "dependency error is detected (default off).")
@click.option("--no-cache", "no_cache", is_flag=True, default=False,
              show_default=True,
              help="Disable the caching layer: LLM memo cache and "
                   "targeted-test verdict cache (env PATCHCRAFT_NO_CACHE).")
@click.option("--allow-dirty", "allow_dirty", is_flag=True, default=False,
              show_default=True,
              help="Allow running even if the git working tree is not clean.")
@click.option("--push", "push_to_github", is_flag=True, default=False,
              show_default=True,
              help="After success: push the patchcraft/* branch to origin and "
                   "open/update a pull request (draft per pr.draft, default "
                   "true). Default is local-only.")
@click.pass_obj
def fix(
    obj: dict,
    github_repo: str | None,
    issue_ref: str,
    local_repo_path: str,
    max_retries: int,
    token_budget: int | None,
    time_budget: float | None,
    min_credits: float | None,
    auto_install: bool,
    no_cache: bool,
    allow_dirty: bool,
    push_to_github: bool,
) -> None:
    """Solve ISSUE_REF headlessly in LOCAL_REPO_PATH.

    ISSUE_REF is a GitHub issue URL (https://github.com/owner/repo/issues/123
    — then --repo may be omitted) or a bare number (123 or #123 — then
    --repo owner/repo is required).

    Exit codes: 0 success · 1 no convergence · 2 config error · 3 budget halt.
    """
    from src.core.config import ConfigError, resolve_issue_reference
    from src.core.gitflow import GitSafetyError
    from src.github.issue_fetcher import GitHubAPIError, get_issue
    from src.orchestrator import run_patchcraft_loop

    try:
        url_repo, number = resolve_issue_reference(issue_ref)
    except ConfigError as exc:
        CONSOLE.print(Panel(str(exc), title="[bold red]Invalid issue reference[/]", border_style="red"))
        raise SystemExit(EXIT_CONFIG_ERROR) from exc

    repo = github_repo or url_repo
    if not repo:
        CONSOLE.print(Panel(
            "A bare issue number requires --repo owner/repo; alternatively "
            "pass the full issue URL.",
            title="[bold red]Missing repository[/]",
            border_style="red",
        ))
        raise SystemExit(EXIT_CONFIG_ERROR)

    # Step 3.3: configuration file provides the defaults; CLI flags win.
    cfg = _load_repo_config(local_repo_path)
    effective_model = _resolve_model(obj, cfg)
    _apply_fallback_models(cfg)
    effective_token_budget = token_budget if token_budget is not None else cfg.token_budget
    effective_time_budget = time_budget if time_budget is not None else cfg.time_budget
    effective_min_credits = min_credits if min_credits is not None else cfg.min_credits
    effective_max_retries = (
        max_retries if max_retries > 0 else (cfg.max_retries or None)
    )

    render_credits_panel(CONSOLE)  # non-blocking widget at pipeline start
    try:
        with CONSOLE.status(f"Fetching issue #{number} of [bold]{repo}[/] ..."):
            issue = get_issue(repo, number)
    except (GitHubAPIError, ValueError) as exc:
        CONSOLE.print(Panel(str(exc), title="[bold red]GitHub error[/]", border_style="red"))
        raise SystemExit(EXIT_CONFIG_ERROR) from exc

    title = issue.get("title", "")
    body = issue.get("body") or ""
    issue_description = f"Title: {title}\n\nBody:\n{body}"
    CONSOLE.print(f"[green]Selected:[/] #{issue.get('number', '?')} — {title}")

    # Headless by design (--yes is implied): there are no prompts here.
    from src.gui.live_panel import LiveRunView

    live_view = LiveRunView(token_budget=effective_token_budget)
    live_view.start()
    try:
        result = run_patchcraft_loop(
            repo_path=local_repo_path,
            issue_description=issue_description,
            model=effective_model,
            max_retries=effective_max_retries,
            token_budget=effective_token_budget,
            time_budget_seconds=effective_time_budget,
            min_remaining_credits=effective_min_credits,
            auto_install=auto_install,
            use_cache=not no_cache,
            event_sink=live_view.sink,
            issue_number=int(issue.get("number")) if issue.get("number") is not None else None,
            issue_title=title or None,
            allow_dirty=allow_dirty,
            github_repo=repo,
        )
    except (NotADirectoryError, ValueError) as exc:
        CONSOLE.print(Panel(str(exc), title="[bold red]Configuration error[/]", border_style="red"))
        raise SystemExit(EXIT_CONFIG_ERROR) from exc
    except GitSafetyError as exc:
        CONSOLE.print(Panel(str(exc), title="[bold red]Git safety stop[/]", border_style="red"))
        raise SystemExit(EXIT_CONFIG_ERROR) from exc
    finally:
        live_view.finish()

    _print_git_summary(result)

    # -- Step 4.2: opt-in pull-request publishing ---------------------------
    if push_to_github and result.success and result.git_branch:
        if result.commit_sha is None:
            CONSOLE.print("[yellow]--push skipped:[/] no commit was created.")
        else:
            report = result.report
            pr_title = getattr(report, "title", None) or title or f"Fix issue #{number}"
            pr_body = (
                getattr(report, "pr_markdown", None)
                or f"Automated PatchCraft fix.\n\nFiles changed:\n"
                + "\n".join(f"- {f}" for f in result.files_changed)
            )
            try:
                from src.github.pr_publisher import publish_pr

                with CONSOLE.status("Publishing pull request ..."):
                    pr_url = publish_pr(
                        repo=repo,
                        repo_path=local_repo_path,
                        branch=result.git_branch,
                        title=pr_title,
                        body=pr_body,
                        draft=cfg.pr.draft,
                        issue_number=number,
                    )
            except (GitHubAPIError, GitSafetyError) as exc:
                CONSOLE.print(Panel(
                    str(exc),
                    title="[bold red]Publishing failed[/]",
                    border_style="red",
                ))
                raise SystemExit(EXIT_CONFIG_ERROR) from exc
            CONSOLE.print(Panel(
                f"[bold]{pr_url}[/]",
                title="[bold green]🔗 Pull Request ready[/]",
                border_style="green",
            ))

    if not result.success:
        if result.halt_reason:
            CONSOLE.print(f"[red]Halted:[/] {result.halt_reason}")
        raise SystemExit(_halt_exit_code(result))


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