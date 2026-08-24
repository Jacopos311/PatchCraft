"""PatchCraft TUI application built with Textual.

Layout:

┌──────────────────────────────────────────────────────────────┐
│ 💳 Credits │ Local repo / GitHub issue selector │ Start btn  │
│            ├─────────────────────────────────────────────── │
│            │ Status line                                    │
│            │ Tabs: Agent log · PR diff · Report             │
└──────────────────────────────────────────────────────────────┘

The heavy pipeline runs on a background thread worker
(:func:`src.gui.pipeline.run_pipeline`); milestones stream into the UI
through an ``event_sink`` marshalled back to the main thread with
``call_from_thread``.
"""

from __future__ import annotations

from typing import Optional

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from src.gui.live_panel import RunState
from src.gui.pipeline import PipelineEvent, credits_snapshot, format_credits_line, run_pipeline


CREDITS_PLACEHOLDER = "[dim]💳 OpenRouter credits…[/dim]"
DEFAULT_ISSUE_PROMPT = "Select an issue…"

class PatchCraftApp(App[None]):
    """PatchCraft interactive TUI."""

    TITLE = "PatchCraft"
    SUB_TITLE = "AI patch generation studio"

    CSS = """
    #root { height: 1fr; }
    #sidebar { width: 46; padding: 0 1; border-right: solid $primary; }
    #main { padding: 0 1; }
    #credits { height: auto; padding: 0 1; margin-bottom: 1; }
    .lbl { margin-top: 1; }
    Button { width: 100%; margin-bottom: 1; }
    Select { margin-bottom: 1; }
    #status-line { height: 3; content-align: center middle; border: round $accent; margin-bottom: 1; }
    #status-line.pass { border: round $success; }
    #status-line.fail { border: round $error; }
    #pipeline-status { height: 1; padding: 0 1; color: $accent; margin-bottom: 1; }
    RichLog, TextArea { height: 1fr; }
    """

    def __init__(
        self,
        default_model: Optional[str] = None,
        max_retries: Optional[int] = None,
        load_credits_on_mount: bool = True,
    ) -> None:
        super().__init__()
        self.default_model = default_model or "openrouter/deepseek/deepseek-chat"
        self.max_retries = max_retries
        self.load_credits_on_mount = load_credits_on_mount
        self._issues: list[dict] = []
        self._running = False
        # Step 3.2: mirrors stage/iteration/tokens/verdict in the footer line.
        self._run_state: Optional[RunState] = None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="root"):
            with Vertical(id="sidebar"):
                yield Static(CREDITS_PLACEHOLDER, id="credits")
                yield Label("[b]Local repository[/b]", classes="lbl")
                yield Input(placeholder="./path/to/project", id="repo-path")
                yield Label("[b]GitHub repo (owner/repo)[/b]", classes="lbl")
                yield Input(placeholder="e.g. langchain-ai/langgraph", id="gh-repo")
                yield Label("[b]Issue label[/b]", classes="lbl")
                yield Input(value="bug", id="label")
                yield Button("Load GitHub issues", id="load-issues", variant="default")
                yield Select([], prompt=DEFAULT_ISSUE_PROMPT, id="issue-select")
                yield Button("▶  Start pipeline", id="start", variant="primary")
            with Vertical(id="main"):
                yield Static("Ready.", id="status-line")
                yield Static("", id="pipeline-status")
                with TabbedContent(initial="log-tab"):
                    with TabPane("Agent log", id="log-tab"):
                        yield RichLog(id="log", highlight=False, markup=False, wrap=True)
                    with TabPane("PR diff", id="diff-tab"):
                        yield TextArea("", read_only=True, id="diff")
                    with TabPane("PR markdown", id="report-tab"):
                        yield TextArea("", read_only=True, id="report")
        yield Footer()

    def on_mount(self) -> None:
        self._set_status("Ready. Load a local repository and press ▶ to start.", state=None)
        if self.load_credits_on_mount:
            self._refresh_credits_worker()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_status(self, message: str, state: Optional[str] = None) -> None:
        """Update the top status line. ``state`` is None | 'pass' | 'fail'."""
        self.status_text = message
        status = self.query_one("#status-line", Static)
        icon = {"pass": "✅ ", "fail": "❌ "}.get(state, "")
        status.update(icon + message)
        status.remove_class("pass")
        status.remove_class("fail")
        if state:
            status.add_class(state)

    def _append_log(self, stage: str, message: str) -> None:
        log = self.query_one("#log", RichLog)
        tag = {
            "start": "🚀",
            "context": "📚",
            "diagnosis": "🔍",
            "iteration": "🔁",
            "patch": "✍️ ",
            "test": "🧪",
            "error": "❌",
            "diff": "📄",
            "report": "📋",
            "done": "🏁",
        }.get(stage, "•")
        log.write(f"{tag} [{stage}] {message}")

    def _update_pipeline_status(self, stage: str, message: str) -> None:
        """Step 3.2: mirror the run state into the compact footer line."""
        if self._run_state is None:
            self._run_state = RunState()
        try:
            self._run_state.observe(stage, message)
            self.query_one("#pipeline-status", Static).update(
                self._run_state.summary_line()
            )
        except Exception:  # noqa: BLE001 - presentation must never break the UI
            pass

# ------------------------------------------------------------------
    # Credits widget (background thread → call_from_thread)
    # ------------------------------------------------------------------
    @work(thread=True, exclusive=True, group="credits")
    def _refresh_credits_worker(self) -> None:
        snapshot = credits_snapshot()
        line = format_credits_line(snapshot)
        self.call_from_thread(self._update_credits, line)

    def _update_credits(self, line: str) -> None:
        self.query_one("#credits", Static).update(line)

    def refresh_credits(self) -> None:
        """Public entry point to re-fetch the OpenRouter credits."""
        self._refresh_credits_worker()

    # ------------------------------------------------------------------
    # Issue loading
    # ------------------------------------------------------------------
    @work(thread=True, exclusive=True, group="issues")
    def _load_issues_worker(self) -> None:
        from src.github.issue_fetcher import GitHubAPIError, get_open_issues

        gh_repo = self.query_one("#gh-repo", Input).value.strip()
        label = self.query_one("#label", Input).value.strip() or "bug"
        if not gh_repo:
            self.call_from_thread(
                self.notify, "GitHub repo is required (owner/repo).", severity="warning"
            )
            return
        try:
            issues = get_open_issues(gh_repo, label=label, limit=20)
        except (ValueError, GitHubAPIError) as exc:
            self.call_from_thread(self._issues_failed, str(exc))
            return
        self.call_from_thread(self._populate_issues, issues)

    def _populate_issues(self, issues: list[dict]) -> None:
        self._issues = issues
        select = self.query_one("#issue-select", Select)
        options = [
            (f"#{i.get('number', '?')} — {str(i.get('title', ''))[:60]}", idx)
            for idx, i in enumerate(issues)
        ]
        select.set_options(options)
        if not issues:
            self.notify("No open issues found for this label.", severity="warning")
        else:
            self.notify(f"Loaded {len(issues)} open issues.")

    def _issues_failed(self, message: str) -> None:
        self.notify(f"Failed to load issues: {message}", severity="error")


# ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    @work(thread=True, exclusive=True, group="pipeline")
    def _run_pipeline_worker(
        self,
        repo_path: str,
        issue_description: str,
        model: str,
        max_retries: int,
    ) -> None:
        try:
            result = run_pipeline(
                repo_path=repo_path,
                issue_description=issue_description,
                model=model,
                max_retries=max_retries,
                on_event=self._emit_event,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.call_from_thread(self._pipeline_crashed, f"{type(exc).__name__}: {exc}")
            return
        self.call_from_thread(self._pipeline_finished, result)

    def _emit_event(self, event: PipelineEvent) -> None:
        """Called on the worker thread; marshals the event to the UI thread."""
        self.call_from_thread(self._handle_event, event)

    def _handle_event(self, event: PipelineEvent) -> None:
        self._append_log(event.stage, event.message)
        self._update_pipeline_status(event.stage, event.message)
        if event.stage == "test":
            first = event.message.splitlines()[0] if event.message else ""
            if "success=True" in first:
                self._set_status("Tests passed.", state="pass")
            else:
                self._set_status("Running tests…", state=None)
        elif event.stage == "diff":
            self.query_one("#diff", TextArea).text = event.message
        elif event.stage == "report":
            self.query_one("#report", TextArea).text = event.message
        elif event.stage == "diagnosis":
            self._set_status("Diagnosis ready. Generating patch…", state=None)

    def _pipeline_crashed(self, message: str) -> None:
        self._running = False
        self._set_status(f"Pipeline crashed: {message}", state="fail")
        self._append_log("error", message)
        self.query_one("#start", Button).disabled = False

    def _pipeline_finished(self, result) -> None:
        self._running = False
        self.query_one("#start", Button).disabled = False
        if result.success:
            self._set_status(
                f"✅ Success in {result.iterations} iteration(s). "
                f"Files changed: {len(result.files_changed)}.",
                state="pass",
            )
            self._append_log("done", "Pipeline finished successfully.")
            self.query_one(TabbedContent).active = "diff-tab"
        else:
            self._set_status(
                f"❌ Failed after {result.iterations} iteration(s) — changes rolled back.",
                state="fail",
            )
            self._append_log("done", "Pipeline failed; changes were rolled back.")

    # ------------------------------------------------------------------
    # UI events
    # ------------------------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "load-issues":
            self._load_issues_worker()
        elif event.button.id == "start":
            self._start_clicked()

    def _start_clicked(self) -> None:
        if self._running:
            self.notify("A pipeline is already running.", severity="warning")
            return
        repo_path = self.query_one("#repo-path", Input).value.strip()
        selected_index = self.query_one("#issue-select", Select).selection

        issue_description: Optional[str]
        if isinstance(selected_index, int):
            issue = self._issues[selected_index]
            title = issue.get("title", "")
            body = issue.get("body") or ""
            issue_description = f"Title: {title}\n\nBody:\n{body}"
        else:
            issue_description = None

        if not repo_path or not issue_description:
            self.notify(
                "Provide a local repository path and pick an issue first.",
                severity="warning",
            )
            return

        self._running = True
        self.query_one("#start", Button).disabled = True
        log = self.query_one("#log", RichLog)
        log.clear()
        # Step 3.2: fresh status footer for the new run.
        self._run_state = RunState()
        try:
            self.query_one("#pipeline-status", Static).update(self._run_state.summary_line())
        except Exception:  # noqa: BLE001 - widget may not exist in odd test setups
            pass
        self.query_one(TabbedContent).active = "log-tab"
        self._set_status("Pipeline running…", state=None)
        self._run_pipeline_worker(repo_path, issue_description, self.default_model, self.max_retries)


def run_app(default_model: Optional[str] = None, max_retries: int = 3) -> None:
    """Create and run the PatchCraft TUI."""
    PatchCraftApp(default_model=default_model, max_retries=max_retries).run()


__all__ = ["PatchCraftApp", "run_app"]
