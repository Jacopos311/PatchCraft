"""Live iteration view shared by the CLI and the TUI (Roadmap Step 3.2).

Presentation only: the orchestrator is untouched. Milestones already flow
through ``event_sink(stage, message)``; this module turns them into:

* **TTY mode** — a compact ``rich.Live`` panel pinned to *stderr* (current
  stage, iteration n/∞, tokens spent vs budget, last test verdict, elapsed
  time, tail of recent milestones). A dedicated stderr console keeps the
  panel independent from the orchestrator's stdout output and spinners.
* **Pipe mode (no TTY)** — one plain ``[stage] message`` line per milestone,
  no ANSI escapes, so CI logs stay clean.

:class:`RunState` holds the parsed state and is also reused by the Textual
TUI to mirror the same summary in its status footer.
"""

from __future__ import annotations

import os
import re
import sys
import time
from collections import deque
from typing import Deque, Optional

from src.core.runstats import current_run

ENV_TOKEN_BUDGET = "PATCHCRAFT_TOKEN_BUDGET"

_STAGE_ICONS = {
    "start": "🚀",
    "context": "📚",
    "diagnosis": "🔍",
    "iteration": "🔁",
    "patch": "✍️",
    "test": "🧪",
    "error": "❌",
    "diff": "📄",
    "report": "📋",
    "done": "🏁",
}

_ITERATION_RE = re.compile(r"^Iteration (\d+)/(.+)$")


def _first_line(message: str) -> str:
    return message.splitlines()[0].strip() if message else ""


def _fmt_clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _verdict_label(verdict: Optional[bool]) -> str:
    if verdict is True:
        return "✅ PASS"
    if verdict is False:
        return "❌ FAIL"
    return "—"


class RunState:
    """Parsed snapshot of a pipeline run, fed by ``(stage, message)`` events."""

    def __init__(self, token_budget: Optional[int] = None) -> None:
        self.stage: str = "starting"
        self.iteration: int = 0
        # None means unbounded ("∞"): the goal-driven loop has no hard cap.
        self.iteration_total: Optional[int] = None
        self.last_verdict: Optional[bool] = None
        self.token_budget = token_budget
        self.started_at: float = time.monotonic()

    # ------------------------------------------------------------------
    # Event ingestion
    # ------------------------------------------------------------------
    def observe(self, stage: str, message: str) -> None:
        """Update the state from one orchestrator milestone."""
        if stage == "iteration":
            match = _ITERATION_RE.match(_first_line(message))
            if match:
                self.iteration = int(match.group(1))
                total = match.group(2).strip()
                self.iteration_total = None if total in {"∞", "inf"} else int(total)
                self.stage = "iteration"
                return
        if stage == "test":
            first = _first_line(message)
            if "success=True" in first or "success=False" in first:
                self.last_verdict = "success=True" in first
            self.stage = "test"
            return
        if stage in _STAGE_ICONS:
            self.stage = stage
            return
        self.stage = stage or self.stage

    # ------------------------------------------------------------------
    # Derived values
    # ------------------------------------------------------------------
    @property
    def elapsed_seconds(self) -> float:
        """Wall-clock time since the run started."""
        return time.monotonic() - self.started_at

    @property
    def tokens_spent(self) -> Optional[int]:
        """Total tokens reported by :mod:`src.core.runstats`, if any."""
        stats = current_run()
        return stats.total if stats is not None else None

    @property
    def iteration_label(self) -> str:
        total = "∞" if self.iteration_total is None else str(self.iteration_total)
        return f"{self.iteration}/{total}"

    def summary_line(self) -> str:
        """Compact single-line status mirroring the CLI panel."""
        spent = self.tokens_spent
        if spent is None:
            tokens_part = "tokens —"
        elif self.token_budget:
            tokens_part = f"tokens {spent:,}/{self.token_budget:,}"
        else:
            tokens_part = f"tokens {spent:,}"
        icon = _STAGE_ICONS.get(self.stage, "•")
        return (
            f"{icon} {self.stage} · iter {self.iteration_label} · "
            f"{tokens_part} · ⏱ {_fmt_clock(self.elapsed_seconds)} · "
            f"tests: {_verdict_label(self.last_verdict)}"
        )


class LiveRunView:
    """Event sink rendering pipeline milestones for humans.

    Implements the ``event_sink(stage, message)`` contract, so it can be
    passed straight to :func:`src.orchestrator.run_patchcraft_loop`.

    * Attached to a TTY → rich Live panel on **stderr** (stdout keeps the
      orchestrator's detailed output; CI redirection stays clean).
    * Piped → plain, ANSI-free lines on stdout: ``[stage] first line``.
    """

    TAIL_LINES = 8

    def __init__(self, token_budget: Optional[int] = None) -> None:
        if token_budget is None:
            raw = os.getenv(ENV_TOKEN_BUDGET, "").strip()
            try:
                token_budget = int(raw) if raw else None
            except ValueError:
                token_budget = None
        self.state = RunState(token_budget=token_budget)
        self.interactive = sys.stdout.isatty()
        self._tail: Deque[str] = deque(maxlen=self.TAIL_LINES)
        self._live = None  # rich.live.Live when active
        self._console = None

    # ------------------------------------------------------------------
    # event_sink interface
    # ------------------------------------------------------------------
    def __call__(self, stage: str, message: str) -> None:
        self.sink(stage, message)

    def sink(self, stage: str, message: str) -> None:
        """Consume one milestone (never raises — sinks must not break runs)."""
        try:
            self.state.observe(stage, message)
            self._tail.append(
                f"{_STAGE_ICONS.get(stage, '•')} [{stage}] {_first_line(message)}"
            )
            if self.interactive:
                self._refresh_panel()
            else:
                # Pipe-friendly: plain text, no ANSI escapes, flushed at once.
                print(f"[{stage}] {_first_line(message)}", flush=True)
        except Exception as exc:  # noqa: BLE001 - presentation must never crash a run
            print(f"[live-ui] render error: {type(exc).__name__}: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Activate the live panel (interactive terminals only)."""
        if not self.interactive or self._live is not None:
            return
        try:
            from rich.console import Console
            from rich.live import Live

            self._console = Console(file=sys.stderr, highlight=False)
            self._live = Live(
                self._build_panel(),
                console=self._console,
                refresh_per_second=4,
                transient=False,
            )
            self._live.start()
        except Exception as exc:  # noqa: BLE001 - degrade to pipe mode silently
            print(f"[live-ui] panel unavailable ({exc}); using plain lines.", file=sys.stderr)
            self._live = None
            self.interactive = False

    def finish(self) -> None:
        """Stop the live panel cleanly (safe to call multiple times)."""
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            self._live = None

    # ------------------------------------------------------------------
    # Rendering (interactive mode)
    # ------------------------------------------------------------------
    def _refresh_panel(self) -> None:
        if self._live is not None:
            self._live.update(self._build_panel(), refresh=True)

    def _build_panel(self):
        from rich.console import Group
        from rich.text import Text

        summary = Text(self.state.summary_line(), style="bold cyan")
        tail = Text("\n".join(self._tail), style="dim") if self._tail else Text("")
        return Group(summary, tail)


__all__ = ["ENV_TOKEN_BUDGET", "LiveRunView", "RunState"]
