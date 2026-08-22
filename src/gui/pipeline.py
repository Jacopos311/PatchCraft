"""Pipeline bridge for the PatchCraft GUI.

Wraps :func:`src.orchestrator.run_patchcraft_loop` so the Textual app can
stream structured milestones through an ``event_sink(stage, message)``
callback, and exposes a small helper to build the OpenRouter credits snapshot
shown in the sidebar widget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from src.core.credits import CreditsError, build_usage_bar, fetch_credits

logger = logging.getLogger(__name__)

# Milestone stages emitted by the orchestrator's event_sink.
STAGE_START = "start"
STAGE_CONTEXT = "context"
STAGE_DIAGNOSIS = "diagnosis"
STAGE_ITERATION = "iteration"
STAGE_PATCH = "patch"
STAGE_TEST = "test"
STAGE_ERROR = "error"
STAGE_DIFF = "diff"
STAGE_REPORT = "report"
STAGE_DONE = "done"


@dataclass(frozen=True)
class PipelineEvent:
    """A single milestone emitted while the pipeline runs."""

    stage: str
    message: str


def run_pipeline(
    repo_path: str,
    issue_description: str,
    model: str,
    max_retries: int = 3,
    on_event: Optional[Callable[[PipelineEvent], None]] = None,
) -> Any:
    """Run the PatchCraft pipeline, streaming milestones to ``on_event``.

    This is a thin, blocking wrapper around
    :func:`src.orchestrator.run_patchcraft_loop` designed to be invoked from
    a background thread/worked. ``on_event`` receives a :class:`PipelineEvent`
    for every milestone; exceptions raised by the callback are swallowed by
    the orchestrator so they can never break the pipeline.
    """
    from src.orchestrator import run_patchcraft_loop

    def sink(stage: str, message: str) -> None:
        if on_event is not None:
            on_event(PipelineEvent(stage=stage, message=message))

    return run_patchcraft_loop(
        repo_path=repo_path,
        issue_description=issue_description,
        model=model,
        max_retries=max_retries,
        event_sink=sink,
    )


def credits_snapshot(api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Build the credits snapshot for the GUI sidebar widget.

    Returns ``None`` when no API key is configured or the endpoint fails
    (the GUI then renders a disabled state — never raises).
    """
    try:
        return fetch_credits(api_key=api_key)
    except CreditsError as exc:
        logger.warning("Credits snapshot unavailable: %s", exc)
        return None


def format_credits_line(credits: Optional[Dict[str, Any]]) -> str:
    """Render the credits snapshot as a compact one/two-line string."""
    if not credits:
        return "[dim]💳 OpenRouter: no API key configured[/dim]"
    usage: float = credits.get("usage", 0.0)
    limit: Optional[float] = credits.get("limit")
    if limit and limit > 0:
        bar = build_usage_bar(usage, limit, width=16)
        pct = min(100.0, max(0.0, usage / limit * 100.0))
        return f"💳 [b]${usage:.4f}[/b] of ${limit:.2f}\n[cyan]{bar}[/cyan] {pct:.1f}%"
    line = f"💳 Total spent: [b]${usage:.4f}[/b]"
    if credits.get("is_free_tier"):
        line += "\n[dim]Free tier — no limit applied.[/dim]"
    return line


__all__ = [
    "PipelineEvent",
    "run_pipeline",
    "credits_snapshot",
    "format_credits_line",
]