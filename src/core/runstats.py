"""Per-run statistics shared with the presentation layer (Step 3.2).

A tiny, thread-safe registry holding the token consumption of the CURRENT
pipeline run. The orchestrator's internal ``usage_sink`` feeds it (pure
instrumentation: signatures and pipeline behavior are unchanged) so a live
UI can display "tokens spent vs budget" without the orchestrator knowing
anything about presentation.
"""

from __future__ import annotations

import threading
from typing import Optional


class RunStats:
    """Accumulated LLM token usage of one pipeline run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record one usage report (called from the LLM layer)."""
        with self._lock:
            self.prompt_tokens += int(prompt_tokens)
            self.completion_tokens += int(completion_tokens)

    @property
    def total(self) -> int:
        """Total tokens (prompt + completion) spent so far."""
        with self._lock:
            return self.prompt_tokens + self.completion_tokens


_current: Optional[RunStats] = None
_lock = threading.Lock()


def begin_run() -> RunStats:
    """Start a fresh stats window and make it the current one."""
    global _current
    with _lock:
        _current = RunStats()
        return _current


def current_run() -> Optional[RunStats]:
    """Stats of the run in progress, or ``None`` outside of any run."""
    with _lock:
        return _current


def reset() -> None:
    """Drop the current window (used by tests)."""
    global _current
    with _lock:
        _current = None


__all__ = ["RunStats", "begin_run", "current_run", "reset"]
