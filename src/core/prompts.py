"""Prompt compiler: budgeted, priority-driven prompt assembly (Roadmap Step 1.4).

Every agent prompt is assembled from named *sections* with an explicit
priority and a trimming strategy. When the combined size exceeds the token
budget, the compiler trims (or drops) the LOWEST-priority sections first —
per the roadmap trim order:

    issue text > affected-file current content > repo map > test feedback tail > rest

Trim strategies:
* ``head`` — keep the beginning of the section (issue text).
* ``tail`` — keep the end of the section (test summaries live at the end).
* ``drop``  — remove the section entirely (default for large optional blocks).

The compiler never lets the compiled text exceed the budget (hard cap as a
last resort) and debug-logs exactly what was included, dropped or trimmed.
Token estimates use the ~4 characters-per-token heuristic.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "CHARS_PER_TOKEN",
    "estimate_tokens",
    "TRIM_HEAD",
    "TRIM_TAIL",
    "TRIM_DROP",
    "Section",
    "CompiledPrompt",
    "PromptCompiler",
]

CHARS_PER_TOKEN = 4
SEP = "\n\n"
TRUNCATION_MARKER = "…[truncated]"

TRIM_HEAD = "head"
TRIM_TAIL = "tail"
TRIM_DROP = "drop"

#: Sections are never trimmed below this many characters; below that they are
#: dropped entirely (a useless stub is worse than no section).
MIN_USABLE_CHARS = 200


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` (~4 characters per token)."""
    if not text:
        return 0
    return math.ceil(len(text) / CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Sections and compiled result
# ---------------------------------------------------------------------------
@dataclass
class Section:
    """One named block of a prompt."""

    name: str
    content: str
    priority: int = 5          # 1 = keep at all costs, higher = trimmed first
    trim: str = TRIM_DROP      # how this section shrinks when over budget
    min_chars: int = MIN_USABLE_CHARS


@dataclass
class CompiledPrompt:
    """Result of :meth:`PromptCompiler.compile`."""

    text: str
    included: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    trimmed: list[str] = field(default_factory=list)
    estimated_tokens: int = 0


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------
class PromptCompiler:
    """Assembles sections into one prompt within an explicit token budget.

    Trim order: lowest priority first; among equal priorities the
    last-added section is trimmed first (later context is less load-bearing).
    Priority-1 sections are never dropped or trimmed by the loop — only the
    final hard cap may cut them.
    """

    def __init__(self, max_tokens: int) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        self.max_tokens = max_tokens
        self._sections: list[Section] = []

    def add(
        self,
        name: str,
        content: str,
        *,
        priority: int = 5,
        trim: str = TRIM_DROP,
        min_chars: int = MIN_USABLE_CHARS,
    ) -> None:
        """Register a section. Empty content is ignored silently."""
        if not content:
            return
        self._sections.append(Section(name=name, content=content, priority=priority, trim=trim, min_chars=min_chars))

    def compile(self) -> CompiledPrompt:
        budget_chars = self.max_tokens * CHARS_PER_TOKEN
        # active: [insertion_index, section, working_content]
        active: list[tuple[int, Section, str]] = [
            (i, s, s.content) for i, s in enumerate(self._sections) if s.content
        ]
        dropped: list[str] = []
        trimmed: list[str] = []

        def total_length() -> int:
            return (
                sum(len(content) for _, _, content in active)
                + max(0, len(active) - 1) * len(SEP)
            )

        while total_length() > budget_chars:
            trimmable = [(s.priority, i, j) for j, (i, s, _) in enumerate(active) if s.priority > 1]
            if not trimmable:
                break
            # Highest priority NUMBER = least important => trimmed/dropped first;
            # among equal priorities the last-added section goes first.
            _, _, pick = max(trimmable, key=lambda item: (item[0], item[1]))
            position, section, content = active[pick]

            # A section that was already truncated cannot shrink further via a
            # tail/head cut (the cut index starts from the marker, keeping the
            # length stable) — drop it so the loop always makes progress.
            if (section.trim in (TRIM_HEAD, TRIM_TAIL)) and TRUNCATION_MARKER in content:
                active.pop(pick)
                dropped.append(section.name)
                continue

            others_length = total_length() - len(content) - (len(active) - 1) * len(SEP)
            allowed = budget_chars - others_length

            if section.trim in (TRIM_HEAD, TRIM_TAIL) and allowed >= max(section.min_chars, len(TRUNCATION_MARKER)):
                marker_len = len(TRUNCATION_MARKER) + 1
                # Reserve room for the JOIN separators that stay in the final text.
                keep = allowed - marker_len - (len(active) - 1) * len(SEP)
                if keep < 1:
                    # Nothing usable fits: drop the section instead of
                    # looping forever on significant zero-width trims.
                    active.pop(pick)
                    dropped.append(section.name)
                    continue
                if section.trim == TRIM_TAIL:
                    new_content = f"{TRUNCATION_MARKER}\n{content[-keep:]}"
                else:
                    new_content = f"{content[:keep]}\n{TRUNCATION_MARKER}"
                active[pick] = (position, section, new_content)
                trimmed.append(section.name)
                continue  # re-evaluate: maybe more trimming is needed

            active.pop(pick)
            dropped.append(section.name)

        included = [s.name for _, s, _ in active]
        text = SEP.join(content for _, _, content in active)

        # Hard cap (last resort): guarantees the budget even when only
        # priority-1 sections remain.
        if len(text) > budget_chars:
            text = text[: max(0, budget_chars - len(TRUNCATION_MARKER))] + TRUNCATION_MARKER
            trimmed.append("<hard-cut>")

        logger.debug(
            "Prompt compiled: ~%d/%d tokens | included=%s | dropped=%s | trimmed=%s",
            estimate_tokens(text), self.max_tokens, included, dropped, trimmed,
        )
        return CompiledPrompt(
            text=text,
            included=included,
            dropped=dropped,
            trimmed=trimmed,
            estimated_tokens=estimate_tokens(text),
        )
