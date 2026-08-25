"""PR writer agent (Roadmap Step 4.3).

Rebuilds the old reporter into a writer that produces human-grade pull
request content. Two LLM passes:

1. **Draft** — given ONLY verified inputs (final diff, deterministic diff
   stat, test evidence, issue text, and the repo voice profile), write the
   PR body following the repo template / tone.
2. **Self-review** — a second call critiques the draft against the voice
   profile and a template-compliance checklist, then revises **once**.

Anti-hallucination by construction: the writer prompt states that any claim
not derivable from the given inputs is forbidden, and a deterministic
``sanitize_honesty`` pass strips review/approval phrasing that would
fabricate social facts.
"""

from __future__ import annotations

import re
from typing import Callable, Optional, Sequence, Type, Union

from pydantic import BaseModel, Field

from src.core.llm import call_llm
from src.core.repo_profile import RepoVoice


class PatchReport(BaseModel):
    """Final report: title, summary, diff and the PR description body."""

    title: str = Field(description="Pull Request title (repo-style).")
    summary: str = Field(description="Summary of the applied changes.")
    diff: str = Field(description="Diff of the proposed changes.")
    pr_markdown: str = Field(description="Markdown ready for the PR description.")


class PRReview(BaseModel):
    """Second-pass critique + single revision of a PR draft."""

    conformance_notes: str = Field(
        description="What the draft got wrong vs the template/voice profile.")
    revise_pr_markdown: bool = Field(
        default=False, description="True when the body needs revision.")
    pr_markdown: str = Field(
        default="", description="Revised PR body (when ``revise_pr_markdown``).")
    title: Optional[str] = Field(
        default=None, description="Optional corrected PR title.")


SYSTEM_PROMPT = (
    "You are a senior engineer writing a pull request exactly the way an "
    "experienced human maintainer of this repository would. You only ever "
    "describe what is provable from the inputs you are given."
)

# Hard anti-hallucination rule included in every writer prompt.
_HONESTY_RULE = (
    "HARD RULES: (1) Only claims derivable from the provided INPUTS (diff, "
    "diff stat, test evidence, issue text) are allowed. Never invent facts, "
    "benchmark numbers, coverage percentages, or other people's opinions. "
    "(2) Never claim any review, approval, CI status, or sign-off exists. "
    "(3) Do not mention any file outside the diff. (4) Follow the provided "
    "template/checkboxes exactly; tick only what is truly done."
)

# Phrases that fabricate non-existent social/CI facts — always removed.
_FABRICATION_PATTERNS = (
    re.compile(r"(?im)^[ \t]*[-*]?[ \t]*(approved|reviewed|sign[- ]?off)\b[^\n]*$"),
    re.compile(r"(?im)\b(lgtm|:tada:|status: passed)\b"),
)
def sanitize_honesty(text: str) -> str:
    """Remove lines/words that would fabricate reviews, approvals or CI facts."""
    lines = []
    for line in (text or "").splitlines():
        if any(pattern.search(line) for pattern in _FABRICATION_PATTERNS):
            continue
        lines.append(line)
    return "\n".join(lines).strip() + ("\n" if text and text.strip() else "")


def build_diff_stat(diff: str) -> str:
    """Deterministic ``%d files changed, +%d / -%d`` summary of a unified diff.

    This is derived directly from the diff text (never from the LLM), so the
    numbers the PR body cites are always honest and verified.
    """
    files = additions = deletions = 0
    for line in (diff or "").splitlines():
        if line.startswith("diff --git"):
            files += 1
        elif line.startswith("+++") or line.startswith("--- "):
            continue  # file headers, not content
        elif line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    noun = "file" if files == 1 else "files"
    return f"{files} {noun} changed, +{additions} insertions, -{deletions} deletions"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def _voice_section(voice: RepoVoice) -> str:
    parts = []
    if voice.contribution_guidelines:
        parts.append(
            "REPO GUIDELINES (CONTRIBUTING.md excerpts; follow them):\n"
            + voice.contribution_guidelines
        )
    if voice.pull_request_template:
        parts.append(
            "PULL REQUEST TEMPLATE — reproduce its sections and checkboxes "
            "EXACTLY, honestly ticked:\n" + voice.pull_request_template
        )
    if voice.recent_title_examples:
        parts.append(
            "RECENT MERGED PR TITLES (match their style and length):\n"
            + "\n".join(f"- {t}" for t in voice.recent_title_examples)
        )
    return "\n\n".join(parts)


def _build_writer_prompt(
    *,
    diff: str,
    diff_stat: Optional[str],
    test_evidence: Optional[str],
    issue_text: Optional[str],
    voice: Optional[RepoVoice],
    max_diff_chars: int = 12_000,
) -> str:
    """Compose the first-pass writer prompt from ONLY verified data."""
    sections: list[str] = []
    if issue_text:
        sections.append(
            f"ISSUE (context only — never restate opinions as facts):\n{issue_text[:2000]}"
        )
    sections.append(f"VERIFIED DIFF:\n{diff[:max_diff_chars]}")
    if diff_stat:
        sections.append(f"DIFF STAT (computed by a tool, accurate):\n{diff_stat}")
    if test_evidence:
        sections.append(f"TEST EVIDENCE (what actually ran and passed):\n{test_evidence}")
    if voice is not None:
        profile = _voice_section(voice)
        if profile:
            sections.append(profile)
    return "\n\n".join(sections) + f"\n\n{_HONESTY_RULE}"


def _build_review_prompt(
    *,
    draft: Union[PatchReport, BaseModel],
    voice: Optional[RepoVoice],
    diff_stat: Optional[str],
    test_evidence: Optional[str],
) -> str:
    """Ask the reviewer to check template fields + honesty before one revision."""
    report = draft if isinstance(draft, PatchReport) else PatchReport.model_validate(draft)
    voice_block = _voice_section(voice) if voice is not None else ""
    return "\n\n".join((
        "Review this PR draft like a strict maintainer. Criticize: template "
        "compliance (every required field present), title style, honesty "
        "(strip any claim not backed by the verified data), and tone fit.",
        f"DIFF STAT:\n{diff_stat or 'n/a'}",
        f"TEST EVIDENCE: {test_evidence or 'n/a'}",
        f"VOICE/TEMPLATE:\n{voice_block if voice_block else '(none — generic structure)'}",
        f"DRAFT TITLE: {report.title}",
        f"DRAFT BODY:\n{report.pr_markdown}",
        _HONESTY_RULE,
    ))
def generate_report(
    diff: str,
    provider_model: str,
    json_schema: Optional[Type[BaseModel]] = None,
    usage_sink: Optional[Callable[[int, int], None]] = None,
    *,
    repo_voice: Optional[RepoVoice] = None,
    issue_text: Optional[str] = None,
    diff_stat: Optional[str] = None,
    test_evidence: Optional[str] = None,
    self_review: bool = True,
) -> Union[str, BaseModel]:
    """Generate a human-grade :class:`PatchReport` from verified data only.

    Extends the old ``generate_report`` with the Step 4.3 inputs (voice
    profile, issue text, deterministic diff stat, test evidence).
    ``self_review=True`` performs one LLM critique-and-revise pass when a
    template or recent-PR examples are available.
    """
    effective_schema = json_schema or PatchReport
    user_prompt = _build_writer_prompt(
        diff=diff,
        diff_stat=diff_stat,
        test_evidence=test_evidence,
        issue_text=issue_text,
        voice=repo_voice,
    )
    draft = call_llm(
        provider_model=provider_model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        json_schema=effective_schema,
        usage_sink=usage_sink,
    )

    if (
        self_review
        and isinstance(draft, PatchReport)
        and repo_voice is not None
        and (repo_voice.pull_request_template or repo_voice.recent_title_examples)
    ):
        try:
            review = call_llm(
                provider_model=provider_model,
                system_prompt=(
                    "You are the strict final reviewer of a pull request "
                    "draft. Be specific and correction-oriented."
                ),
                user_prompt=_build_review_prompt(
                    draft=draft,
                    voice=repo_voice,
                    diff_stat=diff_stat,
                    test_evidence=test_evidence,
                ),
                json_schema=PRReview,
                usage_sink=usage_sink,
            )
        except Exception:  # noqa: BLE001 - self-review must never break the report
            review = None
        if isinstance(review, PRReview) and review.revise_pr_markdown:
            updates: dict = {}
            if review.pr_markdown.strip():
                updates["pr_markdown"] = review.pr_markdown
            if review.title:
                updates["title"] = review.title
            if updates:
                draft = draft.model_copy(update=updates)

    if isinstance(draft, PatchReport):
        draft = draft.model_copy(update={"pr_markdown": sanitize_honesty(draft.pr_markdown)})
    return draft


__all__ = [
    "PatchReport",
    "PRReview",
    "generate_report",
    "build_diff_stat",
    "sanitize_honesty",
]