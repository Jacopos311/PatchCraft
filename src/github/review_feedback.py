"""Review-comment fetching, classification and replies (Roadmap Step 4.4).

Powers ``patchcraft followup <pr-url>``:

* fetches PR review comments (inline) AND general PR comments via the REST
  API, normalized into :class:`ReviewComment`;
* classifies each comment as ``must-fix`` / ``nit`` / ``question`` with a
  deterministic keyword classifier (labels like ``[must-fix]`` win first);
* builds the correction-loop task description from the must-fix items;
* posts polite English replies to addressed comments and re-requests review
  from their authors (per repo norms).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from src.github.issue_fetcher import GitHubAPIError, _auth_headers

logger = logging.getLogger(__name__)

GITHUB_API_URL = "https://api.github.com"
TIMEOUT_SECONDS = 15.0

CommentKind = Literal["must-fix", "nit", "question"]

_BOT_SUFFIX = "[bot]"

# Explicit labels win over everything else.
_LABEL_RE = re.compile(
    r"^\s*\[?(must-fix|must fix|nit|question)\b[:\]]?", re.IGNORECASE
)

_MUST_FIX_KEYWORDS = (
    "must", "blocker", "blocking", "needs work", "request changes",
    "do not merge", "don't merge", "incorrect", "wrong", "broken",
    "bug", "fails", "failing", "error", "crash", "regression",
    "security", "vulnerability", "leak", "race condition",
)
_NIT_KEYWORDS = (
    "nit:", "nit ", "minor", "typo", "cosmetic", "style",
    "optional", "suggestion", "consider ", "prefer ",
)
_QUESTION_KEYWORDS = ("question", "why ", "how come", "could you clarify",
                      "curious", "thoughts")


class ReviewComment(BaseModel):
    """One reviewer comment, normalized across inline/general endpoints."""

    id: int
    author: str = Field(default="unknown", description="Reviewer login.")
    body: str = Field(default="", description="Comment text (markdown).")
    path: Optional[str] = Field(default=None, description="File path (inline only).")
    line: Optional[int] = Field(default=None, description="Line number (inline only).")
    is_inline: bool = Field(default=False, description="True for inline review comments.")
    kind: CommentKind = Field(default="must-fix", description="Classifier verdict.")


def parse_pr_url(reference: str) -> Tuple[str, int]:
    """Parse a GitHub PR URL into ``(owner/repo, number)``.

    Accepts ``https://github.com/{owner}/{repo}/pull/{n}`` (optional trailing
    slash / www). Raises :class:`GitHubAPIError` for anything else.
    """
    text = (reference or "").strip()
    match = re.match(
        r"^https?://(?:www\.)?github\.com/([^/\s]+)/([^/\s]+)/pull/(\d+)/?$",
        text,
    )
    if match:
        owner, name, number = match.groups()
        return f"{owner}/{name}", int(number)
    raise GitHubAPIError(
        f"Pull request reference {reference!r} is not valid: use a GitHub PR "
        f"URL (https://github.com/owner/repo/pull/123)."
    )


def classify_comment(body: str) -> CommentKind:
    """Deterministic keyword classification of one review comment."""
    text = (body or "").lower()
    label = _LABEL_RE.match(text.strip())
    if label:
        kind = label.group(1).lower().replace(" ", "-")
        if kind in ("must-fix", "nit", "question"):
            return kind  # type: ignore[return-value]
    if any(kw in text for kw in _MUST_FIX_KEYWORDS):
        return "must-fix"
    if "?" in text or any(kw in text for kw in _QUESTION_KEYWORDS):
        return "question"
    if any(kw in text for kw in _NIT_KEYWORDS):
        return "nit"
    # Unclassifiable feedback is treated as actionable (safe default).
    return "must-fix"


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """Authenticated GET returning parsed JSON; raises on HTTP errors."""
    import requests

    try:
        response = requests.get(
            f"{GITHUB_API_URL}{path}",
            params=params,
            headers=_auth_headers(),
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise GitHubAPIError(f"Network error calling GitHub API ({path}): {exc}") from exc
    if response.status_code == 401:
        raise GitHubAPIError("GitHub API returned 401: check GITHUB_TOKEN/GH_TOKEN.")
    if response.status_code == 403:
        raise GitHubAPIError(
            "GitHub rate limit exceeded (403): set GITHUB_TOKEN/GH_TOKEN "
            "to raise the limit."
        )
    if response.status_code != 200:
        raise GitHubAPIError(
            f"GitHub API returned HTTP {response.status_code} for {path}."
        )
    return response.json()


def fetch_review_comments(repo: str, pr_number: int) -> List[ReviewComment]:
    """Fetch + classify every human comment on a pull request.

    Merges inline review comments (``/pulls/N/comments``) with general
    conversation comments (``/issues/N/comments``); bot accounts are skipped.
    """
    out: List[ReviewComment] = []
    for is_inline, path in (
        (True, f"/repos/{repo}/pulls/{int(pr_number)}/comments"),
        (False, f"/repos/{repo}/issues/{int(pr_number)}/comments"),
    ):
        for item in _get(path) or []:
            if not isinstance(item, dict):
                continue
            author = ((item.get("user") or {}).get("login")) or "unknown"
            if author.lower().endswith(_BOT_SUFFIX):
                continue
            comment_id = item.get("id")
            if comment_id is None:
                continue
            out.append(ReviewComment(
                id=int(comment_id),
                author=author,
                body=item.get("body") or "",
                path=item.get("path"),
                line=item.get("line") or item.get("original_line"),
                is_inline=is_inline,
                kind=classify_comment(item.get("body") or ""),
            ))
    return out


def get_pull_request(repo: str, pr_number: int) -> Dict[str, Any]:
    """PR metadata dict (head.ref, base.ref, title, html_url...)."""
    data = _get(f"/repos/{repo}/pulls/{int(pr_number)}")
    if not isinstance(data, dict):
        raise GitHubAPIError(f"Unexpected response shape for PR #{pr_number}.")
    return data


# ---------------------------------------------------------------------------
# Task building + replies
# ---------------------------------------------------------------------------
def build_followup_task(pr_number: int, comments: List[ReviewComment]) -> Optional[str]:
    """Format must-fix comments into the correction-loop description.

    Returns ``None`` when there is nothing that must be fixed.
    """
    must_fix = [c for c in comments if c.kind == "must-fix"]
    if not must_fix:
        return None
    lines = [f"Address these must-fix review comments on PR #{pr_number}:"]
    for i, comment in enumerate(must_fix, start=1):
        where = ""
        if comment.path:
            where = f" [{comment.path}" + (f":{comment.line}]" if comment.line else "]")
        body = " ".join(comment.body.split())[:400]
        lines.append(f"{i}.{where} (@{comment.author}) {body}")
    return "\n".join(lines)


def reply_to_comment(
    repo: str,
    pr_number: int,
    comment: ReviewComment,
    body: str,
) -> None:
    """Reply to one addressed comment (inline threads; general appends).

    Failures are logged as warnings — replying is best-effort and must never
    mask the outcome of the underlying fix.
    """
    import requests

    if comment.is_inline:
        url = (f"{GITHUB_API_URL}/repos/{repo}/pulls/{int(pr_number)}"
               f"/comments/{comment.id}/replies")
    else:
        url = f"{GITHUB_API_URL}/repos/{repo}/issues/{int(pr_number)}/comments"
    try:
        response = requests.post(url, json={"body": body},
                                 headers=_auth_headers(), timeout=TIMEOUT_SECONDS)
        if response.status_code not in (200, 201):
            logger.warning("Reply to comment %s failed (HTTP %s).",
                           comment.id, response.status_code)
    except requests.RequestException as exc:
        logger.warning("Reply to comment %s failed: %s", comment.id, exc)


def request_rereview(repo: str, pr_number: int, reviewers: List[str]) -> None:
    """Re-request review from the given logins (best-effort, never raises)."""
    humans = [r for r in reviewers if r and not r.lower().endswith(_BOT_SUFFIX)]
    if not humans:
        return
    import requests

    try:
        requests.post(
            f"{GITHUB_API_URL}/repos/{repo}/pulls/{int(pr_number)}/requested_reviewers",
            json={"reviewers": humans},
            headers=_auth_headers(),
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        logger.warning("Re-review request failed: %s", exc)


__all__ = [
    "ReviewComment",
    "build_followup_task",
    "classify_comment",
    "fetch_review_comments",
    "get_pull_request",
    "parse_pr_url",
    "reply_to_comment",
    "request_rereview",
]