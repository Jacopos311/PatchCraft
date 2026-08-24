"""GitHub pull-request publishing (Roadmap Step 4.2).

Built on the same thin HTTP layer as :mod:`src.github.issue_fetcher`
(``GITHUB_TOKEN``/``GH_TOKEN`` auth, no extra SDK):

* resolves ``owner/repo`` and the default branch from the local git config
  first, falling back to the GitHub API;
* pushes the ``patchcraft/*`` branch to ``origin``;
* opens a pull request (draft per ``pr.draft``, default true until M5) via
  ``POST /repos/{owner}/{repo}/pulls``;
* IDEMPOTENT: the PR body carries a hidden marker
  ``<!-- patchcraft-issue:N -->``; an existing open PR for the same head
  branch OR the same marker is UPDATED instead of duplicated;
* never leaves a half-pushed state: if PR creation fails, the just-pushed
  remote branch reference is deleted again.

All failures raise :class:`GitHubAPIError` (or :class:`GitSafetyError` for
pure git problems) with clear English messages and actionable hints.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from src.core.gitflow import run_git
from src.github.issue_fetcher import GitHubAPIError, _auth_headers

logger = logging.getLogger(__name__)

GITHUB_API_URL = "https://api.github.com"
TIMEOUT_SECONDS = 15.0

# Hidden HTML comment embedded in the PR body for idempotent lookups.
_MARKER_TEMPLATE = "<!-- patchcraft-issue:{number} -->"


def pr_marker(issue_number: int) -> str:
    """Hidden marker identifying PRs created for ``issue_number``."""
    return _MARKER_TEMPLATE.format(number=int(issue_number))


def append_marker(body: str, issue_number: int) -> str:
    """Append the idempotency marker to ``body`` (once)."""
    marker = pr_marker(issue_number)
    if marker in (body or ""):
        return body
    return f"{(body or '').rstrip()}\n\n{marker}\n"


def _request(
    method: str,
    path: str,
    *,
    json_payload: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> requests.Response:
    """One authenticated GitHub API call (never raises for HTTP status)."""
    url = f"{GITHUB_API_URL}{path}"
    try:
        return requests.request(
            method,
            url,
            json=json_payload,
            params=params,
            headers=_auth_headers(),
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise GitHubAPIError(
            f"Network error while calling GitHub API ({method} {path}): {exc}"
        ) from exc


def _hint_for(status: int, detail: str) -> str:
    lowered = detail.lower()
    if status == 401:
        return "Check GITHUB_TOKEN/GH_TOKEN — it looks invalid or expired."
    if status == 403:
        if "rate limit" in lowered:
            return ("GitHub rate limit exceeded: wait for the reset window or "
                    "set GITHUB_TOKEN/GH_TOKEN to raise the limit.")
        return ("Access denied: the token may lack the 'pull-requests: write' "
                "scope, or the resource is protected.")
    if status == 404:
        return "Not found: verify the repository name and that you have access to it."
    if status == 422:
        return ("Validation failed: a pull request for this branch may "
                "already exist, or the base branch rejects new PRs "
                "(branch protection).")
    return f"Unexpected HTTP {status}."


def _fail(response: requests.Response, action: str, context: str) -> GitHubAPIError:
    """Build an actionable English error for a failed API call."""
    try:
        payload = response.json()
        detail = str(payload.get("message", "")) if isinstance(payload, dict) else ""
    except ValueError:  # pragma: no cover - non-JSON bodies are rare
        detail = (response.text or "")[:200]
    return GitHubAPIError(
        f"GitHub API refused to {action} (HTTP {response.status_code}) "
        f"[{context}]: {detail} — {_hint_for(response.status_code, detail)}"
    )


# ---------------------------------------------------------------------------
# Local git facts
# ---------------------------------------------------------------------------
def origin_owner_repo(repo_path: Path) -> Optional[str]:
    """Derive ``owner/repo`` from ``remote.origin.url`` (https or ssh form)."""
    url = run_git(
        ["config", "--get", "remote.origin.url"], Path(repo_path), check=False
    ).stdout.strip()
    if not url:
        return None
    match = re.search(r"github\.com[:/]([^/\s]+)/([^/\s]+?)(?:\.git)?/?$", url)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    logger.warning("Cannot derive owner/repo from remote URL: %r", url)
    return None


def resolve_base_branch(repo_path: Path, repo: str) -> str:
    """Default branch of the BASE repo: git config first, GitHub API second."""
    completed = run_git(
        ["symbolic-ref", "refs/remotes/origin/HEAD"],
        Path(repo_path),
        check=False,
    )
    if completed.returncode == 0:
        ref = completed.stdout.strip()  # e.g. refs/remotes/origin/main
        candidate = ref.rsplit("/", 1)[-1] if ref else ""
        if candidate and candidate != "origin":
            return candidate
    response = _request("GET", f"/repos/{repo}")
    if response.status_code != 200:
        raise _fail(response, "fetch repository info", f"repo '{repo}'")
    return response.json().get("default_branch") or "main"


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------
def push_branch(
    repo_path: Path,
    branch: str,
    remote: str = "origin",
) -> None:
    """Push ``branch`` to ``remote`` with actionable errors (no force)."""
    remotes = run_git(["remote"], Path(repo_path), check=False).stdout.split()
    if remote not in remotes:
        raise GitHubAPIError(
            f"No git remote '{remote}' is configured. Add one first: "
            f"`git remote add origin https://github.com/<owner>/<repo>.git`."
        )
    completed = run_git(["push", "-u", remote, branch], Path(repo_path), check=False)
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise GitHubAPIError(
            f"git push of branch '{branch}' failed: {stderr} — Hints: fetch and "
            f"rebase onto the latest base branch, check branch protection "
            f"rules, and verify your credentials."
        )


def _find_open_pr_by_head(repo: str, branch: str) -> Optional[Dict[str, Any]]:
    """Open PR whose head is exactly ``owner:branch``, or None."""
    owner = repo.split("/", 1)[0]
    response = _request(
        "GET",
        f"/repos/{repo}/pulls",
        params={"state": "open", "head": f"{owner}:{branch}", "per_page": 100},
    )
    if response.status_code != 200:
        raise _fail(response, "search open pull requests", f"repo '{repo}'")
    items = response.json()
    return items[0] if isinstance(items, list) and items else None


def find_open_pr_by_marker(repo: str, issue_number: int) -> Optional[Dict[str, Any]]:
    """Open PR carrying this issue's hidden marker, or None."""
    response = _request(
        "GET",
        f"/repos/{repo}/pulls",
        params={"state": "open", "per_page": 100},
    )
    if response.status_code != 200:
        raise _fail(response, "list open pull requests", f"repo '{repo}'")
    marker = pr_marker(issue_number)
    for item in response.json() or []:
        if isinstance(item, dict) and marker in (item.get("body") or ""):
            return item
    return None


def publish_pr(
    *,
    repo: str,
    repo_path: Path,
    branch: str,
    title: str,
    body: str,
    draft: bool,
    issue_number: Optional[int] = None,
    base: Optional[str] = None,
) -> str:
    """Push ``branch`` and open (or UPDATE) the pull request; returns its URL.

    Idempotency: an existing open PR for the same head branch, or carrying
    this issue's hidden marker, is updated in place. When creation fails the
    remote branch reference is deleted again so nothing is left half-pushed.
    """
    # 1. Push the branch under the patchcraft/* namespace.
    push_branch(Path(repo_path), branch)

    # 2. Resolve the base branch (git config first, API fallback).
    effective_base = base or resolve_base_branch(Path(repo_path), repo)

    final_body = append_marker(body, issue_number) if issue_number else body

    # 3. Idempotency: same head branch, else same issue marker.
    existing = _find_open_pr_by_head(repo, branch)
    if existing is None and issue_number is not None:
        existing = find_open_pr_by_marker(repo, issue_number)

    if existing is not None:
        number = existing.get("number")
        response = _request(
            "PATCH",
            f"/repos/{repo}/pulls/{number}",
            json_payload={"title": title, "body": final_body},
        )
        if response.status_code != 200:
            raise _fail(response, f"update pull request #{number}", f"repo '{repo}'")
        updated = response.json()
        return updated.get("html_url") or f"https://github.com/{repo}/pull/{number}"

    # 4. Create the PR; roll back the remote ref if this fails.
    payload = {
        "title": title,
        "head": f"{repo.split('/', 1)[0]}:{branch}",
        "base": effective_base,
        "body": final_body,
        "draft": bool(draft),
    }
    response = _request("POST", f"/repos/{repo}/pulls", json_payload=payload)
    if response.status_code not in (200, 201):
        delete_response = _request(
            "DELETE", f"/repos/{repo}/git/refs/heads/{branch}"
        )
        rolled_back = delete_response.status_code in (204, 200)
        suffix = (
            " The remote branch was deleted again — nothing left half-pushed."
            if rolled_back
            else " WARNING: the remote branch could NOT be deleted automatically;"
                 f" remove it manually ('git push origin --delete {branch}')."
        )
        error = _fail(response, "open the pull request", f"repo '{repo}'")
        raise GitHubAPIError(str(error) + suffix)

    created = response.json()
    url = created.get("html_url")
    return url or f"https://github.com/{repo}/pull/(unknown)"


__all__ = [
    "append_marker",
    "find_open_pr_by_marker",
    "origin_owner_repo",
    "pr_marker",
    "publish_pr",
    "push_branch",
    "resolve_base_branch",
]
