"""GitHub issue fetching for PatchCraft.

Main function: :func:`get_open_issues`, which queries the public GitHub API
``GET /repos/{repo_name}/issues`` and returns only real **issues** (Pull
Requests are excluded, since the ``/issues`` endpoint includes them by design).

An optional token via ``GITHUB_TOKEN``/``GH_TOKEN`` avoids anonymous rate
limits; for public repositories it also works without one.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

GITHUB_API_URL = "https://api.github.com"
TIMEOUT_SECONDS = 10.0


class GitHubAPIError(RuntimeError):
    """Error communicating with the GitHub API (HTTP or network)."""


def _auth_headers() -> dict[str, str]:
    """Headers required by the GitHub API (mandatory User-Agent + optional auth)."""
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "PatchCraft",
    }
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _validate_repo_name(repo_name: str) -> str:
    """Validate and normalize the ``owner/repo`` repository name."""
    normalized = repo_name.strip().strip("/")
    if normalized.count("/") != 1:
        raise ValueError(
            "repo_name must be in 'owner/repo' format "
            "(e.g. langchain-ai/langgraph)."
        )
    owner, name = normalized.split("/", 1)
    if not owner or not name:
        raise ValueError("owner and repo must not be empty.")
    return normalized


def get_open_issues(
    repo_name: str,
    label: str = "bug",
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Fetch the open issues labeled with ``label`` for ``repo_name``.

    Parameters
    ----------
    repo_name : str
        Repository in ``owner/repo`` format (e.g. ``langchain-ai/langgraph``).
    label : str
        GitHub label to filter by (default ``bug``). An empty string
        disables the label filter.
    limit : int
        Maximum number of issues to fetch (1..100).

    Returns
    -------
    List[Dict[str, Any]]
        Open issues (without Pull Requests), as GitHub API dicts.
        Useful fields: ``number``, ``title``, ``body``, ``html_url``.

    Raises
    ------
    GitHubAPIError
        On network errors or non-2xx HTTP responses.
    ValueError
        If ``repo_name`` is not in ``owner/repo`` format or ``limit`` is invalid.
    """
    repo_name = _validate_repo_name(repo_name)
    if not 1 <= int(limit) <= 100:
        raise ValueError("limit must be between 1 and 100.")

    url = f"{GITHUB_API_URL}/repos/{repo_name}/issues"
    params: dict[str, Any] = {"state": "open", "per_page": int(limit)}
    if label:
        params["labels"] = label

    try:
        response = requests.get(
            url, params=params, headers=_auth_headers(), timeout=TIMEOUT_SECONDS
        )
    except requests.RequestException as exc:
        raise GitHubAPIError(
            f"Network error while contacting the GitHub API for '{repo_name}': {exc}"
        ) from exc

    if response.status_code == 404:
        raise GitHubAPIError(f"Repository '{repo_name}' not found (404).")
    if response.status_code == 403:
        raise GitHubAPIError(
            "GitHub rate limit exceeded (403). "
            "Set GITHUB_TOKEN/GH_TOKEN to raise the limit."
        )
    if response.status_code == 422:
        raise GitHubAPIError(
            f"Filter rejected for '{repo_name}' (422): the label "
            f"'{label}' may not exist."
        )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise GitHubAPIError(
            f"The GitHub API responded with HTTP {response.status_code} "
            f"for '{repo_name}'."
        ) from exc

    payload = response.json()
    # The /issues endpoint includes Pull Requests too: exclude items that
    # contain the `pull_request` key.
    issues = [
        item for item in payload if isinstance(item, dict) and "pull_request" not in item
    ]
    return [dict(item) for item in issues[:limit]]


__all__ = ["get_open_issues", "GitHubAPIError"]