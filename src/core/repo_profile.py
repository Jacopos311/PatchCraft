"""Repository voice profile for human-grade PR writing (Roadmap Step 4.3).

Ingests the repo's actual guidance — ``CONTRIBUTING.md``, pull-request /
issue templates, ``CODEOWNERS`` and recent merged PR titles — and compiles
a compact "voice" profile that the PR writer uses to match the project's
conventions and tone.

The profile is **cached** at ``<repo>/.patchcraft/repo_voice.json`` keyed
by the content hashes of its source files, so it is rebuilt only when any
source changes. Everything is best-effort: a repo without template files
gets a clean generic profile, and the optional GitHub fetch (recent merged
PRs, used only for tone calibration) degrades silently on errors.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

VOICE_CACHE_NAME = "repo_voice.json"
MAX_GUIDELINE_CHARS = 4000
MAX_TEMPLATE_CHARS = 8000
MAX_TITLE_EXAMPLES = 12
MAX_BODY_EXAMPLE_CHARS = 500

# Where PR/issue templates conventionally live (first match wins).
TEMPLATE_CANDIDATES = (
    (".github", "PULL_REQUEST_TEMPLATE.md"),
    ("PULL_REQUEST_TEMPLATE.md",),
    ("docs", "PULL_REQUEST_TEMPLATE.md"),
    (".github", "pull_request_template.md"),
)
GUIDELINE_CANDIDATES = (
    ("CONTRIBUTING.md",),
    (".github", "CONTRIBUTING.md"),
    ("docs", "CONTRIBUTING.md"),
)


class RepoVoice(BaseModel):
    """Compiled style/tone fingerprint of a repository."""

    contribution_guidelines: str = Field(
        default="", description="Contents of CONTRIBUTING.md (trimmed).")
    pull_request_template: str = Field(
        default="", description="PULL_REQUEST_TEMPLATE.md contents (trimmed).")
    codeowners: str = Field(
        default="", description="CODEOWNERS contents (trimmed).")
    recent_title_examples: List[str] = Field(
        default_factory=list, description="Titles of recent merged PRs.")
    source_hashes: Dict[str, str] = Field(
        default_factory=dict, description="Rel-path -> content hash keying the cache.")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _read_first(root: Path, candidates) -> str:
    """Contents of the first existing candidate file (or '')."""
    for parts in candidates:
        path = root.joinpath(*parts)
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.debug("Cannot read %s: %s", path, exc)
                return ""
    return ""


def _read_path(path: Path, max_chars: int) -> str:
    """First ``max_chars`` characters of ``path`` ('' when unreadable)."""
    try:
        return path.read_bytes().decode("utf-8", errors="replace")[:max_chars]
    except OSError as exc:
        logger.debug("Cannot read %s: %s", path, exc)
        return ""
def _source_hashes(root: Path) -> Dict[str, str]:
    """Content hashes of every file contributing to the voice profile."""
    sources: List[Path] = []
    for parts in (*TEMPLATE_CANDIDATES, *GUIDELINE_CANDIDATES):
        path = root.joinpath(*parts)
        if path.is_file():
            sources.append(path)
    codeowners = root / "CODEOWNERS"
    if codeowners.is_file():
        sources.append(codeowners)
    return {
        str(path.relative_to(root)): _sha256(_read_path(path, MAX_TEMPLATE_CHARS * 4))
        for path in sorted(sources)
    }


def _fetch_recent_merged(repo_name: str) -> List[Dict[str, str]]:
    """Titles/bodies of recent MERGED PRs (tone calibration only)."""
    try:
        import requests

        from src.github.issue_fetcher import GITHUB_API_URL, _auth_headers

        response = requests.get(
            f"{GITHUB_API_URL}/repos/{repo_name}/pulls",
            params={"state": "closed", "sort": "updated",
                    "direction": "desc", "per_page": 20},
            headers=_auth_headers(),
            timeout=10.0,
        )
        if response.status_code != 200:
            logger.debug("Recent-PR fetch skipped (HTTP %s).", response.status_code)
            return []
        out: List[Dict[str, str]] = []
        for item in response.json() or []:
            if not isinstance(item, dict) or not item.get("merge_commit_sha"):
                continue  # only merged pulls calibrate the "voice"
            out.append({
                "title": (item.get("title") or "").strip(),
                "body": (item.get("body") or "").strip()[:MAX_BODY_EXAMPLE_CHARS],
            })
            if len(out) >= MAX_TITLE_EXAMPLES:
                break
        return out
    except Exception as exc:  # noqa: BLE001 - profile build must never crash
        logger.debug("Recent-PR fetch skipped: %s: %s", type(exc).__name__, exc)
        return []


def build_repo_voice(
    repo_root: Union[str, Path],
    github_repo: Optional[str] = None,
    fetch_prs: bool = True,
) -> RepoVoice:
    """Build (or reuse the cached) voice profile for ``repo_root``.

    The cache at ``<root>/.patchcraft/repo_voice.json`` is keyed by the
    content hashes of its source files; any change triggers a rebuild.
    Without templates the profile fields stay empty — the reporter falls
    back to a clean generic structure.
    """
    root = Path(repo_root).expanduser().resolve()
    cache_path = root / ".patchcraft" / VOICE_CACHE_NAME

    hashes = _source_hashes(root)
    try:
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("source_hashes") == hashes:
                return RepoVoice.model_validate(cached)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        logger.debug("Ignoring repo voice cache: %s", exc)

    contrib = _read_first(root, GUIDELINE_CANDIDATES)[:MAX_GUIDELINE_CHARS]
    template = _read_first(root, TEMPLATE_CANDIDATES)[:MAX_TEMPLATE_CHARS]
    codeowners = _read_path(root / "CODEOWNERS", MAX_GUIDELINE_CHARS)

    recent: List[Dict[str, str]] = []
    if fetch_prs and github_repo:
        recent = _fetch_recent_merged(github_repo)

    profile = RepoVoice(
        contribution_guidelines=contrib,
        pull_request_template=template,
        codeowners=codeowners,
        recent_title_examples=[r["title"] for r in recent if r.get("title")],
        source_hashes=hashes,
    )

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(profile.model_dump(), ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.debug("Repo voice cache not persisted: %s", exc)
    return profile


__all__ = [
    "VOICE_CACHE_NAME",
    "RepoVoice",
    "build_repo_voice",
]