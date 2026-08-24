"""Caching layer (Roadmap Step 3.1): aggressive-but-safe caching.

Two independent, disk-backed caches live under ``<repo>/.patchcraft/cache/``:

* :class:`LLMMemoCache` — memoizes LLM completions. Identical
  ``(model, messages)`` calls reuse the earlier response, saving money when
  retries re-send identical prompts. Entries are scoped PER ACCOUNT (the
  ``OPENROUTER_API_KEY`` hash is part of the storage path), so responses are
  never shared across different accounts.
* :class:`TestResultCache` — memoizes the verdict of *TARGETED* test runs,
  keyed by the post-patch content of every touched file plus the exact test
  subset. It is NEVER used for the final full-suite gate (that must always
  execute for real).

Global kill switch: setting ``PATCHCRAFT_NO_CACHE=1`` (truthy values: 1,
true, yes, on) disables every cache regardless of constructor flags.

Invalidation rules are documented in ``docs/caching.md``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence, Union

logger = logging.getLogger(__name__)

# Environment kill switch (documented in docs/caching.md).
ENV_NO_CACHE = "PATCHCRAFT_NO_CACHE"

_TRUTHY = {"1", "true", "yes", "on"}

# Output tails larger than this are truncated before being stored, so a huge
# failure dump cannot bloat the cache directory.
MAX_CACHED_OUTPUT_CHARS = 20_000


def env_disables_cache() -> bool:
    """True when ``PATCHCRAFT_NO_CACHE`` is set to a truthy value."""
    return (os.getenv(ENV_NO_CACHE, "").strip().lower() in _TRUTHY)


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()


class _JsonDiskCache:
    """Tiny JSON-file store shared by both caches (one file per key)."""

    VERSION = 1

    def __init__(self, directory: Union[str, Path]) -> None:
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def lookup(self, key: str) -> Optional[dict[str, Any]]:
        """Return the stored payload for ``key`` or ``None`` on miss."""
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Ignoring unreadable cache entry %s: %s", path.name, exc)
            return None
        if (
            not isinstance(payload, dict)
            or payload.get("version") != self.VERSION
            or payload.get("key") != key
        ):
            logger.debug("Ignoring stale/corrupt cache entry %s.", path.name)
            return None
        return payload

    def store(self, key: str, data: dict[str, Any]) -> None:
        """Persist ``data`` under ``key`` (best effort, never raises)."""
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            payload = {"version": self.VERSION, "key": key, **data}
            self._path(key).write_text(
                json.dumps(payload, indent=1, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Could not write cache entry %s: %s", key, exc)

    def clear(self) -> int:
        """Delete every entry; returns how many files were removed."""
        removed = 0
        if not self.directory.is_dir():
            return 0
        for path in self.directory.glob("*.json"):
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                logger.debug("Could not remove %s: %s", path, exc)
        return removed

class LLMMemoCache:
    """Disk-backed memo cache for LLM completions (Roadmap Step 3.1).

    Keys combine the account hash (so different ``OPENROUTER_API_KEY``
    accounts NEVER share entries), the requested model, the full message
    texts, the response schema (when structured output is requested) and the
    output token budget — i.e. everything that meaningfully affects the
    completion.

    Parameters
    ----------
    base_dir : str | Path | None
        Cache root directory. ``None`` defaults to
        ``<cwd>/.patchcraft/cache``; the orchestrator pins it to the target
        repository's ``.patchcraft/cache`` for reproducible per-repo scope.
    enabled : bool
        Master switch (default off; the orchestrator enables it). Requesting
        ``True`` still resolves to ``False`` when ``PATCHCRAFT_NO_CACHE`` is
        set.
    """

    def __init__(
        self,
        base_dir: Union[str, Path, None] = None,
        enabled: bool = False,
    ) -> None:
        self.base_dir = (
            Path(base_dir) if base_dir is not None
            else Path.cwd() / ".patchcraft" / "cache"
        )
        self.enabled = bool(enabled) and not env_disables_cache()

    # ------------------------------------------------------------------
    # Account scoping
    # ------------------------------------------------------------------
    @staticmethod
    def account_hash() -> str:
        """Short hash of the current ``OPENROUTER_API_KEY`` (or empty key)."""
        raw = (os.getenv("OPENROUTER_API_KEY") or "").strip()
        return _sha256_hex(f"openrouter|{raw}")[:16]

    def _store(self) -> _JsonDiskCache:
        # Per-account directory: responses never leak across accounts even
        # if two accounts work on the same machine.
        return _JsonDiskCache(self.base_dir / "llm" / self.account_hash())

    # ------------------------------------------------------------------
    # Key construction
    # ------------------------------------------------------------------
    def make_key(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema_json: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Stable key covering everything that shapes the completion."""
        payload = json.dumps(
            {
                "account": self.account_hash(),
                "model": model,
                "system": system_prompt,
                "user": user_prompt,
                "schema": schema_json,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return _sha256_hex(payload)

    # ------------------------------------------------------------------
    # Lookup / store
    # ------------------------------------------------------------------
    def lookup(self, key: str) -> Optional[str]:
        """Cached completion text for ``key``, or ``None``."""
        if not self.enabled:
            return None
        payload = self._store().lookup(key)
        if payload is None:
            return None
        content = payload.get("content")
        return content if isinstance(content, str) else None

    def store(self, key: str, content: str) -> None:
        """Cache ``content`` for ``key`` (ignored when disabled)."""
        if not self.enabled:
            return
        self._store().store(key, {"content": content})

    def clear(self) -> int:
        """Remove every cached completion for the current account."""
        return self._store().clear()


class TestResultCache:
    """Verdict cache for TARGETED test runs (Roadmap Step 3.1).

    A cached verdict is reused ONLY when the exact same patch (fingerprinted
    through the post-patch content of every touched file) is re-proposed for
    the exact same test subset. The FULL-SUITE gate is never served from this
    cache — see ``docs/caching.md`` for the full rules.
    """

    # Prevents pytest from collecting this class as a *test class*.
    __test__ = False

    def __init__(
        self,
        repo_root: Union[str, Path],
        enabled: bool = False,
    ) -> None:
        self.directory = Path(repo_root) / ".patchcraft" / "cache" / "test_results"
        self.enabled = bool(enabled) and not env_disables_cache()

    @staticmethod
    def make_key(patch_fingerprint: str, targets: Sequence[str]) -> str:
        """Key = post-patch fingerprint + exact (sorted) target subset."""
        payload = json.dumps(
            {"fp": patch_fingerprint, "targets": sorted(targets)},
            sort_keys=True,
        )
        return _sha256_hex(payload)

    def lookup(self, key: str) -> Optional[dict[str, Any]]:
        """Stored verdict payload for ``key``, or ``None``."""
        if not self.enabled:
            return None
        payload = _JsonDiskCache(self.directory).lookup(key)
        if payload is None:
            return None
        verdict = payload.get("verdict")
        return verdict if isinstance(verdict, dict) else None

    def store(
        self,
        key: str,
        *,
        success: bool,
        exit_code: int,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        """Cache a targeted-run verdict (output tails truncated)."""
        if not self.enabled:
            return
        _JsonDiskCache(self.directory).store(key, {
            "verdict": {
                "success": bool(success),
                "exit_code": int(exit_code),
                "stdout": (stdout or "")[-MAX_CACHED_OUTPUT_CHARS:],
                "stderr": (stderr or "")[-MAX_CACHED_OUTPUT_CHARS:],
            },
        })

    def clear(self) -> int:
        """Remove every cached verdict."""
        return _JsonDiskCache(self.directory).clear()


# ---------------------------------------------------------------------------
# Module-level singleton used by call_llm (agents never touch the cache
# directly; they just call call_llm as usual).
# ---------------------------------------------------------------------------
_memo_cache: Optional[LLMMemoCache] = None


def get_memo_cache() -> LLMMemoCache:
    """Process-wide memo cache instance (created lazily, disabled default)."""
    global _memo_cache
    if _memo_cache is None:
        _memo_cache = LLMMemoCache()
    return _memo_cache


def configure_memo_cache(
    enabled: Optional[bool] = None,
    base_dir: Union[str, Path, None] = None,
) -> LLMMemoCache:
    """Configure the process-wide memo cache (used by the orchestrator/CLI).

    ``PATCHCRAFT_NO_CACHE`` always wins over ``enabled=True``.
    """
    cache = get_memo_cache()
    if base_dir is not None:
        cache.base_dir = Path(base_dir)
    if enabled is not None:
        cache.enabled = bool(enabled) and not env_disables_cache()
    return cache


def reset_memo_cache() -> None:
    """Drop the singleton (used by tests to isolate configurations)."""
    global _memo_cache
    _memo_cache = None


__all__ = [
    "ENV_NO_CACHE",
    "LLMMemoCache",
    "TestResultCache",
    "env_disables_cache",
    "get_memo_cache",
    "configure_memo_cache",
    "reset_memo_cache",
]
