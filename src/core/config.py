"""Typed ``.patchcraft.yml`` configuration (Roadmap Step 3.3).

An OPTIONAL file at the repository root; every field has a sensible default,
so repositories work without it::

    # .patchcraft.yml
    model: openrouter/deepseek/deepseek-chat
    fallback_models:
      - openrouter/anthropic/claude-3.5-sonnet
      - openrouter/openai/gpt-4o
    retrieval_k: 12
    token_budget: 200000        # LLM tokens per task
    time_budget: 1800           # wall-clock seconds per task
    min_credits: 1.0            # OpenRouter credit floor
    max_retries: 8              # patch+test iteration cap
    ignore_globs:               # excluded from indexing/context
      - "vendor/**"
    commit_style: conventional  # conventional | repo-derived (used by M4 git flow)
    pr:
      draft: true
    test:
      command: "python -m pytest"
      timeout_full_suite: 600

Rules:

* Validation errors raise :class:`ConfigError` with clear ENGLISH messages
  (the CLI maps them to exit code 2).
* Unknown keys produce WARNINGS but never fail the run.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

CONFIG_FILENAMES = (".patchcraft.yml", ".patchcraft.yaml")


class ConfigError(RuntimeError):
    """Invalid PatchCraft configuration (maps to CLI exit code 2)."""


class TestConfig(BaseModel):
    """Test-execution overrides."""

    command: Optional[str] = Field(
        default=None,
        description="Explicit test command override (e.g. 'python -m pytest').",
    )
    timeout_full_suite: Optional[float] = Field(
        default=None,
        gt=0,
        description="Full-suite run timeout in seconds.",
    )


class PRConfig(BaseModel):
    """Pull-request preferences."""

    # Default TRUE per Step 4.2: every published PR is a draft until the
    # evaluation/safety milestone (M5) proves the pipeline reliable enough.
    draft: bool = Field(default=True, description="Open pull requests as drafts.")


class PatchcraftConfig(BaseModel):
    """Typed view of ``.patchcraft.yml``."""

    model: Optional[str] = Field(default=None, description="Primary LLM model.")
    fallback_models: Optional[List[str]] = Field(
        default=None,
        description="Fallback LLM chain (requested model is always tried first).",
    )
    retrieval_k: Optional[int] = Field(
        default=None, ge=1, le=200,
        description="Files retrieved by BM25 for the diagnostic context.",
    )
    token_budget: Optional[int] = Field(
        default=None, ge=1, description="Max total LLM tokens per task."
    )
    time_budget: Optional[float] = Field(
        default=None, ge=1, description="Wall-clock budget per task (seconds)."
    )
    min_credits: Optional[float] = Field(
        default=None, ge=0, description="Halt if OpenRouter credits drop below this."
    )
    max_retries: Optional[int] = Field(
        default=None, ge=1, description="Hard cap on patch+test iterations."
    )
    ignore_globs: List[str] = Field(
        default_factory=list,
        description="Glob patterns excluded from indexing and context building.",
    )
    commit_style: Literal["conventional", "repo-derived"] = Field(
        default="repo-derived",
        description="Commit message style for the git workflow (Milestone 4).",
    )
    test: TestConfig = Field(default_factory=TestConfig)
    pr: PRConfig = Field(default_factory=PRConfig)


def find_config_file(repo_root: Union[str, Path]) -> Optional[Path]:
    """Path of the repository's ``.patchcraft.yml``, or ``None``."""
    root = Path(repo_root)
    return next(
        (root / name for name in CONFIG_FILENAMES if (root / name).is_file()),
        None,
    )


# Top-level keys understood by this schema (everything else warns).
_KNOWN_TOP_LEVEL_KEYS = frozenset(
    {
        "model", "fallback_models", "retrieval_k", "token_budget",
        "time_budget", "min_credits", "max_retries", "ignore_globs",
        "commit_style", "test", "pr",
    }
)


def _collect_unknown_keys(data: Any, prefix: str = "") -> List[str]:
    """Dotted paths of unrecognized keys (top level and known sub-sections)."""
    unknown: List[str] = []
    if not isinstance(data, dict):
        return unknown
    for key, value in data.items():
        dotted = f"{prefix}{key}"
        if prefix == "" and key == "sandbox":
            continue  # legacy section: tolerated silently (runner owns it)
        if prefix == "" and key not in _KNOWN_TOP_LEVEL_KEYS:
            unknown.append(dotted)
        elif dotted in {"test", "pr"} and isinstance(value, dict):
            allowed = set(TestConfig.model_fields) | set(PRConfig.model_fields)
            unknown.extend(
                f"{dotted}.{sub}" for sub in value if sub not in allowed
            )
    return unknown


def _legacy_sandbox_into_test(data: dict[str, Any], warnings: List[str]) -> None:
    """Map the legacy ``sandbox:`` keys onto the typed ``test:`` section.

    The dedicated ``test:`` section always wins when both are present.
    """
    sandbox = data.pop("sandbox", None)
    if not isinstance(sandbox, dict):
        return
    test = data.setdefault("test", {})
    if not isinstance(test, dict):
        return
    for legacy_key in ("command", "timeout_full_suite"):
        if legacy_key in sandbox and legacy_key not in test:
            test[legacy_key] = sandbox[legacy_key]
            warnings.append(
                f"Key 'sandbox.{legacy_key}' is legacy and will be removed: "
                f"move it to 'test.{legacy_key}'."
            )


def load_config_with_warnings(
    repo_root: Union[str, Path],
) -> Tuple[PatchcraftConfig, List[str]]:
    """Load and validate ``<repo>/.patchcraft.yml``.

    Returns ``(config, warnings)``. Raises :class:`ConfigError` for unreadable
    files, malformed YAML or schema violations (message includes the file
    name and an English explanation suitable for CI logs).
    """
    root = Path(repo_root)
    config_path = find_config_file(root)
    if config_path is None:
        return PatchcraftConfig(), []

    try:
        import yaml  # soft dependency: litellm already ships PyYAML
    except ImportError as exc:  # pragma: no cover - PyYAML ships with litellm
        raise ConfigError(
            f"{config_path.name} found but PyYAML is not installed "
            f"(`pip install pyyaml`)."
        ) from exc

    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read {config_path.name}: {exc}") from exc

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"{config_path.name} is not valid YAML: {str(exc).splitlines()[0]}"
        ) from exc

    if data is None:
        data = {}  # empty file == defaults
    if not isinstance(data, dict):
        raise ConfigError(
            f"{config_path.name} must contain a YAML mapping of settings "
            f"(found {type(data).__name__})."
        )

    warnings: List[str] = []
    for unknown in sorted(_collect_unknown_keys(data)):
        warnings.append(f"Unknown configuration key '{unknown}' was ignored.")

    _legacy_sandbox_into_test(data, warnings)

    try:
        config = PatchcraftConfig.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(
            f"'{'.'.join(str(p) for p in err['loc'])}': {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigError(
            f"{config_path.name} is invalid: {details}. "
            f"See the README configuration reference for the schema."
        ) from exc

    # Normalize glob patterns early so consumers get clean posix-style globs.
    config.ignore_globs = [
        g.strip().replace("\\\\", "/") for g in config.ignore_globs if g.strip()
    ]
    return config, warnings


def load_config(repo_root: Union[str, Path]) -> PatchcraftConfig:
    """Load the repository configuration; warnings are logged, not returned."""
    config, warnings = load_config_with_warnings(repo_root)
    for warning in warnings:
        logger.warning("[patchcraft.config] %s", warning)
    return config


def resolve_issue_reference(reference: str) -> Tuple[Optional[str], int]:
    """Parse an issue URL or number into ``(owner/repo | None, number)``.

    Accepted forms::

        https://github.com/{owner}/{repo}/issues/{number}
        {number}
        #{number}

    Raises :class:`ConfigError` for anything else (English message).
    """
    text = (reference or "").strip()
    url_match = re.match(
        r"^https?://(?:www\.)?github\.com/([^/\s]+)/([^/\s]+)/issues/(\d+)/?$",
        text,
    )
    if url_match:
        owner, name, number = url_match.groups()
        return f"{owner}/{name}", int(number)
    number_match = re.match(r"^#?(\d+)$", text)
    if number_match:
        return None, int(number_match.group(1))
    raise ConfigError(
        f"Issue reference {reference!r} is not valid: use a GitHub issue URL "
        f"(https://github.com/owner/repo/issues/123) or a plain number (123)."
    )


__all__ = [
    "CONFIG_FILENAMES",
    "ConfigError",
    "PatchcraftConfig",
    "TestConfig",
    "PRConfig",
    "find_config_file",
    "load_config",
    "load_config_with_warnings",
    "resolve_issue_reference",
]
