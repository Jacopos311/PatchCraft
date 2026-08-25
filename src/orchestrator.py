"""PatchCraft orchestrator: main loop Diagnosis -> Patch -> Test -> Self-Correction.

:func:`run_patchcraft_loop` coordinates the end-to-end flow:

1. **Context**: reads ``architecture.md``, ``README.md`` and the relevant
   source files of the target repository.
2. **Diagnosis**: the :mod:`src.agents.diagnostic` agent analyzes the problem.
3. **Loop up to ``max_retries``**:
   a. the :mod:`src.agents.coder` agent generates (or corrects) the patch and
      applies it to the **real files**;
   b. :class:`src.sandbox.runner.SandboxRunner` runs the test suite;
   c. if tests pass, the :mod:`src.agents.reporter` agent generates the report;
   d. if they fail, ``stdout``/``stderr`` are passed to the self-correction
      agent to fix the patch.

Progress is shown on screen with the ``rich`` library; an optional
``event_sink`` callback streams structured milestones for GUIs/Loggers.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from src.agents.coder import (
    EditHunk,
    Patch,
    correct_patch,
    dynamic_patch_budget,
    generate_patch,
)
from src.agents.diagnostic import diagnose
from src.agents.reporter import PatchReport, build_diff_stat, generate_report
from src.core.repo_profile import RepoVoice, build_repo_voice
from src.sandbox.runner import SandboxRunner, TestResult
from src.sandbox.failures import extract_failures, format_failure_report
from src.core.cache import (
    TestResultCache,
    configure_memo_cache,
    get_memo_cache,
)
from src.core.runstats import begin_run
from src.core.gitflow import (
    GitFlow,
    GitSafetyError,
    build_branch_name,
    build_commit_message,
    detect_commit_style,
    get_recent_subjects,
    pop_worktree_cleanup,
    register_worktree_cleanup,
)


# Lazy-imported targeted test selection (Step 2.1) — see _select_test_targets().
from src.core.targeted_tests import (
    TestSelectionResult as _TestSelectionResult,
    select_targeted_tests as _select_targeted_tests,
)

console = Console()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Limits for context collection (protect the LLM context window)
# ---------------------------------------------------------------------------
MAX_SOURCE_FILES = 20            # maximum number of source files read
MAX_FILE_CHARS = 12_000          # characters per single file
MAX_CONTEXT_CHARS = 60_000       # overall cap for the LLM context

SOURCE_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".yaml", ".yml",
    ".mjs", ".cjs", ".go", ".rs", ".java", ".c", ".h", ".cpp", ".hpp", ".sh",
}

# Maximum characters of raw test output appended to the corrector feedback
# after the structured failure report (Step 2.2): structured data leads,
# raw output stays available as fallback without flooding the prompt.
RAW_FEEDBACK_TAIL_CHARS = 4000

IGNORED_DIR_PARTS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__",
    ".venv", "venv", ".env", "dist", "build", ".pytest_cache",
    ".mypy_cache", ".idea", ".vscode",
}


class RunResult(BaseModel):
    """Overall outcome of the PatchCraft loop."""

    success: bool = Field(description="True if tests passed within max_retries.")
    iterations: int = Field(description="Number of executed iterations.")
    report: Optional[PatchReport] = Field(default=None, description="Final report if successful.")
    test_errors: list[str] = Field(default_factory=list, description="stderr of every failed iteration.")
    files_changed: list[str] = Field(
        default_factory=list, description="Files actually modified."
    )
    halt_reason: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable reason the loop stopped without success "
            "(retry limit, stagnation, token/time budget or credit floor)."
        ),
    )
    git_branch: Optional[str] = Field(
        default=None,
        description=(
            "patchcraft/* branch holding the result when the target repo is "
            "a git repository (Step 4.1). None on failure or non-git repos."
        ),
    )
    commit_sha: Optional[str] = Field(
        default=None,
        description="SHA of the commit created on git_branch (Step 4.1).",
    )


# ---------------------------------------------------------------------------
# Goal-driven loop guardrails (loop detection + budget/safety limits)
# ---------------------------------------------------------------------------
# The self-correction loop runs until tests pass. To guarantee it can never
# spin forever or burn unbounded tokens, three safety nets are enforced:
#
# * STAGNATION: when the SAME failure signature (or an identical patch)
#   repeats over consecutive iterations, the model first receives an
#   explicit strategy-change directive; if it keeps producing the same
#   failing result the loop halts gracefully with a clear report.
STAGNATION_STRATEGY_AFTER = 2   # consecutive identical failures before a strategy change is forced
STAGNATION_HALT_AFTER = 5       # consecutive identical failures before the loop halts


def _error_signature(test_result: object) -> str:
    """Stable fingerprint of a test failure used for loop detection.

    Only *volatile* tokens (execution times, memory addresses) are
    normalized; assertion values and failure names are preserved, because a
    changed assertion value means the agent is making progress. Only the
    tail of the output is kept, where test frameworks print their summary.
    """
    text = f"{getattr(test_result, 'stdout', '')}\n{getattr(test_result, 'stderr', '')}"
    text = re.sub(r"\d+(?:\.\d+)?\s*(?:s|sec|seconds|ms)\b", "<t>", text)
    text = re.sub(r"0x[0-9a-fA-F]+", "<addr>", text)
    lines = [re.sub(r"\s+", " ", line.strip()) for line in text.splitlines() if line.strip()]
    blob = "\n".join(lines[-40:])
    return hashlib.sha1(blob.encode("utf-8", "replace")).hexdigest()


def _remaining_credits() -> Optional[float]:
    """Remaining OpenRouter credits (``limit - usage``), ``None`` if unknown."""
    try:
        from src.core.credits import fetch_credits

        data = fetch_credits()
    except Exception as exc:  # noqa: BLE001 - the credit guard must never break the loop
        logger.debug("Credit check skipped: %s: %s", type(exc).__name__, exc)
        return None
    if not data:
        return None
    limit = data.get("limit")
    usage = data.get("usage")
    if isinstance(limit, (int, float)) and isinstance(usage, (int, float)):
        return max(0.0, float(limit) - float(usage))
    return None


def _env_optional(name: str, cast: Callable[[str], object]) -> Optional[object]:
    """Read an optional numeric environment variable."""
    raw = os.getenv(name)
    if not raw or not raw.strip():
        return None
    try:
        return cast(raw)
    except ValueError:
        logger.warning("Invalid value for %s: %r — ignored.", name, raw)
        return None


def _retrieval_k_for(repo_root: Path) -> int:
    """BM25 retrieval width for ``repo_root`` (Step 3.3).

    Precedence: ``PATCHCRAFT_RETRIEVAL_K`` env > ``retrieval_k`` in
    ``<repo>/.patchcraft.yml`` > built-in default. Invalid values always
    degrade gracefully to the built-in default.
    """
    from src.core.retrieval import DEFAULT_RETRIEVAL_K

    raw_env = os.getenv("PATCHCRAFT_RETRIEVAL_K")
    if raw_env and raw_env.strip():
        try:
            return max(1, int(raw_env))
        except ValueError:
            logger.warning(
                "Invalid PATCHCRAFT_RETRIEVAL_K=%r — using default %d.",
                raw_env, DEFAULT_RETRIEVAL_K,
            )
            return DEFAULT_RETRIEVAL_K
    try:
        from src.core.config import load_config

        configured = load_config(repo_root).retrieval_k
    except Exception as exc:  # noqa: BLE001 - config must never break context
        logger.debug("Retrieval config unavailable (%s: %s).", type(exc).__name__, exc)
        return DEFAULT_RETRIEVAL_K
    return configured if configured is not None else DEFAULT_RETRIEVAL_K


# ---------------------------------------------------------------------------
# Context collection (documentation + sources)
# ---------------------------------------------------------------------------
def _read_text_limited(path: Path, limit: int = MAX_FILE_CHARS) -> str:
    """Read a text file, truncating it to ``limit`` characters."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) > limit:
        text = f"{text[:limit]}\n... [truncated to {limit} characters]"
    return text

def _discover_source_files(repo_root: Path) -> list[Path]:
    """Find the relevant source files, ignoring venv, node_modules, etc."""
    sources: list[Path] = []
    
    for root, dirs, files in os.walk(repo_root, topdown=True):
        # Filter dirs in-place BEFORE walking them (avoids Windows symlink crashes / node_modules)
        dirs[:] = [d for d in dirs if d not in IGNORED_DIR_PARTS and not d.startswith(".")]
        
        for file in sorted(files):
            path = Path(root) / file
            if path.suffix.lower() in SOURCE_EXTENSIONS:
                sources.append(path)
                if len(sources) >= MAX_SOURCE_FILES:
                    return sources

    return sources


def _prompt_budget(env_name: str, default_tokens: int) -> int:
    """Read an optional prompt-budget env override (positive tokens)."""
    try:
        from src.core.prompts import estimate_tokens
    except Exception:  # noqa: BLE001 - a broken override must never break context building
        estimate_tokens = None  # type: ignore[assignment]
    raw = os.getenv(env_name)
    if not raw or not raw.strip():
        return default_tokens
    try:
        value = int(raw)
        return value if value > 0 else default_tokens
    except ValueError:
        logger.warning("Invalid %s=%r — using default %d tokens.", env_name, raw, default_tokens)
        return default_tokens


def build_context(repo_root: Path, issue_description: str, max_tokens: Optional[int] = None) -> str:
    """Compose the context for the model: issue + repo map + docs + sources.

    Assembled with the priority-aware prompt compiler (Step 1.4): issue text
    is always kept, repo map comes before docs/sources, and the lowest-
    priority sections (raw source files) are trimmed/dropped first when the
    budget runs out.
    """
    if max_tokens is None:
        max_tokens = _prompt_budget("PATCHCRAFT_CONTEXT_BUDGET", int(MAX_CONTEXT_CHARS / 5))
    from src.core.prompts import TRIM_HEAD, PromptCompiler

    compiler = PromptCompiler(max_tokens=max_tokens)

    issue = issue_description.strip()
    if issue:
        compiler.add("issue", f"# ISSUE TO SOLVE\n{issue}", priority=1, trim=TRIM_HEAD)

    # Structural overview first (Roadmap Step 1.1): a compact symbol map of
    # the whole repository, built incrementally and cached on disk.
    repo_index_obj = None
    try:
        from src.core.repo_index import RepoIndex

        repo_index_obj = RepoIndex.build(repo_root)
        repo_map_text = repo_index_obj.repo_map()
        if repo_map_text:
            compiler.add("repo_map", f"# REPOSITORY MAP\n{repo_map_text}", priority=2)
    except Exception as exc:  # noqa: BLE001 - indexing must never break context building
        logger.debug("Repo index unavailable (%s: %s); continuing without map.", type(exc).__name__, exc)

    docs: list[Path] = []
    for name in ("architecture.md", "README.md"):
        candidate = repo_root / name
        if candidate.is_file():
            docs.append(candidate)
        else:
            base = name.rsplit(".", 1)[0]  # e.g. "README" from "README.md"
            for variant in (
                repo_root / f"{base}.markdown",
                repo_root / f"{base}.MD",
            ):
                if variant.is_file() and variant not in docs:
                    docs.append(variant)

    for doc in docs:
        content = _read_text_limited(doc, limit=MAX_FILE_CHARS)
        if content:
            compiler.add(f"doc:{doc.relative_to(repo_root).as_posix()}", f"# FILE: {doc.relative_to(repo_root)}\n```\n{content}\n```", priority=3)

    sources = _discover_source_files(repo_root)
    if repo_index_obj is not None and sources:
        try:
            from src.core.retrieval import select_files

            k = _retrieval_k_for(repo_root)
            retrieved = [
                repo_root / rel
                for rel in select_files(issue_description, repo_index_obj, k=k)
                if (repo_root / rel).is_file()
            ]
        except Exception as exc:  # noqa: BLE001 - retrieval must never break context building
            logger.debug("Retrieval skipped (%s: %s).", type(exc).__name__, exc)
            retrieved = []
    else:
        retrieved = []

    resolved_retrieved = {p.resolve() for p in retrieved}
    rest = [p for p in sources if p.resolve() not in resolved_retrieved]
    ordered_sources = [*retrieved, *rest[:MAX_SOURCE_FILES]]

    for source in ordered_sources:
        content = _read_text_limited(source)
        if not content:
            continue
        marker = " [RETRIEVED: relevant to this issue]" if source in retrieved else ""
        compiler.add(
            f"source:{source.relative_to(repo_root).as_posix()}",
            f"# FILE: {source.relative_to(repo_root)}{marker}\n```\n{content}\n```",
            priority=4,
        )

    return compiler.compile().text


def build_coder_context(repo_root: Path, affected_files: Sequence[str]) -> str:
    """Collect the ACTUAL contents of the diagnosed affected files.

    The coder agent needs the real source code to produce patches that match
    the files on disk (without it, patches are invented and tests always
    fail, triggering the final rollback). Paths are validated and
    fuzzy-matched against the repository index (Step 1.2) before use, and
    the overall size is capped like ``build_context``.
    """
    try:
        from src.core.retrieval import resolve_affected_files

        affected_files = resolve_affected_files(repo_root, list(affected_files))
    except Exception as exc:  # noqa: BLE001 - resolution must never break context building
        logger.debug("Affected-path resolution skipped (%s: %s).", type(exc).__name__, exc)

    from src.core.prompts import PromptCompiler

    compiler = PromptCompiler(max_tokens=_prompt_budget("PATCHCRAFT_CODER_BUDGET", int(MAX_CONTEXT_CHARS / 2)))
    seen: set[Path] = set()
    for rel in affected_files:
        if not rel or not rel.strip():
            continue
        target = _resolve_patch_path(rel.strip(), repo_root)
        if target is None or not target.is_file() or target in seen:
            continue
        seen.add(target)
        content = _read_text_limited(target)
        if not content:
            continue
        compiler.add(
            f"src:{target.relative_to(repo_root).as_posix()}",
            f"# FILE: {target.relative_to(repo_root).as_posix()}\n```\n{content}\n```",
            priority=2,  # affected-file content: kept before everything optional
        )
    return compiler.compile().text


# ---------------------------------------------------------------------------
# Applying patches to the real files
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PatchApplyResult:
    """Outcome of :func:`apply_patch_detailed`."""

    snapshots: dict[Path, Optional[str]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def _normalize_line(line: str) -> str:
    """Whitespace-insensitive form of a line used for tolerant matching."""
    return " ".join(line.split())


def apply_edits_to_text(
    content: str,
    hunks: Sequence[EditHunk],
) -> tuple[str, list[str]]:
    """Apply surgical search/replace hunks to ``content``.

    Matching strategy per hunk, in order: exact substring; then a single
    whitespace-normalized line-window match (tolerates indentation drift and
    EOL differences). Ambiguous or missing anchors abort the whole file edit:
    the original content is returned together with actionable problem
    descriptions for the self-correction loop (atomic per file).
    """
    problems: list[str] = []
    working = content

    for idx, hunk in enumerate(hunks, start=1):
        find, replace = hunk.find, hunk.replace
        if not find.strip():
            problems.append(f"edit #{idx}: the 'find' snippet is empty.")
            break  # offsets of later hunks are unreliable after a failure

        occurrences = working.count(find)
        if occurrences == 1:
            working = working.replace(find, replace, 1)
            continue
        if occurrences > 1:
            problems.append(
                f"edit #{idx}: 'find' snippet matches {occurrences} locations "
                f"(ambiguous). Include more surrounding lines to make it unique."
            )
            break

        # Whitespace-tolerant fallback on normalized lines.
        src_keep = working.splitlines(keepends=True)
        src_norm = [_normalize_line(line) for line in src_keep]
        norm_find = [_normalize_line(line) for line in find.splitlines()]
        span = len(norm_find)
        matches = (
            [i for i in range(len(src_norm) - span + 1) if src_norm[i:i + span] == norm_find]
            if span and span <= len(src_norm)
            else []
        )
        if len(matches) != 1:
            detail = f"matches {len(matches)} locations approximately" if matches else "was not found"
            preview = find if len(find) <= 120 else find[:117] + "..."
            problems.append(
                f"edit #{idx}: 'find' snippet {detail}: {preview!r}. Copy it "
                f"EXACTLY from the current file content shown in the context."
            )
            break

        start = matches[0]
        end = start + span
        last_original = src_keep[end - 1]
        eol = "\r\n" if last_original.endswith("\r\n") else ("\n" if last_original.endswith("\n") else "")
        repl_lines = replace.splitlines()
        replacement = [line + "\n" for line in repl_lines[:-1]]
        if repl_lines:
            replacement.append(repl_lines[-1] + eol)
        src_keep[start:end] = replacement
        working = "".join(src_keep)

    if problems:
        # Atomic per file: never leave a half-applied multi-hunk edit.
        return content, problems
    return working, []


def _resolve_patch_path(patch_file: str, repo_root: Path) -> Optional[Path]:
    """Resolve a patch path safely inside ``repo_root``.

    Returns ``None`` if the patch attempts to escape the repo root
    (path traversal) — in that case the change is discarded.
    """
    root = Path(repo_root).expanduser().resolve()  # tolerate relative roots
    raw = Path(patch_file.replace("\\", "/"))
    if raw.is_absolute():
        # If the model emitted an absolute path, try to map it back to the repo.
        try:
            raw = raw.relative_to(root)
        except ValueError:
            return None
    candidate = (root / raw).resolve()
    if candidate != root and root not in candidate.parents:
        console.print(f"[yellow]⚠ Path outside the repo discarded: {patch_file}[/]")
        return None
    return candidate


def apply_patch(patch: Patch, repo_root: Path) -> dict[Path, Optional[str]]:
    """Write patch files to disk.

    Returns a snapshot ``{path: original_content|None}`` of the modified
    files, used to compute the diff or roll back. ``None`` means the file did
    not exist before (it will be created). Use :func:`apply_patch_detailed`
    when per-file application problems are needed as self-correction feedback.
    """
    return apply_patch_detailed(patch, repo_root).snapshots


def apply_patch_detailed(patch: Patch, repo_root: Path) -> PatchApplyResult:
    """Apply a patch, returning snapshots plus actionable application problems."""
    repo_root = repo_root.expanduser().resolve()  # tolerate relative roots
    applied: dict[Path, Optional[str]] = {}
    problems: list[str] = []
    if not patch.files:
        console.print("[yellow]The patch contains no files: no changes applied.[/]")
        return PatchApplyResult(applied, problems)

    for file_patch in patch.files:
        target = _resolve_patch_path(file_patch.file_path, repo_root)
        if target is None:
            continue

        if file_patch.edits:
            # Surgical mode: modify an existing file via search/replace hunks.
            if not target.is_file():
                problems.append(
                    f"{file_patch.file_path}: cannot apply edits — file does not "
                    f"exist. Use 'new_content' instead to create it."
                )
                console.print(f"[yellow]⚠ {file_patch.file_path}: edits skipped (missing file)[/]")
                continue
            try:
                original = target.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                problems.append(f"{file_patch.file_path}: cannot read file ({exc}).")
                continue
            new_content, edit_problems = apply_edits_to_text(original, file_patch.edits)
            if edit_problems:
                problems.extend(f"{file_patch.file_path}: {p}" for p in edit_problems)
                console.print(f"[red]✗ {file_patch.file_path}: {len(edit_problems)} edit(s) failed[/]")
                continue
            if new_content == original:
                console.print(f"[dim]{target.relative_to(repo_root)} already up to date[/]")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(new_content, encoding="utf-8")
            applied[target] = original
            console.print(
                f"[green] ✔ {target.relative_to(repo_root)} "
                f"(modified, {len(file_patch.edits)} surgical edit(s))[/]"
            )
            continue

        # Whole-content mode: create a brand-new file or fully rewrite one.
        original: Optional[str] = None
        if target.is_file():
            original = target.read_text(encoding="utf-8", errors="replace")
        if original == file_patch.new_content:
            console.print(f"[dim]{target.relative_to(repo_root)} already up to date[/]")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(file_patch.new_content, encoding="utf-8")
        applied[target] = original
        rel = target.relative_to(repo_root)
        status = "created" if original is None else "modified"
        console.print(f"[green] ✔ {rel} ({status})[/]")
    return PatchApplyResult(applied, problems)


def compute_diff(repo_root: Path, snapshots: dict[Path, Optional[str]]) -> str:
    """Generate a unified diff between the original snapshot and current state."""
    parts: list[str] = []
    for target, original in sorted(snapshots.items()):
        current = target.read_text(encoding="utf-8", errors="replace")
        rel = str(target.relative_to(repo_root)).replace("\\", "/")
        before = (original or "").splitlines(keepends=True)
        after = current.splitlines(keepends=True)
        fromfile = f"a/{rel}" if original is not None else "/dev/null"
        tofile = f"b/{rel}"
        diff = "".join(
            difflib.unified_diff(
                before, after, fromfile=fromfile, tofile=tofile, lineterm="\n"
            )
        )
        if diff:
            parts.append(diff)
    return "\n".join(parts)


def rollback(repo_root: Path, snapshots: dict[Path, Optional[str]]) -> None:
    """Restore the original files (used when tests never pass)."""
    for target, original in snapshots.items():
        if original is None:
            target.unlink(missing_ok=True)
        else:
            target.write_text(original, encoding="utf-8")
    console.print("[yellow]↩ Files restored to their original contents.[/]")


# ---------------------------------------------------------------------------
# Structured correction feedback (Step 2.2)
# ---------------------------------------------------------------------------
def _build_correction_feedback(test_result) -> str:
    """Build the self-corrector feedback for a failed test run.

    Priority (Step 2.2):
    1. Missing-dependency warning when the environment, not the patch, broke.
    2. Structured failure report (pytest/Jest/Vitest parsing) — small and
       actionable.
    3. Raw stdout/stderr TAIL as fallback/context (truncated so huge dumps do
       not flood the prompt).
    """
    parts: list[str] = []

    missing = getattr(test_result, "missing_dependency", None)
    if missing:
        parts.append(
            "MISSING DEPENDENCY DETECTED (environment problem, not a code "
            f"bug):\n{missing}\n"
            "These tests cannot pass until the dependency/import issue is "
            "resolved; do not try to fix it by changing unrelated code."
        )

    failures = extract_failures(test_result)
    if failures:
        report = format_failure_report(failures)
        parts.append(
            f"STRUCTURED FAILURE REPORT ({len(failures)} failing test(s)):\n{report}"
        )

    raw_parts: list[str] = []
    if test_result.stdout:
        raw_parts.append("--- stdout ---\n" + test_result.stdout[-RAW_FEEDBACK_TAIL_CHARS:])
    if test_result.stderr:
        raw_parts.append("--- stderr ---\n" + test_result.stderr[-RAW_FEEDBACK_TAIL_CHARS:])
    if raw_parts:
        parts.append("RAW OUTPUT (tail):\n" + "\n".join(raw_parts))

    if not parts:
        parts.append(f"(no output; exit code {test_result.exit_code})")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Targeted test execution (Step 2.1) + targeted verdict cache (Step 3.1)
# ---------------------------------------------------------------------------
# Maximum output tail stored per cached targeted verdict, so a huge failure
# dump cannot bloat .patchcraft/cache/test_results/.
_CACHED_OUTPUT_TAIL_CHARS = 20_000


def _cacheable_test_result(result: object) -> bool:
    """Whether a test verdict is stable enough to be cached (Step 3.1).

    Missing-dependency failures depend on the machine environment and
    timeouts (exit code 124) depend on machine load — neither is a property
    of the patch itself, so both are never cached.
    """
    if getattr(result, "missing_dependency", None) is not None:
        return False
    return getattr(result, "exit_code", 1) != 124


def _patch_fingerprint(repo_root: Path, snapshots: dict[Path, Optional[str]]) -> str:
    """Stable fingerprint of the post-patch state of every touched file.

    Hashes each touched file's CURRENT content (not the patch text), so the
    fingerprint is identical no matter whether the change arrived as a
    surgical edit or as a whole-file rewrite. Used by the targeted-test
    result cache: same fingerprint + same subset => same verdict.
    """
    digest = hashlib.sha256()
    for path in sorted(snapshots):
        rel = path.relative_to(repo_root).as_posix().replace("\\\\", "/")
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        digest.update(rel.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(
            hashlib.sha256(content.encode("utf-8", "replace")).hexdigest().encode("ascii")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _build_test_evidence(selection, test_result) -> str:
    """Compact, honest summary of what actually ran and passed (Step 4.3).

    Only called on the success path, so ``exit code 0`` is a verified fact.
    """
    parts: list[str] = []
    if selection is not None and getattr(selection, "has_targets", False):
        targets = ", ".join(getattr(selection, "node_ids", []) or [])
        parts.append(f"pytest (targeted): {targets}")
    subset = getattr(test_result, "subset", "full")
    parts.append(f"{subset} run: exit code 0")
    return "; ".join(parts)


def _run_tests_or_fallback(
    runner: SandboxRunner,
    repo_root: Path,
    changed_files: list[str],
    selection,
    emit: Callable[[str, str], None],
    result_cache: Optional[TestResultCache] = None,
    patch_fingerprint: Optional[str] = None,
):
    """Run targeted tests when available; fall back to the full suite.

    Two-phase strategy:
    1. If targeted tests were found for the changed files, run them first.
       A failure here is definitive (the patch broke something specific) and
       saves a full-suite run. When the EXACT same patch (post-patch
       fingerprint) is re-proposed for the exact same subset, the previous
       verdict is reused from ``result_cache`` instead of re-executing.
    2. Only when targeted tests pass, run the FULL suite as the final gate —
       this catches regressions in unrelated code that the import graph
       couldn't predict. The gate NEVER uses the cache.

    Returns the final :class:`TestResult` (always from the full suite when
    targeted passed, from targeted when they failed, or from the full suite
    directly when no targets were available).
    """
    if selection is not None and selection.has_targets:
        targets = selection.node_ids

        # -- Step 3.1: reuse the cached verdict for an identical patch -----
        cache_key: Optional[str] = None
        if (
            result_cache is not None
            and result_cache.enabled
            and patch_fingerprint
        ):
            cache_key = result_cache.make_key(patch_fingerprint, targets)
            cached_verdict = result_cache.lookup(cache_key)
            if cached_verdict is not None:
                console.print(
                    "[cyan]♻ Cached targeted verdict reused "
                    "(identical patch + test subset).[/]"
                )
                emit(
                    "test",
                    "Targeted run skipped: cached verdict reused for the "
                    "identical patch and test subset.",
                )
                return TestResult(
                    success=bool(cached_verdict.get("success")),
                    stdout=str(cached_verdict.get("stdout", "")),
                    stderr=str(cached_verdict.get("stderr", "")),
                    exit_code=int(cached_verdict.get("exit_code", 1)),
                    subset="targeted",
                    cached=True,
                )

        emit("test", f"Running {len(targets)} targeted test file(s): {', '.join(targets)}")
        console.print(f"[cyan]🎯 Targeted tests:[/] {len(targets)} file(s)")
        targeted_result = runner.run_tests(targets=targets)

        if cache_key is not None and _cacheable_test_result(targeted_result):
            result_cache.store(  # type: ignore[union-attr]
                cache_key,
                success=targeted_result.success,
                exit_code=targeted_result.exit_code,
                stdout=(targeted_result.stdout or "")[-_CACHED_OUTPUT_TAIL_CHARS:],
                stderr=(targeted_result.stderr or "")[-_CACHED_OUTPUT_TAIL_CHARS:],
            )

        if not targeted_result.success:
            console.print("[bold red]❌ Targeted tests failed — skipping full suite.[/]")
            return targeted_result

        # Targeted green → run the full suite as the final gate.
        console.print("[green]✅ Targeted tests passed — running full suite as gate.[/]")
        full_result = runner.run_tests()
        if not full_result.success:
            console.print(
                "[bold red]❌ Full suite failed even though targeted tests "
                "passed — possible regression outside the changed files.[/]"
            )
            return full_result
        # Both green → report the full-suite result (authoritative).
        return full_result

    if selection is not None and selection.notes:
        for note in selection.notes:
            console.print(f"[dim]{note}[/]")

    return runner.run_tests()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_patchcraft_loop(
    repo_path: str,
    issue_description: str,
    model: str,
    max_retries: Optional[int] = None,
    event_sink: Optional[Callable[[str, str], None]] = None,
    token_budget: Optional[int] = None,
    time_budget_seconds: Optional[float] = None,
    min_remaining_credits: Optional[float] = None,
    auto_install: bool = False,
    use_cache: bool = True,
    issue_number: Optional[int] = None,
    issue_title: Optional[str] = None,
    allow_dirty: bool = False,
    github_repo: Optional[str] = None,
    use_git_flow: bool = True,
) -> RunResult:
    """Run the goal-driven Diagnosis -> Patch -> Test -> Self-Correction flow.

    Thin wrapper that scopes the process-wide caching configuration
    (Step 3.1) to THIS run only: whatever memo-cache settings existed before
    the call are restored afterwards, so long-lived processes (GUI, test
    suites) never inherit stale cache state. See
    :func:`_run_patchcraft_loop_impl` for the full documentation.
    """
    memo = get_memo_cache()
    saved_enabled, saved_base_dir = memo.enabled, memo.base_dir
    try:
        return _run_patchcraft_loop_impl(
            repo_path=repo_path,
            issue_description=issue_description,
            model=model,
            max_retries=max_retries,
            event_sink=event_sink,
            token_budget=token_budget,
            time_budget_seconds=time_budget_seconds,
            min_remaining_credits=min_remaining_credits,
            auto_install=auto_install,
            use_cache=use_cache,
            issue_number=issue_number,
            issue_title=issue_title,
            allow_dirty=allow_dirty,
            github_repo=github_repo,
            use_git_flow=use_git_flow,
        )
    finally:
        memo.enabled, memo.base_dir = saved_enabled, saved_base_dir
        # Step 4.1 safety net: if the pipeline crashed without consuming its
        # worktree cleanup, run it now with failure semantics.
        crash_cleanup = pop_worktree_cleanup()
        if crash_cleanup is not None:
            try:
                crash_cleanup()
                logger.warning("Pending git worktree cleaned up after a crash.")
            except Exception as exc:  # noqa: BLE001 - best effort
                logger.warning("Crash cleanup of the git worktree failed: %s", exc)


def _run_patchcraft_loop_impl(
    repo_path: str,
    issue_description: str,
    model: str,
    max_retries: Optional[int] = None,
    event_sink: Optional[Callable[[str, str], None]] = None,
    token_budget: Optional[int] = None,
    time_budget_seconds: Optional[float] = None,
    min_remaining_credits: Optional[float] = None,
    auto_install: bool = False,
    use_cache: bool = True,
    issue_number: Optional[int] = None,
    issue_title: Optional[str] = None,
    allow_dirty: bool = False,
    github_repo: Optional[str] = None,
    use_git_flow: bool = True,
) -> RunResult:
    """Run the goal-driven Diagnosis -> Patch -> Test -> Self-Correction flow.

    The loop is **dynamic**: it keeps iterating (analyze, patch, test) until
    all tests pass or one of the safety guardrails stops it — there is no
    arbitrary retry cut-off unless you explicitly set ``max_retries``.

    Parameters
    ----------
    repo_path : str
        Target repository directory.
    issue_description : str
        Description of the bug/issue to fix.
    model : str
        Primary LLM model (litellm format).
    max_retries : int | None
        Hard cap on patch+test iterations. ``None`` (default) means the loop
        runs until tests pass or a guardrail halts it.
    event_sink : Callable[[str, str], None] | None
        Optional callback invoked as ``event_sink(stage, message)`` at every
        milestone, for GUIs/loggers that need structured streaming. It must
        never raise; exceptions from the sink are swallowed and logged.
    token_budget : int | None
        Maximum total LLM tokens (prompt + completion) for the whole task.
        ``None`` disables the check; falls back to ``PATCHCRAFT_TOKEN_BUDGET``.
    time_budget_seconds : float | None
        Wall-clock budget for the whole task; when exceeded the loop halts
        gracefully. ``None`` disables the check; falls back to
        ``PATCHCRAFT_TIME_BUDGET``.
    min_remaining_credits : float | None
        Halt when the OpenRouter remaining credit balance drops below this
        value. ``None`` disables the check; falls back to
        ``PATCHCRAFT_MIN_CREDITS``.
    auto_install : bool
        When True, a failed test run showing a missing-dependency error
        triggers ONE dependency install + retry in the sandbox (default off;
        see :class:`src.sandbox.runner.SandboxRunner`).
    use_cache : bool
        Enable the caching layer (Roadmap Step 3.1): the LLM memo cache and
        the targeted-test verdict cache, both scoped to
        ``<repo>/.patchcraft/cache``. ``False`` disables both for this run
        (equivalent to the CLI ``--no-cache`` flag). The environment variable
        ``PATCHCRAFT_NO_CACHE=1`` always wins.

    Returns
    -------
    :class:`RunResult` with the outcome, the report on success, the test
    errors of the failed iterations and, on failure, ``halt_reason``
    explaining which guardrail stopped the loop.
    """

    def emit(stage: str, message: str) -> None:
        """Forward a milestone to the event sink, never raising."""
        if event_sink is None:
            return
        try:
            event_sink(stage, message)
        except Exception as exc:  # noqa: BLE001 - the sink must never break the loop
            logger.warning("event_sink raised for stage '%s': %s", stage, exc)

    repo_root = Path(repo_path).expanduser().resolve()
    if not repo_root.is_dir():
        raise NotADirectoryError(f"Repository does not exist: {repo_root}")
    if not issue_description.strip():
        raise ValueError("issue_description must not be empty.")

    # -- Step 4.1: safe git workflow ---------------------------------------
    # Git repos get an isolated worktree on a patchcraft/* branch; the
    # user's checkout is never touched. Non-git repos keep today's behavior.
    main_repo_root = repo_root  # caches stay anchored to the user's checkout
    git_flow: Optional[GitFlow] = None
    worktree_path: Optional[Path] = None
    git_branch: Optional[str] = None
    # Step 4.4: `use_git_flow=False` lets callers (e.g. `patchcraft followup`)
    # manage git themselves and work directly on an existing branch.
    if use_git_flow and GitFlow.is_git_repo(repo_root):
        git_flow = GitFlow(repo_root)
        git_flow.ensure_ready(allow_dirty=allow_dirty)  # raises GitSafetyError
        git_branch = build_branch_name(issue_number, issue_title)
        worktree_path = git_flow.create_worktree(git_branch)
        register_worktree_cleanup(
            lambda wt=worktree_path, br=git_branch: git_flow.cleanup(wt, True, br)
        )
        repo_root = worktree_path
        console.print(f"[bold]Git branch:[/] {git_branch} (isolated worktree)")

    # -- Step 3.1: caching layer (LLM memo + targeted-test verdicts) -------
    # Both caches live under <repo>/.patchcraft/cache of the USER'S checkout;
    # PATCHCRAFT_NO_CACHE always disables them regardless of use_cache.
    configure_memo_cache(
        enabled=use_cache,
        base_dir=main_repo_root / ".patchcraft" / "cache",
    )
    test_result_cache = TestResultCache(main_repo_root, enabled=use_cache)

    # -- Step 4.3: repo voice profile (PR writer) ---------------------------
    # Built from the user's checkout (so the cache persists), best-effort.
    repo_voice: Optional[RepoVoice] = None
    try:
        repo_voice = build_repo_voice(
            main_repo_root, github_repo=github_repo, fetch_prs=bool(github_repo)
        )
    except Exception as exc:  # noqa: BLE001 - voice must never break a run
        logger.debug("Repo voice unavailable: %s: %s", type(exc).__name__, exc)
        repo_voice = None

    # Guardrail defaults from the environment (explicit arguments win).
    if token_budget is None:
        token_budget = _env_optional("PATCHCRAFT_TOKEN_BUDGET", int)
    if time_budget_seconds is None:
        time_budget_seconds = _env_optional("PATCHCRAFT_TIME_BUDGET", float)
    if min_remaining_credits is None:
        min_remaining_credits = _env_optional("PATCHCRAFT_MIN_CREDITS", float)

    limit_desc = str(max_retries) if max_retries is not None else "until green"
    console.rule("[bold blue] PatchCraft — starting [/]")
    console.print(f"[bold]Repository:[/] {repo_root}")
    console.print(f"[bold]Model:[/] {model}  |  [bold]Iterations:[/] {limit_desc} (goal-driven)")
    emit(
        "start",
        f"Repository: {repo_root} | Model: {model} | Iterations: {limit_desc}",
    )

    started_at = time.monotonic()
    usage: dict[str, int] = {"prompt": 0, "completion": 0}

    # Step 3.2: mirror token accounting into the shared run-stats registry
    # so live views can show "tokens spent vs budget". Pure instrumentation:
    # pipeline behavior and budgets are computed from `usage` as before.
    run_stats = begin_run()

    def usage_sink(prompt_tokens: int, completion_tokens: int) -> None:
        usage["prompt"] += prompt_tokens
        usage["completion"] += completion_tokens
        run_stats.add(prompt_tokens, completion_tokens)

    with console.status("📖 Reading documentation and sources ..."):
        context = build_context(repo_root, issue_description)
    console.print("[green]✓[/] Context built.")
    emit("context", "Documentation and sources collected.")

    console.rule("[bold magenta]🔍 Diagnostic analysis")
    with console.status("Calling the Diagnostic agent ..."):
        diagnosis = diagnose(context, model, usage_sink=usage_sink)
    console.print(f"[bold]Summary:[/] {diagnosis.summary}")
    console.print(f"[bold]Root cause:[/] {diagnosis.root_cause}")
    console.print(f"[bold]Affected files:[/] {', '.join(diagnosis.affected_files)}")
    emit(
        "diagnosis",
        f"{diagnosis.summary}\nRoot cause: {diagnosis.root_cause}\n"
        f"Files: {', '.join(diagnosis.affected_files)}",
    )

    original_snapshots: dict[Path, Optional[str]] = {}
    current_patch: Optional[Patch] = None
    test_errors: list[str] = []
    files_changed: list[str] = []

    # Ground the coder in the real source code: without it every patch is
    # invented, tests always fail and the final rollback erases all changes.
    with console.status("📚 Collecting affected files for the coder ..."):
        coder_context = build_coder_context(repo_root, diagnosis.affected_files)
    if coder_context:
        console.print(f"[green]✓[/] Coder context ready ({len(coder_context)} chars).")
    else:
        console.print("[yellow]⚠ No affected files found on disk; coder will rely on the diagnosis only.[/]")

    # Goal-driven loop state (loop detection + budget accounting).
    attempt = 0
    halt_reason: Optional[str] = None
    strategy_directive = ""
    previous_signature: Optional[str] = None
    previous_patch_json: Optional[str] = None
    stagnation_repeats = 0

    def _register_failure(signature: str, patch_json: Optional[str]) -> bool:
        """Track consecutive identical failures; True when the loop must halt."""
        nonlocal stagnation_repeats, previous_signature, previous_patch_json
        nonlocal strategy_directive, halt_reason
        same_error = signature == previous_signature
        same_patch = patch_json is not None and patch_json == previous_patch_json
        stagnation_repeats = stagnation_repeats + 1 if (same_error or same_patch) else 1
        previous_signature = signature
        previous_patch_json = patch_json
        if stagnation_repeats >= STAGNATION_HALT_AFTER:
            halt_reason = (
                f"Stagnation detected: the same failure repeated "
                f"{stagnation_repeats} consecutive iterations without progress."
            )
            return True
        if stagnation_repeats >= STAGNATION_STRATEGY_AFTER:
            strategy_directive = (
                f"STAGNATION WARNING: the same failure has now repeated "
                f"{stagnation_repeats} times. You MUST change strategy: question "
                f"your assumptions, try a fundamentally different approach and "
                f"re-read the failing test output before patching again."
            )
        else:
            strategy_directive = ""
        return False

    while True:
        attempt += 1

        # -- Guardrail: explicit retry cap (disabled unless configured) ------
        if max_retries is not None and max_retries > 0 and attempt > max_retries:
            attempt -= 1  # the aborted check does not count as an iteration
            halt_reason = f"Iteration limit reached ({max_retries})."
            break

        # -- Guardrail: wall-clock time budget --------------------------------
        if time_budget_seconds is not None and time.monotonic() - started_at > time_budget_seconds:
            attempt -= 1
            halt_reason = f"Time budget exhausted (limit: {time_budget_seconds} s)."
            break

        # -- Guardrail: per-task token budget ---------------------------------
        tokens_used = usage["prompt"] + usage["completion"]
        if token_budget is not None and tokens_used > token_budget:
            attempt -= 1
            halt_reason = f"Token budget exhausted ({tokens_used} tokens used > {token_budget})."
            break

        # -- Guardrail: OpenRouter credit floor -------------------------------
        if min_remaining_credits is not None:
            remaining = _remaining_credits()
            if remaining is not None and remaining < min_remaining_credits:
                attempt -= 1
                halt_reason = (
                    f"OpenRouter credits below the safety floor "
                    f"({remaining:.2f} remaining < {min_remaining_credits:.2f})."
                )
                break

        total_desc = str(max_retries) if max_retries is not None else "∞"
        console.rule(f"[bold cyan] Iteration {attempt}/{total_desc} [/]")
        emit("iteration", f"Iteration {attempt}/{total_desc}")

        # Dynamic budget: grows with the number of files and each correction
        # iteration, so later (larger) corrections are never truncated.
        patch_budget = dynamic_patch_budget(max(1, len(diagnosis.affected_files)), attempt - 1)

        with console.status("✍️  Generating patch ..."):
            if attempt == 1:
                candidate = generate_patch(
                    diagnosis.model_dump_json(indent=2),
                    model,
                    repo_context=coder_context,
                    max_tokens=patch_budget,
                    usage_sink=usage_sink,
                )
            else:
                feedback = "\n---\n".join(test_errors)
                if strategy_directive:
                    feedback = f"{strategy_directive}\n\n{feedback}"
                candidate = correct_patch(
                    previous_patch=current_patch,
                    test_feedback=feedback,
                    provider_model=model,
                    repo_context=coder_context,
                    max_tokens=patch_budget,
                    iteration=attempt - 1,
                    usage_sink=usage_sink,
                )
                emit("patch", "Self-correcting patch generated.")
        current_patch = _coerce_patch(candidate)
        if current_patch is None or not current_patch.files:
            console.print("[red]✗ No files in the patch: iteration skipped.[/]")
            test_errors.append(
                "The generated patch was missing, not valid JSON or contained no "
                "files. IMPORTANT: respond with a single JSON object like "
                '{"files": [{"file_path": "rel/path.py", "new_content": "COMPLETE file content"}]} '
                "and make sure the output is NOT truncated."
            )
            emit("error", "The generated patch was invalid or contained no modifiable files.")
            if _register_failure("invalid-patch-output", None):
                break
            continue

        apply_result = apply_patch_detailed(current_patch, repo_root)
        new_snapshots = apply_result.snapshots
        if apply_result.problems:
            for problem in apply_result.problems:
                console.print(f"[red]✗ {problem}[/]")
                test_errors.append(
                    f"Surgical patch problem: {problem} Re-read the CURRENT file "
                    f"content in the context and copy the 'find' snippet EXACTLY "
                    f"as it appears (2-4 surrounding lines)."
                )
            emit("error", "Some surgical edits failed to apply:\n" + "\n".join(apply_result.problems))
        if not new_snapshots:
            console.print("[yellow]The patch produced no applicable changes.[/]")
        else:
            emit(
                "patch",
                "Applied changes:\n"
                + "\n".join(str(p.relative_to(repo_root)) for p in new_snapshots),
            )
        for path, _orig in new_snapshots.items():
            original_snapshots.setdefault(path, _orig)
        files_changed = [str(p.relative_to(repo_root)) for p in original_snapshots]

        console.rule("[bold cyan]🧪 Running tests")
        with console.status("Running tests ..."):
            runner = SandboxRunner(repo_root, auto_install=auto_install)

            # -- Step 2.1: Targeted test selection -----------------------
            changed = [str(p.relative_to(repo_root).as_posix())
                       for p in new_snapshots]
            selection = None
            if changed:
                try:
                    selection = _select_targeted_tests(repo_root, changed)
                except Exception as exc:
                    logger.debug("Targeted test selection skipped (%s).", exc)

            # -- Step 3.1: targeted verdict cache (identical patch reuse) --
            patch_fp = (
                _patch_fingerprint(repo_root, new_snapshots)
                if new_snapshots else None
            )
            test_result = _run_tests_or_fallback(
                runner, repo_root, changed, selection, emit,
                result_cache=test_result_cache,
                patch_fingerprint=patch_fp,
            )
        emit(
            "test",
            f"exit_code={test_result.exit_code} success={test_result.success}\n"
            + (test_result.stdout or "")
            + ("\n" + test_result.stderr if test_result.stderr else ""),
        )

        if test_result.stdout:
            console.print(test_result.stdout, soft_wrap=False)
        if test_result.stderr:
            console.print(f"[red]{test_result.stderr}[/]")

        if test_result.success:
            console.rule("[bold green]✅ Tests passed")
            diff = compute_diff(repo_root, original_snapshots)
            emit("diff", diff)
            # Step 4.3: verified inputs only for the human-grade PR writer.
            test_evidence = _build_test_evidence(selection, test_result)
            diff_stat = build_diff_stat(diff)
            with console.status("📝 Generating report ..."):
                report = generate_report(
                    diff,
                    model,
                    usage_sink=usage_sink,
                    repo_voice=repo_voice,
                    issue_text=issue_description,
                    diff_stat=diff_stat,
                    test_evidence=test_evidence,
                )
            emit("report", report.pr_markdown if isinstance(report, PatchReport) else str(report))
            if isinstance(report, PatchReport):
                console.print(Panel(
                    f"[bold]{report.title}[/]\n\n{report.summary}\n\n"
                    f"[dim]PR diff: {len(report.diff)} characters.[/]",
                    title="[bold green]📋 Final report",
                    border_style="green",
                ))
            emit("done", "Pipeline completed successfully.")

            # -- Step 4.1: commit ONLY the touched files on the branch ------
            commit_sha: Optional[str] = None
            if git_flow is not None and worktree_path is not None:
                style = detect_commit_style(get_recent_subjects(main_repo_root))
                summary = issue_title or issue_description.splitlines()[0]
                message = build_commit_message(style, summary, issue_number)
                commit_sha = git_flow.commit_touched(
                    worktree_path, files_changed, message
                )
                # Worktree removed; branch + commit are kept as deliverable.
                git_flow.cleanup(worktree_path, delete_branch=False, branch=git_branch)
                pop_worktree_cleanup()  # consumed: no crash-cleanup needed
                worktree_path = None
                console.print(
                    f"[green]Committed on branch[/] [bold]{git_branch}[/]"
                    + (f" ([dim]{commit_sha[:10]}[/])" if commit_sha else " (no changes)")
                )

            return RunResult(
                success=True,
                iterations=attempt,
                report=report if isinstance(report, PatchReport) else None,
                test_errors=test_errors,
                files_changed=files_changed,
                git_branch=git_branch if git_flow is not None else None,
                commit_sha=commit_sha,
            )

        console.print("[bold red]❌ Tests failed — structured feedback passed to the self-corrector.[/]")
        if test_result.missing_dependency:
            console.print(
                f"[yellow]⚠ Missing dependency detected:[/] {test_result.missing_dependency}"
            )
        if failures_extracted := extract_failures(test_result):
            console.print(
                f"[cyan]🔎 Structured failure report:[/] {len(failures_extracted)} "
                "failing test(s) parsed."
            )
        # Step 2.2: structured report first (small + actionable), raw output
        # tail as fallback — Jest/Vitest print details on stdout.
        feedback = _build_correction_feedback(test_result)
        test_errors.append(feedback)
        emit("error", feedback)

        # -- Guardrail: loop detection ----------------------------------------
        patch_json = current_patch.model_dump_json() if isinstance(current_patch, Patch) else None
        if _register_failure(_error_signature(test_result), patch_json):
            break

    # A guardrail stopped the loop: roll back and fail with a clear report.
    rollback(repo_root, original_snapshots)
    # Step 4.1: on failure the worktree AND the branch are removed; HEAD of
    # the user's checkout was never touched.
    if git_flow is not None and worktree_path is not None:
        git_flow.cleanup(worktree_path, delete_branch=True, branch=git_branch)
        pop_worktree_cleanup()
        worktree_path = None
        console.print("[yellow]Git worktree removed and branch deleted.[/]")
    reason = halt_reason or "The loop stopped without converging."
    tokens_summary = usage["prompt"] + usage["completion"]
    console.print(Panel(
        f"The self-correction loop halted before all tests passed.\n"
        f"[bold]Reason:[/] {reason}\n"
        f"Iterations executed: {attempt} | LLM tokens used: {tokens_summary}\n"
        f"Detected errors:\n" + "\n".join(f"- {e[:400]}" for e in test_errors),
        title="[bold red]❌ Loop did not converge",
        border_style="red",
    ))
    emit(
        "done",
        f"The pipeline halted: {reason} Changes rolled back "
        f"({attempt} iteration(s), {tokens_summary} LLM tokens used).",
    )
    return RunResult(
        success=False,
        iterations=attempt,
        report=None,
        test_errors=test_errors,
        files_changed=[],
        halt_reason=halt_reason,
    )


def _coerce_patch(candidate: object) -> Optional[Patch]:
    """Convert agent output (Patch | str) into :class:`Patch`.

    Smart-guardrails for raw string output:
    * markdown code fences (```json ... ```) are stripped;
    * a JSON object embedded in surrounding prose is extracted;
    * truncated JSON is reported explicitly so the self-correction loop can
      react instead of silently skipping the iteration.
    """
    if isinstance(candidate, Patch):
        return candidate
    if isinstance(candidate, BaseModel):
        return Patch.model_validate(candidate.model_dump())
    if isinstance(candidate, str):
        text = candidate.strip()
        # 1) direct parse
        attempts = [text]
        # 2) strip markdown fences
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            while lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            attempts.append("\n".join(lines).strip())
        # 3) extract outermost JSON object from surrounding prose
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            attempts.append(text[start:end + 1])

        for attempt_text in attempts:
            if not attempt_text:
                continue
            try:
                data = json.loads(attempt_text)
            except json.JSONDecodeError:
                continue
            try:
                return Patch.model_validate(data)
            except ValidationError as exc:
                console.print(f"[red]✗ Patch JSON does not match the schema: {exc}[/]")
                return None

        truncated = "```" in text or text.rstrip().endswith(('"', ",", "}", "]")) is False
        hint = (
            " The output looks TRUNCATED: raise the max_tokens budget."
            if len(text) >= 4000 or truncated
            else ""
        )
        console.print(f"[red]✗ Invalid patch JSON (no parsable object found).{hint}[/]")
        return None
    return None


__all__ = [
    "run_patchcraft_loop",
    "RunResult",
    "build_context",
    "build_coder_context",
    "apply_patch",
    "apply_patch_detailed",
    "apply_edits_to_text",
    "compute_diff",
    "rollback",
]