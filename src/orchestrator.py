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
import json
import logging
import os
from pathlib import Path
from typing import Callable, Optional, Sequence

from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from src.agents.coder import Patch, correct_patch, dynamic_patch_budget, generate_patch
from src.agents.diagnostic import diagnose
from src.agents.reporter import PatchReport, generate_report
from src.sandbox.runner import SandboxRunner

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


def build_context(repo_root: Path, issue_description: str) -> str:
    """Compose the context for the model: docs + sources + issue description."""
    sections: list[str] = [f"# ISSUE TO SOLVE\n{issue_description.strip()}"]

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
            sections.append(f"# FILE: {doc.relative_to(repo_root)}\n```\n{content}\n```")

    sources = _discover_source_files(repo_root)
    total = sum(len(s) if isinstance(s, str) else 0 for s in sections)
    for source in sources:
        content = _read_text_limited(source)
        header_len = len(f"\n# FILE: {source.relative_to(repo_root)}\n```\n```\n")
        if total + len(content) + header_len > MAX_CONTEXT_CHARS:
            break
        sections.append(
            f"\n# FILE: {source.relative_to(repo_root)}\n```\n{content}\n```\n"
        )
        total += len(content) + header_len

    return "\n".join(sections)


def build_coder_context(repo_root: Path, affected_files: Sequence[str]) -> str:
    """Collect the ACTUAL contents of the diagnosed affected files.

    The coder agent needs the real source code to produce patches that match
    the files on disk (without it, patches are invented and tests always
    fail, triggering the final rollback). Paths are resolved safely inside
    the repo root and the overall size is capped like ``build_context``.
    """
    sections: list[str] = []
    total = 0
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
        header = f"\n# FILE: {target.relative_to(repo_root).as_posix()}\n```\n```\n"
        if total + len(content) + len(header) > MAX_CONTEXT_CHARS:
            console.print(f"[yellow]⚠ Coder context budget reached, skipping: {rel}[/]")
            break
        sections.append(f"# FILE: {target.relative_to(repo_root).as_posix()}\n```\n{content}\n```")
        total += len(content) + len(header)
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Applying patches to the real files
# ---------------------------------------------------------------------------
def _resolve_patch_path(patch_file: str, repo_root: Path) -> Optional[Path]:
    """Resolve a patch path safely inside ``repo_root``.

    Returns ``None`` if the patch attempts to escape the repo root
    (path traversal) — in that case the change is discarded.
    """
    raw = Path(patch_file.replace("\\", "/"))
    if raw.is_absolute():
        # If the model emitted an absolute path, try to map it back to the repo.
        try:
            raw = raw.relative_to(repo_root)
        except ValueError:
            return None
    candidate = (repo_root / raw).resolve()
    if candidate != repo_root and repo_root not in candidate.parents:
        console.print(f"[yellow]⚠ Path outside the repo discarded: {patch_file}[/]")
        return None
    return candidate


def apply_patch(patch: Patch, repo_root: Path) -> dict[Path, Optional[str]]:
    """Write patch files to disk.

    Returns a snapshot ``{path: original_content|None}`` of the modified
    files, used to compute the diff or roll back. ``None`` means the file did
    not exist before (it will be created).
    """
    applied: dict[Path, Optional[str]] = {}
    if not patch.files:
        console.print("[yellow]The patch contains no files: no changes applied.[/]")
        return applied

    for file_patch in patch.files:
        target = _resolve_patch_path(file_patch.file_path, repo_root)
        if target is None:
            continue
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
    return applied


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
# Main loop
# ---------------------------------------------------------------------------
def run_patchcraft_loop(
    repo_path: str,
    issue_description: str,
    model: str,
    max_retries: int = 3,
    event_sink: Optional[Callable[[str, str], None]] = None,
) -> RunResult:
    """Run the full Diagnosis -> Patch -> Test -> Self-Correction flow.

    Parameters
    ----------
    repo_path : str
        Target repository directory.
    issue_description : str
        Description of the bug/issue to fix.
    model : str
        Primary LLM model (litellm format).
    max_retries : int
        Maximum number of patch+test iterations (default 3).
    event_sink : Callable[[str, str], None] | None
        Optional callback invoked as ``event_sink(stage, message)`` at every
        milestone, for GUIs/loggers that need structured streaming. It must
        never raise; exceptions from the sink are swallowed and logged.

    Returns
    -------
    :class:`RunResult` with the outcome, the report on success, and the test
    errors of the failed iterations.
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

    console.rule("[bold blue] PatchCraft — starting [/]")
    console.print(f"[bold]Repository:[/] {repo_root}")
    console.print(f"[bold]Model:[/] {model}  |  [bold]Max retries:[/] {max_retries}")
    emit("start", f"Repository: {repo_root} | Model: {model} | Max retries: {max_retries}")

    with console.status("📖 Reading documentation and sources ..."):
        context = build_context(repo_root, issue_description)
    console.print("[green]✓[/] Context built.")
    emit("context", "Documentation and sources collected.")

    console.rule("[bold magenta]🔍 Diagnostic analysis")
    with console.status("Calling the Diagnostic agent ..."):
        diagnosis = diagnose(context, model)
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
    patch_budget = dynamic_patch_budget(max(1, len(diagnosis.affected_files)))

    for attempt in range(1, max_retries + 1):
        console.rule(f"[bold cyan] Iteration {attempt}/{max_retries} [/]")
        emit("iteration", f"Iteration {attempt}/{max_retries}")

        with console.status("✍️  Generating patch ..."):
            if attempt == 1:
                candidate = generate_patch(
                    diagnosis.model_dump_json(indent=2),
                    model,
                    repo_context=coder_context,
                    max_tokens=patch_budget,
                )
            else:
                candidate = correct_patch(
                    previous_patch=current_patch,
                    test_feedback="\n---\n".join(test_errors),
                    provider_model=model,
                    repo_context=coder_context,
                    max_tokens=patch_budget,
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
            continue

        new_snapshots = apply_patch(current_patch, repo_root)
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
        with console.status("Running the test suite ..."):
            runner = SandboxRunner(repo_root)
            test_result = runner.run_tests()
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
            with console.status("📝 Generating report ..."):
                report = generate_report(diff, model)
            emit("report", report.pr_markdown if isinstance(report, PatchReport) else str(report))
            if isinstance(report, PatchReport):
                console.print(Panel(
                    f"[bold]{report.title}[/]\n\n{report.summary}\n\n"
                    f"[dim]PR diff: {len(report.diff)} characters.[/]",
                    title="[bold green]📋 Final report",
                    border_style="green",
                ))
            emit("done", "Pipeline completed successfully.")
            return RunResult(
                success=True,
                iterations=attempt,
                report=report if isinstance(report, PatchReport) else None,
                test_errors=test_errors,
                files_changed=files_changed,
            )

        console.print("[bold red]❌ Tests failed — stdout+stderr passed to the self-corrector.[/]")
        # Jest/Vitest print failed-test details on stdout: the feedback for
        # the self-corrector includes both stdout and stderr.
        feedback_parts: list[str] = []
        if test_result.stdout:
            feedback_parts.append(f"--- stdout ---\n{test_result.stdout}")
        if test_result.stderr:
            feedback_parts.append(f"--- stderr ---\n{test_result.stderr}")
        if not feedback_parts:
            feedback_parts.append(f"(no output; exit code {test_result.exit_code})")
        test_errors.append("\n\n".join(feedback_parts))
        emit("error", "\n\n".join(feedback_parts))

    # max_retries exhausted: roll back and fail.
    rollback(repo_root, original_snapshots)
    console.print(Panel(
        f"The test suite did not pass within {max_retries} iterations.\n"
        f"Detected errors:\n" + "\n".join(f"- {e[:400]}" for e in test_errors),
        title="[bold red]❌ Loop did not converge",
        border_style="red",
    ))
    emit("done", f"The pipeline did not converge within {max_retries} iterations; changes rolled back.")
    return RunResult(
        success=False,
        iterations=max_retries,
        report=None,
        test_errors=test_errors,
        files_changed=[],
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
    "compute_diff",
    "rollback",
]