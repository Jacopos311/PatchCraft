"""Safe git workflow (Roadmap Step 4.1).

Safety-first design:

* the TARGET REPO must be clean before patching (refused otherwise unless
  explicitly allowed);
* all patching happens in an ISOLATED WORKTREE on a dedicated branch
  ``patchcraft/<issue-number>-<slug>``, so the user's checkout is never
  disturbed mid-run;
* on SUCCESS only the files PatchCraft touched are committed, with a
  message mirroring the repository's own style (Conventional Commits vs
  free-form, detected from ``git log --oneline -n 30``);
* on FAILURE/HALT/CRASH the worktree is removed and the branch deleted,
  leaving HEAD exactly as before;
* repositories without git keep the previous behavior (handled upstream);
* NOTHING is ever pushed here — pushing/PRs come later, explicitly (4.2).
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)

BRANCH_PREFIX = "patchcraft"
WORKTREE_DIR_NAME = "worktrees"

# Conventional Commits subject line: <type>(optional scope)!: description
_CONVENTIONAL_RE = re.compile(
    r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(\([\w./-]+\))?!?:\s\S"
)


class GitSafetyError(RuntimeError):
    """The repository state does not allow a safe PatchCraft run."""


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without a repo)
# ---------------------------------------------------------------------------
def slugify(text: str, max_length: int = 40) -> str:
    """Lowercase kebab slug: 'Fix off-by-one in Cart!' -> 'fix-off-by-one-in-cart'."""
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:max_length].rstrip("-")


def build_branch_name(
    issue_number: Optional[int] = None,
    title: Optional[str] = None,
) -> str:
    """Branch name ``patchcraft/<issue-number>-<slug>`` (Step 4.1).

    Without an issue number (local ``run``) the form is
    ``patchcraft/fix-<slug-or-timestamp>``.
    """
    slug = slugify(title or "")
    if issue_number is not None:
        return f"{BRANCH_PREFIX}/{issue_number}-{slug or 'fix'}"
    if slug:
        return f"{BRANCH_PREFIX}/fix-{slug}"
    return f"{BRANCH_PREFIX}/fix-{int(time.time())}"


def detect_commit_style(log_output: str) -> str:
    """``'conventional'`` when >= half of the recent subjects follow the spec."""
    subjects: List[str] = []
    for line in (log_output or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)  # drop the abbreviated hash
        if len(parts) == 2:
            subjects.append(parts[1])
    if not subjects:
        return "conventional"
    matches = sum(1 for s in subjects if _CONVENTIONAL_RE.match(s))
    return "conventional" if matches * 2 >= len(subjects) else "plain"


def build_commit_message(
    style: str,
    summary: str,
    issue_number: Optional[int] = None,
) -> str:
    """Compose the commit message following the detected repo style."""
    text = " ".join((summary or "").split())  # collapse whitespace/newlines
    if not text:
        text = "apply PatchCraft fix"
    if len(text) > 72:
        text = text[:69].rstrip("- ") + "..."
    if style == "conventional":
        subject = f"fix: {text}"
    else:
        subject = text[0].upper() + text[1:]
    body_lines = ["Applied by PatchCraft."]
    if issue_number is not None:
        body_lines.append(f"Fixes #{issue_number}")
    return subject + "\n\n" + "\n".join(body_lines) + "\n"


# ---------------------------------------------------------------------------
# Cleanup registry: guarantees worktree removal even on crashes. The
# orchestrator registers a failure-cleanup when it creates a worktree; the
# public entry point pops and runs it in its own ``finally`` if the pipeline
# did not consume it earlier.
# ---------------------------------------------------------------------------
_pending_cleanups: List[Callable[[], None]] = []


def register_worktree_cleanup(cleanup: Callable[[], None]) -> None:
    """Register a failure-cleanup callback for the active worktree."""
    _pending_cleanups.append(cleanup)


def pop_worktree_cleanup() -> Optional[Callable[[], None]]:
    """Remove and return the most recent registered cleanup, if any."""
    return _pending_cleanups.pop() if _pending_cleanups else None


# ---------------------------------------------------------------------------
# Git plumbing
# ---------------------------------------------------------------------------
def run_git(
    args: Sequence[str], cwd: Path, check: bool = True
) -> "subprocess.CompletedProcess[str]":
    """Run a git command; raises :class:`GitSafetyError` when ``check`` fails."""
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and completed.returncode != 0:
        raise GitSafetyError(
            f"git {' '.join(args)} failed (exit {completed.returncode}): "
            f"{(completed.stderr or completed.stdout or '').strip()}"
        )
    return completed


def get_recent_subjects(path: Path, count: int = 30) -> str:
    """Raw ``git log --oneline -n <count>`` output used for style detection."""
    return run_git(["log", "--oneline", f"-n{count}"], path, check=False).stdout


def _rmtree_best_effort(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


class GitFlow:
    """Safety-first git operations for one PatchCraft run."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)

    # ------------------------------------------------------------------
    # Preconditions
    # ------------------------------------------------------------------
    @staticmethod
    def is_git_repo(path: Path) -> bool:
        """True when ``path`` is inside a git working tree."""
        if not path.is_dir():
            return False
        completed = run_git(["rev-parse", "--is-inside-work-tree"], path, check=False)
        return completed.returncode == 0 and completed.stdout.strip() == "true"

    @staticmethod
    def is_clean(path: Path) -> bool:
        """True when ``git status --porcelain`` reports nothing."""
        completed = run_git(["status", "--porcelain"], path)
        return completed.stdout.strip() == ""

    def ensure_ready(self, allow_dirty: bool = False) -> None:
        """Verify the repo can be patched safely (Step 4.1 point 1)."""
        run_git(["rev-parse", "--verify", "HEAD"], self.repo_root)  # need a commit
        if not self.is_clean(self.repo_root) and not allow_dirty:
            raise GitSafetyError(
                "Working tree is not clean: commit or stash your changes "
                "first, or pass --allow-dirty to proceed anyway."
            )

    # ------------------------------------------------------------------
    # Worktree lifecycle
    # ------------------------------------------------------------------
    def create_worktree(self, branch: str) -> Path:
        """Create an isolated worktree on a NEW branch from current HEAD."""
        dir_name = branch.replace("/", "__")
        worktree_path = self.repo_root / ".patchcraft" / WORKTREE_DIR_NAME / dir_name
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        completed = run_git(
            ["worktree", "add", "-b", branch, str(worktree_path), "HEAD"],
            self.repo_root,
            check=False,
        )
        if completed.returncode != 0:
            # Branch may linger from an interrupted previous run: reset it.
            run_git(["branch", "-D", branch], self.repo_root, check=False)
            run_git(["worktree", "prune"], self.repo_root, check=False)
            run_git(["worktree", "add", "-b", branch, str(worktree_path), "HEAD"],
                    self.repo_root)
        return worktree_path

    def commit_touched(
        self,
        worktree: Path,
        files: Sequence[str],
        message: str,
    ) -> Optional[str]:
        """Stage ONLY ``files`` and commit them; returns the SHA (or None).

        ``git add`` handles both modified and patch-deleted paths. When
        nothing ends up staged the commit is skipped and ``None`` returned.
        """
        if not files:
            return None
        # Only stage paths that exist on disk or are already tracked (a patch
        # may delete files; it can never invent ones that were never there).
        known = set(run_git(["ls-files"], worktree).stdout.splitlines())
        tracked = [
            f.replace("\\\\", "/") for f in files
            if (worktree / f).exists() or f.replace("\\\\", "/") in known
        ]
        if not tracked:
            return None
        run_git(["add", "--", *tracked], worktree)
        staged = run_git(["diff", "--cached", "--name-only"], worktree).stdout.strip()
        if not staged:
            logger.info("No staged changes; skipping the commit.")
            return None
        # Fall back to a local-only identity when the repo has none, so the
        # commit never fails for configuration reasons.
        identity_args: List[str] = []
        email = run_git(["config", "user.email"], worktree, check=False).stdout.strip()
        name = run_git(["config", "user.name"], worktree, check=False).stdout.strip()
        if not email:
            identity_args += ["-c", "user.email=patchcraft@localhost"]
        if not name:
            identity_args += ["-c", "user.name=PatchCraft"]
        run_git([*identity_args, "commit", "-m", message], worktree)
        sha = run_git(["rev-parse", "HEAD"], worktree).stdout.strip()
        return sha or None

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------
    def cleanup(
        self,
        worktree: Path,
        delete_branch: bool,
        branch: Optional[str] = None,
    ) -> None:
        """Remove the worktree on every exit path (Step 4.1 point 4).

        ``delete_branch`` is True on failure/rollback (branch deleted; the
        user's checkout was never touched) and False on success (branch +
        commit are the deliverable).
        """
        try:
            run_git(["worktree", "remove", "--force", str(worktree)],
                    self.repo_root, check=False)
        finally:
            if worktree.exists():
                _rmtree_best_effort(worktree)
            run_git(["worktree", "prune"], self.repo_root, check=False)
            if delete_branch and branch:
                run_git(["branch", "-D", branch], self.repo_root, check=False)


__all__ = [
    "BRANCH_PREFIX",
    "GitFlow",
    "GitSafetyError",
    "build_branch_name",
    "build_commit_message",
    "detect_commit_style",
    "get_recent_subjects",
    "pop_worktree_cleanup",
    "register_worktree_cleanup",
    "run_git",
    "slugify",
]
