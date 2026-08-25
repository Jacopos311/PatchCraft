"""Review-response loop orchestration (Roadmap Step 4.4).

Implements ``patchcraft followup <pr-url>``:

1. resolves the PR's head branch and materializes it in an isolated
   worktree of the user's clone;
2. fetches reviewer comments, classifies them and keeps ONLY must-fix items
   (deterministic classifier — see :mod:`src.github.review_feedback`);
3. runs the existing self-correction loop scoped to those items, reusing
   every guardrail (stagnation, tokens, time, credits) with a hard iteration
   cap from ``followup_max_iterations``;
4. on success commits ON THE SAME BRANCH, pushes it (never opens new PRs),
   replies politely to each addressed comment and re-requests review.

Failures leave the PR untouched: without a green run there is no push, no
commit and no replies.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from src.core.gitflow import (
    GitFlow,
    GitSafetyError,
    detect_commit_style,
    get_recent_subjects,
    register_worktree_cleanup,
    run_git,
)
from src.core.config import PatchcraftConfig
from src.github.pr_publisher import push_branch
from src.github.review_feedback import (
    ReviewComment,
    build_followup_task,
    fetch_review_comments,
    get_pull_request,
    parse_pr_url,
    reply_to_comment,
    request_rereview,
)

logger = logging.getLogger(__name__)

FOLLOWUP_WORKTREE_DIR = "followup"
DEFAULT_MAX_ITERATIONS = 3


class FollowupError(RuntimeError):
    """Follow-up could not even start (bad URL, missing branch, ...)."""


class FollowupOutcome(BaseModel):
    """Result of one ``patchcraft followup`` invocation."""

    success: bool = Field(description="Tests green after addressing feedback.")
    nothing_to_fix: bool = Field(
        default=False,
        description="True when no must-fix comments were found.",
    )
    branch: Optional[str] = Field(default=None, description="PR head branch.")
    commit_sha: Optional[str] = Field(default=None, description="New commit SHA.")
    replied_count: int = Field(default=0, description="Reviewer replies posted.")
    reviewers_requested: List[str] = Field(
        default_factory=list, description="Re-review requested from these logins.")
    iterations: int = Field(default=0)
    halt_reason: Optional[str] = None


def _materialize_pr_branch_worktree(
    repo_path: Path, branch: str
) -> tuple[Path, GitFlow]:
    """Worktree on the EXISTING PR branch (local or fetched from origin)."""
    if not GitFlow.is_git_repo(repo_path):
        raise GitSafetyError(
            f"{repo_path} is not a git repository: 'patchcraft followup' "
            f"needs the local clone of the pull request."
        )
    flow = GitFlow(repo_path)

    local = run_git(
        ["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"],
        repo_path, check=False,
    )
    if local.returncode != 0:
        fetch = run_git(["fetch", "origin", branch], repo_path, check=False)
        if fetch.returncode != 0:
            raise GitSafetyError(
                f"Branch '{branch}' does not exist locally and fetching it "
                f"from origin failed: {(fetch.stderr or '').strip()} — Hint: "
                f"check that origin points at the pull request repository."
            )
        run_git(
            ["worktree", "add", "--track", "-b", branch,
             str(_worktree_path(repo_path, branch)), f"origin/{branch}"],
            repo_path,
        )
    else:
        run_git(
            ["worktree", "add", str(_worktree_path(repo_path, branch)), branch],
            repo_path,
        )
    return _worktree_path(repo_path, branch), flow


def _worktree_path(repo_path: Path, branch: str) -> Path:
    safe = branch.replace("/", "__")
    return Path(repo_path) / ".patchcraft" / "followup" / safe


def _commit_message(style: str, pr_number: int) -> str:
    subject = (
        f"fix(review): address must-fix feedback on PR #{pr_number}"
        if style == "conventional"
        else f"Address review feedback on PR #{pr_number}"
    )
    body = "Applied by PatchCraft in response to reviewer comments.\n"
    return f"{subject}\n\n{body}"


def run_followup(
    *,
    pr_url: str,
    repo_path: Path,
    model: str,
    max_iterations: Optional[int] = None,
    token_budget: Optional[int] = None,
    time_budget_seconds: Optional[float] = None,
    min_remaining_credits: Optional[float] = None,
    auto_install: bool = False,
    use_cache: bool = True,
    event_sink: Optional[Callable[[str, str], None]] = None,
    loop_fn: Optional[Callable[..., Any]] = None,
) -> FollowupOutcome:
    """Address must-fix review comments on a PR and push to the SAME branch.

    ``loop_fn`` is injectable for tests; it defaults to
    :func:`src.orchestrator.run_patchcraft_loop` (called with
    ``use_git_flow=False`` because git is managed here).
    """
    from src.orchestrator import run_patchcraft_loop as _default_loop

    loop = loop_fn or _default_loop
    cap = max_iterations or DEFAULT_MAX_ITERATIONS

    # 1. Identify the pull request and its head branch.
    repo, pr_number = parse_pr_url(pr_url)
    pr = get_pull_request(repo, pr_number)
    branch = ((pr.get("head") or {}).get("ref")) or ""
    if not branch:
        raise FollowupError(f"Pull request #{pr_number} has no head branch.")

    # 2. Materialize the head branch in an isolated worktree.
    worktree, flow = _materialize_pr_branch_worktree(Path(repo_path), branch)
    register_worktree_cleanup(
        lambda wt=worktree, fl=flow: fl.cleanup(wt, delete_branch=False)
    )

    # 3. Collect + classify reviewer feedback.
    comments = fetch_review_comments(repo, pr_number)
    task = build_followup_task(pr_number, comments)
    must_fix = [c for c in comments if c.kind == "must-fix"]
    reviewers = sorted({c.author for c in must_fix})

    def _teardown():
        flow.cleanup(worktree, delete_branch=False)

    if task is None:
        outcome = FollowupOutcome(
            success=True, nothing_to_fix=True, branch=branch)
        _teardown()
        pop_pending()
        return outcome

    # 4. Run the existing self-correction loop, scoped to the feedback.
    result = loop(
        repo_path=str(worktree),
        issue_description=task,
        model=model,
        max_retries=cap,
        token_budget=token_budget,
        time_budget_seconds=time_budget_seconds,
        min_remaining_credits=min_remaining_credits,
        auto_install=auto_install,
        use_cache=use_cache,
        event_sink=event_sink,
        use_git_flow=False,
        allow_dirty=True,
    )

    if not result.success:
        _teardown()
        pop_pending()
        return FollowupOutcome(
            success=False,
            branch=branch,
            iterations=result.iterations,
            halt_reason=result.halt_reason,
        )

    # 5. Commit on the SAME branch and push it (never a new PR).
    style = detect_commit_style(get_recent_subjects(worktree))
    sha = flow.commit_touched(
        worktree, list(result.files_changed),
        _commit_message(style, pr_number),
    )
    push_branch(worktree, branch)

    # 6. Reply to every addressed comment + re-request review.
    short_sha = (sha or "")[:10] or "the latest commit"
    reply_text = (
        f"Thanks for the review! This has been addressed in commit "
        f"{short_sha}, pushed to this branch."
    )
    replied = 0
    for comment in must_fix:
        reply_to_comment(repo, pr_number, comment, reply_text)
        replied += 1
    request_rereview(repo, pr_number, reviewers)

    _teardown()
    pop_pending()
    return FollowupOutcome(
        success=True,
        branch=branch,
        commit_sha=sha,
        replied_count=replied,
        reviewers_requested=reviewers,
        iterations=result.iterations,
    )


def pop_pending() -> None:
    """Consume the registered crash-cleanup after a normal teardown."""
    from src.core.gitflow import pop_worktree_cleanup

    pop_worktree_cleanup()


__all__ = [
    "FollowupError",
    "FollowupOutcome",
    "DEFAULT_MAX_ITERATIONS",
    "run_followup",
]