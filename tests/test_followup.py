"""Tests for the review-response loop (Roadmap Step 4.4)."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

import pytest

import src.github.followup as followup_mod
from src.core.config import load_config
from src.github.review_feedback import (
    ReviewComment,
    build_followup_task,
    classify_comment,
    parse_pr_url,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _git(args: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    )
    return completed.stdout


def _comment(cid: int, body: str, *, author: str = "alice",
             inline: bool = True, path: str | None = "src/app.py") -> ReviewComment:
    return ReviewComment(
        id=cid, author=author, body=body, path=path if inline else None,
        line=12 if inline else None, is_inline=inline,
        kind=classify_comment(body),
    )


def _init_repo_with_branch(tmp_path: Path) -> Path:
    repo = tmp_path / "clone"
    repo.mkdir()
    _git(["init"], repo)
    _git(["config", "user.email", "tester@example.com"], repo)
    _git(["config", "user.name", "Tester"], repo)
    _write = repo / "src"
    _write.mkdir(parents=True)
    (repo / "src" / "app.py").write_text("def add(a, b):\n    return a - b\n",
                                         encoding="utf-8")
    _git(["add", "."], repo)
    _git(["commit", "-m", "feat: calculator"], repo)
    _git(["branch", "patchcraft/123-fix-sign"], repo)
    _git(["remote", "add", "origin", "https://github.com/owner/repo.git"], repo)
    return repo


PR_URL = "https://github.com/owner/repo/pull/123"


class TestClassification:
    @pytest.mark.parametrize("body,expected", [
        ("This must be fixed before merge.", "must-fix"),
        ("[must-fix] The null check is wrong.", "must-fix"),
        ("This breaks the payment flow — request changes.", "must-fix"),
        ("Security: this leaks the API key into logs.", "must-fix"),
        ("nit: rename this variable please", "nit"),
        ("Minor style thing, optional.", "nit"),
        ("Why did you choose a dict here?", "question"),
        ("Could you clarify the retry semantics?", "question"),
        ("Suggestion: consider caching this lookup", "nit"),
    ])
    def test_classification(self, body: str, expected: str) -> None:
        assert classify_comment(body) == expected

    def test_explicit_label_wins_over_keywords(self) -> None:
        # Contains "wrong" but explicitly labelled as nit.
        assert classify_comment("[nit] naming looks wrong-ish to me") == "nit"


class TestParsePrUrl:
    def test_valid(self) -> None:
        assert parse_pr_url(PR_URL) == ("owner/repo", 123)
        assert parse_pr_url("https://github.com/o/r/pull/9/") == ("o/r", 9)

    def test_issue_url_rejected(self) -> None:
        with pytest.raises(Exception):
            parse_pr_url("https://github.com/o/r/issues/9")


class TestFollowupTaskBuilder:
    def test_includes_only_must_fix_with_location(self) -> None:
        comments = [
            _comment(1, "This must be fixed.", path="src/app.py"),
            _comment(2, "nit: typo here"),
            _comment(3, "Why this approach?"),
        ]
        task = build_followup_task(123, comments)
        assert task is not None
        assert "[src/app.py:12]" in task
        assert "@alice" in task
        assert "must be fixed" in task
        assert "typo" not in task and "approach" not in task

    def test_none_when_nothing_must_be_fixed(self) -> None:
        comments = [_comment(1, "nit: typo"), _comment(2, "Why?")]
        assert build_followup_task(123, comments) is None


# ---------------------------------------------------------------------------
# run_followup (real git repo + mocked API/loop)
# ---------------------------------------------------------------------------
def _result(success: bool = True, halt_reason: str | None = None,
            files: list[str] | None = None):
    from src.orchestrator import RunResult

    return RunResult(
        success=success,
        iterations=1 if success else 2,
        files_changed=files or ["src/app.py"],
        halt_reason=halt_reason,
    )


@pytest.fixture()
def pr_env(tmp_path: Path, monkeypatch):
    """Real clone with the PR branch + fully mocked GitHub API surface."""
    repo = _init_repo_with_branch(tmp_path)

    recorded: dict = {}

    def fake_get_pull_request(repo_name, number):
        return {"head": {"ref": "patchcraft/123-fix-sign"},
                "base": {"ref": "main"}, "number": number}

    def fake_fetch(repo_name, number):
        return [
            _comment(1, "This must be fixed: wrong operator.", path="src/app.py"),
            _comment(2, "nit: add a docstring"),
        ]

    def fake_push(path, branch, remote="origin"):
        recorded["pushed_branch"] = branch
        recorded["pushed_from"] = str(path)

    def fake_reply(repo_name, number, comment, body):
        recorded.setdefault("replies", []).append((comment.id, body))

    def fake_rereview(repo_name, number, reviewers):
        recorded["rereview"] = list(reviewers)

    monkeypatch.setattr(followup_mod, "get_pull_request", fake_get_pull_request)
    monkeypatch.setattr(followup_mod, "fetch_review_comments", fake_fetch)
    monkeypatch.setattr(followup_mod, "push_branch", fake_push)
    monkeypatch.setattr(followup_mod, "reply_to_comment", fake_reply)
    monkeypatch.setattr(followup_mod, "request_rereview", fake_rereview)
    return repo, recorded


class TestRunFollowup:
    def test_success_flow_same_branch(self, pr_env) -> None:
        repo, recorded = pr_env

        loop_kwargs: dict = {}

        def fake_loop(**kwargs):
            loop_kwargs.update(kwargs)
            # Simulate PatchCraft fixing the file inside its worktree.
            wt = Path(kwargs["repo_path"])
            app = wt / "src" / "app.py"
            app.write_text(
                app.read_text(encoding="utf-8").replace("-", "+"),
                encoding="utf-8",
            )
            return _result(files=["src/app.py"])

        outcome = followup_mod.run_followup(
            pr_url=PR_URL, repo_path=repo, model="mock", max_iterations=2,
            loop_fn=fake_loop,
        )

        assert outcome.success is True
        assert outcome.branch == "patchcraft/123-fix-sign"
        assert outcome.commit_sha
        assert outcome.replied_count == 1          # only the must-fix comment
        assert outcome.reviewers_requested == ["alice"]

        # Loop ran WITHOUT the internal git workflow, on an isolated worktree.
        assert loop_kwargs["use_git_flow"] is False
        assert loop_kwargs["allow_dirty"] is True
        assert loop_kwargs["max_retries"] == 2     # config cap passed through
        assert "must-fix review comments" in loop_kwargs["issue_description"]

        # Commit exists ON THE SAME BRANCH; user's checkout untouched.
        assert "patchcraft/123-fix-sign" in _git(["branch", "--list"], repo)
        message = _git(["log", "-1", "--pretty=%B",
                        "patchcraft/123-fix-sign"], repo)
        assert message.startswith("fix(review):")
        committed = _git(["show", "--name-only", "--pretty=format:",
                          "patchcraft/123-fix-sign"], repo)
        assert "src/app.py" in committed
        assert "return a - b" in (repo / "src" / "app.py").read_text(encoding="utf-8")

        # Same-branch push + polite reply mentioning the commit.
        assert recorded["pushed_branch"] == "patchcraft/123-fix-sign"
        cid, body = recorded["replies"][0]
        assert cid == 1 and "Thanks for the review!" in body

        # Worktree cleaned up on the normal exit path.
        wt_root = repo / ".patchcraft" / "followup"
        assert not wt_root.exists() or not any(wt_root.iterdir())

    def test_no_must_fix_short_circuits(self, pr_env) -> None:
        repo, _ = pr_env

        def fake_fetch(repo_name, number):
            return [_comment(2, "nit: typo")]

        with mock.patch.object(followup_mod, "fetch_review_comments", fake_fetch):
            outcome = followup_mod.run_followup(
                pr_url=PR_URL, repo_path=repo, model="mock",
                loop_fn=lambda **k: pytest.fail("loop must not run"),
            )

        assert outcome.nothing_to_fix is True
        assert outcome.success is True

    def test_guardrail_halt_blocks_push_and_replies(self, pr_env) -> None:
        repo, recorded = pr_env

        outcome = followup_mod.run_followup(
            pr_url=PR_URL, repo_path=repo, model="mock",
            loop_fn=lambda **k: _result(False,
                                        halt_reason="Token budget exhausted."),
        )

        assert outcome.success is False
        assert outcome.halt_reason == "Token budget exhausted."
        assert "pushed_branch" not in recorded       # nothing published
        assert "replies" not in recorded             # no replies on failure
        wt_root = repo / ".patchcraft" / "followup"
        assert not wt_root.exists() or not any(wt_root.iterdir())


class TestConfigCap:
    def test_followup_max_iterations_configurable(self, tmp_path: Path) -> None:
        (tmp_path / ".patchcraft.yml").write_text(
            "followup_max_iterations: 5\n", encoding="utf-8")
        cfg = load_config(tmp_path)
        assert cfg.followup_max_iterations == 5