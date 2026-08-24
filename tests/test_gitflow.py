"""Tests for the safe git workflow (Roadmap Step 4.1).

Uses REAL temporary git repositories (subprocess) covering:
* dirty-repo refusal and --allow-dirty;
* branch naming;
* commit style detection and message building;
* worktree isolation, commit staging, rollback cleanliness;
* non-git fallback behavior through the full pipeline.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

import pytest

from src.agents.coder import Patch
from src.core import gitflow
from src.core.gitflow import (
    GitFlow,
    GitSafetyError,
    build_branch_name,
    build_commit_message,
    detect_commit_style,
    slugify,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _git(args: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    )
    return completed.stdout


def _write(path: Path, rel: str, content: str) -> None:
    target = path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _init_repo(tmp_path: Path, *, conventional: bool = True) -> Path:
    """Create a real git repo with one initial commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init"], repo)
    _git(["config", "user.email", "tester@example.com"], repo)
    _git(["config", "user.name", "Tester"], repo)
    first_message = (
        "chore: initial commit" if conventional else "Initial import of the project"
    )
    _write(repo, "README.md", "# demo\n")
    _git(["add", "."], repo)
    _git(["commit", "-m", first_message], repo)
    return repo


def _seed_pytest_project(repo: Path) -> None:
    """Minimal buggy project whose targeted tests can be fixed by a patch."""
    _write(repo, "src/app.py", "def add(a, b):\n    return a - b\n")
    _write(
        repo, "tests/test_app.py",
        "from src.app import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
    )
    _git(["add", "."], repo)
    _git(["commit", "-m", "feat: calculator module"], repo)


def _diagnosis():
    from src.agents.diagnostic import Diagnosis

    return Diagnosis(
        summary="offset bug",
        root_cause="off-by-one",
        affected_files=["src/app.py"],
        confidence=0.9,
    )


def _report():
    from src.agents.reporter import PatchReport

    return PatchReport(title="t", summary="s", diff="d", pr_markdown="m")


_SIGN_PATCH = lambda: Patch(files=[{  # noqa: E731
    "file_path": "src/app.py",
    "edits": [{"find": "    return a - b\n", "replace": "    return a + b\n"}],
}])


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
class TestSlugAndBranchName:
    def test_slugify_normalizes(self) -> None:
        assert slugify("Fix off-by-one in Cart!") == "fix-off-by-one-in-cart"
        assert slugify("Ümlaut & spaces") == "mlaut-spaces"

    def test_slugify_truncates(self) -> None:
        assert len(slugify("a" * 100, max_length=40)) <= 40

    def test_branch_with_issue_number(self) -> None:
        assert build_branch_name(123, "Fix sign bug!") == "patchcraft/123-fix-sign-bug"

    def test_branch_issue_number_without_title(self) -> None:
        assert build_branch_name(7) == "patchcraft/7-fix"

    def test_branch_without_issue_number(self) -> None:
        assert build_branch_name(title="Add retry logic") == "patchcraft/fix-add-retry-logic"

    def test_branch_fully_anonymous_is_stable_enough(self) -> None:
        name = build_branch_name()
        assert name.startswith("patchcraft/fix-")


class TestCommitStyleDetection:
    def test_conventional_repo(self) -> None:
        log = "\n".join([
            "a1 chore: initial commit",
            "a2 fix(core): handle refunds",
            "a3 docs: update readme",
        ])
        assert detect_commit_style(log) == "conventional"

    def test_freeform_repo(self) -> None:
        log = "\n".join([
            "a1 Initial import of the project",
            "a2 Make the parser tolerate empty lines",
            "a3 Small cleanups around the CLI",
        ])
        assert detect_commit_style(log) == "plain"

    def test_empty_log_defaults_to_conventional(self) -> None:
        assert detect_commit_style("") == "conventional"


class TestCommitMessageBuilding:
    def test_conventional_with_issue_trailer(self) -> None:
        msg = build_commit_message("conventional", "Fix sign bug", issue_number=123)
        assert msg.startswith("fix: Fix sign bug")
        assert "Fixes #123" in msg

    def test_plain_capitalized_no_trailer_without_issue(self) -> None:
        msg = build_commit_message("plain", "fix the sign handling")
        assert msg.startswith("Fix the sign handling")
        assert "Fixes #" not in msg

    def test_long_summary_truncated(self) -> None:
        msg = build_commit_message("conventional", "x" * 200)
        subject = msg.splitlines()[0]
        assert len(subject) <= 80


class TestCleanupRegistry:
    def test_register_then_pop(self) -> None:
        calls: list[str] = []
        gitflow.register_worktree_cleanup(lambda: calls.append("x"))
        cb = gitflow.pop_worktree_cleanup()
        assert cb is not None
        cb()
        assert calls == ["x"]
        assert gitflow.pop_worktree_cleanup() is None


# ---------------------------------------------------------------------------
# Real git repositories
# ---------------------------------------------------------------------------
class TestGitFlowWithRealRepos:
    def test_is_git_repo_detection(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        plain = tmp_path / "plain"
        plain.mkdir()
        assert GitFlow.is_git_repo(repo) is True
        assert GitFlow.is_git_repo(plain) is False

    def test_dirty_repo_refused_unless_allowed(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _write(repo, "README.md", "# changed\n")
        flow = GitFlow(repo)
        with pytest.raises(GitSafetyError) as excinfo:
            flow.ensure_ready()
        assert "--allow-dirty" in str(excinfo.value)
        # Explicit opt-in proceeds.
        flow.ensure_ready(allow_dirty=True)

    def test_unborn_branch_refused(self, tmp_path: Path) -> None:
        repo = tmp_path / "empty"
        repo.mkdir()
        _git(["init"], repo)
        with pytest.raises(GitSafetyError):
            GitFlow(repo).ensure_ready()

    def test_worktree_isolation(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        flow = GitFlow(repo)
        branch = build_branch_name(5, "isolation check")
        worktree = flow.create_worktree(branch)

        assert worktree.is_dir()
        assert branch in _git(["branch", "--list", branch], repo)

        # Writing inside the worktree never touches the user's checkout.
        original = (repo / "README.md").read_text(encoding="utf-8")
        (worktree / "README.md").write_text("mutated\n", encoding="utf-8")
        assert (repo / "README.md").read_text(encoding="utf-8") == original

        flow.cleanup(worktree, delete_branch=True, branch=branch)
        assert not worktree.exists()
        assert branch not in _git(["branch", "--list"], repo)
        assert (repo / "README.md").read_text(encoding="utf-8") == original

    def test_commit_stages_only_touched_files(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        flow = GitFlow(repo)
        branch = build_branch_name(9, "targeted commit")
        worktree = flow.create_worktree(branch)

        _write(worktree, "src/app.py", "changed\n")   # PatchCraft touched
        _write(worktree, "other.py", "unrelated\n")   # NOT touched
        sha = flow.commit_touched(
            worktree,
            ["src/app.py"],
            build_commit_message("conventional", "fix app", issue_number=9),
        )

        assert sha
        committed_files = _git(
            ["show", "--name-only", "--pretty=format:", branch], repo
        )
        assert "src/app.py" in committed_files
        assert "other.py" not in committed_files          # untouched stays out
        assert "other.py" in _git(["status", "--porcelain"], worktree)
        message = _git(["log", "-1", "--pretty=%B", branch], repo)
        assert message.startswith("fix: fix app")
        assert "Fixes #9" in message

        flow.cleanup(worktree, delete_branch=True, branch=branch)

    def test_commit_skipped_when_nothing_changed(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        flow = GitFlow(repo)
        worktree = flow.create_worktree(build_branch_name(1))
        assert flow.commit_touched(worktree, ["src/app.py"], "msg") is None
        flow.cleanup(worktree, delete_branch=True, branch="patchcraft/1-fix")


# ---------------------------------------------------------------------------
# Full pipeline on real git repos (orchestrator integration)
# ---------------------------------------------------------------------------
def _diagnosis():
    from src.agents.diagnostic import Diagnosis

    return Diagnosis(
        summary="offset bug",
        root_cause="off-by-one",
        affected_files=["src/app.py"],
        confidence=0.9,
    )


def _report():
    from src.agents.reporter import PatchReport

    return PatchReport(title="t", summary="s", diff="d", pr_markdown="m")


def _sign_patch():
    return Patch(files=[{
        "file_path": "src/app.py",
        "edits": [{"find": "    return a - b\n", "replace": "    return a + b\n"}],
    }])


def _seed_pytest_project(repo: Path) -> None:
    """Minimal buggy project whose targeted tests a patch can fix."""
    _write(repo, "src/app.py", "def add(a, b):\n    return a - b\n")
    _write(
        repo, "tests/test_app.py",
        "from src.app import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
    )
    _git(["add", "."], repo)
    _git(["commit", "-m", "feat: calculator module"], repo)


class TestLoopIntegration:
    def _run_loop(self, repo: Path, *, failing: bool):
        """Run the real pipeline on a git repo with mocked agents."""

        def run_tests(targets=None):
            from src.sandbox.runner import TestResult

            if failing:
                return TestResult(success=False, stdout="FAILED tests/test_app.py",
                                  exit_code=1)
            return TestResult(success=True, stdout="exit_code=0 success=True", exit_code=0)

        with (
            mock.patch("src.orchestrator.diagnose", return_value=_diagnosis()),
            mock.patch("src.orchestrator.generate_patch", return_value=_sign_patch()),
            mock.patch("src.orchestrator.correct_patch", return_value=_sign_patch()),
            mock.patch("src.sandbox.runner.SandboxRunner.run_tests",
                       side_effect=run_tests),
            mock.patch("src.orchestrator.generate_report", return_value=_report()),
        ):
            from src.orchestrator import run_patchcraft_loop

            return run_patchcraft_loop(
                str(repo), "Fix sign", model="mock",
                issue_number=123, issue_title="Fix sign bug",
                max_retries=2 if failing else None,
            )

    def test_success_commits_branch_and_cleans_worktree(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _seed_pytest_project(repo)
        head_before = _git(["rev-parse", "HEAD"], repo).strip()

        result = self._run_loop(repo, failing=False)

        assert result.success is True
        assert result.git_branch == "patchcraft/123-fix-sign-bug"
        assert result.commit_sha
        # Branch exists in the user's repo and carries the commit.
        assert result.git_branch in _git(["branch", "--list"], repo)
        message = _git(["log", "-1", "--pretty=%B", result.git_branch], repo)
        assert message.startswith("fix:")
        assert "Fixes #123" in message
        committed = _git(["show", "--name-only", "--pretty=format:", result.git_branch],
                         repo)
        assert "src/app.py" in committed
        assert "tests/test_app.py" not in committed
        # Worktree cleaned up; user's checkout untouched.
        wt_root = repo / ".patchcraft" / "worktrees"
        assert not wt_root.exists() or not any(wt_root.iterdir())
        assert "return a - b" in (repo / "src" / "app.py").read_text(encoding="utf-8")
        assert _git(["rev-parse", "HEAD"], repo).strip() == head_before

    def test_failure_deletes_branch_and_restores_state(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _seed_pytest_project(repo)
        head_before = _git(["rev-parse", "HEAD"], repo).strip()

        result = self._run_loop(repo, failing=True)

        assert result.success is False
        assert result.git_branch is None
        assert "patchcraft/123" not in _git(["branch", "--list"], repo)
        wt_root = repo / ".patchcraft" / "worktrees"
        assert not wt_root.exists() or not any(wt_root.iterdir())
        assert _git(["rev-parse", "HEAD"], repo).strip() == head_before
        assert "return a - b" in (repo / "src" / "app.py").read_text(encoding="utf-8")

    def test_non_git_repo_keeps_old_behavior(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain"
        plain.mkdir()
        _write(plain, "src/app.py", "def add(a, b):\n    return a - b\n")
        _write(
            plain, "tests/test_app.py",
            "from src.app import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        )

        with (
            mock.patch("src.orchestrator.diagnose", return_value=_diagnosis()),
            mock.patch("src.orchestrator.generate_patch", return_value=_sign_patch()),
            mock.patch(
                "src.sandbox.runner.SandboxRunner.run_tests",
                return_value=mock.Mock(success=True, exit_code=0, stdout="", stderr="",
                                       missing_dependency=None, subset="full"),
            ),
            mock.patch("src.orchestrator.generate_report", return_value=_report()),
        ):
            from src.orchestrator import run_patchcraft_loop

            result = run_patchcraft_loop(str(plain), "Fix sign", model="mock")

        assert result.success is True
        assert result.git_branch is None
        assert result.commit_sha is None
        # Applied in place, exactly like before Step 4.1.
        assert "return a + b" in (plain / "src" / "app.py").read_text(encoding="utf-8")
