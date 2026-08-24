"""Tests for GitHub PR publishing (Roadmap Step 4.2).

Uses a mocked ``requests`` transport (no network) covering: the happy path,
duplicate-PR update paths (head branch AND hidden body marker), error paths
with actionable hints, draft flag propagation and the half-pushed-state
rollback of the remote branch reference.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import src.github.pr_publisher as pub
from src.core.config import PatchcraftConfig
from src.github.issue_fetcher import GitHubAPIError
from src.github.pr_publisher import (
    append_marker,
    pr_marker,
    publish_pr,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _response(status: int, payload: dict | list | None = None):
    response = mock.Mock()
    response.status_code = status
    response.json.return_value = payload if payload is not None else {}
    response.text = ""
    return response


@pytest.fixture()
def git_ok(monkeypatch):
    """Fake local git: origin exists, push succeeds, base branch unknown."""
    calls: list[list[str]] = []

    def fake_run_git(args, cwd, check=False):
        calls.append(list(args))
        if args[:1] == ["remote"]:
            return SimpleNamespace(returncode=0, stdout="origin\n", stderr="")
        if args[:1] == ["push"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(pub, "run_git", fake_run_git)
    return calls


@pytest.fixture()
def http_log():
    """Router over ``requests.request`` recording every call."""
    calls: list[dict] = []

    def make_router(routes):
        # routes: callable(method, path, json_payload, params) -> Response|None
        def handler(method, url, **kwargs):
            from src.github.issue_fetcher import GITHUB_API_URL

            path = url[len(GITHUB_API_URL):]
            json_payload = kwargs.get("json")
            params = kwargs.get("params")
            for matcher, responder in routes:
                if matcher(method, path, json_payload, params):
                    response = responder(method, path, json_payload, params)
                    break
            else:
                response = _response(404, {"message": "unmatched in test"})
            calls.append({
                "method": method, "path": path,
                "json": json_payload, "params": params,
                "status": response.status_code,
            })
            return response
        return handler

    return SimpleNamespace(make_router=make_router, calls=calls)


REPO = "owner/repo"
EXISTING_PR = {
    "number": 55,
    "html_url": "https://github.com/owner/repo/pull/55",
    "body": f"old body\n\n{pr_marker(123)}\n",
}


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
class TestMarkers:
    def test_marker_format(self) -> None:
        assert pr_marker(123) == "<!-- patchcraft-issue:123 -->"

    def test_append_marker_once(self) -> None:
        once = append_marker("body", 9)
        twice = append_marker(once, 9)
        assert once.count(pr_marker(9)) == 1
        assert once == twice


class TestOriginParsing:
    def test_https_url(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(
            pub, "run_git",
            lambda *a, **k: SimpleNamespace(returncode=0, check=False,
                                            stdout="https://github.com/o/r.git\n",
                                            stderr=""),
        )
        assert pub.origin_owner_repo(tmp_path) == "o/r"

    def test_ssh_url(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(
            pub, "run_git",
            lambda *a, **k: SimpleNamespace(returncode=0, check=False,
                                            stdout="git@github.com:o/r.git\n",
                                            stderr=""),
        )
        assert pub.origin_owner_repo(tmp_path) == "o/r"

    def test_no_remote(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(
            pub, "run_git",
            lambda *a, **k: SimpleNamespace(returncode=0, check=False,
                                            stdout="", stderr=""),
        )
        assert pub.origin_owner_repo(tmp_path) is None


# ---------------------------------------------------------------------------
# publish_pr
# ---------------------------------------------------------------------------
class TestPublishPr:
    def test_happy_path_creates_draft_pr(self, git_ok, http_log, tmp_path: Path):
        routes = [
            (lambda m, p, j, q: p == "/repos/owner/repo/pulls"
             and (q or {}).get("head"),
             lambda *a: _response(200, [])),                       # head search: none
            (lambda m, p, j, q: p.endswith("/pulls") and m == "GET",
             lambda *a: _response(200, [])),                       # marker search: none
            (lambda m, p, j, q: p.endswith("/pulls") and m == "POST",
             lambda *a: _response(
                 201,
                 {"html_url": "https://github.com/owner/repo/pull/77"}),
             ),
        ]
        with mock.patch.object(pub.requests, "request",
                               side_effect=http_log.make_router(routes)):
            url = publish_pr(
                repo=REPO, repo_path=tmp_path, branch="patchcraft/123-fix-sign-bug",
                title="fix: sign", body="Automated fix.", draft=True,
                issue_number=123, base="main",
            )

        assert url == "https://github.com/owner/repo/pull/77"
        posts = [c for c in http_log.calls if c["method"] == "POST"]
        assert len(posts) == 1
        payload = posts[0]["json"]
        assert payload["draft"] is True                      # flag propagation
        assert payload["head"] == "owner:patchcraft/123-fix-sign-bug"
        assert pr_marker(123) in payload["body"]             # idempotency marker
        pushed = [c for c in git_ok if c[:1] == ["push"]]
        assert pushed and pushed[0][3] == "patchcraft/123-fix-sign-bug"

    def test_duplicate_head_branch_is_updated(self, git_ok, http_log, tmp_path: Path):
        routes = [
            (lambda m, p, j, q: p.endswith("/pulls") and m == "GET"
             and (q or {}).get("head") == f"owner:patchcraft/123-fix-sign-bug",
             lambda *a: _response(200, [EXISTING_PR])),
            (lambda m, p, j, q: p == "/repos/owner/repo/pulls/55" and m == "PATCH",
             lambda *a: _response(200, {"html_url": EXISTING_PR["html_url"]})),
        ]
        with mock.patch.object(pub.requests, "request",
                               side_effect=http_log.make_router(routes)):
            url = publish_pr(
                repo=REPO, repo_path=tmp_path, branch="patchcraft/123-fix-sign-bug",
                title="updated", body="new body", draft=False, issue_number=123,
                base="main",
            )

        assert url == EXISTING_PR["html_url"]
        methods = [(c["method"], c["path"]) for c in http_log.calls]
        assert ("POST", "/repos/owner/repo/pulls") not in methods  # no duplicates
        patches = [c for c in http_log.calls if c["method"] == "PATCH"]
        assert len(patches) == 1
        assert patches[0]["json"]["title"] == "updated"

    def test_duplicate_found_by_hidden_marker_across_branch_names(
        self, git_ok, http_log, tmp_path: Path,
    ):
        """Same issue re-run on a NEW branch name still updates the old PR."""
        old_pr = {**EXISTING_PR, "body": f"old\n\n{pr_marker(123)}\n"}
        routes = [
            (lambda m, p, j, q: p.endswith("/pulls") and m == "GET"
             and (q or {}).get("head") == "owner:patchcraft/123-other-slug",
             lambda *a: _response(200, [])),                   # new branch: no PR
            (lambda m, p, j, q: p.endswith("/pulls") and m == "GET"
             and not (q or {}).get("head"),
             lambda *a: _response(200, [old_pr])),             # marker match!
            (lambda m, p, j, q: p == "/repos/owner/repo/pulls/55" and m == "PATCH",
             lambda *a: _response(200, {"html_url": old_pr["html_url"]})),
        ]
        with mock.patch.object(pub.requests, "request",
                               side_effect=http_log.make_router(routes)):
            url = publish_pr(
                repo=REPO, repo_path=tmp_path, branch="patchcraft/123-other-slug",
                title="t", body="b", draft=True, issue_number=123, base="main",
            )
        assert url == old_pr["html_url"]
        assert any(c["method"] == "PATCH" for c in http_log.calls)


class TestErrorPaths:
    def test_creation_failure_rolls_back_remote_branch(
        self, git_ok, http_log, tmp_path: Path,
    ):
        routes = [
            (lambda m, p, j, q: p.endswith("/pulls") and m == "GET",
             lambda *a: _response(200, [])),
            (lambda m, p, j, q: p.endswith("/pulls") and m == "POST",
             lambda *a: _response(422, {"message": "Validation Failed"})),
            (lambda m, p, j, q: p.endswith("/git/refs/heads/patchcraft/123-x")
             and m == "DELETE",
             lambda *a: _response(204)),
        ]
        with mock.patch.object(pub.requests, "request",
                               side_effect=http_log.make_router(routes)):
            with pytest.raises(GitHubAPIError) as excinfo:
                publish_pr(
                    repo=REPO, repo_path=tmp_path, branch="patchcraft/123-x",
                    title="t", body="b", draft=True, issue_number=123, base="main",
                )

        message = str(excinfo.value)
        assert "deleted again" in message          # nothing left half-pushed
        deletes = [c for c in http_log.calls if c["method"] == "DELETE"]
        assert len(deletes) == 1
        assert deletes[0]["path"].endswith("/refs/heads/patchcraft/123-x")

    def test_failed_rollback_warns_about_manual_cleanup(
        self, git_ok, http_log, tmp_path: Path,
    ):
        routes = [
            (lambda m, p, j, q: p.endswith("/pulls") and m == "GET",
             lambda *a: _response(200, [])),
            (lambda m, p, j, q: p.endswith("/pulls") and m == "POST",
             lambda *a: _response(422, {"message": "Validation Failed"})),
            (lambda m, p, j, q: "/git/refs/" in p and m == "DELETE",
             lambda *a: _response(500, {"message": "boom"})),
        ]
        with mock.patch.object(pub.requests, "request",
                               side_effect=http_log.make_router(routes)):
            with pytest.raises(GitHubAPIError) as excinfo:
                publish_pr(
                    repo=REPO, repo_path=tmp_path, branch="patchcraft/123-x",
                    title="t", body="b", draft=True, issue_number=123, base="main",
                )
        assert "manually" in str(excinfo.value)

    @pytest.mark.parametrize("status,detail,expected_hint", [
        (401, {"message": "Bad credentials"}, "GITHUB_TOKEN"),
        (403, {"message": "API rate limit exceeded"}, "rate limit"),
        (404, {"message": "Not Found"}, "verify the repository"),
        (422, {"message": "Validation Failed"}, "branch protection"),
    ])
    def test_actionable_hints(self, status, detail, expected_hint,
                              git_ok, http_log, tmp_path: Path):
        routes = [
            (lambda m, p, j, q: p.endswith("/pulls") and m == "GET",
             lambda *a: _response(200, [])),
            (lambda m, p, j, q: p.endswith("/pulls") and m == "POST",
             lambda *a: _response(status, detail)),
            (lambda m, p, j, q: "/git/refs/" in p and m == "DELETE",
             lambda *a: _response(204)),
        ]
        with mock.patch.object(pub.requests, "request",
                               side_effect=http_log.make_router(routes)):
            with pytest.raises(GitHubAPIError) as excinfo:
                publish_pr(
                    repo=REPO, repo_path=tmp_path, branch="patchcraft/1-x",
                    title="t", body="b", draft=True, issue_number=1, base="main",
                )
        assert expected_hint.lower() in str(excinfo.value).lower()

    def test_push_failure_blocks_every_api_call(self, monkeypatch, tmp_path: Path):
        def fake_run_git(args, cwd, check=False):
            if args[:1] == ["remote"]:
                return SimpleNamespace(returncode=0, stdout="origin\n", stderr="")
            return SimpleNamespace(returncode=1, stdout="",
                                   stderr="protected branch hint")

        monkeypatch.setattr(pub, "run_git", fake_run_git)
        with mock.patch.object(pub.requests, "request") as api_mock:
            with pytest.raises(GitHubAPIError) as excinfo:
                publish_pr(
                    repo=REPO, repo_path=tmp_path, branch="patchcraft/1-x",
                    title="t", body="b", draft=True, issue_number=1, base="main",
                )
        api_mock.assert_not_called()  # nothing half-pushed into the API layer
        assert "protected" in str(excinfo.value)

    def test_missing_remote_gives_add_hint(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr(pub, "run_git", lambda *a, **k:
                            SimpleNamespace(returncode=0, stdout="", stderr=""))
        with pytest.raises(GitHubAPIError) as excinfo:
            publish_pr(
                repo=REPO, repo_path=tmp_path, branch="b",
                title="t", body="b2", draft=True, issue_number=1,
            )
        assert "git remote add origin" in str(excinfo.value)


class TestBaseBranchResolution:
    def test_git_config_first_then_api_fallback(
        self, git_ok, http_log, tmp_path: Path,
    ):
        """origin/HEAD unknown locally -> default branch from the API."""
        captured: dict = {}

        def router(method, url, **kwargs):
            from src.github.issue_fetcher import GITHUB_API_URL

            path = url[len(GITHUB_API_URL):]
            if path == f"/repos/{REPO}" and method == "GET":
                return _response(200, {"default_branch": "develop"})
            if path.endswith("/pulls"):
                if method == "GET":
                    return _response(200, [])
                captured["base"] = (kwargs.get("json") or {}).get("base")
                return _response(201, {"html_url": "https://github.com/x"})
            return _response(404)

        with mock.patch.object(pub.requests, "request", side_effect=router):
            url = publish_pr(
                repo=REPO, repo_path=tmp_path, branch="patchcraft/1-x",
                title="t", body="b", draft=True, issue_number=1,
            )
        assert url == "https://github.com/x"
        assert captured["base"] == "develop"

    def test_pr_draft_defaults_to_true_in_config(self) -> None:
        cfg = PatchcraftConfig()
        assert cfg.pr.draft is True  # always draft until Milestone 5


# ---------------------------------------------------------------------------
# CLI wiring: fix --push
# ---------------------------------------------------------------------------
@pytest.fixture()
def cli_env(monkeypatch):
    import main as main_module

    monkeypatch.setattr(main_module, "render_credits_panel",
                        lambda *a, **k: None)
    return main_module


def _success_result(**overrides):
    from src.agents.reporter import PatchReport
    from src.orchestrator import RunResult

    fields = dict(
        success=True,
        iterations=1,
        files_changed=["src/app.py"],
        git_branch="patchcraft/123-fix-sign-bug",
        commit_sha="abc123def4567",
        report=PatchReport(title="fix: sign bug", summary="s",
                           diff="d", pr_markdown="# PR\nbody"),
    )
    fields.update(overrides)
    return RunResult(**fields)


class TestFixPushWiring:
    def _invoke(self, cli_env, tmp_path: Path, args: list[str], result,
                publish_mock=None):
        from click.testing import CliRunner

        with (
            mock.patch("src.github.issue_fetcher.get_issue",
                       return_value={"number": 123, "title": "Fix sign bug",
                                     "body": ""}),
            mock.patch("src.orchestrator.run_patchcraft_loop",
                       return_value=result),
        ):
            cmd = ["fix", *args]
            if publish_mock is not None:
                with mock.patch("src.github.pr_publisher.publish_pr",
                                side_effect=publish_mock) as pub_spy:
                    outcome = CliRunner().invoke(cli_env.cli, cmd)
                    return outcome, pub_spy
            outcome = CliRunner().invoke(cli_env.cli, cmd)
            return outcome, None

    def test_push_flag_publishes_and_prints_url(self, cli_env, tmp_path: Path) -> None:
        captured: dict = {}

        def fake_publish(**kwargs):
            captured.update(kwargs)
            return "https://github.com/owner/repo/pull/88"

        outcome, spy = self._invoke(
            cli_env, tmp_path,
            ["--push", "https://github.com/owner/repo/issues/123", str(tmp_path)],
            _success_result(),
            publish_mock=fake_publish,
        )
        assert outcome.exit_code == 0
        assert "Pull Request ready" in outcome.output
        assert captured["branch"] == "patchcraft/123-fix-sign-bug"
        assert captured["draft"] is True          # config default honored
        assert captured["issue_number"] == 123
        assert captured["title"].startswith("fix:")
        spy.assert_called_once()

    def test_without_push_flag_nothing_is_published(self, cli_env, tmp_path: Path) -> None:
        outcome, spy = self._invoke(
            cli_env, tmp_path,
            ["https://github.com/owner/repo/issues/123", str(tmp_path)],
            _success_result(),
            publish_mock=lambda **k: "https://x",
        )
        assert outcome.exit_code == 0
        spy.assert_not_called()  # default remains local-only

    def test_publish_failure_maps_to_exit_two(self, cli_env, tmp_path: Path) -> None:
        outcome, _ = self._invoke(
            cli_env, tmp_path,
            ["--push", "https://github.com/owner/repo/issues/123", str(tmp_path)],
            _success_result(report=None),
            publish_mock=_raise_github_error,
        )
        assert outcome.exit_code == 2
        assert "Publishing failed" in outcome.output


def _raise_github_error(**kwargs):
    raise GitHubAPIError("GitHub API refused (HTTP 401)")
