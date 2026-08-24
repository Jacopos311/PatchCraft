"""Tests for configuration & one-command flows (Roadmap Step 3.3)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from src.core.config import (
    ConfigError,
    PatchcraftConfig,
    find_config_file,
    load_config_with_warnings,
    resolve_issue_reference,
)
from src.core.llm import (
    get_default_fallback_chain,
    set_default_fallback_chain,
)
from src.core.repo_index import RepoIndex
from src.sandbox.runner import load_runner_config


@pytest.fixture(autouse=True)
def _restore_fallback_chain():
    yield
    set_default_fallback_chain(None)


def _write(root: Path, rel: str, content: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestLoaderBasics:
    def test_missing_file_gives_defaults(self, tmp_path: Path) -> None:
        cfg, warnings = load_config_with_warnings(tmp_path)
        assert isinstance(cfg, PatchcraftConfig)
        assert cfg.model is None
        assert cfg.ignore_globs == []
        assert warnings == []
        assert find_config_file(tmp_path) is None

    def test_full_valid_file_parses_everything(self, tmp_path: Path) -> None:
        _write(tmp_path, ".patchcraft.yml", """
model: openrouter/deepseek/deepseek-chat
fallback_models:
  - openrouter/anthropic/claude-3.5-sonnet
retrieval_k: 7
token_budget: 123456
time_budget: 900
min_credits: 0.5
max_retries: 4
ignore_globs:
  - "vendor/**"
commit_style: conventional
pr:
  draft: true
test:
  command: "python -m pytest -q"
  timeout_full_suite: 600
""")
        cfg, warnings = load_config_with_warnings(tmp_path)
        assert warnings == []
        assert cfg.model == "openrouter/deepseek/deepseek-chat"
        assert cfg.fallback_models == ["openrouter/anthropic/claude-3.5-sonnet"]
        assert cfg.retrieval_k == 7
        assert cfg.token_budget == 123456
        assert cfg.time_budget == 900
        assert cfg.min_credits == 0.5
        assert cfg.max_retries == 4
        assert cfg.ignore_globs == ["vendor/**"]
        assert cfg.commit_style == "conventional"
        assert cfg.pr.draft is True
        assert cfg.test.command == "python -m pytest -q"
        assert cfg.test.timeout_full_suite == 600

    def test_empty_file_is_defaults(self, tmp_path: Path) -> None:
        _write(tmp_path, ".patchcraft.yml", "")
        cfg, warnings = load_config_with_warnings(tmp_path)
        assert warnings == []
        assert cfg == PatchcraftConfig()

    def test_unknown_top_level_key_warns_but_loads(self, tmp_path: Path) -> None:
        _write(tmp_path, ".patchcraft.yml", "model: m1\ntotally_unknown: 42\n")
        cfg, warnings = load_config_with_warnings(tmp_path)
        assert cfg.model == "m1"
        assert any("totally_unknown" in w for w in warnings)

    def test_unknown_nested_key_warns_but_loads(self, tmp_path: Path) -> None:
        _write(tmp_path, ".patchcraft.yml", "test:\n  command: x\n  bogus: 1\n")
        cfg, warnings = load_config_with_warnings(tmp_path)
        assert cfg.test.command == "x"
        assert any("test.bogus" in w for w in warnings)


class TestLoaderErrors:
    def test_malformed_yaml_raises_english_error(self, tmp_path: Path) -> None:
        _write(tmp_path, ".patchcraft.yml", "{not: valid: yaml:\n")
        with pytest.raises(ConfigError) as excinfo:
            load_config_with_warnings(tmp_path)
        assert ".patchcraft.yml" in str(excinfo.value)
        assert "YAML" in str(excinfo.value)

    def test_non_mapping_file_raises(self, tmp_path: Path) -> None:
        _write(tmp_path, ".patchcraft.yml", "- just\n- a\n- list\n")
        with pytest.raises(ConfigError, match="mapping"):
            load_config_with_warnings(tmp_path)

    @pytest.mark.parametrize("body", [
        "commit_style: semverish\n",
        "token_budget: -5\n",
        "retrieval_k: 0\n",
        "test:\n  timeout_full_suite: 0\n",
    ])
    def test_schema_violations_raise(self, tmp_path: Path, body: str) -> None:
        _write(tmp_path, ".patchcraft.yml", body)
        with pytest.raises(ConfigError) as excinfo:
            load_config_with_warnings(tmp_path)
        message = str(excinfo.value)
        assert ".patchcraft.yml is invalid" in message


class TestLegacySandboxCompat:
    def test_sandbox_keys_map_into_test_section(self, tmp_path: Path) -> None:
        _write(tmp_path, ".patchcraft.yml",
               "sandbox:\n  command: old-cmd\n  timeout_full_suite: 111\n")
        cfg, warnings = load_config_with_warnings(tmp_path)
        assert cfg.test.command == "old-cmd"
        assert cfg.test.timeout_full_suite == 111
        assert any("legacy" in w.lower() for w in warnings)

    def test_test_section_wins_over_sandbox(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            ".patchcraft.yml",
            "sandbox:\n  command: old\n  timeout_full_suite: 1\n"
            "test:\n  command: new\n",
        )
        cfg, _ = load_config_with_warnings(tmp_path)
        assert cfg.test.command == "new"
        assert cfg.test.timeout_full_suite == 1

    def test_runner_loader_merges_test_over_sandbox(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            ".patchcraft.yml",
            "sandbox:\n  command: old\n  timeout_full_suite: 1\n"
            "test:\n  command: new\n  timeout_full_suite: 222\n",
        )
        merged = load_runner_config(tmp_path)
        assert merged["command"] == "new"
        assert merged["timeout_full_suite"] == 222


class TestIssueReferences:
    def test_full_url(self) -> None:
        repo, number = resolve_issue_reference(
            "https://github.com/owner/repo/issues/123"
        )
        assert (repo, number) == ("owner/repo", 123)

    def test_url_with_trailing_slash_and_www(self) -> None:
        repo, number = resolve_issue_reference(
            "https://www.github.com/o/r/issues/7/"
        )
        assert (repo, number) == ("o/r", 7)

    def test_bare_number(self) -> None:
        assert resolve_issue_reference("42") == (None, 42)
        assert resolve_issue_reference("#42") == (None, 42)

    @pytest.mark.parametrize("bad", ["", "abc", "issues/3", "https://gitlab.com/o/r/issues/1"])
    def test_invalid_references_raise_english_error(self, bad: str) -> None:
        with pytest.raises(ConfigError) as excinfo:
            resolve_issue_reference(bad)
        assert "not valid" in str(excinfo.value)


class TestIgnoreGlobs:
    def test_explicit_globs_exclude_files(self, tmp_path: Path) -> None:
        _write(tmp_path, "src/keep.py", "x = 1\n")
        _write(tmp_path, "vendor/lib.py", "y = 2\n")
        index = RepoIndex.build(tmp_path, ignore_globs=["vendor/**"])
        assert "src/keep.py" in index.files
        assert "vendor/lib.py" not in index.files

    def test_component_glob_matches_directory_name(self, tmp_path: Path) -> None:
        _write(tmp_path, "src/keep.py", "x = 1\n")
        _write(tmp_path, "build/out.py", "y = 2\n")
        index = RepoIndex.build(tmp_path, ignore_globs=["build"])
        assert "src/keep.py" in index.files
        assert "build/out.py" not in index.files

    def test_globs_auto_loaded_from_config_file(self, tmp_path: Path) -> None:
        _write(tmp_path, "src/keep.py", "x = 1\n")
        _write(tmp_path, "generated/gen.py", "y = 2\n")
        _write(tmp_path, ".patchcraft.yml", 'ignore_globs:\n  - "generated/**"\n')
        index = RepoIndex.build(tmp_path)
        assert "src/keep.py" in index.files
        assert "generated/gen.py" not in index.files


class TestRetrievalK:
    def test_config_used_when_env_unset(self, tmp_path: Path, monkeypatch) -> None:
        from src.orchestrator import _retrieval_k_for

        monkeypatch.delenv("PATCHCRAFT_RETRIEVAL_K", raising=False)
        _write(tmp_path, ".patchcraft.yml", "retrieval_k: 5\n")
        assert _retrieval_k_for(tmp_path) == 5

    def test_env_beats_config(self, tmp_path: Path, monkeypatch) -> None:
        from src.orchestrator import _retrieval_k_for

        monkeypatch.setenv("PATCHCRAFT_RETRIEVAL_K", "9")
        _write(tmp_path, ".patchcraft.yml", "retrieval_k: 5\n")
        assert _retrieval_k_for(tmp_path) == 9

    def test_bad_env_falls_back_to_default(self, tmp_path: Path, monkeypatch) -> None:
        from src.core.retrieval import DEFAULT_RETRIEVAL_K
        from src.orchestrator import _retrieval_k_for

        monkeypatch.setenv("PATCHCRAFT_RETRIEVAL_K", "garbage")
        assert _retrieval_k_for(tmp_path) == DEFAULT_RETRIEVAL_K


class TestFallbackChainOverride:
    def test_requested_model_first_then_configured_order(self) -> None:
        from src.core.llm import call_llm
        import litellm

        set_default_fallback_chain([
            "openrouter/openai/gpt-4o",
            "openrouter/deepseek/deepseek-chat",
        ])
        # Every model fails; we only observe the ORDER litellm is called with.
        seen_models: list[str] = []

        def fake_completion(**kwargs):
            seen_models.append(kwargs["model"])
            raise RuntimeError("stop at first")

        with mock.patch.object(litellm, "completion", side_effect=fake_completion):
            with pytest.raises(Exception):
                call_llm("openrouter/anthropic/claude-3.5-sonnet", "s", "u",
                         max_retries_per_model=1, backoff_base=0)
        assert seen_models[0] == "openrouter/anthropic/claude-3.5-sonnet"
        assert seen_models[1] == "openrouter/openai/gpt-4o"
        assert seen_models[2] == "openrouter/deepseek/deepseek-chat"
        assert call_llm is not None  # silence linters about the import

    def test_reset_restores_automatic_chain(self) -> None:
        set_default_fallback_chain(["m1"])
        assert get_default_fallback_chain() == ("m1",)
        set_default_fallback_chain(None)
        assert get_default_fallback_chain() is None


# ---------------------------------------------------------------------------
# CLI: fix command (exit codes + config precedence)
# ---------------------------------------------------------------------------
def _run_result(success: bool, halt_reason: str | None = None):
    from src.orchestrator import RunResult

    return RunResult(
        success=success,
        iterations=1,
        test_errors=[],
        files_changed=[],
        halt_reason=halt_reason,
    )


@pytest.fixture()
def cli_env(monkeypatch):
    """Common CLI scaffolding: mocked credits panel."""
    import main as main_module

    monkeypatch.setattr(main_module, "render_credits_panel", lambda *a, **k: None)
    return main_module


class TestFixCommand:
    def _invoke_fix(self, cli_env, args: list[str], issue: dict | None = None,
                    result=None):
        with (
            mock.patch("src.github.issue_fetcher.get_issue",
                       return_value=issue or {"number": 123, "title": "Bug", "body": "b"}),
            mock.patch("src.orchestrator.run_patchcraft_loop",
                       return_value=result or _run_result(True)) as loop_mock,
        ):
            exit_code = CliRunner().invoke(cli_env.cli, ["fix", *args]).exit_code
        return exit_code, loop_mock

    def test_url_success_exits_zero(self, cli_env, tmp_path: Path) -> None:
        exit_code, loop_mock = self._invoke_fix(
            cli_env,
            ["https://github.com/owner/repo/issues/123", str(tmp_path)],
        )
        assert exit_code == 0
        kwargs = loop_mock.call_args.kwargs
        assert kwargs["issue_description"].startswith("Title: Bug")
        assert kwargs["repo_path"] == str(tmp_path)

    def test_bare_number_with_repo_success(self, cli_env, tmp_path: Path) -> None:
        exit_code, loop_mock = self._invoke_fix(
            cli_env, ["--repo", "owner/repo", "123", str(tmp_path)],
        )
        assert exit_code == 0
        assert loop_mock.call_args.kwargs["model"]

    def test_budget_halt_exits_three(self, cli_env, tmp_path: Path) -> None:
        exit_code, _ = self._invoke_fix(
            cli_env,
            ["--repo", "owner/repo", "#9", str(tmp_path)],
            result=_run_result(False, halt_reason="Token budget exhausted."),
        )
        assert exit_code == 3

    def test_no_convergence_exits_one(self, cli_env, tmp_path: Path) -> None:
        exit_code, _ = self._invoke_fix(
            cli_env,
            ["--repo", "owner/repo", "9", str(tmp_path)],
            result=_run_result(False),
        )
        assert exit_code == 1

    def test_missing_repo_for_bare_number_exits_two(self, cli_env, tmp_path: Path) -> None:
        exit_code, loop_mock = self._invoke_fix(cli_env, ["9", str(tmp_path)])
        assert exit_code == 2
        loop_mock.assert_not_called()

    def test_invalid_reference_exits_two(self, cli_env, tmp_path: Path) -> None:
        exit_code, loop_mock = self._invoke_fix(cli_env, ["not-a-ref", str(tmp_path)])
        assert exit_code == 2
        loop_mock.assert_not_called()

    def test_config_file_provides_model_and_budget(self, cli_env, tmp_path: Path) -> None:
        _write(tmp_path, ".patchcraft.yml",
               "model: openrouter/x/y\ntoken_budget: 7777\n")
        exit_code, loop_mock = self._invoke_fix(
            cli_env, ["https://github.com/owner/repo/issues/1", str(tmp_path)],
        )
        assert exit_code == 0
        kwargs = loop_mock.call_args.kwargs
        assert kwargs["model"] == "openrouter/x/y"
        assert kwargs["token_budget"] == 7777

    def test_cli_flag_beats_config(self, cli_env, tmp_path: Path) -> None:
        _write(tmp_path, ".patchcraft.yml", "token_budget: 7777\n")
        with (
            mock.patch("src.github.issue_fetcher.get_issue",
                       return_value={"number": 1, "title": "t", "body": ""}),
            mock.patch("src.orchestrator.run_patchcraft_loop",
                       return_value=_run_result(True)) as loop_mock,
        ):
            CliRunner().invoke(cli_env.cli, [
                "-m", "openrouter/cli/model",
                "fix", "--token-budget", "42",
                "https://github.com/owner/repo/issues/1", str(tmp_path),
            ])
        kwargs = loop_mock.call_args.kwargs
        assert kwargs["model"] == "openrouter/cli/model"
        assert kwargs["token_budget"] == 42


class TestRunAndSelectExitCodes:
    def test_run_uses_config_defaults(self, cli_env, tmp_path: Path) -> None:
        _write(tmp_path, ".patchcraft.yml",
               "max_retries: 3\ntime_budget: 555\nmin_credits: 0.25\n")
        with mock.patch("src.orchestrator.run_patchcraft_loop",
                        return_value=_run_result(True)) as loop_mock:
            exit_code = CliRunner().invoke(
                cli_env.cli, ["run", str(tmp_path), "Fix the thing"]
            ).exit_code
        assert exit_code == 0
        kwargs = loop_mock.call_args.kwargs
        assert kwargs["max_retries"] == 3
        assert kwargs["time_budget_seconds"] == 555
        assert kwargs["min_remaining_credits"] == 0.25

    def test_run_no_convergence_exits_one(self, cli_env, tmp_path: Path) -> None:
        with mock.patch("src.orchestrator.run_patchcraft_loop",
                        return_value=_run_result(False)):
            exit_code = CliRunner().invoke(
                cli_env.cli, ["run", str(tmp_path), "Fix"]
            ).exit_code
        assert exit_code == 1

    def test_select_yes_picks_first_issue_headlessly(self, cli_env, tmp_path: Path) -> None:
        issues = [
            {"number": 11, "title": "first", "body": ""},
            {"number": 22, "title": "second", "body": ""},
        ]
        with (
            mock.patch("src.github.issue_fetcher.get_open_issues", return_value=issues),
            mock.patch("src.orchestrator.run_patchcraft_loop",
                       return_value=_run_result(True)) as loop_mock,
        ):
            exit_code = CliRunner().invoke(cli_env.cli, [
                "--yes", "select", "owner/repo", str(tmp_path),
            ]).exit_code

        assert exit_code == 0
        assert "Title: first" in loop_mock.call_args.kwargs["issue_description"]
