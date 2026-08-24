"""Tests for the caching layer (Roadmap Step 3.1).

Covers hit/miss paths of:
* the LLM memo cache (including account isolation and env kill switch);
* the targeted-test verdict cache (targeted-only, never the full gate);
* a regression check that the repo index cache is never stale.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import litellm
import pytest

from src.core.cache import (
    ENV_NO_CACHE,
    LLMMemoCache,
    TestResultCache,
    configure_memo_cache,
    get_memo_cache,
    reset_memo_cache,
)
from src.core.llm import call_llm
from src.core.repo_index import RepoIndex


@pytest.fixture(autouse=True)
def _isolated_memo_cache(monkeypatch):
    """Keep every test's memo-cache configuration isolated."""
    monkeypatch.delenv(ENV_NO_CACHE, raising=False)
    reset_memo_cache()
    yield
    reset_memo_cache()


def _completion_response(content: str):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _write(root: Path, rel: str, content: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestLLMMemoCacheUnit:
    def test_store_then_lookup_hit(self, tmp_path: Path) -> None:
        cache = LLMMemoCache(base_dir=tmp_path, enabled=True)
        key = cache.make_key("model-x", "sys", "user")
        assert cache.lookup(key) is None  # miss
        cache.store(key, "hello")
        assert cache.lookup(key) == "hello"  # hit

    def test_disabled_cache_never_stores_or_hits(self, tmp_path: Path) -> None:
        cache = LLMMemoCache(base_dir=tmp_path, enabled=False)
        key = cache.make_key("model-x", "sys", "user")
        cache.store(key, "hello")
        assert cache.lookup(key) is None
        assert not list(tmp_path.rglob("*.json"))

    def test_env_kill_switch_beats_enabled(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv(ENV_NO_CACHE, "1")
        cache = LLMMemoCache(base_dir=tmp_path, enabled=True)
        assert cache.enabled is False
        key = cache.make_key("m", "s", "u")
        cache.store(key, "hello")
        assert cache.lookup(key) is None

    def test_different_prompts_produce_different_keys(self, tmp_path: Path) -> None:
        cache = LLMMemoCache(base_dir=tmp_path, enabled=True)
        k1 = cache.make_key("m", "sys", "user-1")
        k2 = cache.make_key("m", "sys", "user-2")
        k3 = cache.make_key("other-model", "sys", "user-1")
        assert len({k1, k2, k3}) == 3

    def test_account_isolation(self, tmp_path: Path, monkeypatch) -> None:
        """Different OPENROUTER_API_KEY accounts never share entries."""
        cache = LLMMemoCache(base_dir=tmp_path, enabled=True)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-account-a")
        key_a = cache.make_key("m", "s", "u")
        cache.store(key_a, "answer-a")

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-account-b")
        key_b = cache.make_key("m", "s", "u")
        assert key_b != key_a
        assert cache.lookup(key_a) is None  # other account's entry invisible

        # ...and the on-disk layout keeps accounts in separate folders.
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-account-a")
        assert cache.lookup(key_a) == "answer-a"

    def test_corrupt_entry_is_a_miss(self, tmp_path: Path) -> None:
        from src.core.cache import _JsonDiskCache

        store = _JsonDiskCache(tmp_path / "llm" / LLMMemoCache.account_hash())
        key = "deadbeef"
        store._path(key).parent.mkdir(parents=True, exist_ok=True)
        store._path(key).write_text("{not json", encoding="utf-8")

        cache = LLMMemoCache(base_dir=tmp_path, enabled=True)
        assert cache.lookup(key) is None



class TestCallLLMMemoIntegration:
    def test_identical_call_hits_cache_once(self, tmp_path: Path) -> None:
        configure_memo_cache(enabled=True, base_dir=tmp_path / "cache")
        calls = {"n": 0}

        def fake_completion(**kwargs):
            calls["n"] += 1
            return _completion_response(f"reply-{calls['n']}")

        with mock.patch.object(litellm, "completion", side_effect=fake_completion):
            first = call_llm("openrouter/deepseek/deepseek-chat", "sys", "user",
                             max_retries_per_model=1)
            second = call_llm("openrouter/deepseek/deepseek-chat", "sys", "user",
                              max_retries_per_model=1)

        assert first == second == "reply-1"
        assert calls["n"] == 1  # second call served from cache

    def test_usage_sink_not_called_on_cache_hit(self, tmp_path: Path) -> None:
        configure_memo_cache(enabled=True, base_dir=tmp_path / "cache")
        sink_calls: list[tuple[int, int]] = []

        def sink(p: int, c: int) -> None:
            sink_calls.append((p, c))

        with mock.patch.object(
            litellm, "completion", return_value=_completion_response("hi")
        ):
            call_llm("openrouter/deepseek/deepseek-chat", "s", "u",
                     max_retries_per_model=1, usage_sink=sink)
            call_llm("openrouter/deepseek/deepseek-chat", "s", "u",
                     max_retries_per_model=1, usage_sink=sink)

        assert len(sink_calls) == 1  # only the live call reported usage

    def test_use_cache_false_overrides_enabled_config(self, tmp_path: Path) -> None:
        configure_memo_cache(enabled=True, base_dir=tmp_path / "cache")
        calls = {"n": 0}

        def fake_completion(**kwargs):
            calls["n"] += 1
            return _completion_response(f"reply-{calls['n']}")

        with mock.patch.object(litellm, "completion", side_effect=fake_completion):
            a = call_llm("openrouter/deepseek/deepseek-chat", "s", "u",
                         max_retries_per_model=1, use_cache=False)
            b = call_llm("openrouter/deepseek/deepseek-chat", "s", "u",
                         max_retries_per_model=1, use_cache=False)

        assert a != b
        assert calls["n"] == 2

    def test_json_schema_response_round_trips_through_cache(self, tmp_path: Path) -> None:
        from pydantic import BaseModel

        class Movie(BaseModel):
            title: str
            year: int

        configure_memo_cache(enabled=True, base_dir=tmp_path / "cache")

        with mock.patch.object(
            litellm, "completion",
            return_value=_completion_response('{"title": "Alien", "year": 1979}'),
        ) as mocked:
            first = call_llm("openrouter/deepseek/deepseek-chat", "s", "u",
                             Movie, max_retries_per_model=1)
            second = call_llm("openrouter/deepseek/deepseek-chat", "s", "u",
                              Movie, max_retries_per_model=1)

        assert isinstance(first, Movie) and isinstance(second, Movie)
        assert second == first
        assert mocked.call_count == 1

    def test_unparseable_cached_payload_degrades_to_live_call(self, tmp_path: Path) -> None:
        from pydantic import BaseModel

        class Movie(BaseModel):
            title: str
            year: int

        configure_memo_cache(enabled=True, base_dir=tmp_path / "cache")
        # Seed an unusable cached payload for the exact request about to be made.
        memo = get_memo_cache()
        key = memo.make_key(
            model="openrouter/deepseek/deepseek-chat",
            system_prompt="s",
            user_prompt="u",
            schema_json=json.dumps(Movie.model_json_schema(), sort_keys=True),
            max_tokens=None,
        )
        memo.store(key, "this is not JSON at all")

        calls = {"n": 0}

        def fake_completion(**kwargs):
            calls["n"] += 1
            return _completion_response('{"title": "Dune", "year": 1984}')

        with mock.patch.object(litellm, "completion", side_effect=fake_completion):
            result = call_llm("openrouter/deepseek/deepseek-chat", "s", "u",
                              Movie, max_retries_per_model=1)

        assert isinstance(result, Movie)
        assert calls["n"] == 1  # fell through to a live call instead of failing


class TestTestResultCache:
    def test_store_lookup_roundtrip(self, tmp_path: Path) -> None:
        cache = TestResultCache(tmp_path, enabled=True)
        key = cache.make_key("fp-1", ["tests/test_a.py"])
        assert cache.lookup(key) is None  # miss
        cache.store(key, success=False, exit_code=1, stdout="out", stderr="err")
        verdict = cache.lookup(key)
        assert verdict == {
            "success": False, "exit_code": 1, "stdout": "out", "stderr": "err",
        }

    def test_different_subset_is_a_miss(self, tmp_path: Path) -> None:
        cache = TestResultCache(tmp_path, enabled=True)
        cache.store(cache.make_key("fp", ["tests/test_a.py"]),
                    success=True, exit_code=0)
        assert cache.lookup(cache.make_key("fp", ["tests/test_b.py"])) is None

    def test_different_patch_fingerprint_is_a_miss(self, tmp_path: Path) -> None:
        cache = TestResultCache(tmp_path, enabled=True)
        cache.store(cache.make_key("fp-old", ["tests/test_a.py"]),
                    success=True, exit_code=0)
        assert cache.lookup(cache.make_key("fp-new", ["tests/test_a.py"])) is None

    def test_target_order_does_not_change_the_key(self, tmp_path: Path) -> None:
        cache = TestResultCache(tmp_path, enabled=True)
        assert (
            cache.make_key("fp", ["b_test.py", "a_test.py"])
            == cache.make_key("fp", ["a_test.py", "b_test.py"])
        )

    def test_disabled_cache_is_inert(self, tmp_path: Path) -> None:
        cache = TestResultCache(tmp_path, enabled=False)
        key = cache.make_key("fp", ["t.py"])
        cache.store(key, success=True, exit_code=0)
        assert cache.lookup(key) is None

    def test_corrupt_verdict_file_is_a_miss(self, tmp_path: Path) -> None:
        cache = TestResultCache(tmp_path, enabled=True)
        key = cache.make_key("fp", ["t.py"])
        cache.directory.mkdir(parents=True, exist_ok=True)
        (cache.directory / f"{key}.json").write_text("{broken", encoding="utf-8")
        assert cache.lookup(key) is None


class TestRepoIndexNeverStale:
    def test_changed_file_is_reindexed(self, tmp_path: Path) -> None:
        """Regression guard for Step 3.1 point 1: index must track edits."""
        _write(tmp_path, "pkg/mod.py", "def old_name():\n    pass\n")
        first = RepoIndex.build(tmp_path)
        assert any(s.name == "old_name" for s in first.files["pkg/mod.py"].symbols)

        # Modify the file AFTER the index was built...
        _write(tmp_path, "pkg/mod.py", "def new_name():\n    pass\n")
        second = RepoIndex.build(tmp_path)

        # ...and verify the rebuilt index reflects the new content.
        names = {s.name for s in second.files["pkg/mod.py"].symbols}
        assert "new_name" in names
        assert "old_name" not in names

    def test_deleted_file_disappears_from_index(self, tmp_path: Path) -> None:
        _write(tmp_path, "pkg/gone.py", "x = 1\n")
        RepoIndex.build(tmp_path)
        (tmp_path / "pkg" / "gone.py").unlink()
        refreshed = RepoIndex.build(tmp_path)
        assert "pkg/gone.py" not in refreshed.files
