"""Test del layer LLM (con ``litellm.completion`` mockato)."""
from __future__ import annotations

import json
from unittest import mock

import litellm
import pytest
from litellm.exceptions import APIConnectionError, AuthenticationError, RateLimitError
from pydantic import BaseModel

from src.core.llm import MaxRetriesExceeded, build_fallback_chain, call_llm


class Movie(BaseModel):
    title: str
    year: int


def _completion_response(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}], "model": "mocked"}


def test_build_fallback_chain_deepseek_first() -> None:
    """DeepSeek richiesto -> la chain è DeepSeek, Anthropic, OpenAI."""
    chain = build_fallback_chain("deepseek/deepseek-chat")
    assert chain == [
        "deepseek/deepseek-chat",
        "anthropic/claude-3-7-sonnet-latest",
        "openai/gpt-4o",
    ]


def test_build_fallback_chain_rotates_requested_provider() -> None:
    """Il modello richiesto viene sempre provato per primo."""
    chain = build_fallback_chain("anthropic/claude-3-7-sonnet-latest")
    assert chain[0].startswith("anthropic")
    assert chain[1].startswith("deepseek")
    assert chain[2].startswith("openai")


def test_call_llm_text_output() -> None:
    with mock.patch.object(litellm, "completion", return_value=_completion_response("hello")) as mocked:
        result = call_llm("deepseek/deepseek-chat", "sys", "user", max_retries_per_model=1)
    assert result == "hello"
    assert mocked.call_args.kwargs["model"] == "deepseek/deepseek-chat"


def test_call_llm_fallback_after_rate_limit() -> None:
    """Rate limit su DeepSeek -> retry (transiente) e poi fallback su Anthropic."""
    seen: list[str] = []

    def fake_completion(**kwargs):
        seen.append(kwargs["model"])
        if kwargs["model"].startswith("deepseek"):
            raise RateLimitError("rate limited", llm_provider="deepseek", model=kwargs["model"])
        return _completion_response("recovered")

    with mock.patch.object(litellm, "completion", side_effect=fake_completion):
        result = call_llm(
            "deepseek/deepseek-chat", "sys", "user", max_retries_per_model=2, backoff_base=0
        )

    assert result == "recovered"
    # Il rate limit è transiente: DeepSeek viene riprovato per i 2 tentativi
    # previsti, e solo dopo scatta il fallback automatico su Anthropic.
    assert seen == [
        "deepseek/deepseek-chat",
        "deepseek/deepseek-chat",
        "anthropic/claude-3-7-sonnet-latest",
    ]


def test_call_llm_fallback_after_rate_limit_single_try() -> None:
    """Con max_retries_per_model=1 il fallback è immediato."""
    seen: list[str] = []

    def fake_completion(**kwargs):
        seen.append(kwargs["model"])
        if kwargs["model"].startswith("deepseek"):
            raise RateLimitError("rate limited", llm_provider="deepseek", model=kwargs["model"])
        return _completion_response("recovered")

    with mock.patch.object(litellm, "completion", side_effect=fake_completion):
        result = call_llm(
            "deepseek/deepseek-chat", "sys", "user", max_retries_per_model=1, backoff_base=0
        )

    assert result == "recovered"
    assert seen == ["deepseek/deepseek-chat", "anthropic/claude-3-7-sonnet-latest"]


def test_call_llm_skips_retries_on_auth_error() -> None:
    """Errori non retryable: niente retry, si passa subito al provider successivo."""
    seen: list[str] = []

    def fake_completion(**kwargs):
        seen.append(kwargs["model"])
        if kwargs["model"].startswith("deepseek"):
            raise AuthenticationError("invalid key", llm_provider="deepseek", model=kwargs["model"])
        return _completion_response("ok")

    with mock.patch.object(litellm, "completion", side_effect=fake_completion):
        result = call_llm(
            "deepseek/deepseek-chat", "sys", "user", max_retries_per_model=3, backoff_base=0
        )

    assert result == "ok"
    assert seen == ["deepseek/deepseek-chat", "anthropic/claude-3-7-sonnet-latest"]


def test_call_llm_structured_output() -> None:
    """json_schema Pydantic -> la funzione restituisce un'istanza validata."""
    payload = json.dumps({"title": "Alien", "year": 1979})
    with mock.patch.object(litellm, "completion", return_value=_completion_response(payload)) as mocked:
        result = call_llm("deepseek/deepseek-chat", "sys", "user", Movie, max_retries_per_model=1)

    assert isinstance(result, Movie)
    assert result.title == "Alien"
    assert result.year == 1979
    # DeepSeek/OpenAI supportano response_format json_object
    assert mocked.call_args.kwargs["response_format"] == {"type": "json_object"}


def test_call_llm_no_response_format_for_anthropic() -> None:
    """Anthropic non riceve response_format (JSON estratto e validato a valle)."""
    payload = json.dumps({"title": "Dune", "year": 1984})
    with mock.patch.object(litellm, "completion", return_value=_completion_response(payload)) as mocked:
        result = call_llm("anthropic/claude-3-7-sonnet-latest", "sys", "user", Movie, max_retries_per_model=1)
    assert isinstance(result, Movie)
    assert "response_format" not in mocked.call_args.kwargs


def test_call_llm_malformed_json_falls_back() -> None:
    """JSON malformato su DeepSeek -> fallback sul provider successivo."""

    def fake_completion(**kwargs):
        if kwargs["model"].startswith("deepseek"):
            return _completion_response("scusa, niente JSON")
        return _completion_response(json.dumps({"title": "Dune", "year": 1984}))

    with mock.patch.object(litellm, "completion", side_effect=fake_completion):
        result = call_llm(
            "deepseek/deepseek-chat", "sys", "user", Movie, max_retries_per_model=1, backoff_base=0
        )

    assert isinstance(result, Movie)
    assert result.title == "Dune"


def test_call_llm_all_providers_fail_raises() -> None:
    """Se tutti i modelli della chain falliscono -> MaxRetriesExceeded."""

    def fake_completion(**kwargs):
        raise APIConnectionError("connection lost", llm_provider=kwargs["model"], model=kwargs["model"])

    with mock.patch.object(litellm, "completion", side_effect=fake_completion):
        with pytest.raises(MaxRetriesExceeded):
            call_llm(
                "deepseek/deepseek-chat", "sys", "user", max_retries_per_model=1, backoff_base=0
            )


def test_call_llm_structured_prompt_contains_schema() -> None:
    """Con json_schema il prompt contiene lo schema JSON da rispettare."""
    payload = json.dumps({"title": "Alien", "year": 1979})
    with mock.patch.object(litellm, "completion", return_value=_completion_response(payload)) as mocked:
        call_llm("deepseek/deepseek-chat", "sys", "user question", Movie, max_retries_per_model=1)
    user_msg = mocked.call_args.kwargs["messages"][1]["content"]
    assert "JSON Schema" in user_msg
    assert '"title"' in user_msg