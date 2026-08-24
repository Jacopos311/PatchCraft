"""LLM access for PatchCraft.

Wraps :func:`litellm.completion` adding:

* **Automatic fallback across providers** (DeepSeek -> Anthropic -> OpenAI) on
  exceptions or rate limits. The requested model is always tried first; the
  others follow in deterministic order.
* **Structured JSON output**: passing a Pydantic ``json_schema`` constrains the
  response to compliant JSON validated with Pydantic.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Type, Union, overload

import litellm
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    Timeout,
)
from pydantic import BaseModel

from src.core.cache import get_memo_cache

logger = logging.getLogger(__name__)

# Drop unsupported parameters instead of raising exceptions that would
# unnecessarily trigger the fallback.
litellm.drop_params = True

DEFAULT_MAX_RETRIES_PER_MODEL = 2
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_BACKOFF_SECONDS = 1.0
DEFAULT_MAX_BACKOFF_SECONDS = 16.0

# Type alias for the optional token-usage callback: invoked as
# ``usage_sink(prompt_tokens, completion_tokens)`` after every successful
# completion so callers can enforce per-task token budgets.
UsageSink = Callable[[int, int], None]

# Cap on completion output tokens. Requesting very large budgets (e.g. 16384)
# makes OpenRouter reject the call with HTTP 402 (insufficient credits) even
# when the account still has enough balance for a smaller generation.
# Overridable via the PATCHCRAFT_MAX_OUTPUT_TOKENS environment variable.
DEFAULT_MAX_OUTPUT_TOKENS = max(1, int(os.getenv("PATCHCRAFT_MAX_OUTPUT_TOKENS", "3000")))

DEFAULT_MODELS: Mapping[str, str] = {
    "deepseek": "openrouter/deepseek/deepseek-chat",
    "anthropic": "openrouter/anthropic/claude-3.5-sonnet",
    "openai": "openrouter/openai/gpt-4o",
}

_FALLBACK_ORDER = ("deepseek", "anthropic", "openai")

_JSON_RESPONSE_FORMAT_PROVIDERS = frozenset(
    {"deepseek", "openai", "azure", "mistral", "groq", "together", "gemini", "vertex_ai", "fireworks_ai"}
)

_TRANSIENT_ERRORS = (RateLimitError, APIConnectionError, Timeout)
_NON_RETRYABLE_ERRORS = (AuthenticationError, BadRequestError, NotFoundError, PermissionDeniedError)


class LLMError(RuntimeError):
    """Generic error from the LLM layer."""


class LLMResponseError(LLMError):
    """The provider response does not have the expected structure."""


class MaxRetriesExceeded(LLMError):
    """Every model in the fallback chain failed."""

    def __init__(
        self,
        provider_model: str,
        chain: Sequence[str],
        last_error: Optional[BaseException] = None,
    ) -> None:
        self.provider_model = provider_model
        self.chain = tuple(chain)
        self.last_error = last_error
        detail = (
            f"{type(last_error).__name__}: {last_error}"
            if last_error is not None
            else "no error recorded"
        )
        super().__init__(
            f"call_llm exhausted every model in the fallback chain for "
            f"'{provider_model}' (chain: {', '.join(self.chain)}). Last error: {detail}"
        )


def _provider_of(model: str) -> str:
    """Extract the provider (prefix) from a litellm-style model id."""
    name = model.strip()
    for sep in ("/", ":"):
        if sep in name:
            return name.split(sep, 1)[0].lower()
    return name.lower()


def _is_openrouter(model: str) -> bool:
    """Whether the model id is routed through OpenRouter (``openrouter/...``)."""
    return _provider_of(model) == "openrouter"


def _effective_provider(model: str) -> str:
    """Resolve the underlying vendor of a litellm-style model id.

    OpenRouter-routed ids (``openrouter/<vendor>/<model>``) map back to the
    vendor segment (e.g. ``deepseek``, ``anthropic``, ``openai``); plain ids
    keep their own prefix.
    """
    name = model.strip()
    if _is_openrouter(name):
        remainder = name.split("/", 1)[1].strip() if "/" in name else ""
        return _provider_of(remainder) if remainder else "openrouter"
    return _provider_of(name)


def build_fallback_chain(provider_model: str) -> list[str]:
    """Build the fallback chain.

    The requested model is always tried first, then the other canonical
    providers in DeepSeek -> Anthropic -> OpenAI order.
    """
    requested = provider_model.strip()
    if not requested:
        raise ValueError("provider_model must not be empty")
    provider = _effective_provider(requested)
    chain = [requested]
    for name in _FALLBACK_ORDER:
        if name == provider:
            continue
        default_model = DEFAULT_MODELS[name]
        if default_model not in chain:
            chain.append(default_model)
    return chain


# ---------------------------------------------------------------------------
# Process-wide fallback-chain override (Roadmap Step 3.3, `fallback_models`)
# ---------------------------------------------------------------------------
_default_fallback_chain: Optional[tuple[str, ...]] = None


def set_default_fallback_chain(models: Optional[Sequence[str]]) -> None:
    """Override the automatic fallback chain process-wide.

    Used by the CLI to apply ``fallback_models`` from ``.patchcraft.yml``.
    Pass ``None``/empty to restore the automatic canonical chain.
    """
    global _default_fallback_chain
    cleaned = tuple(m.strip() for m in (models or []) if m and m.strip())
    _default_fallback_chain = cleaned or None


def get_default_fallback_chain() -> Optional[tuple[str, ...]]:
    """Currently configured override chain, or ``None`` when automatic."""
    return _default_fallback_chain


def _build_messages(
    system_prompt: str,
    user_prompt: str,
    json_schema: Optional[Type[BaseModel]],
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if json_schema is not None:
        schema_json = json.dumps(json_schema.model_json_schema(), indent=2)
        instruction = (
            "\n\nIMPORTANT: respond ONLY with a single valid JSON object "
            "(no markdown code fences and no extra text) that matches "
            f"exactly the following JSON Schema:\n{schema_json}"
        )
        messages[1]["content"] = f"{user_prompt}{instruction}"
    return messages


def _build_completion_kwargs(
    model: str,
    messages: list[dict[str, str]],
    json_schema: Optional[Type[BaseModel]],
    timeout: float,
    max_tokens: Optional[int] = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "timeout": timeout,
        # Keep the default output budget modest so calls fit within the
        # remaining OpenRouter credit allowance instead of failing with
        # HTTP 402. Callers can raise it for large completions (e.g. the
        # coder agent emitting complete file contents).
        "max_tokens": max_tokens if max_tokens and max_tokens > 0 else DEFAULT_MAX_OUTPUT_TOKENS,
    }
    if _is_openrouter(model):
        # Route through OpenRouter natively with the single shared key so
        # litellm never looks for provider-specific keys such as
        # ANTHROPIC_API_KEY or OPENAI_API_KEY.
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key and api_key.strip():
            kwargs["api_key"] = api_key.strip()
        else:
            logger.warning(
                "[patchcraft.llm] OPENROUTER_API_KEY is not set; falling back "
                "to litellm's own environment resolution for '%s'.",
                model,
            )
    if json_schema is not None and _effective_provider(model) in _JSON_RESPONSE_FORMAT_PROVIDERS:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


def _extract_content(response: Any) -> Any:
    """Extract the content from a :func:`litellm.completion` response."""
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMResponseError(f"Unexpected provider response: {response!r}") from exc


def _extract_usage(response: Any) -> tuple[int, int]:
    """Extract ``(prompt_tokens, completion_tokens)`` from a response.

    Returns ``(0, 0)`` when the provider does not report usage information.
    """
    try:
        if isinstance(response, Mapping):
            usage = response.get("usage")
        else:
            usage = getattr(response, "usage", None)
        if not usage:
            return 0, 0
        if isinstance(usage, Mapping):
            prompt = int(usage.get("prompt_tokens") or 0)
            completion = int(usage.get("completion_tokens") or 0)
        else:
            prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion = int(getattr(usage, "completion_tokens", 0) or 0)
        return prompt, completion
    except (TypeError, ValueError, AttributeError):
        return 0, 0


def _as_text(content: Any) -> str:
    """Normalize content into plain text (supports strings and blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(block["text"]) if isinstance(block, Mapping) and "text" in block else str(block)
            for block in content
        ]
        return "".join(parts)
    return str(content)


def _strip_json_fences(text: str) -> str:
    """Remove any markdown code fences around the returned JSON."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _coerce_to_model(content: Any, json_schema: Type[BaseModel]) -> BaseModel:
    """Convert provider content into a Pydantic model instance."""
    if isinstance(content, (dict, list)):
        data = content
    elif isinstance(content, str):
        data = json.loads(_strip_json_fences(content))
    else:
        raise LLMResponseError(
            f"Content cannot be converted to JSON: {type(content)!r}"
        )
    return json_schema.model_validate(data)


@overload
def call_llm(
    provider_model: str,
    system_prompt: str,
    user_prompt: str,
    json_schema: None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries_per_model: int = DEFAULT_MAX_RETRIES_PER_MODEL,
    backoff_base: float = DEFAULT_BACKOFF_SECONDS,
    fallback_chain: Optional[Sequence[str]] = None,
    max_tokens: Optional[int] = None,
    usage_sink: Optional[Callable[[int, int], None]] = None,
    use_cache: Optional[bool] = None,
) -> str: ...


@overload
def call_llm(
    provider_model: str,
    system_prompt: str,
    user_prompt: str,
    json_schema: Type[BaseModel],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries_per_model: int = DEFAULT_MAX_RETRIES_PER_MODEL,
    backoff_base: float = DEFAULT_BACKOFF_SECONDS,
    fallback_chain: Optional[Sequence[str]] = None,
    max_tokens: Optional[int] = None,
    usage_sink: Optional[Callable[[int, int], None]] = None,
    use_cache: Optional[bool] = None,
) -> BaseModel: ...



def call_llm(
    provider_model: str,
    system_prompt: str,
    user_prompt: str,
    json_schema: Optional[Type[BaseModel]] = None,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries_per_model: int = DEFAULT_MAX_RETRIES_PER_MODEL,
    backoff_base: float = DEFAULT_BACKOFF_SECONDS,
    fallback_chain: Optional[Sequence[str]] = None,
    max_tokens: Optional[int] = None,
    usage_sink: Optional[Callable[[int, int], None]] = None,
    use_cache: Optional[bool] = None,
) -> Union[str, BaseModel]:
    """Query the LLM with automatic cross-model fallback.

    Parameters
    ----------
    provider_model : str
        Primary model in litellm format (e.g. ``deepseek/deepseek-chat``).
    system_prompt, user_prompt : str
        System and user prompts.
    json_schema : Type[BaseModel] | None
        When provided, the output is constrained to JSON matching the schema
        and the function returns an instance of ``json_schema`` (instead of a
        plain ``str``).
    timeout : float
        Timeout (seconds) for each single call.
    max_retries_per_model : int
        Attempts per model before moving to the next one in the chain.
    backoff_base : float
        Base of the exponential wait on transient errors (e.g. rate limits).
    fallback_chain : Sequence[str] | None
        Overrides the automatic chain (default DeepSeek -> Anthropic -> OpenAI).
    max_tokens : int | None
        Maximum completion output tokens. ``None`` uses the conservative
        default (:data:`DEFAULT_MAX_OUTPUT_TOKENS`); raise it for calls that
        must emit large payloads (e.g. complete-file patches).
    usage_sink : Callable[[int, int], None] | None
        Optional callback invoked as ``usage_sink(prompt_tokens,
        completion_tokens)`` after every successful completion, so callers
        can enforce per-task token budgets. Never raises into the loop:
        sink exceptions are swallowed and logged. NOT invoked for memo-cache
        hits (a cached response costs zero tokens).
    use_cache : bool | None
        Per-call override of the LLM memo cache (Step 3.1). ``None`` (the
        default) follows the process-wide configuration (see
        :func:`src.core.cache.configure_memo_cache` and the
        ``PATCHCRAFT_NO_CACHE`` environment variable).

    Returns
    -------
    ``str`` when ``json_schema`` is ``None``, otherwise an instance of
    ``json_schema``.

    Raises
    ------
    ``MaxRetriesExceeded`` when every model in the fallback chain fails.
    """
    if fallback_chain:
        chain = list(fallback_chain)
    elif _default_fallback_chain:
        # Configured chain (Step 3.3): the requested model is always tried
        # first, then the configured order, deduplicated.
        seen: set[str] = set()
        chain = []
        for candidate in (provider_model, *_default_fallback_chain):
            stripped = candidate.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                chain.append(stripped)
    else:
        chain = build_fallback_chain(provider_model)
    messages = _build_messages(system_prompt, user_prompt, json_schema)
    logger.debug("[patchcraft.llm] fallback chain: %s", chain)

    # -- Step 3.1: LLM memo cache --------------------------------------
    # Identical (account, model, messages, schema, max_tokens) calls reuse
    # the earlier response instead of hitting the network again. The entry
    # is stored only AFTER the response validates, so a corrupt or unusable
    # cached payload degrades gracefully to a normal cache miss.
    memo = get_memo_cache()
    caching_enabled = memo.enabled if use_cache is None else bool(use_cache)
    cache_key: Optional[str] = None
    if caching_enabled:
        schema_json = (
            json.dumps(json_schema.model_json_schema(), sort_keys=True)
            if json_schema is not None
            else None
        )
        cache_key = memo.make_key(
            model=provider_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_json=schema_json,
            max_tokens=max_tokens,
        )
        cached_content = memo.lookup(cache_key)
        if cached_content is not None:
            logger.info("[patchcraft.llm] memo-cache HIT (%s)", provider_model)
            try:
                if json_schema is None:
                    return _as_text(cached_content)
                return _coerce_to_model(cached_content, json_schema)
            except Exception as exc:  # noqa: BLE001 - unusable entry == miss
                logger.warning(
                    "[patchcraft.llm] memo-cache entry unusable (%s: %s); "
                    "falling through to a live call.",
                    type(exc).__name__, exc,
                )

    last_error: Optional[BaseException] = None
    for model in chain:
        completion_kwargs = _build_completion_kwargs(model, messages, json_schema, timeout, max_tokens)
        for attempt in range(1, max_retries_per_model + 1):
            try:
                response = litellm.completion(**completion_kwargs)
                if usage_sink is not None:
                    try:
                        prompt_toks, completion_toks = _extract_usage(response)
                        if prompt_toks or completion_toks:
                            usage_sink(prompt_toks, completion_toks)
                    except Exception as exc:  # noqa: BLE001 - sink must never break the loop
                        logger.warning("usage_sink raised: %s: %s", type(exc).__name__, exc)
                content = _as_text(_extract_content(response))
                result: Union[str, BaseModel]
                if json_schema is None:
                    result = content
                else:
                    result = _coerce_to_model(content, json_schema)
                # Only validated responses are cached.
                if cache_key is not None:
                    memo.store(cache_key, content)
                return result
            except Exception as exc:  # noqa: BLE001 - any error triggers fallback
                last_error = exc
                logger.warning(
                    "[patchcraft.llm] '%s' attempt %d/%d failed: %s: %s",
                    model, attempt, max_retries_per_model, type(exc).__name__, exc,
                )
                if isinstance(exc, _NON_RETRYABLE_ERRORS):
                    break  # no retry: skip straight to the next model
                if isinstance(exc, _TRANSIENT_ERRORS) and attempt < max_retries_per_model:
                    wait = min(backoff_base * (2 ** (attempt - 1)), DEFAULT_MAX_BACKOFF_SECONDS)
                    logger.info("[patchcraft.llm] waiting %.1fs before retrying '%s'", wait, model)
                    time.sleep(wait)

    raise MaxRetriesExceeded(
        provider_model=provider_model,
        chain=chain,
        last_error=last_error,
    )


__all__ = [
    "call_llm",
    "build_fallback_chain",
    "set_default_fallback_chain",
    "get_default_fallback_chain",
    "LLMError",
    "LLMResponseError",
    "MaxRetriesExceeded",
]
