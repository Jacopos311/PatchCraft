"""Diagnostic agent: analyzes context and bugs, producing a structured diagnosis."""
from __future__ import annotations

from typing import Callable, Optional, Type, Union

from pydantic import BaseModel, Field

from src.core.llm import call_llm

DEFAULT_MODEL = "openrouter/deepseek/deepseek-chat"


class Diagnosis(BaseModel):
    """Structured outcome of a bug analysis."""

    summary: str = Field(description="One-sentence summary of the problem.")
    root_cause: str = Field(description="Root cause of the bug.")
    affected_files: list[str] = Field(description="Files involved in the bug.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1.")


SYSTEM_PROMPT = (
    "You are a senior debugging engineer. Analyze the provided context and "
    "identify the bug with precision, citing the affected files and the root cause."
)


def diagnose(
    context: str,
    provider_model: str = DEFAULT_MODEL,
    json_schema: Optional[Type[BaseModel]] = None,
    usage_sink: Optional[Callable[[int, int], None]] = None,
) -> Union[str, BaseModel]:
    """Analyze the context and return a :class:`Diagnosis`.

    Uses the automatic DeepSeek -> Anthropic -> OpenAI fallback chain.
    When ``json_schema=None`` the structured :class:`Diagnosis` is returned;
    for free text provide your own schema or call :func:`src.core.llm.call_llm`.
    ``usage_sink`` optionally receives ``(prompt_tokens, completion_tokens)``
    after every successful completion (per-task token accounting).
    """
    return call_llm(
        provider_model=provider_model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=context,
        json_schema=json_schema or Diagnosis,
        usage_sink=usage_sink,
    )