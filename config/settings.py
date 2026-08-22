"""PatchCraft settings: API keys and models.

Values are read from environment variables; defaults mirror the canonical
DeepSeek -> Anthropic -> OpenAI fallback chain.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

DEFAULT_PRIMARY_MODEL = "deepseek/deepseek-chat"
DEFAULT_FALLBACK_CHAIN: tuple[str, ...] = (
    "deepseek/deepseek-chat",
    "anthropic/claude-3-7-sonnet-latest",
    "openai/gpt-4o",
)


@dataclass(frozen=True)
class Settings:
    """Runtime settings for PatchCraft."""

    primary_model: str = DEFAULT_PRIMARY_MODEL
    fallback_chain: tuple[str, ...] = DEFAULT_FALLBACK_CHAIN
    deepseek_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    llm_max_retries: int = 2
    llm_timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables.

        Read variables:
            PATCHCRAFT_PRIMARY_MODEL, PATCHCRAFT_FALLBACK_MODELS (CSV),
            DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY,
            PATCHCRAFT_MAX_RETRIES, PATCHCRAFT_LLM_TIMEOUT
        """
        raw_chain = os.getenv("PATCHCRAFT_FALLBACK_MODELS")
        fallback_chain = (
            tuple(m.strip() for m in raw_chain.split(",") if m.strip())
            if raw_chain
            else DEFAULT_FALLBACK_CHAIN
        )
        return cls(
            primary_model=os.getenv("PATCHCRAFT_PRIMARY_MODEL", DEFAULT_PRIMARY_MODEL),
            fallback_chain=fallback_chain,
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            llm_max_retries=int(os.getenv("PATCHCRAFT_MAX_RETRIES", "2")),
            llm_timeout=float(os.getenv("PATCHCRAFT_LLM_TIMEOUT", "60")),
        )