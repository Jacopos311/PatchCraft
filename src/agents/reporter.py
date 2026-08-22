"""Reporter agent: produces the diff and the markdown for the Pull Request."""
from __future__ import annotations

from typing import Optional, Type, Union

from pydantic import BaseModel, Field

from src.core.llm import call_llm


class PatchReport(BaseModel):
    """Final report: summary, diff and Pull Request boilerplate."""

    title: str = Field(description="Pull Request title.")
    summary: str = Field(description="Summary of the applied changes.")
    diff: str = Field(description="Diff of the proposed changes.")
    pr_markdown: str = Field(description="Markdown ready for the PR description.")


SYSTEM_PROMPT = (
    "You are a senior engineer. Given the diff and the test results, produce "
    "a concise report and a markdown description for the pull request."
)


def generate_report(
    diff: str,
    provider_model: str,
    json_schema: Optional[Type[BaseModel]] = None,
) -> Union[str, BaseModel]:
    """Generate a report (defaults to :class:`PatchReport`) from a diff."""
    return call_llm(
        provider_model=provider_model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=diff,
        json_schema=json_schema or PatchReport,
    )