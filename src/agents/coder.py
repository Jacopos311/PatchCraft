"""Coder agent: generates and corrects patches for a diagnosed bug.

Produced patches carry the **complete content** of the files to create or
modify (:class:`Patch` model), so the orchestrator can write them to the real
files of the target project.

Because patches embed whole files, the completion budget is **dynamic**: it
scales with the number of files involved instead of using the conservative
chat default (which would truncate the JSON and silently discard the patch).
"""
from __future__ import annotations

import json
from typing import Optional, Type, Union

from pydantic import BaseModel, Field

from src.core.llm import call_llm


class FilePatch(BaseModel):
    """A single file to create or overwrite."""

    file_path: str = Field(
        description="Path of the file relative to the project root (e.g. src/calc.py)."
    )
    new_content: str = Field(
        description="COMPLETE final content of the file (not a partial diff)."
    )


class Patch(BaseModel):
    """A complete patch: set of files to create/modify."""

    files: list[FilePatch] = Field(description="List of files to create or modify.")
    notes: str = Field(
        default="", description="Short notes about the changes (for human review)."
    )


# ---------------------------------------------------------------------------
# Dynamic completion budget ("dynamic completion" guardrail)
# ---------------------------------------------------------------------------
# A patch must contain the COMPLETE content of every file it touches. The
# default chat budget (~3000 tokens) truncates such JSON mid-string, the
# payload fails to parse and the whole patch is lost. The budget therefore
# scales with the number of files involved and is hard-capped to stay within
# what OpenRouter models accept.
BASE_PATCH_MAX_TOKENS = 6000       # covers a single focused file rewrite
PER_FILE_EXTRA_TOKENS = 2500       # extra head-room for each additional file
MAX_PATCH_MAX_TOKENS = 16000       # never exceed common model output limits


def dynamic_patch_budget(num_files: int) -> int:
    """Output-token budget for a patch involving ``num_files`` files."""
    files = max(1, num_files)
    return min(MAX_PATCH_MAX_TOKENS, BASE_PATCH_MAX_TOKENS + PER_FILE_EXTRA_TOKENS * (files - 1))


SYSTEM_PROMPT = (
    "You are a senior engineer. Write a focused, minimal patch for the task "
    "described in the diagnosis. For EACH file to create or modify, provide "
    "its path RELATIVE to the project root and its COMPLETE working content. "
    "Base every change on the ACTUAL current file contents provided in the "
    "context — never invent code you have not seen. Do not use placeholders, "
    "'...' elisions or partial diffs: the content will be written verbatim "
    "to disk. Respond with a single valid JSON object matching the schema."
)

CORRECTION_PROMPT = (
    "You are a senior engineer in a self-correction loop. The project tests "
    "FAILED after the previous patch was applied. Analyze the test output, "
    "identify the error and return a CORRECTED patch: provide the complete "
    "working content of every file to create or modify, based on the actual "
    "file contents in the context. Respond with a single valid JSON object "
    "matching the schema."
)


def _files_in_payload(payload: object, key: str) -> int:
    """Best-effort count of files mentioned in a diagnosis/patch payload."""
    text = str(payload)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return text.count(key) or 1
    if isinstance(data, dict):
        value = data.get("affected_files") if key == "affected_files" else data.get("files")
        if isinstance(value, list) and value:
            return len(value)
    return 1


def _append_context(user_prompt: str, repo_context: Optional[str]) -> str:
    """Append the repository context section to a coder prompt."""
    if not repo_context or not repo_context.strip():
        return user_prompt
    return (
        f"{user_prompt}\n\n"
        "# CURRENT FILE CONTENTS (ground truth — base your patch on these)\n"
        f"{repo_context}"
    )


def generate_patch(
    diagnosis: str,
    provider_model: str,
    json_schema: Optional[Type[BaseModel]] = None,
    *,
    repo_context: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Union[str, BaseModel]:
    """Generate a :class:`Patch` (complete file contents) from a diagnosis.

    Parameters
    ----------
    diagnosis : str
        Structured diagnosis produced by the diagnostic agent.
    provider_model : str
        Primary LLM model (litellm format).
    json_schema : Type[BaseModel] | None
        Override the :class:`Patch` response schema.
    repo_context : str | None
        Actual contents of the affected files, so the patch is grounded in
        the real code instead of being invented.
    max_tokens : int | None
        Completion budget; when ``None`` it is derived from the number of
        files mentioned in the diagnosis.
    """
    if max_tokens is None:
        max_tokens = dynamic_patch_budget(_files_in_payload(diagnosis, "affected_files"))
    return call_llm(
        provider_model=provider_model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=_append_context(str(diagnosis), repo_context),
        json_schema=json_schema or Patch,
        max_tokens=max_tokens,
    )


def correct_patch(
    previous_patch: Union[Patch, str],
    test_feedback: str,
    provider_model: str,
    json_schema: Optional[Type[BaseModel]] = None,
    *,
    repo_context: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Union[str, BaseModel]:
    """Self-correction: fix the patch using the failed test output.

    Parameters
    ----------
    previous_patch : Patch | str
        The previously applied patch (files + contents).
    test_feedback : str
        stdout/stderr output of the failed test suite.
    provider_model : str
        Primary LLM model (litellm format).
    json_schema : Type[BaseModel] | None
        Override the :class:`Patch` response schema.
    repo_context : str | None
        Actual contents of the affected files (ground truth for the fix).
    max_tokens : int | None
        Completion budget; when ``None`` it is derived from the size of the
        previous patch.
    """
    if isinstance(previous_patch, Patch):
        previous_patch = previous_patch.model_dump_json(indent=2)
    if max_tokens is None:
        max_tokens = dynamic_patch_budget(_files_in_payload(previous_patch, "files"))
    user_prompt = (
        f"HERE IS THE PREVIOUS PATCH (already applied to files):\n{previous_patch}\n\n"
        f"Here is the TEST FAILURE output:\n{test_feedback}\n\n"
        "Return the corrected patch (COMPLETE file contents)."
    )
    return call_llm(
        provider_model=provider_model,
        system_prompt=CORRECTION_PROMPT,
        user_prompt=_append_context(user_prompt, repo_context),
        json_schema=json_schema or Patch,
        max_tokens=max_tokens,
    )