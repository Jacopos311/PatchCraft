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
import os
from typing import Callable, Optional, Type, Union

from pydantic import BaseModel, Field, model_validator

from src.core.llm import call_llm


class FilePatch(BaseModel):
    """A single file to create (whole content) or modify (surgical edits)."""

    file_path: str = Field(
        description="Path of the file relative to the project root (e.g. src/calc.py)."
    )
    new_content: str = Field(
        default="",
        description=(
            "COMPLETE final content of the file. Use ONLY to create brand-new "
            "files — for existing files prefer surgical 'edits'."
        ),
    )
    edits: list["EditHunk"] = Field(
        default_factory=list,
        description=(
            "Search/replace hunks for an EXISTING file, applied top-to-bottom "
            "on its current content."
        ),
    )

    @model_validator(mode="after")
    def _check_payload(self) -> "FilePatch":
        has_content = bool(self.new_content and self.new_content.strip())
        has_edits = bool(self.edits)
        if has_content and has_edits:
            raise ValueError(
                "use either 'new_content' (to create a new file) or 'edits' "
                "(to modify an existing file), not both"
            )
        if not has_content and not has_edits:
            raise ValueError("provide 'new_content' or a non-empty 'edits' list")
        return self


class EditHunk(BaseModel):
    """One surgical search/replace hunk within an existing file."""

    find: str = Field(
        description=(
            "EXACT snippet copied from the current file content, including "
            "2-4 surrounding lines so the anchor is unique."
        )
    )
    replace: str = Field(description="The replacement snippet (may be empty to delete).")


class Patch(BaseModel):
    """A complete patch: set of files to create or modify."""

    files: list[FilePatch] = Field(description="List of files to create or modify.")
    notes: str = Field(
        default="", description="Short notes about the changes (for human review)."
    )


Patch.model_rebuild()


# ---------------------------------------------------------------------------
# Dynamic completion budget ("dynamic completion" guardrail)
# ---------------------------------------------------------------------------
# Surgical search/replace patches are SMALL: a few anchored hunks instead of
# whole-file rewrites, so the baseline budget is far lower than in the
# whole-file era. Whole content is still emitted for brand-new files, which
# the per-file scaling covers; the hard cap keeps calls within what
# OpenRouter models accept.
BASE_PATCH_MAX_TOKENS = 4000       # covers several surgical hunks or a small new file
PER_FILE_EXTRA_TOKENS = 2000       # extra head-room for each additional file
MIN_FILE_EXTRA_BUDGET = 2000       # minimum extra budget per correction iteration
MAX_PATCH_MAX_TOKENS = 16000       # never exceed common model output limits
MIN_FILE_EXTRA_BUDGET = 2000       # minimum extra budget per correction iteration


def dynamic_patch_budget(num_files: int, correction_iteration: int = 0) -> int:
    """Output-token budget for a patch involving ``num_files`` files."""
    files = max(1, num_files)
    base = BASE_PATCH_MAX_TOKENS + (MIN_FILE_EXTRA_BUDGET * correction_iteration)
    return min(MAX_PATCH_MAX_TOKENS, base + PER_FILE_EXTRA_TOKENS * (files - 1))


SYSTEM_PROMPT = (
    "You are a senior engineer. Write a focused, minimal patch for the task "
    "described in the diagnosis. For each EXISTING file, return surgical "
    "search/replace edits: 'find' must be an EXACT snippet copied from the "
    "current file content including 2-4 surrounding lines so the anchor is "
    "unique, and 'replace' holds the corrected version of that snippet. Use "
    "'new_content' ONLY to create brand-new files. Base every change on the "
    "ACTUAL current file contents provided in the context — never invent code "
    "you have not seen. No placeholders or elisions: hunks are applied "
    "verbatim, top-to-bottom. Respond with a single valid JSON object "
    "matching the schema."
)

CORRECTION_PROMPT = (
    "You are a senior engineer in a self-correction loop. The project tests "
    "FAILED after the previous patch was applied. Analyze the test output, "
    "identify the error and return a CORRECTED patch as surgical "
    "search/replace edits based on the actual file contents in the context "
    "('new_content' only for brand-new files). If previous edit failures were "
    "reported, fix the 'find' snippets to match the real content exactly. "
    "Respond with a single valid JSON object matching the schema."
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


def _prompt_cache_budget() -> int:
    return int(os.getenv("PATCHCRAFT_PROMPT_BUDGET", "16000"))


def generate_patch(
    diagnosis: str,
    provider_model: str,
    json_schema: Optional[Type[BaseModel]] = None,
    *,
    repo_context: Optional[str] = None,
    max_tokens: Optional[int] = None,
    usage_sink: Optional[Callable[[int, int], None]] = None,
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
    usage_sink : Callable[[int, int], None] | None
        Optional ``(prompt_tokens, completion_tokens)`` callback for
        per-task token accounting.
    """
    if max_tokens is None:
        max_tokens = dynamic_patch_budget(_files_in_payload(diagnosis, "affected_files"))
    from src.core.prompts import TRIM_HEAD, PromptCompiler

    compiler = PromptCompiler(max_tokens=_prompt_cache_budget())
    compiler.add("instructions", SYSTEM_PROMPT, priority=1)
    compiler.add("repo_context", _format_repo_context(repo_context), priority=2)
    compiler.add("diagnosis", str(diagnosis), priority=1, trim=TRIM_HEAD)

    return call_llm(
        provider_model=provider_model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=compiler.compile().text,
        json_schema=json_schema or Patch,
        max_tokens=max_tokens,
        usage_sink=usage_sink,
    )


def correct_patch(
    previous_patch: Union[Patch, str],
    test_feedback: str,
    provider_model: str,
    json_schema: Optional[Type[BaseModel]] = None,
    *,
    repo_context: Optional[str] = None,
    max_tokens: Optional[int] = None,
    iteration: int = 0,
    prompt_budget: Optional[int] = None,
    usage_sink: Optional[Callable[[int, int], None]] = None,
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
    iteration : int
        Current correction iteration count (used to increase budget).
    prompt_budget : int | None
        Prompt (context) token budget for the corrector.
    usage_sink : Callable[[int, int], None] | None
        Optional ``(prompt_tokens, completion_tokens)`` callback for
        per-task token accounting.
    """
    if isinstance(previous_patch, Patch):
        previous_patch = previous_patch.model_dump_json(indent=2)
    if max_tokens is None:
        max_tokens = dynamic_patch_budget(_files_in_payload(previous_patch, "files"), iteration)

    from src.core.prompts import TRIM_HEAD, TRIM_TAIL, PromptCompiler

    compiler = PromptCompiler(max_tokens=prompt_budget or _prompt_cache_budget())
    compiler.add("instructions", CORRECTION_PROMPT, priority=1)
    compiler.add("repo_context", _format_repo_context(repo_context), priority=2)
    compiler.add("previous_patch", f"PREVIOUS PATCH (already applied):\n{previous_patch}", priority=3)
    compiler.add(
        "test_feedback",
        f"TEST FAILURE OUTPUT:\n{test_feedback}",
        priority=4,
        trim=TRIM_TAIL,  # summaries live at the end of the feedback
    )

    return call_llm(
        provider_model=provider_model,
        system_prompt=CORRECTION_PROMPT,
        user_prompt=compiler.compile().text,
        json_schema=json_schema or Patch,
        max_tokens=max_tokens,
        usage_sink=usage_sink,
    )


def _format_repo_context(repo_context: Optional[str]) -> str:
    """Wrap raw repo context with its ground-truth header (if any content)."""
    if not repo_context or not repo_context.strip():
        return ""
    return (
        "# CURRENT FILE CONTENTS (ground truth — base your patch on these)\n"
        f"{repo_context}"
    )