"""Coder agent: generates and corrects patches for a diagnosed bug.

Produced patches carry the **complete content** of the files to create or
modify (:class:`Patch` model), so the orchestrator can write them to the real
files of the target project.
"""
from __future__ import annotations

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


SYSTEM_PROMPT = (
    "You are a senior engineer. Write a focused, minimal patch for the bug "
    "described in the diagnosis. For EACH file to modify, provide its relative "
    "path and the COMPLETE working content of the file. Do not use placeholders "
    "or partial diffs: the content will be written verbatim to disk."
)

CORRECTION_PROMPT = (
    "You are a senior engineer in a self-correction loop. The project tests "
    "FAILED after the previous patch was applied. Analyze the stderr, identify "
    "the error and return a CORRECTED patch: provide the complete working "
    "content of every file to create or modify."
)


def generate_patch(
    diagnosis: str,
    provider_model: str,
    json_schema: Optional[Type[BaseModel]] = None,
) -> Union[str, BaseModel]:
    """Generate a :class:`Patch` (complete file contents) from a diagnosis."""
    return call_llm(
        provider_model=provider_model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=diagnosis,
        json_schema=json_schema or Patch,
    )


def correct_patch(
    previous_patch: Union[Patch, str],
    test_feedback: str,
    provider_model: str,
    json_schema: Optional[Type[BaseModel]] = None,
) -> Union[str, BaseModel]:
    """Self-correction: fix the patch using the stderr of the failed tests.

    Parameters
    ----------
    previous_patch : Patch | str
        The previously applied patch (files + contents).
    test_feedback : str
        stderr output of the failed test suite.
    """
    if isinstance(previous_patch, Patch):
        previous_patch = previous_patch.model_dump_json(indent=2)
    user_prompt = (
        f"HERE IS THE PREVIOUS PATCH (already applied to files):\n{previous_patch}\n\n"
        f"Here is the TEST FAILURE (stderr):\n{test_feedback}\n\n"
        "Return the corrected patch (COMPLETE file contents)."
    )
    return call_llm(
        provider_model=provider_model,
        system_prompt=CORRECTION_PROMPT,
        user_prompt=user_prompt,
        json_schema=json_schema or Patch,
    )