"""Test del coder agent (budget dinamico e contesto repository)."""
from __future__ import annotations

import json
from unittest import mock

import litellm

from src.agents.coder import (
    MAX_PATCH_MAX_TOKENS,
    Patch,
    correct_patch,
    dynamic_patch_budget,
    generate_patch,
)


def _completion_response(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}], "model": "mocked"}


_VALID_PATCH_JSON = json.dumps({
    "files": [{"file_path": "src/app.py", "new_content": "x = 1\n"}],
    "notes": "",
})


def test_dynamic_patch_budget_scales_and_is_capped() -> None:
    """Il budget cresce col numero di file ma non supera il cap assoluto."""
    one = dynamic_patch_budget(1)
    three = dynamic_patch_budget(3)
    huge = dynamic_patch_budget(100)
    assert one == 4000
    assert three > one
    assert huge == MAX_PATCH_MAX_TOKENS
    # valori degenerei
    assert dynamic_patch_budget(0) == one
    assert dynamic_patch_budget(-5) == one


def test_generate_patch_passes_repo_context_and_dynamic_budget() -> None:
    """repo_context finisce nel prompt; il budget è quello dinamico."""
    diagnosis = json.dumps({"summary": "s", "affected_files": ["a.py", "b.py"]})
    with mock.patch.object(litellm, "completion", return_value=_completion_response(_VALID_PATCH_JSON)) as mocked:
        generate_patch(
            diagnosis,
            "openrouter/deepseek/deepseek-chat",
            repo_context="# FILE: src/app.py\n```\nprint('hi')\n```",
            max_tokens=9000,
        )
    kwargs = mocked.call_args.kwargs
    assert kwargs["max_tokens"] == 9000
    assert "CURRENT FILE CONTENTS" in kwargs["messages"][1]["content"]
    assert "print('hi')" in kwargs["messages"][1]["content"]


def test_generate_patch_default_budget_from_diagnosis_files() -> None:
    """Senza max_tokens esplicito il budget deriva dai file nel diagnosis."""
    diagnosis = json.dumps({
        "summary": "s",
        "root_cause": "r",
        "affected_files": ["src/a.py", "src/b.py", "src/c.py"],
        "confidence": 0.9,
    })
    with mock.patch.object(litellm, "completion", return_value=_completion_response(_VALID_PATCH_JSON)) as mocked:
        generate_patch(diagnosis, "openrouter/deepseek/deepseek-chat")
    assert mocked.call_args.kwargs["max_tokens"] == dynamic_patch_budget(3)


def test_correct_patch_includes_feedback_and_context() -> None:
    """Il correttore riceve patch precedente, stderr e contesto reale."""
    previous = Patch(files=[{"file_path": "src/app.py", "new_content": "x = 1\n"}])
    with mock.patch.object(litellm, "completion", return_value=_completion_response(_VALID_PATCH_JSON)) as mocked:
        correct_patch(
            previous,
            "FAILED test_add",
            "openrouter/deepseek/deepseek-chat",
            repo_context="# FILE: src/app.py\n```\nx = 1\n```",
        )
    user_msg = mocked.call_args.kwargs["messages"][1]["content"]
    assert "x = 1" in user_msg          # patch precedente
    assert "FAILED test_add" in user_msg  # feedback test
    assert "CURRENT FILE CONTENTS" in user_msg  # contesto reale
    assert mocked.call_args.kwargs["max_tokens"] == dynamic_patch_budget(1)



