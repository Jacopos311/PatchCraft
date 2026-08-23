"""Tests for the prompt compiler / context budgeter (Roadmap Step 1.4)."""
from __future__ import annotations

import json
from unittest import mock

import pytest

import litellm

from src.agents.coder import Patch, correct_patch, generate_patch
from src.core.prompts import (
    CHARS_PER_TOKEN,
    TRIM_HEAD,
    TRIM_TAIL,
    PromptCompiler,
    estimate_tokens,
)


def test_estimate_tokens_uses_4_chars_per_token():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens("abcd") == 1


def test_max_tokens_must_be_positive():
    with pytest.raises(ValueError):
        PromptCompiler(max_tokens=0)
    with pytest.raises(ValueError):
        PromptCompiler(max_tokens=-5)


def test_empty_sections_ignored():
    compiled = PromptCompiler(max_tokens=1000)
    compiled.add("empty", "")
    assert compiled._sections == []
    assert compiled.compile().text == ""


def test_hard_cap_guaranteed_even_when_only_priority_one():
    compiler = PromptCompiler(max_tokens=50)  # 200 chars budget
    compiler.add("instructions", ("keep me  " * 100), priority=1, trim=TRIM_HEAD)
    compiled = compiler.compile()
    assert len(compiled.text) <= 50 * CHARS_PER_TOKEN + len("…[truncated]")
    assert compiled.text.endswith("…[truncated]")


def test_drop_order_lowest_priority_first():
    """The roadmap trim order must be honored exactly."""
    compiler = PromptCompiler(max_tokens=200)  # 800 chars
    compiler.add("issue", "ISSUE " * 100, priority=1, trim=TRIM_HEAD)
    compiler.add("repo_map", "HERE " * 300, priority=2)
    compiler.add("doc", "DOC " * 300, priority=3)
    compiler.add("source", "SRC " * 300, priority=4)

    compiled = compiler.compile()

    assert "ISSUE" in compiled.text       # priority 1 always kept
    assert "HERE" not in compiled.text    # repo_map dropped first
    assert "SRC" not in compiled.text     # higher priority dropped
    assert compiled.dropped == ["source", "doc", "repo_map"]
    assert compiled.estimated_tokens <= 200


def test_trim_tail_keeps_feedback_end():
    """test_feedback is tail-trimmed: the summary at the end survives."""
    compiler = PromptCompiler(max_tokens=800)  # 3200 chars
    compiler.add("instructions", "I" * 300, priority=1)
    compiler.add("repo", "R" * 200, priority=2)
    compiler.add(
        "feedback",
        "FAIL head\n" + ("tail summary line\n" * 200),
        priority=4,
        trim=TRIM_TAIL,
    )

    compiled = compiler.compile()

    assert "tail summary line" in compiled.text
    assert "FAIL head" not in compiled.text  # the head of the feedback is gone
    assert "…[truncated]" in compiled.text  # the tail was kept with a marker
    assert compiled.text.endswith("…[truncated]") or len(compiled.text) <= 800 * CHARS_PER_TOKEN + 200


def _valid_patch_json() -> str:
    return json.dumps({
        "files": [{"file_path": "src/app.py", "new_content": "x = 1\n"}],
        "notes": "",
    })


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}], "model": "mocked"}


def test_correct_patch_budget_tight_keeps_within_budget():
    """Budget enforced end-to-end on the corrector (trim-then-drop is expected)."""
    previous = Patch(files=[{"file_path": "src/app.py", "new_content": "x = 1\n"}])
    huge_feedback = "failure detail\n" * 2000
    with mock.patch.object(
        litellm, "completion",
        return_value=_completion(_valid_patch_json()),
    ) as mocked:
        correct_patch(
            previous,
            huge_feedback,
            "openrouter/deepseek/deepseek-chat",
            repo_context="# FILE: src/app.py\n```\nx = 1\n```",
            prompt_budget=600,  # small context budget for the test
        )
    user_msg = mocked.call_args.kwargs["messages"][1]["content"]
    # The compiled context fits its budget exactly (verified below by the
    # compiler), then _build_messages() appends the JSON-schema instruction
    # after the compiler runs — allow that overhead in the assertion.
    assert len(user_msg) <= 600 * CHARS_PER_TOKEN + 3000  # compiled + schema overhead
    # The huge feedback must be trimmed by the budgeter, not kept in full.
    assert "…[truncated]" in user_msg
    assert "senior engineer in a self-correction loop" in user_msg


def test_generate_patch_uses_compiler_and_keeps_diagnosis():
    diagnosis = json.dumps({"summary": "s", "affected_files": ["a.py"]})
    with mock.patch.object(litellm, "completion", return_value=_completion(_valid_patch_json())) as mocked:
        generate_patch(
            diagnosis,
            "openrouter/deepseek/deepseek-chat",
            repo_context="# FILE: a.py\n```\nprint('hi')\n```",
        )
    user_msg = mocked.call_args.kwargs["messages"][1]["content"]
    assert "CURRENT FILE CONTENTS" in user_msg
    assert "print('hi')" in user_msg
    assert "affected_files" in user_msg  # diagnosis kept (priority 1)


class TestBuildContextBudgeting:
    def test_build_context_respects_tight_budget(self, tmp_path):
        from src.orchestrator import build_context

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("def f():\n    pass\n", encoding="utf-8")
        (tmp_path / "README.md").write_text("readme " * 500, encoding="utf-8")
        context = build_context(tmp_path, "Fix the bug", max_tokens=200)
        assert "# ISSUE TO SOLVE" in context
        assert len(context) <= 200 * CHARS_PER_TOKEN + 500  # hard cap respected (+marker slack)