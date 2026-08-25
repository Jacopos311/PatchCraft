"""Tests for human-grade PR writing (Roadmap Step 4.3)."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from src.agents import reporter
from src.agents.reporter import (
    PatchReport,
    PRReview,
    build_diff_stat,
    generate_report,
    sanitize_honesty,
)
from src.core.repo_profile import RepoVoice, build_repo_voice


def _write(root: Path, rel: str, content: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


GOLDEN_DIFF = (
    "diff --git a/src/app.py b/src/app.py\n"
    "--- a/src/app.py\n"
    "+++ b/src/app.py\n"
    "@@ -1,3 +1,3 @@\n"
    "-def add(a, b):\n"
    "-    return a - b\n"
    "+def add(a, b):\n"
    "+    return a + b\n"
)


class TestDiffStat:
    def test_golden_counts(self) -> None:
        assert build_diff_stat(GOLDEN_DIFF) == "1 file changed, +2 insertions, -2 deletions"

    def test_multiple_files(self) -> None:
        diff = (
            "diff --git a/a.py b/a.py\n+++ b/a.py\n+one\n"
            "diff --git a/b.py b/b.py\n--- a/b.py\n-gone\n"
        )
        stat = build_diff_stat(diff)
        assert stat.startswith("2 files changed")
        assert "+1 insertions" in stat and "-1 deletions" in stat

    def test_empty_diff(self) -> None:
        assert build_diff_stat("") == "0 files changed, +0 insertions, -0 deletions"


class TestRepoVoice:
    def test_local_build_with_templates(self, tmp_path: Path) -> None:
        _write(tmp_path, ".github/PULL_REQUEST_TEMPLATE.md",
               "## What\n## Why\n## Testing\n- [x] tests pass\n")
        _write(tmp_path, "CONTRIBUTING.md", "Always reference the issue number.")
        voice = build_repo_voice(tmp_path)
        assert "## What" in voice.pull_request_template
        assert "reference the issue" in voice.contribution_guidelines

    def test_generic_fallback_without_templates(self, tmp_path: Path) -> None:
        voice = build_repo_voice(tmp_path)
        assert voice.pull_request_template == ""
        assert voice.contribution_guidelines == ""

    def test_cache_roundtrip_and_invalidation(self, tmp_path: Path) -> None:
        _write(tmp_path, ".github/PULL_REQUEST_TEMPLATE.md", "## What\n")
        first = build_repo_voice(tmp_path)
        assert first.pull_request_template == "## What\n"
        # Cached copy is reused without changes.
        again = build_repo_voice(tmp_path)
        assert again == first
        # Changing a source file invalidates the cache.
        _write(tmp_path, ".github/PULL_REQUEST_TEMPLATE.md", "## What\n## Why\n")
        rebuilt = build_repo_voice(tmp_path)
        assert "## Why" in rebuilt.pull_request_template


class TestWriterPromptVerifiedInputs:
    def test_prompt_contains_only_verified_sections(self, tmp_path: Path) -> None:
        _write(tmp_path, ".github/PULL_REQUEST_TEMPLATE.md",
               "## What\n## Why\n## How Tested\n")
        voice = build_repo_voice(tmp_path)
        captured: list[dict] = []

        def fake_call_llm(**kwargs):
            captured.append(kwargs)
            return PatchReport(title="t", summary="s", diff="d", pr_markdown="body")

        with mock.patch.object(reporter, "call_llm", side_effect=fake_call_llm):
            generate_report(
                GOLDEN_DIFF, "mock",
                repo_voice=voice,
                issue_text="Title: sign bug\nBody: returns wrong value",
                diff_stat=build_diff_stat(GOLDEN_DIFF),
                test_evidence="pytest (targeted): tests/test_app.py; exit 0",
            )

        prompt = captured[0]["user_prompt"]  # FIRST call = writer draft
        assert "VERIFIED DIFF" in prompt
        assert "+    return a + b" in prompt
        assert "DIFF STAT" in prompt
        assert "1 file changed" in prompt
        assert "TEST EVIDENCE" in prompt
        assert "PULL REQUEST TEMPLATE" in prompt
        assert "HARD RULES" in prompt
        # The honesty rule forbids fabricated facts.
        assert "Never invent" in prompt
        assert captured[0]["json_schema"] is reporter.PatchReport

    def test_self_review_runs_when_template_present(self, tmp_path: Path) -> None:
        _write(tmp_path, ".github/PULL_REQUEST_TEMPLATE.md", "## What\n## Why\n")
        voice = build_repo_voice(tmp_path)
        calls: list[dict] = []

        def fake_call_llm(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return PatchReport(title="bad title", summary="s", diff="d",
                                   pr_markdown="draft body")
            return PRReview(
                conformance_notes="title style off; body fine",
                revise_pr_markdown=True,
                pr_markdown="revised body",
                title="fix(app): correct the sign handling",
            )

        with mock.patch.object(reporter, "call_llm", side_effect=fake_call_llm):
            report = generate_report(GOLDEN_DIFF, "mock", repo_voice=voice)

        assert isinstance(report, PatchReport)
        assert report.pr_markdown.rstrip("\n") == "revised body"
        assert report.title == "fix(app): correct the sign handling"
        assert len(calls) == 2  # draft + one review pass

    def test_no_self_review_without_voice(self) -> None:
        calls: list[dict] = []

        def fake_call_llm(**kwargs):
            calls.append(kwargs)
            return PatchReport(title="t", summary="s", diff="d", pr_markdown="body")

        with mock.patch.object(reporter, "call_llm", side_effect=fake_call_llm):
            generate_report(GOLDEN_DIFF, "mock")

        assert len(calls) == 1  # no review pass without a voice profile

    def test_review_failure_never_breaks_the_report(self, tmp_path: Path) -> None:
        _write(tmp_path, ".github/PULL_REQUEST_TEMPLATE.md", "## What\n")
        voice = build_repo_voice(tmp_path)
        state = {"n": 0}

        def fake_call_llm(**kwargs):
            state["n"] += 1
            if state["n"] == 1:
                return PatchReport(title="t", summary="s", diff="d",
                                   pr_markdown="body")
            raise RuntimeError("LLM exploded during review")

        with mock.patch.object(reporter, "call_llm", side_effect=fake_call_llm):
            report = generate_report(GOLDEN_DIFF, "mock", repo_voice=voice)

        assert isinstance(report, PatchReport)
        assert report.pr_markdown.rstrip("\n") == "body"


class TestHonestySanitizer:
    def test_strips_fabricated_social_facts(self) -> None:
        body = (
            "## What\nFixed the sign.\n"
            "Approved by @maintainer\n"
            "reviewed by two people\n"
            "LGTM :tada:\n"
            "## Why\nIssue said so.\n"
        )
        clean = sanitize_honesty(body)
        assert "Approved by" not in clean
        assert "reviewed by" not in clean
        assert "LGTM" not in clean
        assert "Fixed the sign." in clean
        assert "Issue said so." in clean

    def test_keeps_legit_content(self) -> None:
        body = "## Testing\n- [x] pytest passed locally\n"
        assert sanitize_honesty(body) == body.strip() + "\n"