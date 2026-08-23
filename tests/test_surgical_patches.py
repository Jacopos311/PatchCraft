"""Tests for surgical search/replace patches (Roadmap Step 1.3)."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from src.agents.coder import EditHunk, Patch
from src.orchestrator import apply_edits_to_text, apply_patch_detailed


ORIGINAL = (
    "def add(a, b):\n"
    "    return a + b\n"
    "\n"
    "def multiply(a, b):\n"
    "    return a * b\n"
)


class TestApplyEditsToText:
    def test_single_exact_hunk(self):
        new, problems = apply_edits_to_text(
            ORIGINAL,
            [EditHunk(
                find="def multiply(a, b):\n    return a * b\n",
                replace="def multiply(a, b):\n    return a * b * 2\n",
            )],
        )
        assert problems == []
        assert "return a * b * 2" in new
        assert "return a + b" in new  # untouched part preserved

    def test_multi_hunk_applied_top_to_bottom(self):
        new, problems = apply_edits_to_text(
            ORIGINAL,
            [
                EditHunk(
                    find="def add(a, b):\n    return a + b\n",
                    replace="def add(a, b, c=0):\n    return a + b + c\n",
                ),
                EditHunk(find="def multiply(a, b):", replace="def multiply(a, b, *, scale=1):"),
            ],
        )
        assert problems == []
        assert "def add(a, b, c=0):" in new
        assert "def multiply(a, b, *, scale=1):" in new

    def test_ambiguous_find_is_rejected_atomically(self):
        ambiguous = ORIGINAL + ORIGINAL
        new, problems = apply_edits_to_text(
            ambiguous,
            [EditHunk(find="def add(a, b):\n    return a + b\n", replace="")],
        )
        assert new == ambiguous, "atomic: failed hunk must not modify the content"
        assert len(problems) == 1
        assert "ambiguous" in problems[0]
        assert "2 locations" in problems[0]

    def test_missing_find_reports_and_keeps_original(self):
        new, problems = apply_edits_to_text(
            ORIGINAL,
            [EditHunk(find="def subtract(a, b):\n    return a - b\n", replace="")],
        )
        assert new == ORIGINAL
        assert len(problems) == 1
        assert "was not found" in problems[0]

    def test_whitespace_tolerant_fallback_matches_once(self):
        drifted = (
            "def add(a, b):\n"
            "\treturn a + b   \n"
            "\n"
            "def multiply(a, b):\n"
            "    return a * b\n"
        )
        new, problems = apply_edits_to_text(
            drifted,
            [EditHunk(
                find="def add(a, b):\n    return a + b\n",
                replace="def add(a, b):\n    return a + b + 0\n",
            )],
        )
        assert problems == []
        assert "return a + b + 0" in new

    def test_deletion_hunk_allowed(self):
        new, problems = apply_edits_to_text(
            ORIGINAL,
            [EditHunk(find="\ndef multiply(a, b):\n    return a * b\n", replace="")],
        )
        assert problems == []
        assert "multiply" not in new

    def test_crlf_files_preserve_line_endings(self):
        crlf_original = ORIGINAL.replace("\n", "\r\n")
        new, problems = apply_edits_to_text(
            crlf_original,
            [EditHunk(find="def add(a, b):\n    return a + b\n", replace="def add(a, b):\n    return a + b\n")],
        )
        assert problems == []
        assert "\r\ndef multiply" in new or new.endswith("\r\n")

    def test_empty_find_rejected(self):
        new, problems = apply_edits_to_text(ORIGINAL, [EditHunk(find="", replace="x")])
        assert new == ORIGINAL
        assert "'find' snippet is empty" in problems[0]


class TestApplyPatchDetailed:
    def _repo_with_file(self, tmp_path: Path) -> Path:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text(ORIGINAL, encoding="utf-8")
        return tmp_path

    def test_surgical_edit_modifies_file_and_snapshots_original(self, tmp_path: Path):
        self._repo_with_file(tmp_path)
        patch = Patch(files=[{
            "file_path": "src/app.py",
            "edits": [{"find": "    return a * b\n", "replace": "    return a * b * 3\n"}],
        }])
        result = apply_patch_detailed(patch, tmp_path)

        assert result.problems == []
        assert (
            (tmp_path / "src" / "app.py").read_text(encoding="utf-8")
            == ORIGINAL.replace("return a * b", "return a * b * 3")
        )
        assert list(result.snapshots.values()) == [ORIGINAL]

    def test_edits_on_missing_file_become_problems(self, tmp_path: Path):
        tmp_path.mkdir(exist_ok=True)
        patch = Patch(files=[{
            "file_path": "src/ghost.py",
            "edits": [{"find": "x", "replace": "y"}],
        }])
        result = apply_patch_detailed(patch, tmp_path)
        assert result.snapshots == {}
        assert any("does not exist" in p for p in result.problems)

    def test_new_content_mode_still_works_for_new_files(self, tmp_path: Path):
        patch = Patch(files=[{"file_path": "src/new.py", "new_content": "print('hi')\n"}])
        result = apply_patch_detailed(patch, tmp_path)
        assert result.problems == []
        assert (tmp_path / "src" / "new.py").exists()
        assert list(result.snapshots.values()) == [None]  # created from nothing

    def test_both_modes_on_same_file_is_invalid(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Patch(files=[{
                "file_path": "src/app.py",
                "new_content": "x = 1\n",
                "edits": [{"find": "a", "replace": "b"}],
            }])


class TestLoopIntegration:
    def _diagnosis(self):
        from src.agents.diagnostic import Diagnosis

        return Diagnosis(
            summary="sign bug",
            root_cause="subtraction instead of addition",
            affected_files=["src/app.py"],
            confidence=0.9,
        )

    def test_loop_applies_surgical_patch_end_to_end(self, tmp_path: Path):
        from src.agents.reporter import PatchReport
        from src.orchestrator import run_patchcraft_loop
        from src.sandbox.runner import TestResult

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text(
            "def add(a, b):\n    return a - b\n", encoding="utf-8"
        )
        surgical_patch = Patch(files=[{
            "file_path": "src/app.py",
            "edits": [{"find": "    return a - b\n", "replace": "    return a + b\n"}],
        }])

        with (
            mock.patch("src.orchestrator.diagnose", return_value=self._diagnosis()),
            mock.patch("src.orchestrator.generate_patch", return_value=surgical_patch),
            mock.patch(
                "src.orchestrator.SandboxRunner.run_tests",
                return_value=TestResult(success=True, exit_code=0),
            ),
            mock.patch(
                "src.orchestrator.generate_report",
                return_value=PatchReport(title="t", summary="s", diff="d", pr_markdown="m"),
            ),
        ):
            result = run_patchcraft_loop(str(tmp_path), "Fix sign", model="mock")

        assert result.success is True
        assert (
            (tmp_path / "src" / "app.py").read_text(encoding="utf-8")
            == "def add(a, b):\n    return a + b\n"
        )

    def test_failed_edit_feedback_reaches_correction_loop(self, tmp_path: Path):
        from src.orchestrator import run_patchcraft_loop
        from src.sandbox.runner import TestResult

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
        feedback_seen: list[str] = []
        counter = {"run": 0}

        bad_patch = Patch(files=[{
            "file_path": "src/app.py",
            "edits": [{"find": "THIS LINE DOES NOT EXIST", "replace": "y = 2\n"}],
        }])

        def _correct(**kwargs) -> Patch:
            feedback_seen.append(kwargs["test_feedback"])
            return Patch(files=[{"file_path": "src/app.py", "new_content": f"x = {len(feedback_seen)}\n"}])

        def _always_fail() -> TestResult:
            counter["run"] += 1
            return TestResult(success=False, stdout=f"fail-{counter['run']}", stderr="", exit_code=1)

        with (
            mock.patch("src.orchestrator.diagnose", return_value=self._diagnosis()),
            mock.patch("src.orchestrator.generate_patch", return_value=bad_patch),
            mock.patch("src.orchestrator.correct_patch", side_effect=_correct),
            mock.patch("src.orchestrator.SandboxRunner.run_tests", side_effect=_always_fail),
        ):
            run_patchcraft_loop(str(tmp_path), "Fix", model="mock", max_retries=2)

        assert feedback_seen, "the corrector must be invoked after the failed edit"
        assert "Surgical patch problem" in feedback_seen[0]
        assert "THIS LINE DOES NOT EXIST" in feedback_seen[0]