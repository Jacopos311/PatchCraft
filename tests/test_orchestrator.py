"""Test dell'orchestratore PatchCraft (agent LLM e sandbox mockati)."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

from src.agents.coder import Patch
from src.agents.diagnostic import Diagnosis
from src.agents.reporter import PatchReport
from src.orchestrator import (
    apply_patch,
    build_context,
    compute_diff,
    rollback,
    run_patchcraft_loop,
)
from src.sandbox.runner import TestResult


def _write(root: Path, relpath: str, content: str) -> Path:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _patch_for(path: str, content: str) -> Patch:
    return Patch(files=[{"file_path": path, "new_content": content}])


def _diagnosis() -> Diagnosis:
    return Diagnosis(
        summary="offset bug",
        root_cause="off-by-one in index",
        affected_files=["src/app.py"],
        confidence=0.9,
    )


def _report() -> PatchReport:
    return PatchReport(
        title="Fix offset",
        summary="Fixed.",
        diff="@@ -1,1 +1,1 @@",
        pr_markdown="## Summary\nFixed.",
    )


class TestBuildContext:
    def test_includes_docs_and_source(self, tmp_path) -> None:
        _write(tmp_path, "architecture.md", "# Arch")
        _write(tmp_path, "README.md", "# Readme")
        _write(tmp_path, "src/app.py", "print('hi')")
        _write(tmp_path, "node_modules/ignored.py", "print('nope')")

        context = build_context(tmp_path, "Fix the bug")
        assert "Fix the bug" in context
        assert "# Arch" in context
        assert "# Readme" in context
        assert "'hi'" in context
        assert "node_modules" not in context

    def test_includes_web_extensions(self, tmp_path) -> None:
        """Contesto diagnostico: .ts/.js/.json/.yaml/.yml vengono inclusi."""
        _write(tmp_path, "src/pricing.ts", "export const price = 9.99;")
        _write(tmp_path, "src/bundle.js", "module.exports = { x: 1 }")
        _write(tmp_path, "pricing.json", "{\"tier\": \"pro\", \"rate\": 0.05}")
        _write(tmp_path, "config.yaml", "key: value")
        _write(tmp_path, "deploy.yml", "stages: [ci, cd]")

        context = build_context(tmp_path, "Fix pricing bug")
        assert "price" in context
        assert "bundle" in context
        assert "pro" in context
        assert "key: value" in context
        assert "stages" in context


class TestApplyAndDiff:
    def test_apply_patch_creates_files(self, tmp_path) -> None:
        patch = _patch_for("src/hello.py", "print('hello')\n")
        snapshots = apply_patch(patch, tmp_path)
        assert (tmp_path / "src/hello.py").exists()
        assert None in snapshots.values()  # file creato (nessun originale)

    def test_apply_patch_rejects_escape(self, tmp_path) -> None:
        patch = _patch_for("../../evil.py", "pwned\n")
        snapshots = apply_patch(patch, tmp_path)
        assert snapshots == {}
        assert not (tmp_path.parent.parent / "evil.py").exists()

    def test_compute_diff_restore(self, tmp_path) -> None:
        path = _write(tmp_path, "a.txt", "uno\n")
        snapshots = {path: "uno\n"}
        path.write_text("uno\ndue\n", encoding="utf-8")
        diff = compute_diff(tmp_path, snapshots)
        assert "+due" in diff

    def test_rollback_restores_and_removes(self, tmp_path) -> None:
        existing = _write(tmp_path, "a.txt", "originale\n")
        new_file = tmp_path / "b.txt"
        snapshots = {existing: "originale\n", new_file: None}
        existing.write_text("cambiato\n", encoding="utf-8")
        new_file.write_text("nuovo\n", encoding="utf-8")

        rollback(tmp_path, snapshots)
        assert existing.read_text(encoding="utf-8") == "originale\n"
        assert not new_file.exists()


class TestRunPatchcraftLoop:
    def test_success_first_attempt(self, tmp_path) -> None:
        """Test verdi al primo giro → success e report generato."""
        _write(tmp_path, "src/app.py", "def add(a, b):\n    return a + b\n")
        ok = TestResult(success=True, stdout="1 passed", stderr="", exit_code=0)

        with (
            mock.patch("src.orchestrator.diagnose", return_value=_diagnosis()),
            mock.patch(
                "src.orchestrator.generate_patch",
                return_value=_patch_for("src/app.py", "def add(a, b):\n    return a + 0\n"),
            ),
            mock.patch("src.orchestrator.SandboxRunner") as runner_cls,
            mock.patch("src.orchestrator.generate_report", return_value=_report()),
        ):
            runner_cls.return_value.run_tests.return_value = ok
            result = run_patchcraft_loop(str(tmp_path), "Fix bug", model="deepseek/x")

        assert result.success is True
        assert result.iterations == 1
        assert result.report is not None
        assert result.report.title == "Fix offset"

    def test_success_after_self_correction(self, tmp_path) -> None:
        """Primo test fallisce → self-correcction, secondo passa → 2 iterazioni."""
        _write(tmp_path, "src/app.py", "def add(a, b):\n    return a - b\n")
        fail = TestResult(success=False, stderr="AssertionError: 1 != 3", exit_code=1)
        ok = TestResult(success=True, exit_code=0)
        corrected: list[dict] = []

        def _correct(**kwargs) -> Patch:
            corrected.append(kwargs)
            assert "AssertionError: 1 != 3" in kwargs["test_feedback"]
            return _patch_for("src/app.py", "def add(a, b):\n    return a + b\n")

        with (
            mock.patch("src.orchestrator.diagnose", return_value=_diagnosis()),
            mock.patch(
                "src.orchestrator.generate_patch",
                return_value=_patch_for("src/app.py", "def add(a, b):\n    return a - b\n"),
            ),
            mock.patch("src.orchestrator.correct_patch", side_effect=_correct),
            mock.patch("src.orchestrator.SandboxRunner.run_tests", side_effect=[fail, ok]),
            mock.patch("src.orchestrator.generate_report", return_value=_report()),
        ):
            result = run_patchcraft_loop(str(tmp_path), "Fix", "auto")

        assert result.success is True
        assert result.iterations == 2
        assert len(corrected) == 1

    def test_self_correction_feedback_includes_stdout_and_stderr(self, tmp_path) -> None:
        """Il feedback per il self-corrector contiene sia stdout sia stderr
        (Jest/Vitest mettono i dettagli dei test falliti su stdout)."""
        _write(tmp_path, "src/app.py", "def add(a, b):\n    return a - b\n")
        fail = TestResult(
            success=False,
            stdout="FAIL  src/app.test.ts › add(a,b)\n  Expected 3, received 1",
            stderr="error Command failed with exit code 1",
            exit_code=1,
        )
        ok = TestResult(success=True, exit_code=0)
        captured: list[dict] = []

        def _correct(**kwargs) -> Patch:
            captured.append(kwargs)
            return _patch_for("src/app.py", "def add(a, b):\n    return a + b\n")

        with (
            mock.patch("src.orchestrator.diagnose", return_value=_diagnosis()),
            mock.patch(
                "src.orchestrator.generate_patch",
                return_value=_patch_for("src/app.py", "def add(a, b):\n    return a + b\n"),
            ),
            mock.patch("src.orchestrator.correct_patch", side_effect=_correct),
            mock.patch("src.orchestrator.SandboxRunner.run_tests", side_effect=[fail, ok]),
            mock.patch("src.orchestrator.generate_report", return_value=_report()),
        ):
            run_patchcraft_loop(str(tmp_path), "Fix", "auto")

        feedback = captured[0]["test_feedback"]
        assert "Expected 3, received 1" in feedback, "stdout dei test deve arrivare al correttore"
        assert "error Command failed" in feedback, "stderr deve arrivare al correttore"

    def test_exhausted_rolls_back(self, tmp_path) -> None:
        """Tutti i tentativi falliscono → rollback dei file e success=False."""
        _write(tmp_path, "src/app.py", "def add(a, b):\n    return a + b\n")
        fail = TestResult(success=False, stderr="fail", exit_code=1)

        with (
            mock.patch("src.orchestrator.diagnose", return_value=_diagnosis()),
            mock.patch(
                "src.orchestrator.generate_patch",
                return_value=_patch_for("src/app.py", "def add(a, b):\n    return a - b\n"),
            ),
            mock.patch(
                "src.orchestrator.correct_patch",
                return_value=_patch_for("src/app.py", "def add(a, b):\n    return a / b\n"),
            ),
            mock.patch("src.orchestrator.SandboxRunner.run_tests", return_value=fail),
        ):
            result = run_patchcraft_loop(str(tmp_path), "Fix", "mock", max_retries=2)

        assert result.success is False
        assert result.iterations == 2
        assert result.report is None
        # rollback: il file torna al contenuto originale
        assert (
            tmp_path / "src/app.py"
        ).read_text(encoding="utf-8") == "def add(a, b):\n    return a + b\n"

    def test_raises_on_invalid_repo(self, tmp_path) -> None:
        import pytest

        with pytest.raises(NotADirectoryError):
            run_patchcraft_loop(str(tmp_path / "nope"), "bug", "mock")