"""Test del sandbox runner (subprocess reale, niente mock)."""
from __future__ import annotations

import json
import shutil
import sys
import time

import pytest

from src.sandbox.runner import (
    EXIT_CODE_NOT_FOUND,
    EXIT_CODE_TIMEOUT,
    TIMEOUT_MAX,
    SandboxRunner,
    TestResult,
)


def _write(root, relpath: str, content: str):
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestPytestAutoDetection:
    def test_pytest_pass(self, tmp_path) -> None:
        _write(tmp_path, "test_sample.py", "def test_ok():\n    assert True\n")
        result = SandboxRunner(tmp_path).run_tests()
        assert isinstance(result, TestResult)
        assert result.success is True
        assert result.exit_code == 0
        assert "1 passed" in result.stdout

    def test_pytest_fail(self, tmp_path) -> None:
        _write(tmp_path, "test_sample.py", "def test_ok():\n    assert 1 + 1 == 3\n")
        result = SandboxRunner(tmp_path).run_tests()
        assert result.success is False
        assert result.exit_code != 0
        assert result.stderr or result.stdout  # c'è traccia dell'errore

    def test_no_pattern_uses_pytest_by_default(self, tmp_path) -> None:
        # nessun package.json: il default è python -m pytest
        assert SandboxRunner(tmp_path)._detect_test_command() == [
            sys.executable,
            "-m",
            "pytest",
        ]


class TestNpmDetection:
    def test_no_package_json_uses_pytest_by_default(self, tmp_path) -> None:
        # nessun package.json e nessun marker: fallback pytest
        assert SandboxRunner(tmp_path)._detect_test_command() == [
            sys.executable,
            "-m",
            "pytest",
        ]

    def test_pyproject_uses_pytest(self, tmp_path) -> None:
        _write(tmp_path, "pyproject.toml", "[tool.pytest.ini_options]\n")
        assert SandboxRunner(tmp_path)._detect_test_command() == [
            sys.executable,
            "-m",
            "pytest",
        ]

    def test_pytest_ini_uses_pytest(self, tmp_path) -> None:
        _write(tmp_path, "pytest.ini", "[pytest]\n")
        assert SandboxRunner(tmp_path)._detect_test_command() == [
            sys.executable,
            "-m",
            "pytest",
        ]

    def test_python_file_uses_pytest(self, tmp_path) -> None:
        _write(tmp_path, "src/app.py", "print('hi')\n")
        assert SandboxRunner(tmp_path)._detect_test_command() == [
            sys.executable,
            "-m",
            "pytest",
        ]

    def test_npm_without_lockfile(self, tmp_path) -> None:
        _write(tmp_path, "package.json", "{\"name\": \"t\", \"scripts\": {}}")
        assert SandboxRunner(tmp_path)._detect_test_command() == ["npm", "test"]

    def test_pnpm_when_pnpm_lock(self, tmp_path) -> None:
        _write(tmp_path, "package.json", "{}")
        _write(tmp_path, "pnpm-lock.yaml", "lockfileVersion: '6.0'\n")
        assert SandboxRunner(tmp_path)._detect_test_command() == ["pnpm", "test"]

    def test_yarn_when_yarn_lock(self, tmp_path) -> None:
        _write(tmp_path, "package.json", "{}")
        _write(tmp_path, "yarn.lock", "# yarn\n")
        assert SandboxRunner(tmp_path)._detect_test_command() == ["yarn", "test"]

    def test_yarn_when_yarn_lock_json(self, tmp_path) -> None:
        _write(tmp_path, "package.json", "{}")
        _write(tmp_path, "yarn-lock.json", "{}\n")
        assert SandboxRunner(tmp_path)._detect_test_command() == ["yarn", "test"]

    def test_pnpm_takes_priority_over_yarn(self, tmp_path) -> None:
        _write(tmp_path, "package.json", "{}")
        _write(tmp_path, "pnpm-lock.yaml", "lockfileVersion: '6.0'\n")
        _write(tmp_path, "yarn.lock", "# yarn\n")
        assert SandboxRunner(tmp_path)._detect_test_command() == ["pnpm", "test"]

    @pytest.mark.skipif(
        shutil.which("npm") is None or shutil.which("node") is None,
        reason="npm/node non disponibili sull'ambiente",
    )
    def test_npm_test(self, tmp_path) -> None:
        package = {
            "name": "target",
            "scripts": {"test": 'node -e "console.log(\'npm-ok\')"'},
        }
        _write(tmp_path, "package.json", json.dumps(package))
        result = SandboxRunner(tmp_path).run_tests()
        assert result.success is True
        assert result.exit_code == 0
        assert "npm-ok" in result.stdout


class TestJsFailureCapturesStdout:
    @pytest.mark.skipif(
        shutil.which("npm") is None or shutil.which("node") is None,
        reason="npm/node non disponibili sull'ambiente",
    )
    def test_failing_js_test_keeps_stdout(self, tmp_path) -> None:
        """Test JS falliti: anche se il dettaglio è su stdout, il TestResult
        deve restituirlo (per il feedback del self-corrector)."""
        package = {
            "name": "target",
            "scripts": {"test": "node -e \"console.log('DETTAGLIO-TEST-FALLITO'); process.exit(1)\""},
        }
        _write(tmp_path, "package.json", json.dumps(package))
        result = SandboxRunner(tmp_path).run_tests()
        assert result.success is False
        assert result.exit_code != 0
        assert "DETTAGLIO-TEST-FALLITO" in f"{result.stdout}\n{result.stderr}"


class TestTimeoutAndErrors:
    def test_timeout_kills_process(self, tmp_path) -> None:
        """Un test che non termina viene ucciso (exit_code=124) e in fretta."""
        command = [sys.executable, "-c", "import time; time.sleep(60)"]
        runner = SandboxRunner(tmp_path, timeout=0.5, command=command)
        start = time.monotonic()
        result = runner.run_tests()
        elapsed = time.monotonic() - start

        assert result.success is False
        assert result.exit_code == EXIT_CODE_TIMEOUT
        assert elapsed < 5, "il timeout deve terminare subito, non dopo 60s"

    def test_command_not_found(self, tmp_path) -> None:
        result = SandboxRunner(tmp_path, command=["comando_che_non_esiste_xyz"]).run_tests()
        assert result.success is False
        assert result.exit_code == EXIT_CODE_NOT_FOUND
        assert "Command not found" in result.stderr

    def test_invalid_project_dir(self, tmp_path) -> None:
        with pytest.raises(NotADirectoryError):
            SandboxRunner(tmp_path / "non_esiste")

    def test_timeout_clamped_to_max(self, tmp_path) -> None:
        runner = SandboxRunner(tmp_path, timeout=9999)
        assert runner.timeout == TIMEOUT_MAX

    def test_timeout_zero_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            SandboxRunner(tmp_path, timeout=0)


class TestExplicitCommand:
    def test_explicit_list_command(self, tmp_path) -> None:
        result = SandboxRunner(
            tmp_path, command=[sys.executable, "-c", "print('esplicito')"]
        ).run_tests()
        assert result.success is True
        assert result.exit_code == 0
        assert "esplicito" in result.stdout

    def test_run_alias(self, tmp_path) -> None:
        runner = SandboxRunner(tmp_path)
        assert runner.run([sys.executable, "-c", "raise SystemExit(0)"]).success is True
        assert runner.run("python -c \"raise SystemExit(3)\"").exit_code == 3