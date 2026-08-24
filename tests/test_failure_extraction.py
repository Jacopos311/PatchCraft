"""Tests for environment hardening & structured failure extraction
(Roadmap Step 2.2): output parsing, dependency-error detection, runner
options (.patchcraft.yml / env), timeouts and one-shot auto-install."""
from __future__ import annotations

import sys
from unittest import mock

import pytest

from src.sandbox.failures import (
    FailureReport,
    detect_dependency_error,
    extract_failures,
    format_failure_report,
)
from src.sandbox.runner import (
    TIMEOUT_FULL_SUITE_DEFAULT,
    TIMEOUT_MAX,
    SandboxRunner,
    TestResult,
    load_runner_config,
)


PYTEST_OUTPUT = """\
============================= test session starts =============================
collected 2 items

test_sample.py F.

================================== FAILURES ===================================
_______________________________ test_add_fails ________________________________

    def test_add_fails():
>       assert add(1, 1) == 3
E       assert 2 == 3
E        +  where 2 = add(1, 1)

test_sample.py:6: AssertionError
========================== short summary info ================================
FAILED test_sample.py::test_add_fails - assert 2 == 3
========================= 1 failed, 1 passed in 0.02s ========================
"""

JEST_OUTPUT = """\
FAIL src/app.test.ts
  ● add(a,b) › sums two numbers

    expect(received).toBe(expected) // Object.is equality

    Expected: 3
    Received: 2

      2 | it("sums", () => {
    > 3 |   expect(add(1, 1)).toBe(3);
        |                   ^
      4 | });

Test Suites: 1 failed, 1 total
"""


def _write(root, relpath: str, content: str):
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestPytestParsing:
    def test_parses_assertion_expected_actual(self) -> None:
        result = TestResult(success=False, stdout=PYTEST_OUTPUT, exit_code=1)
        failures = extract_failures(result)
        assert len(failures) == 1
        failure = failures[0]
        assert isinstance(failure, FailureReport)
        assert failure.test_id == "test_sample.py::test_add_fails"
        assert "assert 2 == 3" in failure.assertion
        assert failure.expected == "3"
        assert failure.actual == "2"
        assert "where 2 = add(1, 1)" in failure.traceback_tail

    def test_formats_compact_report(self) -> None:
        result = TestResult(success=False, stdout=PYTEST_OUTPUT, exit_code=1)
        report = format_failure_report(extract_failures(result))
        assert "[1/1] test_sample.py::test_add_fails" in report
        assert "expected: 3" in report
        assert "actual:" in report


class TestJestVitestParsing:
    def test_parses_expected_received(self) -> None:
        result = TestResult(success=False, stdout=JEST_OUTPUT, exit_code=1)
        failures = extract_failures(result)
        assert len(failures) >= 1
        top = failures[0]
        assert "add(a,b)" in top.test_id or "src/app.test.ts" in top.test_id
        assert top.expected == "3"
        assert top.actual == "2"
        assert "expect(" in top.assertion

    def test_stderr_only_jest_output(self) -> None:
        # The orchestrator feeds details on stdout AND an error on stderr.
        result = TestResult(
            success=False,
            stdout="FAIL  src/app.test.ts › add(a,b)\n  Expected 3, received 1",
            stderr="error Command failed with exit code 1",
            exit_code=1,
        )
        failures = extract_failures(result)
        assert failures and "add(a,b)" in failures[0].test_id


class TestMalformedOutput:
    @pytest.mark.parametrize(
        "stdout,stderr",
        [
            ("", ""),
            ("total garbage \x01\x02 no structure here", ""),
            ("Traceback cut off mid-li", ""),
            ("=" * 80, "= " * 40),
            ("FAIL  ", "\x00binary"),
        ],
    )
    def test_never_raises_and_returns_empty(self, stdout, stderr) -> None:
        result = TestResult(success=False, stdout=stdout, stderr=stderr, exit_code=1)
        assert extract_failures(result) == []

    def test_caps_number_of_failures(self) -> None:
        blocks = []
        for i in range(30):
            blocks.append(f"_____________________ test_case_{i} _____________________\n\n")
            blocks.append(f"E       assert {i} == -1\n\ntest_x.py:{i}: AssertionError\n")
        result = TestResult(success=False, stdout="\n".join(blocks), exit_code=1)
        assert len(extract_failures(result)) <= 10


class TestDependencyErrorDetection:
    def test_python_module_not_found(self) -> None:
        message = detect_dependency_error(
            "", "ModuleNotFoundError: No module named 'leftpad'"
        )
        assert message is not None
        assert "'leftpad'" in message

    def test_node_cannot_find_module(self) -> None:
        message = detect_dependency_error(
            "Error: Cannot find module 'react'", ""
        )
        assert message is not None
        assert "'react'" in message

    def test_err_module_not_found(self) -> None:
        assert detect_dependency_error("", "code: 'ERR_MODULE_NOT_FOUND'") is not None

    def test_pip_no_distribution(self) -> None:
        message = detect_dependency_error(
            "", "ERROR: No matching distribution found for weirdpkg==99"
        )
        assert message is not None and "weirdpkg==99" in message

    def test_plain_failure_is_not_a_dependency_error(self) -> None:
        assert detect_dependency_error("", "AssertionError: 1 != 3") is None
        assert detect_dependency_error("", "") is None


class TestRunnerOptions:
    def test_full_suite_timeout_defaults_and_clamps(self, tmp_path) -> None:
        runner = SandboxRunner(tmp_path)
        assert runner.timeout_full_suite == TIMEOUT_FULL_SUITE_DEFAULT
        big = SandboxRunner(tmp_path, timeout_full_suite=999_999)
        assert big.timeout_full_suite == 1800.0  # TIMEOUT_FULL_SUITE_MAX

    def test_targeted_vs_full_timeout_selection(self, tmp_path) -> None:
        runner = SandboxRunner(tmp_path, timeout=10, timeout_full_suite=200)
        assert runner._effective_timeout(True) == 10
        assert runner._effective_timeout(False) == 200

    def test_explicit_timeout_also_caps_full_suite(self, tmp_path) -> None:
        # Pre-2.2 behavior preserved: an explicit `timeout` (without an
        # explicit timeout_full_suite) applies globally.
        runner = SandboxRunner(tmp_path, timeout=0.5)
        assert runner._effective_timeout(False) == 0.5

    def test_default_timeouts_are_split(self, tmp_path) -> None:
        runner = SandboxRunner(tmp_path)
        assert runner._effective_timeout(True) == TIMEOUT_MAX
        assert runner._effective_timeout(False) == TIMEOUT_FULL_SUITE_DEFAULT

    def test_env_overrides(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("PATCHCRAFT_TIMEOUT_FULL_SUITE", "120")
        monkeypatch.setenv("PATCHCRAFT_AUTO_INSTALL", "1")
        runner = SandboxRunner(tmp_path)
        assert runner.timeout_full_suite == 120.0
        assert runner.auto_install is True

    def test_explicit_args_beat_env(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("PATCHCRAFT_TIMEOUT_FULL_SUITE", "120")
        monkeypatch.setenv("PATCHCRAFT_AUTO_INSTALL", "1")
        runner = SandboxRunner(tmp_path, timeout_full_suite=60, auto_install=False)
        assert runner.timeout_full_suite == 60.0
        assert runner.auto_install is False

    def test_patchcraft_yml_sandbox_section(self, tmp_path) -> None:
        _write(
            tmp_path,
            ".patchcraft.yml",
            "sandbox:\n"
            "  command: python -m pytest -q\n"
            "  timeout: 15\n"
            "  timeout_full_suite: 240\n"
            "  auto_install: true\n"
            "unknown_key: ignored\n",
        )
        config = load_runner_config(tmp_path)
        assert config["timeout_full_suite"] == 240
        runner = SandboxRunner(tmp_path)
        assert runner.timeout == 15
        assert runner.timeout_full_suite == 240.0
        assert runner.auto_install is True
        # explicit args still win over the config file
        override = SandboxRunner(tmp_path, auto_install=False)
        assert override.auto_install is False

    def test_malformed_yaml_is_ignored(self, tmp_path) -> None:
        _write(tmp_path, ".patchcraft.yaml", "sandbox: [unclosed\n  bad yaml")
        assert load_runner_config(tmp_path) == {}
        runner = SandboxRunner(tmp_path)  # must not raise
        assert runner.auto_install is False

    def test_result_reports_subset(self, tmp_path) -> None:
        _write(tmp_path, "test_ok.py", "def test_ok():\n    assert True\n")
        result = SandboxRunner(tmp_path).run_tests()
        assert result.subset == "full"
        targeted = SandboxRunner(tmp_path).run_tests(targets=["test_ok.py"])
        assert targeted.subset == "targeted"


class TestAutoInstallRetry:
    def _flaky_command(self):
        # First execution fails with a dependency error; after the mocked
        # installer creates marker.txt the same command succeeds.
        code = (
            "import sys, pathlib; "
            "sys.exit(0) if pathlib.Path('marker.txt').exists() "
            "else print(\"ModuleNotFoundError: No module named 'leftpad'\") "
            "or sys.exit(1)"
        )
        return [sys.executable, "-c", code]

    def test_auto_install_retries_once(self, tmp_path) -> None:
        runner = SandboxRunner(
            tmp_path, command=self._flaky_command(), auto_install=True
        )

        def fake_install(self):
            _write(tmp_path, "marker.txt", "ok")
            return True

        with mock.patch.object(
            SandboxRunner, "install_dependencies", autospec=True,
            side_effect=fake_install,
        ) as installer:
            result = runner.run_tests()

        installer.assert_called_once()
        assert result.success is True
        assert result.dependency_retried is True
        assert result.missing_dependency is None

    def test_auto_install_disabled_by_default(self, tmp_path) -> None:
        runner = SandboxRunner(tmp_path, command=self._flaky_command())
        assert runner.auto_install is False
        with mock.patch.object(SandboxRunner, "install_dependencies") as installer:
            result = runner.run_tests()
        installer.assert_not_called()
        assert result.success is False
        # detection still runs even without auto-install
        assert result.missing_dependency is not None
        assert "'leftpad'" in result.missing_dependency

    def test_detect_install_commands(self, tmp_path) -> None:
        runner = SandboxRunner(tmp_path)
        assert runner._detect_install_command() is None
        _write(tmp_path, "requirements.txt", "# empty\n")
        assert runner._detect_install_command() == [
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ]
        _write(tmp_path, "package.json", "{}")
        assert runner._detect_install_command() == ["npm", "install"]
        _write(tmp_path, "pnpm-lock.yaml", "lockfileVersion: '6.0'\n")
        assert runner._detect_install_command() == ["pnpm", "install"]


class TestOneShotGuardAndTimeout:
    def test_install_attempted_at_most_once(self, tmp_path) -> None:
        runner = SandboxRunner(
            tmp_path, command=TestAutoInstallRetry()._flaky_command(),
            auto_install=True,
        )
        calls: list[int] = []

        def fake_install(self):  # never fixes anything on purpose
            calls.append(1)
            return True

        with mock.patch.object(SandboxRunner, "install_dependencies", autospec=True,
                               side_effect=fake_install):
            first = runner.run_tests()
            second = runner.run_tests()

        assert len(calls) == 1  # one-shot guard across iterations
        assert first.success is False and second.success is False
        # Only the FIRST failing run gets the install+retry treatment.
        assert first.dependency_retried is True
        assert second.dependency_retried is False

    def test_full_suite_timeout_kills_process(self, tmp_path) -> None:
        import time

        command = [sys.executable, "-c", "import time; time.sleep(60)"]
        runner = SandboxRunner(tmp_path, timeout=20, timeout_full_suite=0.5,
                               command=command)
        start = time.monotonic()
        result = runner.run_tests()
        elapsed = time.monotonic() - start

        assert result.success is False
        assert result.exit_code == 124
        assert result.subset == "full"
        assert elapsed < 5, "the full-suite timeout must fire quickly"

    def test_targeted_timeout_still_capped_at_30s(self, tmp_path) -> None:
        runner = SandboxRunner(tmp_path, timeout=9999)
        assert runner.timeout == TIMEOUT_MAX


