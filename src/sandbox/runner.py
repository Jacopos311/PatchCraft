"""Sandbox runner: isolated, safe execution of the target project's tests.

:class:`SandboxRunner` accepts the target project directory and runs the
test command through ``subprocess``, auto-detecting the best test runner
based on the files in the root:

* **package.json**:
    - ``pnpm-lock.yaml`` -> ``pnpm test``
    - ``yarn.lock`` / ``yarn-lock.json`` -> ``yarn test``
    - otherwise -> ``npm test``
* **Python** (``pyproject.toml``, ``pytest.ini`` or ``.py`` files)
  -> ``python -m pytest``.

On Windows the Node CLI shims (``npm.cmd``/``pnpm.cmd``/``yarn.cmd``) cannot
be executed directly by ``CreateProcess``: the runner invokes them through
``cmd.exe /d /s /c`` (the managed equivalent of ``shell=True`` for these
commands, but without shell-injection risks because the argument is never a
shell string).

Every run is guarded by a **timeout**: targeted runs are capped at 30 seconds
(:data:`TIMEOUT_MAX`), while full-suite runs get a higher, separate limit
(default 300 s, configurable via ``PATCHCRAFT_TIMEOUT_FULL_SUITE`` or
``.patchcraft.yml``). When it expires the whole process tree is killed and
the command is considered failed (``exit_code=124``).

:meth:`SandboxRunner.run_tests` returns a :class:`TestResult` (Pydantic) with
``success``, ``stdout``, ``stderr``, ``exit_code``, plus Step 2.2 fields:
``subset`` (which subset ran), ``missing_dependency`` (detected missing
dependency) and ``dependency_retried`` (one-shot auto-install retry).
Dependency auto-install only happens behind an explicit flag
(``auto_install=True`` / ``PATCHCRAFT_AUTO_INSTALL=1`` / ``.patchcraft.yml``
``sandbox.auto_install: true``) and at most once per runner instance.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from pydantic import BaseModel, Field

from src.sandbox.failures import detect_dependency_error

logger = logging.getLogger(__name__)

# Security limit (seconds) applied to every *targeted* test run.
TIMEOUT_MAX = 30.0

# Default/cap for FULL-SUITE runs (Roadmap Step 2.1/2.2): big suites need far
# more than 30 s, so full runs get their own, higher limit.
TIMEOUT_FULL_SUITE_DEFAULT = 300.0
TIMEOUT_FULL_SUITE_MAX = 1800.0

# One-shot dependency installation (only behind --auto-install).
INSTALL_TIMEOUT = 600.0

# Conventional exit codes for runner errors.
EXIT_CODE_TIMEOUT = 124  # same as GNU coreutils `timeout`
EXIT_CODE_NOT_FOUND = 127  # like the shell "command not found"

# Environment variables (explicit constructor arguments always win).
ENV_TIMEOUT = "PATCHCRAFT_TEST_TIMEOUT"
ENV_TIMEOUT_FULL_SUITE = "PATCHCRAFT_TIMEOUT_FULL_SUITE"
ENV_AUTO_INSTALL = "PATCHCRAFT_AUTO_INSTALL"

_TRUTHY = {"1", "true", "yes", "on"}

# Optional repo-level config file (full schema lands with Step 3.3); only the
# ``sandbox:`` section is consumed here.
CONFIG_FILENAMES = (".patchcraft.yml", ".patchcraft.yaml")


def load_runner_config(project_dir: Union[str, Path]) -> dict[str, Any]:
    """Read optional sandbox options from ``<project>/.patchcraft.yml``.

    Recognized keys under the top-level ``sandbox:`` section (legacy)::

        sandbox:
          command: "python -m pytest"       # explicit test command override
          timeout: 30                        # targeted-run timeout (s)
          timeout_full_suite: 300            # full-suite timeout (s)
          auto_install: false                # one-shot dependency install

    The typed ``test:`` section (Step 3.3) is also recognized and WINS over
    the legacy keys::

        test:
          command: "python -m pytest"
          timeout_full_suite: 600

    Missing file, missing PyYAML or malformed YAML all degrade gracefully to
    an empty mapping (never raises). Unknown keys are ignored silently.
    Full schema validation lives in :mod:`src.core.config`.
    """
    root = Path(project_dir)
    config_path = next(
        (root / name for name in CONFIG_FILENAMES if (root / name).is_file()), None
    )
    if config_path is None:
        return {}
    try:
        import yaml  # soft dependency: litellm already ships PyYAML
    except ImportError:
        logger.debug("PyYAML not available; ignoring %s.", config_path.name)
        return {}
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - malformed YAML must not break runs
        logger.debug("Could not parse %s (%s); ignoring it.", config_path.name, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    sandbox = data.get("sandbox")
    sandbox = sandbox if isinstance(sandbox, dict) else {}
    # Step 3.3: the typed ``test:`` section wins over the legacy one.
    test_section = data.get("test")
    test_section = test_section if isinstance(test_section, dict) else {}
    return {**sandbox, **test_section}




def _has_python_files(root: Path) -> bool:
    """True if any ``*.py`` file exists under ``root`` (ignores venv/node_modules)."""
    ignored = {"node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
    for path in root.rglob("*.py"):
        if not any(part in ignored for part in path.parts):
            return True
    return False


class TestResult(BaseModel):
    """Structured result of running the tests in the sandbox."""

    # Prevents pytest from collecting the class as a *test class*.
    __test__ = False

    success: bool = Field(description="True if the tests finished successfully.")
    stdout: str = Field(default="", description="Captured stdout output.")
    stderr: str = Field(default="", description="Captured stderr output.")
    exit_code: int = Field(description="Exit code of the test command.")
    subset: str = Field(
        default="full",
        description="Which subset ran: 'full' (whole suite) or 'targeted'.",
    )
    missing_dependency: Optional[str] = Field(
        default=None,
        description="Human-readable description of a detected missing dependency.",
    )
    dependency_retried: bool = Field(
        default=False,
        description=(
            "True when the run was retried after a one-shot auto-install "
            "(only possible with auto_install enabled)."
        ),
    )
    cached: bool = Field(
        default=False,
        description=(
            "True when this verdict was reused from the targeted-test "
            "result cache (Roadmap Step 3.1) instead of executing the "
            "tests. Never set for full-suite runs."
        ),
    )


class SandboxRunner:
    """Runs the tests of a target project in isolation.

    Parameters
    ----------
    project_dir : str | Path
        Directory of the target project whose tests must run.
    timeout : float | None
        Maximum timeout in seconds for *targeted* runs (clamped to
        :data:`TIMEOUT_MAX`). ``None`` reads ``PATCHCRAFT_TEST_TIMEOUT`` from
        the environment and falls back to :data:`TIMEOUT_MAX`. An explicitly
        provided timeout (argument or env) also caps FULL-SUITE runs, unless a
        specific ``timeout_full_suite`` is given as well.
    command : str | Sequence[str] | None
        Explicit test command. If ``None`` it is auto-derived from the
        project (npm test when ``package.json`` exists, otherwise pytest).
    timeout_full_suite : float | None
        Maximum timeout for FULL-SUITE runs (clamped to
        :data:`TIMEOUT_FULL_SUITE_MAX`). ``None`` reads
        ``PATCHCRAFT_TIMEOUT_FULL_SUITE``, then ``.patchcraft.yml``
        (``sandbox.timeout_full_suite``) and finally falls back to
        :data:`TIMEOUT_FULL_SUITE_DEFAULT`.
    auto_install : bool | None
        When explicitly ``True``, a failed run that shows a
        missing-dependency error triggers exactly ONE dependency install
        (pip/npm/pnpm/yarn, detected from the project layout) followed by one
        retry. Default (``None``) resolves to the
        ``PATCHCRAFT_AUTO_INSTALL`` environment variable, then
        ``.patchcraft.yml`` (``sandbox.auto_install``), then **off**.
    """

    def __init__(
        self,
        project_dir: Union[str, Path],
        timeout: Optional[float] = None,
        command: Optional[Union[str, Sequence[str]]] = None,
        timeout_full_suite: Optional[float] = None,
        auto_install: Optional[bool] = None,
    ) -> None:
        config = load_runner_config(project_dir)
        self.project_dir = Path(project_dir).resolve()
        if not self.project_dir.is_dir():
            raise NotADirectoryError(
                f"project_dir is not a valid directory: {self.project_dir}"
            )
        # An explicitly provided timeout (argument or env) acts as a GLOBAL
        # cap for every run, preserving pre-2.2 behavior, unless a specific
        # timeout_full_suite is provided as well.
        timeout_given = timeout is not None or bool(os.getenv(ENV_TIMEOUT))
        if timeout is None:
            raw_env = os.getenv(ENV_TIMEOUT)
            if raw_env:
                timeout = float(raw_env)
            elif config.get("timeout") is not None:
                timeout = float(config["timeout"])
            else:
                timeout = TIMEOUT_MAX
        self.timeout: float = min(float(timeout), TIMEOUT_MAX)
        if self.timeout <= 0:
            raise ValueError("The timeout must be greater than zero.")

        tfs_given = (
            timeout_full_suite is not None
            or bool(os.getenv(ENV_TIMEOUT_FULL_SUITE))
            or config.get("timeout_full_suite") is not None
        )
        if timeout_full_suite is None:
            raw_env = os.getenv(ENV_TIMEOUT_FULL_SUITE)
            if raw_env:
                timeout_full_suite = float(raw_env)
            elif config.get("timeout_full_suite") is not None:
                timeout_full_suite = float(config["timeout_full_suite"])
            else:
                timeout_full_suite = TIMEOUT_FULL_SUITE_DEFAULT
        self.timeout_full_suite: float = min(
            float(timeout_full_suite), TIMEOUT_FULL_SUITE_MAX
        )
        if not tfs_given and timeout_given:
            # Global explicit timeout wins over the 300 s default.
            self.timeout_full_suite = self.timeout
        if self.timeout_full_suite <= 0:
            raise ValueError("timeout_full_suite must be greater than zero.")

        if auto_install is None:
            raw_flag = os.getenv(ENV_AUTO_INSTALL)
            if raw_flag is not None:
                auto_install = raw_flag.strip().lower() in _TRUTHY
            else:
                auto_install = bool(config.get("auto_install", False))
        self.auto_install: bool = bool(auto_install)

        if command is not None:
            self.command = command
        else:
            self.command = config.get("command")  # may stay None (auto-detect)

        # One-shot guard: dependencies are installed at most once per runner
        # instance so iterations never re-touch the environment repeatedly.
        self._install_attempted: bool = False


    # ------------------------------------------------------------------
    # Test command detection
    # ------------------------------------------------------------------
    def _detect_test_command(self) -> list[str]:
        """Return the test command to run (explicit or auto-detected).

        Detection is based on the files in the root of ``project_dir``:

        * With ``package.json``:
            - ``pnpm-lock.yaml``  -> ``pnpm test``
            - ``yarn.lock`` / ``yarn-lock.json`` -> ``yarn test``
            - otherwise          -> ``npm test``
        * Without ``package.json`` but with Python markers (``pyproject.toml``,
          ``pytest.ini``) or ``*.py`` files -> ``python -m pytest``.
        * Safe fallback -> ``python -m pytest``.
        """
        if self.command is not None:
            return self._parse_command(self.command)

        if (self.project_dir / "package.json").is_file():
            if (self.project_dir / "pnpm-lock.yaml").is_file():
                return ["pnpm", "test"]
            if (
                (self.project_dir / "yarn.lock").is_file()
                or (self.project_dir / "yarn-lock.json").is_file()
            ):
                return ["yarn", "test"]
            return ["npm", "test"]

        if _has_python_files(self.project_dir) or (
            self.project_dir / "pyproject.toml"
        ).is_file() or (self.project_dir / "pytest.ini").is_file():
            return [sys.executable, "-m", "pytest"]

        # Fallback: no marker detected (the command will fail cleanly if the
        # project has no tests wired for the chosen runner).
        return [sys.executable, "-m", "pytest"]

    @staticmethod
    def _parse_command(command: Union[str, Sequence[str]]) -> list[str]:
        """Normalize the command into a list of args (never passed to a shell).

        Strings are split with :func:`shlex.split` so quoting
        (e.g. ``python -c "raise SystemExit(3)"``) is handled correctly.
        """
        if isinstance(command, str):
            return shlex.split(command)
        return [str(part) for part in command]


# --- environment and execution -------------------------------------------

    def _build_env(self) -> dict[str, str]:
        """Child process environment: inherited but non-interactive."""
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"  # output captured in real time
        env["CI"] = "1"  # non-interactive behavior (pytest/npm)
        env.pop("PYTHONPATH", None)  # no imports from the sandbox
        return env

    @staticmethod
    def _build_popen_kwargs() -> dict:
        """Flags for creating a new process group (to kill the whole tree)."""
        if sys.platform == "win32":
            return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        return {"start_new_session": True}

    @staticmethod
    def _kill_process_tree(proc: subprocess.Popen) -> None:
        """Force-kill the entire process tree of the child."""
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
                check=False,
            )
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()

    @staticmethod
    def _resolve_windows_cmd(command: list[str]) -> list[str]:
        """Resolve Windows `.cmd`/`.bat` executables through ``cmd.exe``.

        On Windows ``CreateProcess`` cannot execute ``.cmd`` files directly
        (e.g. ``npm`` -> ``npm.cmd``, ``pnpm`` -> ``pnpm.cmd``); the command
        is routed through ``cmd.exe``. This is the managed equivalent of
        ``shell=True`` for Node CLIs, but without shell-injection risks: the
        command remains an argument list and on non-Windows systems no wrapper
        is applied.
        """
        if not command or sys.platform != "win32":
            return command
        resolved = shutil.which(command[0])
        if resolved and resolved.lower().endswith((".cmd", ".bat")):
            return [os.environ.get("ComSpec", "cmd.exe"), "/d", "/s", "/c", *command]
        return command

    @staticmethod
    def _with_pytest_targets(
        command: list[str],
        targets: Sequence[str],
        ) -> list[str]:
        """Append ``targets`` to a pytest command (no-op for non-pytest).

        Only appends when the base command is pytest (``python -m pytest``);
        for npm/pnpm/yarn targets are ignored so the full suite runs.
        """
        # Detect pytest-based commands: ["python", "-m", "pytest"] or a direct
        # "pytest" executable.
        is_pytest = (
            "pytest" in command
            or (len(command) >= 3 and command[-3:] == [sys.executable, "-m", "pytest"])
        )
        if not is_pytest:
            return command
        return [*command, "--", *targets]

    def _effective_timeout(self, has_targets: bool) -> float:
        """Timeout for this run: targeted runs use the short limit, the full
        suite uses its own (higher) limit."""
        return self.timeout if has_targets else self.timeout_full_suite

    def _execute(self, command: list[str], timeout: float) -> TestResult:
        """Run ``command`` once and capture the outcome (no retry logic)."""
        try:
            proc = subprocess.Popen(
                command,
                cwd=str(self.project_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self._build_env(),
                **self._build_popen_kwargs(),
            )
        except FileNotFoundError as exc:
            # e.g. `npm` is not installed or not on PATH.
            return TestResult(
                success=False,
                stderr=f"Command not found: {exc.filename}",
                exit_code=EXIT_CODE_NOT_FOUND,
            )

        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_process_tree(proc)
            # Collect the output already produced and wait for termination.
            stdout, stderr = proc.communicate()

        exit_code = EXIT_CODE_TIMEOUT if timed_out else int(proc.returncode)
        return TestResult(
            success=exit_code == 0,
            stdout=stdout or "",
            stderr=stderr or "",
            exit_code=exit_code,
        )

    def _detect_install_command(self) -> Optional[list[str]]:
        """Best dependency-install command for this project layout."""
        root = self.project_dir
        if (root / "package.json").is_file():
            if (root / "pnpm-lock.yaml").is_file():
                return ["pnpm", "install"]
            if (root / "yarn.lock").is_file() or (root / "yarn-lock.json").is_file():
                return ["yarn", "install"]
            return ["npm", "install"]
        if (root / "requirements.txt").is_file():
            return [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        if (root / "pyproject.toml").is_file():
            return [sys.executable, "-m", "pip", "install", "-e", "."]
        return None

    def install_dependencies(self) -> bool:
        """Run the project's dependency install step exactly once per runner.

        Returns ``True`` when the install command completed successfully. The
        one-shot guard (``_install_attempted``) ensures iterations reuse the
        same environment instead of touching it on every failed run.
        """
        if self._install_attempted:
            return False
        self._install_attempted = True

        command = self._detect_install_command()
        if command is None:
            logger.info(
                "Auto-install requested but no known dependency manifest found "
                "in %s.",
                self.project_dir,
            )
            return False

        resolved = self._resolve_windows_cmd(command)
        console_kwargs: dict = {}
        if sys.platform == "win32":
            console_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            completed = subprocess.run(
                resolved,
                cwd=str(self.project_dir),
                capture_output=True,
                text=True,
                timeout=INSTALL_TIMEOUT,
                env=self._build_env(),
                **console_kwargs,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("Dependency install failed: %s", exc)
            return False

        ok = completed.returncode == 0
        if not ok:
            logger.warning(
                "Dependency install exited with %s: %s",
                completed.returncode,
                (completed.stderr or "")[-500:],
            )
        return ok

    def run_tests(self, targets: Optional[Sequence[str]] = None) -> TestResult:
        """Run the tests and return a :class:`TestResult`.

        When a run fails with a missing-dependency error the description is
        stored in ``TestResult.missing_dependency``; with ``auto_install``
        enabled the dependencies are installed ONCE and the tests are retried
        a single time.

        Parameters
        ----------
        targets
            Optional list of pytest node IDs (file paths, ``file::node`` or
            ``file::class::test``).  When provided with a pytest-based project
            the targets are appended to the command so only the relevant tests
            run.  For non-pytest projects (npm/pnpm/yarn) targets are ignored
            and the full suite runs.
        """
        base_command = self._detect_test_command()
        command = self._resolve_windows_cmd(base_command)
        has_targets = bool(targets)
        if has_targets:
            command = self._with_pytest_targets(command, targets)

        timeout = self._effective_timeout(has_targets)
        result = self._execute(command, timeout)
        result.subset = "targeted" if has_targets else "full"

        if not result.success:
            result.missing_dependency = detect_dependency_error(
                result.stdout, result.stderr
            )
            # One-shot environment repair: only when explicitly enabled,
            # and attempted at most once per runner instance.
            if (
                result.missing_dependency
                and self.auto_install
                and not self._install_attempted
            ):
                self._install_attempted = True
                if self.install_dependencies():
                    retry = self._execute(command, timeout)
                    retry.subset = result.subset
                    retry.dependency_retried = True
                    retry.missing_dependency = detect_dependency_error(
                        retry.stdout, retry.stderr
                    )
                    return retry

        return result

    # ------------------------------------------------------------------
    # Compatibility
    # ------------------------------------------------------------------
    def run(self, command: Optional[Union[str, Sequence[str]]] = None) -> TestResult:
        """Alias of :meth:`run_tests`, with an explicit optional test command."""
        if command is not None:
            self.command = command
        return self.run_tests()


__all__ = [
    "SandboxRunner",
    "TestResult",
    "TIMEOUT_MAX",
    "TIMEOUT_FULL_SUITE_DEFAULT",
    "load_runner_config",
]