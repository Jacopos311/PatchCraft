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

Every run is guarded by a **timeout** (maximum 30 seconds) to avoid infinite
loops during tests; when it expires the whole process tree is killed and the
command is considered failed (``exit_code=124``).

:meth:`SandboxRunner.run_tests` returns a :class:`TestResult` (Pydantic) with
``success``, ``stdout``, ``stderr`` and ``exit_code`` — both ``stdout`` and
``stderr`` are always captured (Jest/Vitest print failed-test details on
stdout).
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

from pydantic import BaseModel, Field

# Security limit (seconds) applied to every test run.
TIMEOUT_MAX = 30.0

# Conventional exit codes for runner errors.
EXIT_CODE_TIMEOUT = 124  # same as GNU coreutils `timeout`
EXIT_CODE_NOT_FOUND = 127  # like the shell "command not found"


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


class SandboxRunner:
    """Runs the tests of a target project in isolation.

    Parameters
    ----------
    project_dir : str | Path
        Directory of the target project whose tests must run.
    timeout : float
        Maximum timeout in seconds (clamped to :data:`TIMEOUT_MAX`).
    command : str | Sequence[str] | None
        Explicit test command. If ``None`` it is auto-derived from the
        project (npm test when ``package.json`` exists, otherwise pytest).
    """

    def __init__(
        self,
        project_dir: Union[str, Path],
        timeout: float = TIMEOUT_MAX,
        command: Optional[Union[str, Sequence[str]]] = None,
    ) -> None:
        self.project_dir = Path(project_dir).resolve()
        if not self.project_dir.is_dir():
            raise NotADirectoryError(
                f"project_dir is not a valid directory: {self.project_dir}"
            )
        self.timeout: float = min(float(timeout), TIMEOUT_MAX)
        if self.timeout <= 0:
            raise ValueError("The timeout must be greater than zero.")
        self.command = command

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

    def run_tests(self) -> TestResult:
        """Run the tests and return a :class:`TestResult`.

        On timeout the process and its children are killed; the result reports
        ``success=False``, ``exit_code=124`` and the partial output.
        """
        command = self._resolve_windows_cmd(self._detect_test_command())
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
            stdout, stderr = proc.communicate(timeout=self.timeout)
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

    # ------------------------------------------------------------------
    # Compatibility
    # ------------------------------------------------------------------
    def run(self, command: Optional[Union[str, Sequence[str]]] = None) -> TestResult:
        """Alias of :meth:`run_tests`, with an explicit optional test command."""
        if command is not None:
            self.command = command
        return self.run_tests()


__all__ = ["SandboxRunner", "TestResult", "TIMEOUT_MAX"]