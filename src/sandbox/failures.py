"""Structured failure extraction from test-runner output (Roadmap Step 2.2).

The functions here parse raw pytest / Jest / Vitest output into normalized
:class:`FailureReport` objects::

    FailureReport {
        test_id         -> "tests/test_app.py::test_add"
        assertion       -> "assert 2 == 3"
        expected        -> "3"
        actual          -> "2"
        traceback_tail  -> last lines of the failure block
    }

The orchestrator feeds the compact rendered report to the self-corrector
instead of full raw dumps (the raw output tail remains as fallback), which
makes correction prompts smaller AND more actionable.

All parsers are defensive: malformed, truncated or unrecognized output never
raises — :func:`extract_failures` simply returns an empty list and the caller
falls back to the raw output.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, List, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - import for type hints only
    from src.sandbox.runner import TestResult

logger = logging.getLogger(__name__)

# Safety caps: never let pathological output explode the report.
MAX_FAILURES = 10
TRACEBACK_TAIL_LINES = 12

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class FailureReport(BaseModel):
    """Normalized information about a single failing test."""

    # Prevents pytest from collecting the class as a *test class*.
    __test__ = False

    test_id: str = Field(description="Best-effort test identifier (node id or name).")
    assertion: str = Field(default="", description="The failing assertion or primary error line.")
    expected: str = Field(default="", description="Expected value when it can be extracted.")
    actual: str = Field(default="", description="Received value when it can be extracted.")
    traceback_tail: str = Field(default="", description="Tail of the failure traceback/output.")


# ---------------------------------------------------------------------------
# Dependency-error detection
# ---------------------------------------------------------------------------

_PYTHON_MISSING_RE = re.compile(
    r"(?:ModuleNotFoundError|ImportError)\s*:\s*No module named\s+'(?P<module>[\w\.]+)'"
)
_PYTHON_IMPORT_RE = re.compile(r"^(?:ModuleNotFoundError|ImportError)\b.*$", re.MULTILINE)
_PIP_NO_DIST_RE = re.compile(r"No matching distribution found for\s+(?P<pkg>[\w\.\-\[\]=<>!,]+)")
_NODE_MODULE_RE = re.compile(r"Cannot find (?:module|package)\s+'(?P<module>[^']+)'")
_NODE_ERR_RE = re.compile(r"ERR_MODULE_NOT_FOUND")


def detect_dependency_error(stdout: str, stderr: str) -> Optional[str]:
    """Detect missing-dependency errors in test output.

    Returns a short human-readable description of the missing dependency,
    or ``None`` when the output does not match any known pattern. This never
    raises: unknown formats simply yield ``None``.
    """
    combined = f"{stdout or ''}\n{stderr or ''}"

    match = _PYTHON_MISSING_RE.search(combined)
    if match:
        module = match.group("module")
        return (
            f"Missing Python module '{module}' "
            "(ModuleNotFoundError/ImportError). Install it or fix the import."
        )

    match = _PIP_NO_DIST_RE.search(combined)
    if match:
        pkg = match.group("pkg")
        return (
            f"pip could not resolve '{pkg}' (no matching distribution). "
            "Check the package name/version or the package index."
        )

    match = _NODE_MODULE_RE.search(combined)
    if match:
        module = match.group("module")
        return (
            f"Missing Node.js module/package '{module}' (Cannot find module / "
            "ERR_MODULE_NOT_FOUND). Run the package manager install step."
        )

    if _NODE_ERR_RE.search(combined):
        return (
            "Node.js module resolution failed (ERR_MODULE_NOT_FOUND). "
            "A dependency is probably not installed."
        )

    if _PYTHON_IMPORT_RE.search(combined):
        return (
            "An import error occurred (ModuleNotFoundError/ImportError). "
            "A required Python package may be missing."
        )

    return None


# ---------------------------------------------------------------------------
# pytest parsing
# ---------------------------------------------------------------------------

# Section banners: "================================== FAILURES ============"
_BANNER_RE = re.compile(r"^\s*=+\s*(?P<title>[A-Z ]{4,})\s*=+\s*$")
# Per-test header: "_________ test_add_fails _________" (also ERROR collecting variants)
_TEST_HEADER_RE = re.compile(r"^_{5,}\s*(?P<name>.+?)\s*_{5,}\s*$")
# Short summary line: "FAILED tests/test_app.py::test_add - assert 2 == 3"
_SUMMARY_FAILED_RE = re.compile(r"^\s*FAILED\s+(?P<node>\S+)", re.MULTILINE)

_ASSERT_EQ_RE = re.compile(r"\bassert\s+(?P<actual>.+?)\s*==\s*(?P<expected>.+)")
_ASSERT_NEQ_ERR_RE = re.compile(r"AssertionError:\s*(?P<actual>.+?)\s*!=\s*(?P<expected>.+)")


def _pytest_node_id_for(name: str, combined: str) -> str:
    """Map a failure-block header name to its full node id when possible."""
    escaped = re.escape(name.strip())
    for match in _SUMMARY_FAILED_RE.finditer(combined):
        node = match.group("node")
        # The summary node id usually ends with "::<test_name>" (or the name
        # appears right before a parameterized "[...]" suffix).
        if re.search(rf"::{escaped}(?:\[|\b)", node) or node.endswith(name.strip()):
            return node
    return name.strip()


def _parse_pytest_e_line(line: str, report: FailureReport) -> None:
    """Extract assertion / expected / actual from a single pytest ``E`` line."""
    if report.assertion:
        return
    body = re.sub(r"^E\s+", "", line.rstrip())
    report.assertion = body
    eq = _ASSERT_EQ_RE.search(body)
    if eq:
        report.actual = eq.group("actual").strip()
        report.expected = eq.group("expected").strip()
        return
    neq = _ASSERT_NEQ_ERR_RE.search(body)
    if neq:
        report.actual = neq.group("actual").strip()
        report.expected = neq.group("expected").strip()


def _extract_pytest(output: str) -> List[FailureReport]:
    """Parse pytest ``FAILURES`` / ``ERRORS`` sections into reports."""
    lines = output.splitlines()
    failures: List[FailureReport] = []

    # Collect candidate (header_index, name) pairs inside FAILURES/ERRORS
    # banners; blocks span from one header to the next (or banner end).
    in_section = False
    headers: List[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        banner = _BANNER_RE.match(line)
        if banner:
            title = banner.group("title").strip()
            in_section = title in {"FAILURES", "ERRORS"}
            continue
        if in_section:
            header = _TEST_HEADER_RE.match(line)
            if header:
                headers.append((idx, header.group("name")))

    if not headers:
        return []

    # Compute block boundaries: each header's block ends at the next header
    # or at the closing banner (next all-'=' line after the header).
    bounds: List[int] = []
    for idx, _name in headers:
        end = len(lines)
        for j in range(idx + 1, len(lines)):
            if _TEST_HEADER_RE.match(lines[j]) or _BANNER_RE.match(lines[j]):
                end = j
                break
        bounds.append(end)

    for (start, name), end in zip(headers, bounds):
        if len(failures) >= MAX_FAILURES:
            break
        block_lines = list(lines[start + 1:end])
        report = FailureReport(
            test_id=_pytest_node_id_for(name, output),
        )
        source_line: Optional[str] = None
        for line in block_lines:
            stripped = line.lstrip()
            # E-lines carry the actual assertion result: they take priority
            # over the '>' source line for the assertion text.
            if stripped.startswith("E ") or stripped.startswith("E\t"):
                _parse_pytest_e_line(stripped, report)
            elif source_line is None and stripped.startswith(">"):
                source_line = stripped.lstrip("> ").strip()
        if not report.assertion and source_line:
            report.assertion = source_line
        tail = [l for l in block_lines if l.strip()]
        report.traceback_tail = "\n".join(tail[-TRACEBACK_TAIL_LINES:])
        failures.append(report)

    return failures


# ---------------------------------------------------------------------------
# Jest / Vitest parsing
# ---------------------------------------------------------------------------

_FAIL_FILE_RE = re.compile(r"^FAIL\s+(?P<file>\S+)")
# Single-line variant: "FAIL src/app.test.ts › add(a,b)" (no leading bullet).
_FAIL_TEST_RE = re.compile(r"^FAIL\s+(?P<file>\S+)\s*[\u203a>]\s*(?P<name>.+)$")
_BULLET_RE = re.compile(r"^\s*[\u25cf\u2715]\s*(?P<name>.*)$")
_EXPECTED_RE = re.compile(r"\s*Expected:\s*(?P<value>.*)$")
_RECEIVED_RE = re.compile(r"\s*Received:\s*(?P<value>.*)$")


def _clean_js_name(raw: str) -> str:
    """Normalize a Jest/Vitest test title line into a bare name."""
    name = raw.replace("\u203a", ">").strip()
    name = re.sub(r"\s*\(\d+\s*m?s\)\s*$", "", name)  # drop "(12 ms)"
    return name.rstrip(">").strip()


def _extract_jest(output: str) -> List[FailureReport]:
    """Parse Jest/Vitest failure output into reports."""
    lines = output.splitlines()
    failures: List[FailureReport] = []
    current_file: Optional[str] = None
    current: Optional[FailureReport] = None
    tail_buffer: List[str] = []

    def flush_tail(report: Optional["FailureReport | None"]) -> None:
        if report is not None and tail_buffer:
            report.traceback_tail = "\n".join(tail_buffer[-TRACEBACK_TAIL_LINES:])

    for line in lines:
        fail_test = _FAIL_TEST_RE.match(line)
        if fail_test:
            flush_tail(current)
            tail_buffer.clear()
            current_file = fail_test.group("file")
            name = _clean_js_name(fail_test.group("name"))
            current = FailureReport(test_id=f"{current_file} \u203a {name}")
            if len(failures) < MAX_FAILURES:
                failures.append(current)
            continue

        file_match = _FAIL_FILE_RE.match(line)
        if file_match:
            flush_tail(current)
            current_file = file_match.group("file")
            current = None
            tail_buffer.clear()
            continue

        bullet = _BULLET_RE.match(line)
        if bullet:
            flush_tail(current)
            tail_buffer.clear()
            raw_name = bullet.group("name")
            name = _clean_js_name(raw_name)
            if not name:
                current = None
                continue
            if current_file and "\u203a" not in raw_name:
                test_id = f"{current_file} \u203a {name}"
            else:
                test_id = name.replace(">", "\u203a") if ">" in name else name
            current = FailureReport(test_id=test_id)
            if len(failures) < MAX_FAILURES:
                failures.append(current)
            continue

        if current is None:
            continue

        expected = _EXPECTED_RE.match(line)
        if expected and not current.expected:
            current.expected = expected.group("value").strip().strip("'\"")
            continue
        received = _RECEIVED_RE.match(line)
        if received and not current.actual:
            current.actual = received.group("value").strip().strip("'\"")
            continue
        if not current.assertion and "expect(" in line:
            current.assertion = line.strip()

        stripped = line.strip()
        if stripped.startswith((">", "|")) or "\u276f" in stripped or "\u2715" in stripped:
            tail_buffer.append(line)

    flush_tail(current)

    # Every FAIL/bullet entry is a genuinely failing test id, so all entries
    # are kept even when no Expected/Received values were extractable.
    return failures


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_failures(result: "TestResult | Any") -> List[FailureReport]:
    """Extract structured failures from a :class:`TestResult`-like object.

    Accepts any object exposing ``stdout``/``stderr`` attributes. Tries the
    pytest parser first, then the Jest/Vitest one; returns ``[]`` when nothing
    recognizable is found (callers fall back to the raw output).
    """
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    combined = f"{stdout}\n{stderr}"
    if not combined.strip():
        return []

    try:
        failures = _extract_pytest(combined)
        if not failures:
            failures = _extract_jest(combined)
        return failures
    except Exception:  # noqa: BLE001 - parsing must never break the loop
        logger.debug("Failure extraction failed; falling back to raw output.", exc_info=True)
        return []


def format_failure_report(failures: List[Any]) -> str:
    """Render extracted failures into a compact, corrector-friendly report."""
    lines: List[str] = []
    for i, failure in enumerate(failures, start=1):
        lines.append(f"[{i}/{len(failures)}] {failure.test_id}")
        if failure.expected or failure.actual:
            lines.append(f"    expected: {failure.expected or '?'}")
            lines.append(f"    actual:   {failure.actual or '?'}")
        if failure.assertion:
            lines.append(f"    assertion: {failure.assertion}")
        if failure.traceback_tail:
            indented = "\n".join(
                f"      {l}" for l in failure.traceback_tail.splitlines()
            )
            lines.append(f"    traceback (tail):\n{indented}")
    return "\n".join(lines)


__all__ = [
    "FailureReport",
    "MAX_FAILURES",
    "detect_dependency_error",
    "extract_failures",
    "format_failure_report",
]

