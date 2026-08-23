"""Targeted test selection based on affected files (Roadmap Step 2.1).

Maps source files changed by the patch agent to the set of pytest test node IDs
that exercise them, using the import graph from RepoIndex.

The mapping is reverse-graph traversal: for each affected file we follow
import edges backwards to find files that import it (the dependents), then
collect their companion test modules.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.core.repo_index import RepoIndex

logger = logging.getLogger(__name__)

__all__ = [
    "TestSelectionResult",
    "select_targeted_tests",
    "test_targets_from_files",
]

MAX_IMPORT_DEPTH = 2
MAX_DEPENDENTS = 300
MAX_TARGETS = 15


@dataclass
class TestSelectionResult:
    """Outcome of targeted test selection."""
    node_ids: list[str] = field(default_factory=list)
    has_targets: bool = False
    source_files_covered: int = 0
    notes: list[str] = field(default_factory=list)


def _reverse_import_graph(index: RepoIndex) -> dict[str, set[str]]:
    """Build {file: {files that import it}} from import edges."""
    reverse: dict[str, set[str]] = {rel: set() for rel in index.files}
    for rel, entry in index.files.items():
        for dep in entry.imports:
            if dep in reverse and dep != rel:
                reverse[dep].add(rel)
    return reverse


def _walk_dependents(
    reverse: dict[str, set[str]],
    seed: str,
) -> list[str]:
    """BFS over the reverse graph from *seed*, returning reachable node ids."""
    visited: set[str] = set()
    frontier: set[str] = {seed}
    depth = 0
    while frontier and depth < MAX_IMPORT_DEPTH:
        depth += 1
        next_frontier: set[str] = set()
        for node in frontier:
            for dependent in reverse.get(node, ()):
                if dependent not in visited:
                    visited.add(dependent)
                    next_frontier.add(dependent)
                if len(visited) > MAX_DEPENDENTS:
                    return list(visited)
        frontier = next_frontier
    return list(visited)


def _companion_test_files(
    source_file: str,
    all_files: set[str],
) -> list[str]:
    """Return candidate test modules for *source_file*."""
    stem = Path(source_file).stem
    directory = Path(source_file).parent
    candidates = [
        directory / f"test_{stem}.py",
        Path("tests") / f"test_{stem}.py",
        directory / f"{stem}_test.py",
        Path("tests") / f"{stem}_test.py",
    ]
    candidates.append(Path(f"test_{stem}.py"))
    found: list[str] = []
    for cand in candidates:
        rel = cand.as_posix()
        if rel in all_files and rel not in found:
            found.append(rel)
    return found


def _is_test_file(rel: str) -> bool:
    name = Path(rel).name
    return name.startswith("test_") or name.endswith("_test.py")


def select_targeted_tests(
    repo_root: Path,
    affected_files: list[str],
    *,
    index: Optional[RepoIndex] = None,
) -> TestSelectionResult:
    """Select pytest targets for *affected_files* via the import graph."""
    repo_root = Path(repo_root).resolve()
    try:
        if index is None:
            index = RepoIndex.build(repo_root)
    except Exception as exc:  # noqa: BLE001 - selection must never break the loop
        logger.debug("Cannot build repo index for test selection (%s).", exc)
        return TestSelectionResult(
            has_targets=False, notes=[f"index error: {exc}"]
        )

    reverse = _reverse_import_graph(index)
    all_files = set(index.files)
    affected_clean = [f.replace("\\", "/") for f in affected_files]
    affected_set = {f for f in affected_clean if f in all_files}

    closure: set[str] = set()
    for af in affected_clean:
        closure.add(af)
        closure.update(_walk_dependents(reverse, af))

    test_files: list[str] = []
    seen: set[str] = set()
    for src in sorted(closure):
        for tf in _companion_test_files(src, all_files):
            if tf not in seen:
                seen.add(tf)
                test_files.append(tf)
        for rel, entry in index.files.items():
            if _is_test_file(rel) and any(d in affected_set for d in entry.imports):
                if rel not in seen:
                    seen.add(rel)
                    test_files.append(rel)

    source_covered = len(closure)
    notes: list[str] = []

    if not test_files:
        notes.append(
            "No companion test files found for the changed sources; "
            "falling back to the full test suite."
        )
        return TestSelectionResult(
            node_ids=[],
            has_targets=False,
            source_files_covered=source_covered,
            notes=notes,
        )
    elif len(test_files) > MAX_TARGETS:
        notes.append(
            f"{len(test_files)} candidate tests exceed the practical target "
            "limit; falling back to the full suite for correctness."
        )
        return TestSelectionResult(
            node_ids=[],
            has_targets=False,
            source_files_covered=source_covered,
            notes=notes,
        )

    return TestSelectionResult(
        node_ids=test_files,
        has_targets=True,
        source_files_covered=source_covered,
        notes=notes,
    )


def test_targets_from_files(
    repo_root: Path,
    affected_files: list[str],
    *,
    index: Optional[RepoIndex] = None,
) -> list[str]:
    """Convenience wrapper: return just the ordered pytest node ids."""
    return select_targeted_tests(
        repo_root, affected_files, index=index
    ).node_ids

