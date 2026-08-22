"""Tests for the repository map & symbol index (Roadmap Step 1.1)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

import src.core.repo_index as ri
from src.core.repo_index import (
    IGNORED_DIR_NAMES,
    INDEX_DIR_NAME,
    RepoIndex,
    extract_js,
    extract_python,
)


def _make_fixture_repo(root: Path) -> None:
    """Create a deterministic 30+-file repo with Python and TS sources."""
    pkg = root / "src" / "myapp"
    pkg_models = pkg / "models"
    pkg_services = pkg / "services"
    pkg_models.mkdir(parents=True)
    pkg_services.mkdir(parents=True)

    template = (
        '"""Module {i}."""\n'
        "class Handler{i}:\n"
        '    """Handler."""\n'
        "\n"
        "    def handle(self, payload, retries=3):\n"
        "        return payload\n"
        "\n"
        "\n"
        "def run_task_{i}(job_id, *, dry=False):\n"
        "    return job_id\n"
    )
    for i in range(24):
        (pkg / f"module_{i:02d}.py").write_text(template.format(i=i), encoding="utf-8")
        (pkg_services / f"svc_{i:02d}.py").write_text(
            f"def service_call_{i}():\n    return {i}\n", encoding="utf-8"
        )

    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg_models / "user.py").write_text(
        "class User:\n"
        "    def rename(self, new_name):\n"
        "        self.name = new_name\n",
        encoding="utf-8",
    )
    (root / "web").mkdir()
    (root / "web" / "index.ts").write_text(
        "// bootstrap comment\n"
        "export function bootstrap(app) {\n  return app;\n}\n"
        "export class Router {}\n"
        "const handler = (req, res) => res;\n",
        encoding="utf-8",
    )
    for ignored in ("node_modules", ".venv", "__pycache__", ".git"):
        d = root / ignored
        d.mkdir(exist_ok=True)
        (d / f"junk_{ignored}.py").write_text(
            "def should_not_appear():\n    pass\n", encoding="utf-8"
        )
    (root / "README.md").write_text("# Fixture\n", encoding="utf-8")


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    _make_fixture_repo(tmp_path)
    return tmp_path


class TestExtraction:
    def test_extract_python_symbols(self):
        source = (
            "class Base:\n"
            "    '''doc'''\n"
            "    def method(self, x):\n"
            "        return x\n"
            "\n"
            "def free(a, b=2, *, key='k'):\n"
            "    return a\n"
            "\n"
            "async def afetch(url):\n"
            "    ...\n"
        )
        symbols = extract_python(source)
        by_name = {s.name: s for s in symbols}
        assert set(by_name) == {"Base", "method", "free", "afetch"}
        assert by_name["Base"].kind == "class"
        assert by_name["method"].kind == "method"
        assert by_name["free"].signature == "def free(a, b=2, *, key='k')"
        assert by_name["afetch"].signature.startswith("async def afetch")
        for s in symbols:
            assert s.line_start >= 1
            assert s.line_end >= s.line_start

    def test_extract_python_survives_syntax_errors(self):
        assert extract_python("def broken(:\n    pass\n") == []

    def test_extract_js_declarations(self):
        source = (
            "// comment mentioning function helper\n"
            "export function helper(a) { return a }\n"
            "export class Widget extends Base {}\n"
            "const arrow = (x, y) => x + y;\n"
            "const legacy = function (z) { return z };\n"
        )
        symbols = extract_js(source)
        by_name = {s.name: s for s in symbols}
        assert {"helper", "Widget", "arrow", "legacy"} <= set(by_name)
        assert by_name["helper"].kind == "function"
        assert by_name["Widget"].kind == "class"

class TestBuild:
    def test_build_indexes_all_source_files_and_persists(self, repo: Path):
        index = RepoIndex.build(repo)
        py_files = [p for p in index.files if p.endswith(".py")]
        js_files = [p for p in index.files if p.endswith((".ts", ".js"))]
        assert len(py_files) >= 26  # 24 modules + 24 services + __init__ + user.py
        assert len(js_files) == 1   # web/index.ts

        assert (repo / INDEX_DIR_NAME / "index.json").is_file()
        payload = json.loads(
            (repo / INDEX_DIR_NAME / "index.json").read_text(encoding="utf-8")
        )
        assert payload["version"] == RepoIndex.VERSION
        assert set(payload["files"]) == set(index.files)

    def test_ignored_dirs_never_indexed(self, repo: Path):
        index = RepoIndex.build(repo)
        assert all(p.split("/")[0] not in IGNORED_DIR_NAMES for p in index.files)
        flat_names = {s.name for e in index.files.values() for s in e.symbols}
        assert "should_not_appear" not in flat_names

    def test_missing_repo_raises(self, tmp_path: Path):
        with pytest.raises(NotADirectoryError):
            RepoIndex.build(tmp_path / "nope")

    def test_incremental_rebuild_reuses_cached_symbols(self, repo: Path):
        first = RepoIndex.build(repo)
        cached_entry = first.files["src/myapp/models/user.py"]

        with mock.patch.object(ri, "extract_python", wraps=ri.extract_python) as spy:
            second = RepoIndex.build(repo)

        assert spy.call_count == 0  # nothing changed -> zero re-parsing
        assert second.files["src/myapp/models/user.py"] == cached_entry

    def test_incremental_rebuild_reindexes_only_changed_file(self, repo: Path):
        RepoIndex.build(repo)
        target = repo / "src" / "myapp" / "module_00.py"
        target.write_text("def brand_new_function():\n    pass\n", encoding="utf-8")

        with mock.patch.object(ri, "extract_python", side_effect=ri.extract_python) as spy:
            refreshed = RepoIndex.build(repo)

        assert spy.call_count == 1  # exactly the changed file was re-parsed
        names = {s.name for s in refreshed.files["src/myapp/module_00.py"].symbols}
        assert "brand_new_function" in names

    def test_deleted_files_drop_out_of_the_index(self, repo: Path):
        RepoIndex.build(repo)
        assert "src/myapp/module_05.py" in ri.RepoIndex.load_cached(repo)[0]
        (repo / "src" / "myapp" / "module_05.py").unlink()
        fresh = RepoIndex.build(repo)
        assert "src/myapp/module_05.py" not in fresh.files

    def test_corrupt_cache_is_rebuilt_silently(self, repo: Path):
        RepoIndex.build(repo)
        (repo / INDEX_DIR_NAME / "index.json").write_text("{not json", encoding="utf-8")
        index = RepoIndex.build(repo)
        assert index.files  # rebuilt from scratch without raising

class TestQueries:
    def test_symbols_substring_search(self, repo: Path):
        index = RepoIndex.build(repo)
        hits = index.symbols("handler")
        assert hits, "expected Handler* classes to be found"
        assert all("handler" in h.symbol.name.lower() for h in hits)
        limited = index.symbols("run_task", limit=5)
        assert len(limited) == 5
        assert index.symbols("") == []

    def test_file_summary_format(self, repo: Path):
        index = RepoIndex.build(repo)
        summary = index.file_summary("src/myapp/models/user.py")
        assert summary.startswith("src/myapp/models/user.py :: ")
        assert "class User" in summary
        assert "def rename(self, new_name) L2-3" in summary
        assert index.file_summary("does/not/exist.py") == "does/not/exist.py :: (not indexed)"

    def test_repo_map_budget_and_content(self, repo: Path):
        index = RepoIndex.build(repo)
        small = index.repo_map(max_chars=600)
        assert small.startswith("Repository root:")
        assert "... (repository map truncated)" in small
        assert len(small) < 700

        full = index.repo_map()
        assert "web/index.ts" in full
        assert (
            "src/myapp/models/user.py :: class User L1-3 "
            "| def rename(self, new_name) L2-3" in full
        )
        assert "truncated" not in full


class TestContextIntegration:
    def test_build_context_includes_repository_map(self, tmp_path: Path):
        from src.orchestrator import build_context

        _make_fixture_repo(tmp_path)
        context = build_context(tmp_path, "Fix the user rename bug")
        assert "# REPOSITORY MAP" in context
        assert "src/myapp/models/user.py" in context
        assert "# ISSUE TO SOLVE" in context
        assert "Fix the user rename bug" in context
        # existing behavior preserved: raw sources still included
        assert "new_name" in context

    def test_build_context_stays_within_budget_with_map(self, tmp_path: Path):
        from src.orchestrator import MAX_CONTEXT_CHARS, build_context

        _make_fixture_repo(tmp_path)
        context = build_context(tmp_path, "issue text here")
        # small slack for section join overhead
        assert len(context) <= MAX_CONTEXT_CHARS + 200

    def test_build_context_survives_index_failure(self, tmp_path: Path):
        from src.orchestrator import build_context

        _make_fixture_repo(tmp_path)
        with mock.patch.object(ri.RepoIndex, "build", side_effect=OSError("disk full")):
            context = build_context(tmp_path, "issue")
        assert "# ISSUE TO SOLVE" in context
        assert "# REPOSITORY MAP" not in context

