"""Tests for smart file retrieval (Roadmap Step 1.2)."""
from __future__ import annotations

from pathlib import Path

import pytest

import src.core.retrieval as retrieval
from src.core.repo_index import RepoIndex
from src.core.retrieval import resolve_affected_files, select_files, tokenize


def _write(root: Path, rel: str, content: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_tokenize_splits_snake_and_camel_case():
    tokens = set(tokenize("Fix call_llm and CamelCaseParser in HTTPServer"))
    assert {
        "fix",
        "call_llm", "call", "llm",                      # snake_case -> whole + parts
        "camelcaseparser", "camel", "case", "parser",   # CamelCase -> whole + parts
        "httpserver", "http", "server",                 # acronym boundary handled
    } <= tokens


class TestSelectFiles:
    def test_golden_case_bm25_beats_alphabetical_order(self, tmp_path: Path):
        """The right file sorts LAST alphabetically but BM25 must rank it #1."""
        # 20 alphabetically-early filler files about unrelated topics.
        for i in range(20):
            _write(
                tmp_path,
                f"core/util_{i:02d}.py",
                "def format_string(value):\n"
                "    '''generic string config helper'''\n"
                f"    return value.strip() + '{i}'\n",
            )
        # The target: alphabetically last, topically exact.
        _write(
            tmp_path,
            "payments/z_refund_engine.py",
            "class RefundProcessor:\n"
            "    def refund_payment(self, charge_id):\n"
            "        '''Reverse a chargeback through the payment gateway.'''\n"
            "        return self.gateway.refund(charge_id)\n",
        )

        index = RepoIndex.build(tmp_path)
        issue = "Refund a failed payment through the gateway after a chargeback"

        selected = select_files(issue, index, k=5)

        assert selected, "retrieval must return the target file"
        assert selected[0] == "payments/z_refund_engine.py"
        naive = sorted(p for p in index.files)  # what an alphabetical walk would take
        assert "payments/z_refund_engine.py" not in naive[:5]

    def test_import_graph_boosts_connected_files(self, tmp_path: Path):
        """A low-lexical file connected to a hot file via imports gets boosted."""
        _write(
            tmp_path,
            "models/invoice.py",
            "class Invoice:\n"
            "    def __init__(self, line_items):\n"
            "        self.total = sum(line_items)\n",
        )
        _write(
            tmp_path,
            "api/billing_api.py",
            "from models.invoice import Invoice\n\n\n"
            "def create_invoice():\n    return Invoice([])\n",
        )
        for i in range(10):
            _write(
                tmp_path,
                f"misc/filler_{i}.py",
                f"# unrelated note {i}\ndef placeholder_{i}():\n    return None\n",
            )

        index = RepoIndex.build(tmp_path)
        ranking = select_files("invoice total calculation is wrong", index, k=4)

        assert "models/invoice.py" in ranking[:2]
        assert "api/billing_api.py" in ranking, "the dependent file must be boosted into the results"

    def test_respects_k_and_env_override(self, tmp_path: Path, monkeypatch):
        for i in range(8):
            _write(tmp_path, f"f{i}.py", f"def thing_{i}(payment):\n    return payment\n")
        index = RepoIndex.build(tmp_path)

        assert len(select_files("payment thing", index, k=3)) == 3

        monkeypatch.setenv("PATCHCRAFT_RETRIEVAL_K", "2")
        assert len(select_files("payment thing", index)) == 2

        monkeypatch.setenv("PATCHCRAFT_RETRIEVAL_K", "not-a-number")
        # graceful fallback to the default; only 8 files exist so all match
        assert len(select_files("payment thing", index)) == 8

    def test_empty_query_or_index_return_empty_list(self, tmp_path: Path):
        index = RepoIndex.build(tmp_path)  # empty repo
        assert select_files("anything", index) == []

        _write(tmp_path, "a.py", "x = 1\n")
        index = RepoIndex.build(tmp_path)
        assert select_files("", index) == []
        assert select_files("   ", index) == []

class TestResolveAffectedFiles:
    def _repo(self, tmp_path: Path) -> Path:
        _write(tmp_path, "src/payments/refund.py", "class RefundProcessor:\n    pass\n")
        _write(tmp_path, "src/other/thing.py", "x = 1\n")
        return tmp_path

    def test_exact_paths_are_kept(self, tmp_path: Path):
        root = self._repo(tmp_path)
        out = resolve_affected_files(root, ["src/payments/refund.py"])
        assert out == ["src/payments/refund.py"]

    def test_normalizes_dot_slash_and_backslashes(self, tmp_path: Path):
        root = self._repo(tmp_path)
        out = resolve_affected_files(root, [".\\src\\payments\\refund.py"])
        assert out == ["src/payments/refund.py"]

    def test_unknown_basename_fuzzy_matched_uniquely(self, tmp_path: Path):
        root = self._repo(tmp_path)
        RepoIndex.build(root)
        # agent hallucinated a directory but the basename is unique
        out = resolve_affected_files(root, ["services/refund.py"])
        assert out == ["src/payments/refund.py"]

    def test_unique_symbol_match_fallback(self, tmp_path: Path):
        root = self._repo(tmp_path)
        RepoIndex.build(root)
        out = resolve_affected_files(root, ["somewhere/RefundProcessor.py"])
        assert out == ["src/payments/refund.py"]

    def test_unresolvable_and_escape_paths_discarded(self, tmp_path: Path):
        root = self._repo(tmp_path)
        RepoIndex.build(root)
        out = resolve_affected_files(
            root,
            ["../outside.py", "does/not/exist.py", "", "   ", "src/other/thing.py"],
        )
        assert out == ["src/other/thing.py"]

    def test_duplicates_removed(self, tmp_path: Path):
        root = self._repo(tmp_path)
        out = resolve_affected_files(root, ["src/payments/refund.py", "./src/payments/refund.py"])
        assert out == ["src/payments/refund.py"]


class TestPerformance:
    def test_select_files_under_300ms_on_1000_file_repo(self, tmp_path: Path):
        """Roadmap constraint: retrieval adds <300ms on a 1,000-file repo."""
        for i in range(1000):
            d = tmp_path / f"pkg{i // 100}"
            d.mkdir(exist_ok=True)
            (d / f"mod_{i}.py").write_text(
                f"def handler_{i}(request):\n    'generic handler {i}'\n    return request\n",
                encoding="utf-8",
            )
        index = RepoIndex.build(tmp_path)

        import time

        start = time.perf_counter()
        selected = select_files("handler 512 payment refund gateway", index, k=12)
        elapsed = time.perf_counter() - start

        assert len(selected) <= 12
        assert elapsed < 0.3, f"retrieval took {elapsed:.3f}s (>300ms)"
