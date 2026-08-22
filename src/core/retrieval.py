"""Smart file retrieval: pick the files that matter for an issue (Step 1.2).

Two-stage ranking over a :class:`~src.core.repo_index.RepoIndex`:

1. **Lexical (BM25)** — the issue text is scored against each file's symbol
   names and path (cheap, no I/O), then against the full content of the top
   candidates only (bounded reads keep latency low).
2. **Structural** — files connected through the import graph to high-scoring
   files receive a damped boost, both along dependencies (what hot files use)
   and dependents (who uses hot files).

Pure Python, no external services. ``select_files`` is the public entry point;
``resolve_affected_files`` validates/fuzzy-matches agent-declared paths.
"""
from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Optional

from src.core.repo_index import RepoIndex

logger = logging.getLogger(__name__)

__all__ = ["tokenize", "select_files", "resolve_affected_files", "DEFAULT_RETRIEVAL_K"]

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
DEFAULT_RETRIEVAL_K = 12          # files returned when PATCHCRAFT_RETRIEVAL_K is unset
MAX_CONTENT_CHARS_PER_FILE = 20_000   # content read cap per candidate in pass 2
SYMBOL_DOC_WEIGHT = 3             # symbol/path tokens repeated this often per doc
BM25_K1 = 1.5
BM25_B = 0.75
IMPORT_BOOST_DEPS = 0.35          # damping for dependencies of hot files
IMPORT_BOOST_DEPENDENTS = 0.25    # damping for dependents of hot files


def _retrieval_k() -> int:
    raw = os.getenv("PATCHCRAFT_RETRIEVAL_K")
    if not raw or not raw.strip():
        return DEFAULT_RETRIEVAL_K
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("Invalid PATCHCRAFT_RETRIEVAL_K=%r — using default %d.", raw, DEFAULT_RETRIEVAL_K)
        return DEFAULT_RETRIEVAL_K


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_SUBTOKEN_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens plus camelCase/snake_case sub-tokens."""
    tokens: list[str] = []
    for raw in _WORD_RE.findall(text):
        tokens.append(raw.lower())
        if any(ch.isupper() for ch in raw) or "_" in raw:
            for part in _SUBTOKEN_RE.findall(raw):
                if len(part) > 1:
                    tokens.append(part.lower())
    return tokens


# ---------------------------------------------------------------------------
# BM25 scoring
# ---------------------------------------------------------------------------
def _bm25_scores(query_tokens: list[str], docs: dict[str, list[str]]) -> dict[str, float]:
    """Classic Okapi BM25 scores of ``query_tokens`` against every document."""
    doc_counters = {rel: Counter(tokens) for rel, tokens in docs.items()}
    n_docs = len(doc_counters)
    if n_docs == 0:
        return {}
    total_len = sum(len(tokens) for tokens in docs.values())
    avgdl = (total_len / n_docs) or 1.0

    df: Counter = Counter()
    for counter in doc_counters.values():
        df.update(counter.keys())

    scores: dict[str, float] = {}
    for rel, counter in doc_counters.items():
        doc_len = len(docs[rel])
        norm = BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_len / avgdl))
        score = 0.0
        for term in set(query_tokens):
            tf = counter.get(term, 0)
            if tf == 0:
                continue
            idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * tf * (BM25_K1 + 1.0) / (tf + norm)
        scores[rel] = score
    return scores


def _symbol_doc(rel_path: str, entry) -> list[str]:
    """Cheap document for pass 1: path tokens + boosted symbol-name tokens."""
    symbols_text = " ".join(
        f"{s.name} {s.signature}" for s in entry.symbols
    )
    return tokenize(f"{rel_path} {rel_path} {symbols_text}") * SYMBOL_DOC_WEIGHT


def _read_head(path: Path, max_chars: int) -> str:
    try:
        with open(path, "rb") as fh:
            data = fh.read(max_chars * 2)  # slack for multibyte characters
        return data.decode("utf-8", errors="replace")[:max_chars]
    except OSError as exc:
        logger.debug("Cannot read %s for retrieval: %s", path, exc)
        return ""


# ---------------------------------------------------------------------------
# Structural boost over the import graph
# ---------------------------------------------------------------------------
def _import_maps(files: dict) -> tuple[dict[str, set], dict[str, set]]:
    """``(imports_map: file -> what it imports, imported_by: file -> its users)``."""
    imports_map: dict[str, set] = {rel: set() for rel in files}
    imported_by: dict[str, set] = {rel: set() for rel in files}
    for rel, entry in files.items():
        for target in entry.imports:
            if target in files and target != rel:
                imports_map[rel].add(target)
                imported_by[target].add(rel)
    return imports_map, imported_by


def _apply_import_boost(
    base_scores: dict[str, float],
    imports_map: dict[str, set],
    imported_by: dict[str, set],
    hot_count: int,
) -> dict[str, float]:
    """One-hop damped propagation along the import graph from the hottest files."""
    final = dict(base_scores)
    hot = [
        rel
        for rel, score in sorted(base_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:hot_count]
        if base_scores[rel] > 0
    ]
    for rel in hot:
        weight = base_scores[rel]
        for dependency in imports_map.get(rel, ()):  # what hot files rely on
            final[dependency] = final.get(dependency, 0.0) + IMPORT_BOOST_DEPS * weight
        for dependent in imported_by.get(rel, ()):  # who relies on hot files
            final[dependent] = final.get(dependent, 0.0) + IMPORT_BOOST_DEPENDENTS * weight
    return final


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def select_files(
    issue_text: str,
    repo_index: RepoIndex,
    k: Optional[int] = None,
) -> list[str]:
    """Return the ``k`` repo-relative paths most relevant to ``issue_text``.

    Pass 1 ranks every indexed file on path + symbol tokens (no I/O); pass 2
    re-scores only the top candidates against their actual content; finally a
    damped boost propagates relevance along the import graph. Files with a
    zero score are not returned — an empty list means "nothing matched", and
    callers should fall back to their previous behavior.
    """
    if k is None:
        k = _retrieval_k()
    files = repo_index.files
    if not files:
        return []
    query_tokens = tokenize(issue_text)
    if not query_tokens:
        return []

    # -- Pass 1: symbols + paths (pure in-memory) --------------------------
    symbol_docs = {rel: _symbol_doc(rel, entry) for rel, entry in files.items()}
    symbol_scores = _bm25_scores(query_tokens, symbol_docs)

    # -- Pass 2: content of the top candidates ------------------------------
    ranked_by_symbols = sorted(files, key=lambda rel: (-symbol_scores.get(rel, 0.0), rel))
    pool = ranked_by_symbols[: max(2 * k, 40)]
    content_docs = {
        rel: tokenize(_read_head(repo_index.root / rel, MAX_CONTENT_CHARS_PER_FILE))
        for rel in pool
    }
    merged_docs = {
        rel: [*content_docs.get(rel, []), *symbol_docs[rel]]
        for rel in files
    }
    full_scores = _bm25_scores(query_tokens, merged_docs)

    # Per-file score = best of the two views (content view dilutes tf).
    base_scores = {
        rel: max(full_scores.get(rel, 0.0), symbol_scores.get(rel, 0.0))
        for rel in files
    }

    # -- Structural boost over the import graph ------------------------------
    imports_map, imported_by = _import_maps(files)
    final_scores = _apply_import_boost(base_scores, imports_map, imported_by, hot_count=max(k, 5))

    ranked = sorted(files, key=lambda rel: (-final_scores.get(rel, 0.0), rel))
    selected = [rel for rel in ranked if final_scores.get(rel, 0.0) > 0][:k]
    if selected:
        logger.debug(
            "Retrieval selected %d/%d files for issue (%d chars).",
            len(selected), len(files), len(issue_text),
        )
    return selected


def resolve_affected_files(
    repo_root: Path,
    affected_files: list[str],
    index: Optional[RepoIndex] = None,
) -> list[str]:
    """Validate agent-declared paths and fuzzy-match unknown ones via the index.

    Resolution order per entry: exact relative path inside the repo →
    basename match against the indexed file set → unique symbol-name match.
    Entries that cannot be resolved are dropped with a debug log (never
    silently applied to a wrong file).
    """
    root = Path(repo_root).resolve()
    if index is None:
        cached, ok = RepoIndex.load_cached(root)
        index = RepoIndex(root, cached) if ok else None

    resolved: list[str] = []
    seen: set[str] = set()
    for raw in affected_files:
        candidate_raw = raw.strip().replace("\\", "/")
        if not candidate_raw:
            continue
        while candidate_raw.startswith("./"):
            candidate_raw = candidate_raw[2:]

        target = (root / candidate_raw)
        try:
            resolved_target = target.resolve()
            root_resolved = root.resolve()
            inside = (
                resolved_target == root_resolved or root_resolved in resolved_target.parents
            )
        except OSError:
            inside = False

        if inside and target.is_file():
            rel = target.relative_to(root).as_posix()
        else:
            # Fuzzy matching against the index.
            rel = None
            if index is not None:
                base = PurePosixPath(candidate_raw).name
                basename_hits = [
                    p for p in sorted(index.files)
                    if p.split("/")[-1] == base
                ]
                if len(basename_hits) == 1:
                    rel = basename_hits[0]
                    logger.debug("Affected path '%s' fuzzy-matched to '%s' (basename).", raw, rel)
                else:
                    stem = PurePosixPath(base).stem.lower()
                    hits = index.symbols(stem, limit=2) if stem else []
                    if len(hits) == 1:
                        rel = hits[0].path
                        logger.debug("Affected path '%s' fuzzy-matched to '%s' (symbol).", raw, rel)
            if rel is None:
                logger.debug("Affected path '%s' could not be resolved; discarded.", raw)
                continue

        if rel not in seen:
            seen.add(rel)
            resolved.append(rel)
    return resolved

