"""Repository map & symbol index (Roadmap Step 1.1).

Builds a lightweight structural map of a target repository without executing
any of its code:

* **Python files** are parsed with the standard :mod:`ast` module to extract
  classes, functions and methods with signatures and line ranges.
* **JavaScript/TypeScript files** use a conservative regex extractor
  (function/class declarations and arrow-function constants).
* The index is persisted at ``<repo>/.patchcraft/index.json`` and keyed by
  file content hash, so subsequent builds only re-parse changed files
  (incremental indexing).

The compact map is prepended to the diagnosis context so the LLM gets a
structural overview of the whole repository instead of an alphabetical dump
of the first N files.
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator, Optional, Sequence, Union


from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "SymbolEntry",
    "FileEntry",
    "SymbolHit",
    "RepoIndex",
    "INDEX_DIR_NAME",
]

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
INDEX_DIR_NAME = ".patchcraft"
INDEX_FILE_NAME = "index.json"

# v2: FileEntry gained the `imports` field (import graph for retrieval).
INDEX_VERSION = 2

#: Directories never indexed (superset of the orchestrator's ignore rules).
IGNORED_DIR_NAMES = {
    ".git", ".hg", ".svn", ".patchcraft",
    "node_modules", "__pycache__", ".venv", "venv", ".env", "env",
    "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".idea", ".vscode", ".tox", ".nox", "coverage", ".next", ".nuxt",
}

#: File extensions handled by the Python AST extractor.
PYTHON_EXTENSIONS = {".py", ".pyi"}

#: File extensions handled by the regex extractor.
JS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}

#: Maximum symbols rendered per file in the compact repo map.
MAX_SYMBOLS_PER_FILE_IN_MAP = 12

#: Default character budget for the rendered repo map.
DEFAULT_MAP_MAX_CHARS = 8_000


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
class SymbolEntry(BaseModel):
    """A single extracted symbol (class, function or method)."""

    kind: str = Field(description='"class", "function" or "method".')
    name: str = Field(description="Symbol name.")
    signature: str = Field(description='Compact signature, e.g. call_llm(model, *, timeout=60.0).')
    line_start: int = Field(ge=1, description="First line (1-based).")
    line_end: int = Field(ge=1, description="Last line (1-based, inclusive).")


class FileEntry(BaseModel):
    """Indexed state of one source file."""

    path: str = Field(description="Path relative to the repo root (POSIX separators).")
    hash: str = Field(description="SHA-256 of the file content (hex).")
    symbols: list[SymbolEntry] = Field(default_factory=list)
    imports: list[str] = Field(
        default_factory=list,
        description=(
            "Repo-relative paths this file imports (Python only; resolved "
            "against the indexed file set). Powers the retrieval import graph."
        ),
    )


class SymbolHit(BaseModel):
    """A symbol search result, carrying its owning file."""

    path: str
    symbol: SymbolEntry


# ---------------------------------------------------------------------------
# Python extraction (pure AST — never executes target code)
# ---------------------------------------------------------------------------
def _format_arguments(args: ast.arguments) -> str:
    """Render an :class:`ast.arguments` node as a compact signature tail."""
    parts: list[str] = []
    defaults: list[Optional[ast.expr]] = list(args.defaults)
    pos_args = [*args.posonlyargs, *args.args]
    first_default_at = len(pos_args) - len(defaults)

    for i, a in enumerate(pos_args):
        if i >= first_default_at:
            d = defaults[i - first_default_at]
            rendered = ast.unparse(d) if d is not None else "..."
            parts.append(f"{a.arg}={rendered}")
        else:
            parts.append(a.arg)
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        if d is not None:
            parts.append(f"{a.arg}={ast.unparse(d)}")
        else:
            parts.append(a.arg)
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return ", ".join(parts)


def _symbol_from_function(node: ast.FunctionDef | ast.AsyncFunctionDef, kind: str) -> SymbolEntry:
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    return SymbolEntry(
        kind=kind,
        name=node.name,
        signature=f"{prefix}{node.name}({_format_arguments(node.args)})",
        line_start=node.lineno,
        line_end=node.end_lineno or node.lineno,
    )


def _symbol_from_class(node: ast.ClassDef) -> SymbolEntry:
    bases = ", ".join(ast.unparse(b) for b in node.bases)
    tail = f"({bases})" if bases else ""
    return SymbolEntry(
        kind="class",
        name=node.name,
        signature=f"class {node.name}{tail}",
        line_start=node.lineno,
        line_end=node.end_lineno or node.lineno,
    )


def _parse_python(source: str) -> Optional[ast.Module]:
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        logger.debug("SyntaxError while indexing Python source: %s", exc)
        return None


def _symbols_from_tree(tree: ast.Module) -> list[SymbolEntry]:
    symbols: list[SymbolEntry] = []

    def walk(body: list[ast.stmt], class_depth: int = 0) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "method" if class_depth > 0 else "function"
                symbols.append(_symbol_from_function(node, kind))
            elif isinstance(node, ast.ClassDef):
                symbols.append(_symbol_from_class(node))
                walk(node.body, class_depth + 1)

    walk(tree.body)
    return symbols


def _imports_from_tree(tree: ast.Module) -> list[tuple[int, str]]:
    """Collect ``(level, module)`` pairs from every Import/ImportFrom node."""
    collected: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                collected.append((0, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or (node.names[0].name if node.names else "")
            if module:
                collected.append((node.level or 0, module))
            elif node.level:
                # `from . import x` — only the relative package is known.
                collected.append((node.level or 0, ""))
    return collected


def extract_python(source: str) -> list[SymbolEntry]:
    """Extract symbols from Python source via :mod:`ast` (never executes it)."""
    tree = _parse_python(source)
    if tree is None:
        return []
    return _symbols_from_tree(tree)


# ---------------------------------------------------------------------------
# JS/TS extraction (regex fallback — conservative, declaration-level only)
# ---------------------------------------------------------------------------
_JS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # function name(...) / export function name(...) / async function name(...)
    (
        re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)\s*\("),
        "function",
    ),
    # class Name / export class Name / abstract class Name
    (
        re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)"),
        "class",
    ),
    # const name = (...) => ...  /  const name = async x => ...
    (
        re.compile(
            r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
            r"(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>"
        ),
        "function",
    ),
    # const name = async function(...) / const name = function(...)
    (
        re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?function\b"),
        "function",
    ),
]


def extract_js(source: str) -> list[SymbolEntry]:
    """Extract declaration-level symbols from JS/TS source via regex."""
    symbols: list[SymbolEntry] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "*", "/*")):
            continue
        for pattern, kind in _JS_PATTERNS:
            match = pattern.match(line)
            if match:
                symbols.append(
                    SymbolEntry(
                        kind=kind,
                        name=match.group(1),
                        signature=stripped[:120],
                        line_start=lineno,
                        line_end=lineno,
                    )
                )
                break
    return symbols


# ---------------------------------------------------------------------------
# RepoIndex
# ---------------------------------------------------------------------------
class RepoIndex:
    """Structural index of a repository, persisted for incremental rebuilds."""

    VERSION = INDEX_VERSION

    def __init__(self, root: Path, files: dict[str, FileEntry]) -> None:
        self.root = root
        self.files = files

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @property
    def index_path(self) -> Path:
        return self.root / INDEX_DIR_NAME / INDEX_FILE_NAME

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.VERSION,
            "files": {p: e.model_dump() for p, e in sorted(self.files.items())},
        }
        self.index_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    @classmethod
    def load_cached(cls, root: Path) -> tuple[dict[str, FileEntry], bool]:
        """Load a persisted index; returns ``({}, False)`` when absent/stale."""
        path = root / INDEX_DIR_NAME / INDEX_FILE_NAME
        if not path.is_file():
            return {}, False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Ignoring unreadable index cache: %s", exc)
            return {}, False
        if not isinstance(payload, dict) or payload.get("version") != cls.VERSION:
            logger.debug("Index cache version mismatch; rebuilding.")
            return {}, False
        try:
            files = {
                p: FileEntry.model_validate(entry)
                for p, entry in payload.get("files", {}).items()
            }
        except Exception as exc:  # noqa: BLE001 - corrupt cache must never crash a run
            logger.debug("Ignoring corrupt index cache: %s", exc)
            return {}, False
        return files, True

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    @staticmethod
    def _matches_ignore_globs(rel_posix: str, globs: Sequence[str]) -> bool:
        """Whether a repo-relative posix path matches any ignore glob.

        A pattern matches the full path or any single path component
        (so ``build`` ignores ``build/`` directories and ``vendor/**`` trees).
        """
        from fnmatch import fnmatch

        parts = rel_posix.split("/")
        return any(
            fnmatch(rel_posix, pattern) or any(fnmatch(part, pattern) for part in parts)
            for pattern in globs
        )

    @classmethod
    def _iter_source_files(
        cls, root: Path, ignore_globs: Sequence[str] = ()
    ) -> Iterator[Path]:
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            dirnames[:] = [
                d for d in dirnames
                if d not in IGNORED_DIR_NAMES and not d.startswith(".")
            ]
            for name in sorted(filenames):
                path = Path(dirpath) / name
                if ignore_globs:
                    rel = path.relative_to(root).as_posix()
                    if cls._matches_ignore_globs(rel, ignore_globs):
                        continue
                yield path

    @classmethod
    def _configured_ignore_globs(cls, root: Path) -> list[str]:
        """``ignore_globs`` from ``<root>/.patchcraft.yml`` (defensive)."""
        try:
            from src.core.config import load_config

            return list(load_config(root).ignore_globs)
        except Exception as exc:  # noqa: BLE001 - config must never break indexing
            logger.debug("Ignore globs unavailable (%s: %s).", type(exc).__name__, exc)
            return []

    @staticmethod
    def _hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def _analyze_file(cls, path: Path) -> tuple[list[SymbolEntry], list[tuple[int, str]]]:
        """Extract ``(symbols, raw_imports)`` from one source file."""
        suffix = path.suffix.lower()
        if suffix not in PYTHON_EXTENSIONS and suffix not in JS_EXTENSIONS:
            return [], []
        try:
            data = path.read_bytes()
        except OSError as exc:
            logger.debug("Cannot read %s: %s", path, exc)
            return [], []
        source = data.decode("utf-8", errors="replace")
        try:
            if suffix in PYTHON_EXTENSIONS:
                tree = _parse_python(source)
                if tree is None:
                    return [], []
                return _symbols_from_tree(tree), _imports_from_tree(tree)
            return extract_js(source), []  # JS/TS import graph: not supported yet
        except Exception as exc:  # noqa: BLE001 - indexing must never break a run
            logger.debug("Extraction failed for %s: %s: %s", path, type(exc).__name__, exc)
            return [], []

    @staticmethod
    def _candidate_import_targets(
        level: int, module: str, rel_path: str
    ) -> list[str]:
        """Possible repo-relative file paths for one import statement."""
        parent_parts = list(PurePosixPath(rel_path).parts[:-1])
        if level > 0:
            keep = len(parent_parts) - (level - 1)
            if keep < 0:
                return []
            base = parent_parts[:keep]
        else:
            base = []
        mod_parts = module.split(".") if module else []
        joined = "/".join([*base, *mod_parts])
        if not joined:
            return []
        return [f"{joined}.py", f"{joined}/__init__.py"]

    @classmethod
    def build(
        cls,
        repo_root: Union[str, Path],
        force: bool = False,
        ignore_globs: Optional[Sequence[str]] = None,
    ) -> "RepoIndex":
        """Build (or incrementally refresh) the index of ``repo_root``.

        Unchanged files (same content hash) reuse their cached symbols and
        imports, so a warm rebuild only re-parses what actually changed.
        Import statements are resolved to repo-relative file paths in a
        second pass, once the full file set is known.

        ``ignore_globs`` (Step 3.3) excludes matching paths from the index;
        when ``None`` the patterns are read from ``<repo>/.patchcraft.yml``
        (``ignore_globs`` key), defaulting to no extra exclusions.
        """
        root = Path(repo_root).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Repository does not exist: {root}")

        if ignore_globs is None:
            ignore_globs = cls._configured_ignore_globs(root)

        cached: dict[str, FileEntry] = {}
        if not force:
            cached, _ = cls.load_cached(root)

        files: dict[str, FileEntry] = {}
        pending_imports: dict[str, list[tuple[int, str]]] = {}
        for path in cls._iter_source_files(root, ignore_globs):
            suffix = path.suffix.lower()
            if suffix not in PYTHON_EXTENSIONS and suffix not in JS_EXTENSIONS:
                continue
            rel = path.relative_to(root).as_posix()
            try:
                digest = cls._hash_bytes(path.read_bytes())
            except OSError as exc:
                logger.debug("Cannot hash %s: %s", path, exc)
                continue
            cached_entry = cached.get(rel)
            if cached_entry is not None and cached_entry.hash == digest:
                files[rel] = cached_entry
                continue
            symbols, raw_imports = cls._analyze_file(path)
            files[rel] = FileEntry(path=rel, hash=digest, symbols=symbols)
            if raw_imports:
                pending_imports[rel] = raw_imports

        # Resolve import statements against the complete indexed file set.
        known = set(files)
        for rel, raw_imports in pending_imports.items():
            resolved: set[str] = set()
            for level, module in raw_imports:
                for target in cls._candidate_import_targets(level, module, rel):
                    if target in known and target != rel:
                        resolved.add(target)
                        break
            if resolved:
                files[rel] = files[rel].model_copy(update={"imports": sorted(resolved)})

        index = cls(root, files)
        index.save()
        return index

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def symbols(self, query: str, limit: Optional[int] = None) -> list[SymbolHit]:
        """Case-insensitive substring search over indexed symbol names."""
        needle = query.strip().lower()
        if not needle:
            return []
        hits: list[SymbolHit] = []
        for rel in sorted(self.files):
            for symbol in self.files[rel].symbols:
                if needle in symbol.name.lower():
                    hits.append(SymbolHit(path=rel, symbol=symbol))
                    if limit is not None and len(hits) >= limit:
                        return hits
        return hits

    def file_summary(self, path: str) -> str:
        """One-line compact summary of an indexed file, e.g.

        ``src/core/llm.py :: def call_llm(...) L120-330 | ...``
        """
        entry = self.files.get(path)
        if entry is None:
            return f"{path} :: (not indexed)"
        if not entry.symbols:
            return f"{entry.path} :: (no symbols)"
        rendered = [
            f"{s.signature} L{s.line_start}-{s.line_end}"
            for s in entry.symbols[:MAX_SYMBOLS_PER_FILE_IN_MAP]
        ]
        if len(entry.symbols) > MAX_SYMBOLS_PER_FILE_IN_MAP:
            rendered.append(f"... (+{len(entry.symbols) - MAX_SYMBOLS_PER_FILE_IN_MAP} more)")
        return f"{entry.path} :: " + " | ".join(rendered)

    def repo_map(self, max_chars: int = DEFAULT_MAP_MAX_CHARS) -> str:
        """Render the whole repository as a compact map within ``max_chars``.

        Files with symbols come first (they carry the structure), then files
        without symbols, both in path order. A truncation marker is appended
        when the budget runs out so consumers know the map is partial.
        """
        with_symbols = sorted(p for p, e in self.files.items() if e.symbols)
        without_symbols = sorted(p for p, e in self.files.items() if not e.symbols)
        lines: list[str] = []
        used = 0
        truncated = False
        for rel in [*with_symbols, *without_symbols]:
            line = self.file_summary(rel)
            if used + len(line) + 1 > max_chars:
                truncated = True
                break
            lines.append(line)
            used += len(line) + 1
        if not lines:
            return ""
        header = f"Repository root: {self.root.name} — {len(self.files)} indexed files"
        body = "\n".join(lines)
        marker = "\n... (repository map truncated)" if truncated else ""
        return f"{header}\n{body}{marker}"



