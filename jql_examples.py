"""
JQL example retriever — parses JQL example files and returns the most relevant
examples for a given natural language query.

Uses token-overlap scoring (no external dependencies, no vector DB).

Default dataset: data/ folder (all .md/.json/.csv files loaded and merged)
Custom dataset : --examples-folder PATH  |  --examples-file PATH
                 JQL_EXAMPLES_FOLDER env var  |  JQL_EXAMPLES_FILE env var

Supported file formats:
  .md   — markdown with ```jql blocks containing -- N. Description / jql lines
  .json — [{"description": "...", "jql": "..."}] or [["description", "jql"], ...]
  .csv  — two columns: description,jql  (header row optional)
"""

import csv
import json
import re
from pathlib import Path

_DEFAULT_DATA_FOLDER = Path(__file__).parent / "data"

_STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "all", "any", "some", "no", "not", "that", "this", "with",
    "from", "by", "as", "it", "its", "me", "my", "i", "list", "show",
    "get", "find", "give", "return", "fetch",
}


# ── Parsers ──────────────────────────────────────────────────────────

def _parse_md(path: Path) -> list[tuple[str, str]]:
    """Parse markdown file with ```jql blocks containing -- N. desc / jql lines."""
    examples: list[tuple[str, str]] = []
    in_block = False
    pending_desc: str | None = None
    pending_jql_lines: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()

        if stripped.startswith("```"):
            if not in_block:
                in_block = True
                pending_desc = None
                pending_jql_lines = []
            else:
                if pending_jql_lines:
                    jql = " ".join(pending_jql_lines).strip()
                    examples.append((pending_desc or jql, jql))
                in_block = False
            continue

        if in_block:
            m = re.match(r"^--\s*\d+\.\s*(.+)$", stripped)
            if m:
                if pending_jql_lines and pending_desc is not None:
                    examples.append((pending_desc, " ".join(pending_jql_lines).strip()))
                    pending_jql_lines = []
                pending_desc = m.group(1).strip()
            elif stripped and not stripped.startswith("--"):
                pending_jql_lines.append(stripped)

    return examples


def _parse_json(path: Path) -> list[tuple[str, str]]:
    """Parse JSON: list of {description, jql} dicts or [desc, jql] lists."""
    data = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    for item in data:
        if isinstance(item, dict):
            desc = item.get("description") or item.get("nl") or item.get("query", "")
            jql  = item.get("jql") or item.get("JQL", "")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            desc, jql = str(item[0]), str(item[1])
        else:
            continue
        if jql:
            examples.append((desc or jql, jql))
    return examples


def _parse_csv(path: Path) -> list[tuple[str, str]]:
    """Parse CSV with two columns: description, jql (header row optional)."""
    examples = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) < 2:
                continue
            desc, jql = row[0].strip(), row[1].strip()
            # Skip header row if it looks like column names
            if i == 0 and jql.lower() in ("jql", "query", "jql_query"):
                continue
            if jql:
                examples.append((desc or jql, jql))
    return examples


_SUPPORTED_SUFFIXES = {".md", ".json", ".csv"}


def _load(path: Path) -> list[tuple[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _parse_json(path)
    if suffix == ".csv":
        return _parse_csv(path)
    return _parse_md(path)


def _load_folder(folder: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """Load all supported files from a folder. Returns (examples, loaded_file_names)."""
    examples: list[tuple[str, str]] = []
    loaded: list[str] = []
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in _SUPPORTED_SUFFIXES:
            try:
                batch = _load(f)
                examples.extend(batch)
                loaded.append(f.name)
            except Exception:
                pass  # skip unparseable files silently
    return examples, loaded


# ── ExampleStore ─────────────────────────────────────────────────────

class ExampleStore:
    """Holds a loaded set of (description, jql) examples and supports retrieval."""

    def __init__(
        self,
        path:   str | Path | None = None,
        folder: str | Path | None = None,
    ):
        if folder:
            resolved_folder = Path(folder)
            if resolved_folder.is_dir():
                self._examples, loaded = _load_folder(resolved_folder)
                self._path        = resolved_folder
                self._source_desc = f"folder '{resolved_folder.name}' ({len(loaded)} file(s): {', '.join(loaded)})"
            else:
                self._examples    = []
                self._path        = resolved_folder
                self._source_desc = f"folder '{resolved_folder}' (not found)"
        elif path:
            resolved = Path(path)
            if resolved.exists():
                self._examples    = _load(resolved)
                self._path        = resolved
                self._source_desc = resolved.name
            else:
                self._examples    = []
                self._path        = resolved
                self._source_desc = f"'{resolved}' (not found)"
        else:
            # Default: load all files from the data/ folder
            if _DEFAULT_DATA_FOLDER.is_dir():
                self._examples, loaded = _load_folder(_DEFAULT_DATA_FOLDER)
                self._path        = _DEFAULT_DATA_FOLDER
                self._source_desc = f"folder '{_DEFAULT_DATA_FOLDER.name}' ({len(loaded)} file(s): {', '.join(loaded)})"
            else:
                self._examples    = []
                self._path        = _DEFAULT_DATA_FOLDER
                self._source_desc = f"folder '{_DEFAULT_DATA_FOLDER}' (not found)"

    def count(self) -> int:
        return len(self._examples)

    def path(self) -> Path:
        return self._path

    def source_desc(self) -> str:
        return self._source_desc

    def find(self, query: str, n: int = 5) -> list[tuple[str, str]]:
        """Return top-n (description, jql) pairs most relevant to query."""
        if not self._examples:
            return []

        q_tokens = _tokenize(query)
        if not q_tokens:
            return self._examples[:n]

        scored: list[tuple[float, tuple[str, str]]] = []
        for desc, jql in self._examples:
            overlap   = len(q_tokens & _tokenize(desc))
            jql_boost = len(q_tokens & _tokenize(jql)) * 0.5
            score     = overlap + jql_boost
            if score > 0:
                scored.append((score, (desc, jql)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:n]]


# ── Module-level default store (lazy, uses env var if set) ───────────

_default_store: ExampleStore | None = None


def _get_default_store() -> ExampleStore:
    global _default_store
    if _default_store is None:
        import os
        env_folder = os.getenv("JQL_EXAMPLES_FOLDER")
        env_file   = os.getenv("JQL_EXAMPLES_FILE")
        _default_store = ExampleStore(path=env_file, folder=env_folder)
    return _default_store


def find_examples(query: str, n: int = 5) -> list[tuple[str, str]]:
    return _get_default_store().find(query, n)


def example_count() -> int:
    return _get_default_store().count()


# ── Tokenizer ────────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in _STOP_WORDS and len(t) > 1}
