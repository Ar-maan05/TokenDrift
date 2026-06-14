"""
tokendrift.corpus.loaders
~~~~~~~~~~~~~~~~~~~~~~~~
Load a prompt corpus from disk into a list of ``CorpusEntry`` objects.

Supported formats
-----------------
- **JSONL**: one JSON object per line, must contain a ``"text"`` key.
  Optional ``"id"`` and ``"metadata"`` keys.
- **CSV**: must contain a ``"text"`` column.  Optional ``"id"`` column.
  All other columns are collected into ``metadata``.
- **Plain text**: one entry per line; IDs auto-assigned as ``entry_0``, etc.

Usage
-----
>>> from tokendrift.corpus.loaders import load_corpus
>>> entries = load_corpus("prompts.jsonl")
>>> entries[0].id, entries[0].text
('p001', 'What is the capital of France?')
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from tokendrift.models import CorpusEntry


def load_corpus(path: str | Path) -> list[CorpusEntry]:
    """
    Load a corpus file and return a list of ``CorpusEntry`` objects.

    The format is inferred from the file extension:
    - ``.jsonl`` or ``.ndjson``  → JSONL
    - ``.csv`` or ``.tsv``       → CSV / TSV
    - anything else              → plain text (one entry per line)

    Parameters
    ----------
    path:
        Path to the corpus file.

    Returns
    -------
    list[CorpusEntry]
        Entries in file order.  IDs are auto-assigned as ``entry_0``,
        ``entry_1``, … when no ``id`` field is present in the source file.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If a JSONL line or CSV row is missing the ``"text"`` field.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Corpus file not found: {p}")

    suffix = p.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return _load_jsonl(p)
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        return _load_csv(p, delimiter=delimiter)
    return _load_text(p)


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[CorpusEntry]:
    entries: list[CorpusEntry] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno}: {exc}") from exc

            text = obj.get("text")
            if text is None:
                raise ValueError(f"Line {lineno} is missing the required 'text' field.")
            raw_id = obj.get("id")
            entry_id = str(raw_id) if raw_id not in (None, "") else f"entry_{lineno - 1}"
            metadata = {k: v for k, v in obj.items() if k not in {"id", "text"}}
            entries.append(CorpusEntry(id=entry_id, text=str(text), metadata=metadata))
    return entries


def _load_csv(path: Path, delimiter: str = ",") -> list[CorpusEntry]:
    entries: list[CorpusEntry] = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        if reader.fieldnames is None or "text" not in reader.fieldnames:
            raise ValueError("CSV file must have a 'text' column.")
        for rowno, row in enumerate(reader):
            text = row.get("text")
            if text is None:
                raise ValueError(f"Row {rowno + 1} is missing the 'text' column.")
            raw_id = row.get("id")
            entry_id = str(raw_id) if raw_id not in (None, "") else f"entry_{rowno}"
            metadata = {k: v for k, v in row.items() if k not in {"id", "text"} and v}
            entries.append(CorpusEntry(id=entry_id, text=text, metadata=metadata))
    return entries


def _load_text(path: Path) -> list[CorpusEntry]:
    entries: list[CorpusEntry] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh):
            text = line.rstrip("\n")
            if not text:
                continue
            entries.append(CorpusEntry(id=f"entry_{lineno}", text=text))
    return entries
