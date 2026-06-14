"""Tests for tokendrift.corpus.loaders."""

from __future__ import annotations

import pytest

from tokendrift.corpus.loaders import load_corpus


def _write(tmp_path, name: str, content: str):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


def test_jsonl_basic(tmp_path):
    p = _write(
        tmp_path,
        "c.jsonl",
        '{"id": "p001", "text": "hello"}\n{"id": "p002", "text": "world"}\n',
    )
    entries = load_corpus(p)
    assert [e.id for e in entries] == ["p001", "p002"]
    assert [e.text for e in entries] == ["hello", "world"]


def test_jsonl_metadata_collected(tmp_path):
    p = _write(tmp_path, "c.jsonl", '{"text": "hi", "source": "prod", "user": "u1"}\n')
    entries = load_corpus(p)
    assert entries[0].metadata == {"source": "prod", "user": "u1"}


def test_jsonl_id_autoassigned_when_absent(tmp_path):
    p = _write(tmp_path, "c.jsonl", '{"text": "a"}\n{"text": "b"}\n')
    entries = load_corpus(p)
    assert [e.id for e in entries] == ["entry_0", "entry_1"]


def test_jsonl_blank_id_falls_back(tmp_path):
    p = _write(tmp_path, "c.jsonl", '{"id": "", "text": "a"}\n')
    entries = load_corpus(p)
    assert entries[0].id == "entry_0"


def test_jsonl_skips_blank_lines(tmp_path):
    p = _write(tmp_path, "c.jsonl", '{"text": "a"}\n\n   \n{"text": "b"}\n')
    assert len(load_corpus(p)) == 2


def test_jsonl_missing_text_raises(tmp_path):
    p = _write(tmp_path, "c.jsonl", '{"id": "p1"}\n')
    with pytest.raises(ValueError, match="text"):
        load_corpus(p)


def test_jsonl_invalid_json_raises_with_lineno(tmp_path):
    p = _write(tmp_path, "c.jsonl", '{"text": "ok"}\nnot json\n')
    with pytest.raises(ValueError, match="line 2"):
        load_corpus(p)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def test_csv_basic(tmp_path):
    p = _write(tmp_path, "c.csv", "id,text\np1,hello\np2,world\n")
    entries = load_corpus(p)
    assert [e.id for e in entries] == ["p1", "p2"]
    assert entries[0].text == "hello"


def test_csv_extra_columns_become_metadata(tmp_path):
    p = _write(tmp_path, "c.csv", "text,source\nhi,prod\n")
    entries = load_corpus(p)
    assert entries[0].metadata == {"source": "prod"}


def test_csv_blank_id_falls_back(tmp_path):
    p = _write(tmp_path, "c.csv", "id,text\n,hello\n")
    entries = load_corpus(p)
    assert entries[0].id == "entry_0"


def test_csv_missing_text_column_raises(tmp_path):
    p = _write(tmp_path, "c.csv", "id,prompt\np1,hello\n")
    with pytest.raises(ValueError, match="text"):
        load_corpus(p)


def test_tsv_delimiter(tmp_path):
    p = _write(tmp_path, "c.tsv", "id\ttext\np1\thello world\n")
    entries = load_corpus(p)
    assert entries[0].text == "hello world"


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------


def test_text_one_entry_per_line(tmp_path):
    p = _write(tmp_path, "c.txt", "first line\nsecond line\n")
    entries = load_corpus(p)
    assert [e.text for e in entries] == ["first line", "second line"]
    assert entries[0].id == "entry_0"


def test_text_skips_blank_lines(tmp_path):
    p = _write(tmp_path, "c.txt", "a\n\nb\n")
    assert len(load_corpus(p)) == 2


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nope.jsonl")
