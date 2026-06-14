"""CLI smoke tests using typer's CliRunner.

TokenizerLoader.load is patched to return the offline mock tokenizers so the
whole command surface can be exercised without network access.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from tests.conftest import MockTokenizerA, MockTokenizerB
from tokendrift.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def patch_loader(monkeypatch):
    def fake_load(identifier: str):
        return MockTokenizerB() if identifier.endswith("B") else MockTokenizerA()

    monkeypatch.setattr("tokendrift.cli.main.TokenizerLoader.load", staticmethod(fake_load))


@pytest.fixture
def corpus(tmp_path):
    p = tmp_path / "c.jsonl"
    # Two-char words trigger a MERGE under the char-vs-bigram mock tokenizers
    # (A: 2 char tokens, B: 1 bigram token), exercising boundary detection.
    p.write_text(
        '{"id": "p1", "text": "ab cd ef"}\n{"id": "p2", "text": "gh ij kl"}\n',
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


def test_diff_text(corpus):
    result = runner.invoke(app, ["diff", "mockA", "mockB", "--text", "Hello world", "--no-vocab"])
    assert result.exit_code == 0
    assert "Entry" in result.stdout


def test_diff_corpus(corpus):
    result = runner.invoke(app, ["diff", "mockA", "mockB", "--corpus", str(corpus), "--no-vocab"])
    assert result.exit_code == 0
    assert "Encoding Diff" in result.stdout


def test_diff_corpus_with_boundaries(corpus):
    result = runner.invoke(app, ["diff", "mockA", "mockB", "--corpus", str(corpus), "--no-vocab", "--boundaries"])
    assert result.exit_code == 0
    assert "Boundary changes" in result.stdout


def test_diff_with_cost(corpus):
    result = runner.invoke(
        app,
        ["diff", "mockA", "mockB", "--corpus", str(corpus), "--no-vocab", "--price-a", "0.03", "--price-b", "0.01"],
    )
    assert result.exit_code == 0
    assert "Cost Report" in result.stdout


def test_diff_requires_corpus_or_text():
    result = runner.invoke(app, ["diff", "mockA", "mockB"])
    assert result.exit_code == 1
    assert "provide --corpus or --text" in result.stdout


def test_diff_includes_vocab_by_default(corpus):
    result = runner.invoke(app, ["diff", "mockA", "mockB", "--text", "Hello"])
    assert result.exit_code == 0
    assert "Vocab Diff" in result.stdout


# ---------------------------------------------------------------------------
# vocab-diff / cost / entry
# ---------------------------------------------------------------------------


def test_vocab_diff():
    result = runner.invoke(app, ["vocab-diff", "mockA", "mockB"])
    assert result.exit_code == 0
    assert "Vocab Diff" in result.stdout


def test_cost(corpus):
    result = runner.invoke(
        app, ["cost", "mockA", "mockB", "--corpus", str(corpus), "--price-a", "0.03", "--price-b", "0.01"]
    )
    assert result.exit_code == 0
    assert "Cost Report" in result.stdout


def test_entry():
    result = runner.invoke(app, ["entry", "mockA", "mockB", "--text", "Hello world"])
    assert result.exit_code == 0
    assert "Entry" in result.stdout


def test_diff_empty_text():
    result = runner.invoke(app, ["diff", "mockA", "mockB", "--text", "", "--no-vocab"])
    assert result.exit_code == 0
    assert "Entry" in result.stdout


def test_diff_nonexistent_corpus():
    result = runner.invoke(app, ["diff", "mockA", "mockB", "--corpus", "/nonexistent/file.jsonl", "--no-vocab"])
    assert result.exit_code == 1
    assert "Error loading corpus:" in result.stdout


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "tokendrift version" in result.stdout


# ---------------------------------------------------------------------------
# baseline / ci
# ---------------------------------------------------------------------------


def test_baseline_writes_file(corpus, tmp_path):
    out = tmp_path / "bl.json"
    result = runner.invoke(app, ["baseline", "mockB", "--corpus", str(corpus), "-o", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert "Wrote baseline" in result.stdout


def test_ci_passes_on_identical(corpus, tmp_path):
    out = tmp_path / "bl.json"
    runner.invoke(app, ["baseline", "mockA", "--corpus", str(corpus), "-o", str(out)])
    result = runner.invoke(
        app,
        ["ci", "mockA", "--baseline", str(out), "--corpus", str(corpus), "--max-total-growth-pct", "0"],
    )
    assert result.exit_code == 0
    assert "PASS" in result.stdout


def test_ci_fails_on_growth(corpus, tmp_path):
    out = tmp_path / "bl.json"
    # baseline under bigram (fewer tokens), ci under char-level (more) -> growth
    runner.invoke(app, ["baseline", "mockB", "--corpus", str(corpus), "-o", str(out)])
    result = runner.invoke(
        app,
        ["ci", "mockA", "--baseline", str(out), "--corpus", str(corpus), "--max-total-growth-pct", "1"],
    )
    assert result.exit_code == 1
    assert "FAIL" in result.stdout


def test_ci_missing_baseline_exits_2(corpus, tmp_path):
    result = runner.invoke(
        app,
        ["ci", "mockA", "--baseline", str(tmp_path / "nope.json"), "--corpus", str(corpus)],
    )
    assert result.exit_code == 2
    assert "not found" in result.stdout


def test_ci_cost_delta_requires_price(corpus, tmp_path):
    out = tmp_path / "bl.json"
    runner.invoke(app, ["baseline", "mockA", "--corpus", str(corpus), "-o", str(out)])
    result = runner.invoke(
        app,
        ["ci", "mockA", "--baseline", str(out), "--corpus", str(corpus), "--max-cost-delta", "1.0"],
    )
    assert result.exit_code == 2
    assert "requires --price-per-1k" in result.stdout
