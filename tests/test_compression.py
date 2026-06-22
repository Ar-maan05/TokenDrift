"""Tests for tokendrift.core.compression."""

from __future__ import annotations

import pytest

from tokendrift.core.compression import compression_report, compression_report_corpus
from tokendrift.core.registry import ModelRegistry
from tokendrift.models import CorpusEntry, ModelInfo


@pytest.fixture()
def registry(tok_a, tok_b):
    reg = ModelRegistry()
    reg.add(ModelInfo(name="char", tokenizer="char_id", price_per_1k_input=1.0))
    reg.add(ModelInfo(name="bigram", tokenizer="bigram_id"))  # unpriced
    reg.register_tokenizer("char_id", tok_a)
    reg.register_tokenizer("bigram_id", tok_b)
    return reg


def test_savings_per_model(registry):
    report = compression_report("abcdefgh", "abcd", ["char", "bigram"], registry=registry)
    assert report.original_chars == 8
    assert report.compressed_chars == 4
    assert report.char_pct_saved == pytest.approx(50.0)

    char = report.savings[0]
    assert char.original_tokens == 8
    assert char.compressed_tokens == 4
    assert char.tokens_saved == 4
    assert char.pct_saved == pytest.approx(50.0)
    assert char.cost_saved_usd == pytest.approx(4 / 1000 * 1.0)


def test_unpriced_model_cost_none(registry):
    report = compression_report("abcdefgh", "abcd", ["bigram"], registry=registry)
    assert report.savings[0].cost_saved_usd is None


def test_corpus_aggregation(registry):
    pairs = [
        (CorpusEntry(id="1", text="abcdefgh"), CorpusEntry(id="1c", text="abcd")),
        (CorpusEntry(id="2", text="abcd"), CorpusEntry(id="2c", text="ab")),
    ]
    report = compression_report_corpus(pairs, ["char"], registry=registry)
    # originals: 8 + 4 = 12 tokens; compressed: 4 + 2 = 6 tokens
    s = report.savings[0]
    assert s.original_tokens == 12
    assert s.compressed_tokens == 6
    assert s.tokens_saved == 6


def test_zero_original_chars_no_division_error(registry):
    report = compression_report("", "", ["char"], registry=registry)
    assert report.char_pct_saved == 0.0
    assert report.savings[0].pct_saved == 0.0
