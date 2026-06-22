"""Tests for tokendrift.core.estimate."""

from __future__ import annotations

import pytest

from tokendrift.core.estimate import CostEstimator
from tokendrift.core.registry import ModelRegistry
from tokendrift.models import ModelInfo


@pytest.fixture()
def registry(tok_a, tok_b):
    """
    Registry with two offline mock models:
    - "char": MockTokenizerA (one token per character), priced, 100-token window.
    - "bigram": MockTokenizerB (one token per 2 chars), priced higher, no window.
    """
    reg = ModelRegistry()
    reg.add(
        ModelInfo(
            name="char",
            tokenizer="char_id",
            context_window=100,
            price_per_1k_input=1.0,
            max_output_tokens=10,
        )
    )
    reg.add(ModelInfo(name="bigram", tokenizer="bigram_id", price_per_1k_input=2.0))
    reg.register_tokenizer("char_id", tok_a)
    reg.register_tokenizer("bigram_id", tok_b)
    return reg


def test_estimate_one_token_count_and_cost(registry):
    est = CostEstimator(registry)
    e = est.estimate_one("abcd", "char")  # 4 chars -> 4 tokens
    assert e.token_count == 4
    assert e.cost_usd == pytest.approx(4 / 1000 * 1.0)
    assert e.tokenizer == "char_id"


def test_estimate_fit_and_headroom(registry):
    est = CostEstimator(registry)
    # window 100, max_output 10 reserved by default -> headroom 100 - 4 - 10 = 86
    e = est.estimate_one("abcd", "char")
    assert e.fits is True
    assert e.headroom == 86


def test_estimate_overflow(registry):
    est = CostEstimator(registry)
    text = "x" * 95  # 95 tokens + 10 reserved = 105 > 100
    e = est.estimate_one(text, "char")
    assert e.fits is False
    assert e.headroom == 100 - 95 - 10


def test_reserved_output_override(registry):
    est = CostEstimator(registry)
    e = est.estimate_one("x" * 95, "char", reserved_output=0)
    assert e.fits is True  # 95 + 0 <= 100


def test_unknown_context_window_returns_none_fit(registry):
    est = CostEstimator(registry)
    e = est.estimate_one("abcd", "bigram")
    assert e.fits is None
    assert e.headroom is None
    assert e.context_window is None


def test_unpriced_model_has_none_cost(registry):
    registry.add(ModelInfo(name="free", tokenizer="char_id"))
    est = CostEstimator(registry)
    assert est.estimate_one("abcd", "free").cost_usd is None


def test_multi_model_spread_and_cheapest(registry):
    est = CostEstimator(registry)
    result = est.estimate("abcd", ["char", "bigram"])  # 4 vs 2 tokens
    assert result.text_chars == 4
    assert result.min_tokens == 2
    assert result.max_tokens == 4
    assert result.token_spread == 2
    assert result.divergence_pct == pytest.approx(100.0)
    # char costs 4/1000*1 = 0.004; bigram 2/1000*2 = 0.004 -> tie, first wins
    assert result.cheapest is not None


def test_overflowed_listing(registry):
    est = CostEstimator(registry)
    result = est.estimate("x" * 200, ["char", "bigram"])
    overflowed = result.overflowed
    assert [e.model for e in overflowed] == ["char"]


def test_fits_helper(registry):
    est = CostEstimator(registry)
    assert est.fits("abcd", "char") is True
    assert est.fits("x" * 200, "char") is False
    # Unknown window: nothing to enforce, so it "fits".
    assert est.fits("x" * 200, "bigram") is True
