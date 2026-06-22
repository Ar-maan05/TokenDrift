"""Tests for tokendrift.core.forecast."""

from __future__ import annotations

import pytest

from tokendrift.core.forecast import forecast
from tokendrift.core.registry import ModelRegistry
from tokendrift.models import CorpusEntry, ModelInfo


@pytest.fixture()
def registry(tok_a, tok_b):
    reg = ModelRegistry()
    reg.add(ModelInfo(name="char", tokenizer="char_id", price_per_1k_input=1.0))
    reg.add(ModelInfo(name="bigram", tokenizer="bigram_id", price_per_1k_input=0.5))
    reg.register_tokenizer("char_id", tok_a)
    reg.register_tokenizer("bigram_id", tok_b)
    return reg


@pytest.fixture()
def sample():
    # 4 + 6 = 10 char tokens across 2 requests -> avg 5 tokens/request.
    return [CorpusEntry(id="1", text="abcd"), CorpusEntry(id="2", text="abcdef")]


def test_projection_math(registry, sample):
    report = forecast(sample, ["char"], projected_requests=1000, registry=registry)
    f = report.forecasts[0]
    assert f.sample_requests == 2
    assert f.sample_tokens == 10
    assert f.avg_tokens_per_request == pytest.approx(5.0)
    assert f.projected_tokens == 5000
    assert f.projected_cost_usd == pytest.approx(5000 / 1000 * 1.0)


def test_cheapest_selection(registry, sample):
    report = forecast(sample, ["char", "bigram"], projected_requests=1000, registry=registry)
    # char: avg 5 tok @ $1/1k; bigram: avg ~2.5 tok @ $2/1k.
    assert report.cheapest is not None
    assert report.cheapest.model == "bigram"


def test_to_dict(registry, sample):
    d = forecast(sample, ["char"], projected_requests=1000, registry=registry).to_dict()
    assert d["projected_requests"] == 1000
    assert d["forecasts"][0]["projected_tokens"] == 5000


def test_negative_requests_raises(registry, sample):
    with pytest.raises(ValueError, match="non-negative"):
        forecast(sample, ["char"], projected_requests=-1, registry=registry)


def test_empty_sample_no_division_error(registry):
    report = forecast([], ["char"], projected_requests=1000, registry=registry)
    f = report.forecasts[0]
    assert f.avg_tokens_per_request == 0.0
    assert f.projected_tokens == 0
