"""
Tests for tokendrift.core.baseline: snapshots and CI gating.

The mock tokenizers from conftest give deterministic counts:
- ``tok_a`` (char-level)  -> one token per character
- ``tok_b`` (bigram)      -> roughly half as many tokens

So using ``tok_b`` for the baseline and ``tok_a`` for the run produces growth,
and the reverse produces shrinkage.
"""

from __future__ import annotations

import json

import pytest

from tokendrift.core.baseline import (
    Baseline,
    CIThresholds,
    build_baseline,
    run_ci,
)
from tokendrift.models import CorpusEntry


@pytest.fixture
def corpus() -> list[CorpusEntry]:
    return [
        CorpusEntry(id="a", text="hello world"),
        CorpusEntry(id="b", text="biostatistical significance"),
        CorpusEntry(id="c", text="x"),
    ]


# ---------------------------------------------------------------------------
# build_baseline
# ---------------------------------------------------------------------------


def test_build_baseline_counts_match_tokenizer(tok_a, corpus):
    bl = build_baseline(tok_a, corpus)
    assert bl.tokenizer == "mock_char_level"
    assert bl.entries["a"] == len("hello world")
    assert bl.total_tokens == sum(len(e.text) for e in corpus)
    assert set(bl.entries) == {"a", "b", "c"}
    assert bl.created_at  # timestamp populated


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


def test_baseline_roundtrip(tok_a, corpus, tmp_path):
    bl = build_baseline(tok_a, corpus)
    path = tmp_path / "baseline.json"
    bl.save(path)

    loaded = Baseline.load(path)
    assert loaded.tokenizer == bl.tokenizer
    assert loaded.total_tokens == bl.total_tokens
    assert loaded.entries == bl.entries


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        Baseline.load(tmp_path / "nope.json")


def test_load_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        Baseline.load(p)


def test_load_unsupported_schema(tmp_path):
    p = tmp_path / "v99.json"
    p.write_text(
        json.dumps({"schema_version": 99, "tokenizer": "x", "total_tokens": 0, "entries": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema_version"):
        Baseline.load(p)


def test_load_malformed_entries(tmp_path):
    p = tmp_path / "malformed.json"
    p.write_text(json.dumps({"schema_version": 1, "tokenizer": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed"):
        Baseline.load(p)


# ---------------------------------------------------------------------------
# run_ci: pass / fail logic
# ---------------------------------------------------------------------------


def test_ci_identical_passes(tok_a, corpus):
    bl = build_baseline(tok_a, corpus)
    report = run_ci(bl, tok_a, corpus, CIThresholds(max_total_growth_pct=0))
    assert report.passed
    assert report.token_delta == 0
    assert report.total_pct == 0.0


def test_ci_growth_fails_total_threshold(tok_b, tok_a, corpus):
    # baseline under bigram (fewer tokens), run under char-level (more tokens) -> growth
    bl = build_baseline(tok_b, corpus)
    report = run_ci(bl, tok_a, corpus, CIThresholds(max_total_growth_pct=5))
    assert not report.passed
    assert report.token_delta > 0
    assert any("total token growth" in f for f in report.failures)


def test_ci_growth_within_threshold_passes(tok_b, tok_a, corpus):
    bl = build_baseline(tok_b, corpus)
    # generous threshold absorbs the growth
    report = run_ci(bl, tok_a, corpus, CIThresholds(max_total_growth_pct=10_000))
    assert report.passed


def test_ci_per_entry_threshold(tok_b, tok_a, corpus):
    bl = build_baseline(tok_b, corpus)
    report = run_ci(bl, tok_a, corpus, CIThresholds(max_entry_growth_pct=5))
    assert not report.passed
    assert any("per-entry growth" in f for f in report.failures)


def test_ci_regressions_sorted(tok_b, tok_a, corpus):
    bl = build_baseline(tok_b, corpus)
    report = run_ci(bl, tok_a, corpus)
    deltas = [d.delta for d in report.regressions]
    assert deltas == sorted(deltas, reverse=True)
    assert all(d.delta > 0 for d in report.regressions)


def test_ci_no_thresholds_always_passes(tok_b, tok_a, corpus):
    bl = build_baseline(tok_b, corpus)
    report = run_ci(bl, tok_a, corpus)  # no thresholds = report only
    assert report.passed
    assert report.token_delta > 0  # drift still measured


# ---------------------------------------------------------------------------
# cost gating
# ---------------------------------------------------------------------------


def test_ci_cost_delta_estimated(tok_b, tok_a, corpus):
    bl = build_baseline(tok_b, corpus)
    report = run_ci(bl, tok_a, corpus, CIThresholds(price_per_1k=10.0))
    assert report.cost_delta_usd is not None
    assert report.cost_delta_usd == pytest.approx((report.token_delta / 1000) * 10.0)


def test_ci_cost_threshold_fails(tok_b, tok_a, corpus):
    bl = build_baseline(tok_b, corpus)
    report = run_ci(bl, tok_a, corpus, CIThresholds(price_per_1k=1000.0, max_cost_delta_usd=0.0001))
    assert not report.passed
    assert any("cost delta" in f for f in report.failures)


# ---------------------------------------------------------------------------
# new / missing entries
# ---------------------------------------------------------------------------


def test_ci_new_and_missing_entries(tok_a, corpus):
    bl = build_baseline(tok_a, corpus)
    # drop "c", add "d"
    new_corpus = [e for e in corpus if e.id != "c"] + [CorpusEntry(id="d", text="new entry")]
    report = run_ci(bl, tok_a, new_corpus)
    assert report.new_entries == ["d"]
    assert report.missing_entries == ["c"]
    # totals only count shared ids, so they stay comparable
    assert report.passed  # shared ids identical


def test_ci_fail_on_new(tok_a, corpus):
    bl = build_baseline(tok_a, corpus)
    new_corpus = corpus + [CorpusEntry(id="d", text="extra")]
    report = run_ci(bl, tok_a, new_corpus, CIThresholds(fail_on_new_entries=True))
    assert not report.passed
    assert any("absent from baseline" in f for f in report.failures)


def test_ci_fail_on_missing(tok_a, corpus):
    bl = build_baseline(tok_a, corpus)
    smaller = [e for e in corpus if e.id != "c"]
    report = run_ci(bl, tok_a, smaller, CIThresholds(fail_on_missing_entries=True))
    assert not report.passed
    assert any("absent from corpus" in f for f in report.failures)
