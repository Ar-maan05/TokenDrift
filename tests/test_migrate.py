"""Tests for tokendrift.core.migrate."""

from __future__ import annotations

import pytest

from tokendrift.core.migrate import migrate_report
from tokendrift.core.registry import ModelRegistry
from tokendrift.models import CorpusEntry, ModelInfo


@pytest.fixture()
def registry(tok_a, tok_b):
    reg = ModelRegistry()
    # source: char-level, target: bigram (fewer tokens), small target window.
    reg.add(ModelInfo(name="source", tokenizer="char_id", price_per_1k_input=1.0))
    reg.add(
        ModelInfo(
            name="target",
            tokenizer="bigram_id",
            price_per_1k_input=2.0,
            context_window=10,
            max_output_tokens=0,
        )
    )
    reg.register_tokenizer("char_id", tok_a)
    reg.register_tokenizer("bigram_id", tok_b)
    return reg


def test_token_and_cost_delta(registry):
    entries = [CorpusEntry(id="p1", text="abcd")]  # 4 char tokens -> 2 bigram tokens
    report = migrate_report(entries, "source", "target", registry=registry)
    assert report.total_tokens_source == 4
    assert report.total_tokens_target == 2
    assert report.token_delta == -2
    assert report.pct_token_change == pytest.approx(-50.0)
    assert report.cost_source_usd == pytest.approx(4 / 1000 * 1.0)
    assert report.cost_target_usd == pytest.approx(2 / 1000 * 2.0)
    assert report.cost_delta_usd == pytest.approx(0.0)


def test_overflow_detection(registry):
    # 30 chars -> 15 bigram tokens, target window 10 -> overflow by 5.
    entries = [CorpusEntry(id="big", text="x" * 30), CorpusEntry(id="small", text="ab")]
    report = migrate_report(entries, "source", "target", registry=registry)
    assert [o.entry_id for o in report.overflows] == ["big"]
    o = report.overflows[0]
    assert o.target_tokens == 15
    assert o.context_window == 10
    assert o.overflow == 5


def test_vocab_diff_included_and_skippable(registry):
    entries = [CorpusEntry(id="p1", text="abcd")]
    with_vocab = migrate_report(entries, "source", "target", registry=registry)
    assert with_vocab.vocab is not None
    assert with_vocab.remapped_token_count == len(with_vocab.vocab.remapped)

    without = migrate_report(entries, "source", "target", registry=registry, include_vocab=False)
    assert without.vocab is None
    assert without.remapped_token_count == 0


def test_to_dict_is_machine_readable(registry):
    entries = [CorpusEntry(id="p1", text="abcd")]
    d = migrate_report(entries, "source", "target", registry=registry).to_dict()
    assert d["source_model"] == "source"
    assert d["token_delta"] == -2
    assert d["overflow_count"] == 0
    assert "pct_cost_change" in d


def test_per_prompt_breakdown(registry):
    entries = [CorpusEntry(id="p1", text="ab"), CorpusEntry(id="p2", text="abcd")]
    report = migrate_report(entries, "source", "target", registry=registry)
    assert [p.entry_id for p in report.per_prompt] == ["p1", "p2"]
    assert report.per_prompt[1].tokens_a == 4
