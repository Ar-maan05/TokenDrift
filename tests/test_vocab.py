"""Tests for tokendrift.core.vocab."""

from __future__ import annotations

import pytest

from tokendrift.core.vocab import VocabDiffer
from tokendrift.models import VocabDiff


@pytest.fixture(scope="module")
def diff(tok_a, tok_b):
    return VocabDiffer().diff(tok_a, tok_b)


def test_returns_vocab_diff(diff):
    assert isinstance(diff, VocabDiff)


def test_total_sizes_are_set(diff, tok_a, tok_b):
    assert diff.total_a == len(tok_a.vocab())
    assert diff.total_b == len(tok_b.vocab())


def test_added_not_in_a(diff, tok_a):
    vocab_a = tok_a.vocab()
    for entry in diff.added:
        assert entry.token_str not in vocab_a


def test_deleted_not_in_b(diff, tok_b):
    vocab_b = tok_b.vocab()
    for entry in diff.deleted:
        assert entry.token_str not in vocab_b


def test_remapped_ids_differ(diff, tok_a, tok_b):
    va, vb = tok_a.vocab(), tok_b.vocab()
    for r in diff.remapped:
        assert r.token_str in va and r.token_str in vb
        assert va[r.token_str] == r.old_id
        assert vb[r.token_str] == r.new_id
        assert r.old_id != r.new_id


def test_mock_b_has_bigrams_not_in_a(diff):
    assert len(diff.added) > 0


def test_self_diff_is_empty(tok_a):
    d = VocabDiffer().diff(tok_a, tok_a)
    assert d.added == []
    assert d.deleted == []
    assert d.remapped == []
    assert d.total_a == d.total_b


def test_has_remappings_false_on_self_diff(tok_a):
    d = VocabDiffer().diff(tok_a, tok_a)
    assert not d.has_remappings


@pytest.mark.network
def test_cl100k_to_o200k_has_additions():
    from tokendrift.core.loader import TokenizerLoader

    ta = TokenizerLoader.load("cl100k_base")
    tb = TokenizerLoader.load("o200k_base")
    d = VocabDiffer().diff(ta, tb)
    assert len(d.added) > 0
