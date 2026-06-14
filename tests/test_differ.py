"""Tests for tokenlens.core.differ."""

from __future__ import annotations

import pytest

from tokenlens.core.differ import EncodingDiffer
from tokenlens.models import TokenDiff


def test_returns_token_diff(tok_a, tok_b, differ):
    d = differ.diff("Hello world", tok_a, tok_b)
    assert isinstance(d, TokenDiff)


def test_count_delta_is_b_minus_a(tok_a, tok_b, differ):
    text = "The quick brown fox"
    d = differ.diff(text, tok_a, tok_b)
    assert d.count_delta == d.token_count_b - d.token_count_a


def test_entry_id_propagated(tok_a, tok_b, differ):
    d = differ.diff("Hello", tok_a, tok_b, entry_id="test_42")
    assert d.entry_id == "test_42"


def test_changed_property_false_when_identical(tok_a):
    differ = EncodingDiffer(detect_boundaries=False)
    d = differ.diff("Hello world", tok_a, tok_a)
    assert not d.changed
    assert d.count_delta == 0


def test_changed_property_true_when_different(tok_a, tok_b, differ):
    # char-level vs bigram: any text longer than 1 char differs
    d = differ.diff("ab", tok_a, tok_b)
    assert d.changed


def test_first_divergence_full_text_when_identical(tok_a):
    differ = EncodingDiffer(detect_boundaries=False)
    text = "Hello world"
    d = differ.diff(text, tok_a, tok_a)
    assert d.first_divergence_pos == len(text)


def test_first_divergence_at_zero_for_bigram(tok_a, tok_b):
    # char vs bigram: first token already differs ("H" vs "He")
    d = EncodingDiffer(detect_boundaries=False).diff("Hello", tok_a, tok_b)
    assert d.first_divergence_pos == 0


def test_no_violations_on_identical_tokenizers(tok_a):
    differ = EncodingDiffer(detect_boundaries=True)
    d = differ.diff("Hello world biostatistical", tok_a, tok_a)
    assert d.boundary_violations == []


def test_violations_list_is_list(tok_a, tok_b):
    differ = EncodingDiffer(detect_boundaries=True)
    d = differ.diff("Hello world", tok_a, tok_b)
    assert isinstance(d.boundary_violations, list)


def test_diff_many_length(tok_a, tok_b, differ):
    pairs = [("a", "Hello"), ("b", "World"), ("c", "foo bar")]
    diffs = differ.diff_many(pairs, tok_a, tok_b)
    assert len(diffs) == 3


def test_diff_many_entry_ids(tok_a, tok_b, differ):
    pairs = [("x1", "text one"), ("x2", "text two")]
    diffs = differ.diff_many(pairs, tok_a, tok_b)
    assert diffs[0].entry_id == "x1"
    assert diffs[1].entry_id == "x2"


def test_empty_string(tok_a, tok_b, differ):
    d = differ.diff("", tok_a, tok_b)
    assert d.token_count_a == 0
    assert d.token_count_b == 0
    assert d.count_delta == 0


def test_whitespace_only(tok_a, tok_b, differ):
    d = differ.diff("   ", tok_a, tok_b)
    assert isinstance(d, TokenDiff)


def test_unicode(tok_a, tok_b, differ):
    d = differ.diff("héllo", tok_a, tok_b)
    assert d.token_count_a > 0
    assert d.token_count_b > 0


def test_mock_a_char_level_counts(tok_a):
    differ = EncodingDiffer(detect_boundaries=False)
    text = "abc"
    d = differ.diff(text, tok_a, tok_a)
    assert d.token_count_a == 3
    assert d.token_count_b == 3


def test_mock_b_bigram_counts(tok_b):
    differ = EncodingDiffer(detect_boundaries=False)
    # "abcd" -> 2 bigrams
    d = differ.diff("abcd", tok_b, tok_b)
    assert d.token_count_a == 2
    assert d.token_count_b == 2
    assert d.count_delta == 0


@pytest.mark.network
def test_real_cl100k_encoding():
    from tokenlens.core.loader import TokenizerLoader

    ta = TokenizerLoader.load("cl100k_base")
    tb = TokenizerLoader.load("o200k_base")
    differ = EncodingDiffer(detect_boundaries=False)
    d = differ.diff("biostatistical significance", ta, tb)
    assert isinstance(d, TokenDiff)
