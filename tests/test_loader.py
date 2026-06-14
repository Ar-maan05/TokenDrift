"""Tests for tokendrift.core.loader."""

from __future__ import annotations

import pytest

from tokendrift.core.loader import (
    TIKTOKEN_ENCODINGS,
    TiktokenTokenizer,
    TokenizerLoader,
    _unresolved_message,
)

# ---------------------------------------------------------------------------
# Mock tokenizer interface (no network)
# ---------------------------------------------------------------------------


class TestMockTokenizerAInterface:
    TEXT = "Hello, world!"

    def test_encode_returns_ints(self, tok_a):
        ids = tok_a.encode(self.TEXT)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) == len(self.TEXT)

    def test_decode_roundtrip(self, tok_a):
        ids = tok_a.encode(self.TEXT)
        assert tok_a.decode(ids) == self.TEXT

    def test_decode_single(self, tok_a):
        ids = tok_a.encode(self.TEXT)
        reconstructed = "".join(tok_a.decode_single(i) for i in ids)
        assert reconstructed == self.TEXT

    def test_vocab_nonempty(self, tok_a):
        v = tok_a.vocab()
        assert isinstance(v, dict)
        assert len(v) > 10

    def test_name(self, tok_a):
        assert tok_a.name() == "mock_char_level"

    def test_char_offsets_length(self, tok_a):
        offsets = tok_a.char_offsets(self.TEXT)
        ids = tok_a.encode(self.TEXT)
        assert len(offsets) == len(ids)

    def test_char_offsets_contiguous(self, tok_a):
        offsets = tok_a.char_offsets(self.TEXT)
        for i in range(len(offsets) - 1):
            assert offsets[i][1] == offsets[i + 1][0]

    def test_char_offsets_start_at_zero(self, tok_a):
        offsets = tok_a.char_offsets(self.TEXT)
        if offsets:
            assert offsets[0][0] == 0

    def test_char_offsets_end_at_len(self, tok_a):
        offsets = tok_a.char_offsets(self.TEXT)
        if offsets:
            assert offsets[-1][1] == len(self.TEXT)

    def test_empty_string(self, tok_a):
        assert tok_a.encode("") == []
        assert tok_a.decode([]) == ""
        assert tok_a.char_offsets("") == []

    def test_unicode(self, tok_a):
        text = "héllo"
        ids = tok_a.encode(text)
        assert len(ids) == len(text)
        assert tok_a.decode(ids) == text


class TestMockTokenizerBInterface:
    TEXT = "Hello world"

    def test_encode_bigrams(self, tok_b):
        ids = tok_b.encode(self.TEXT)
        # 11 chars → 5 bigrams + 1 single = 6 tokens
        assert len(ids) == 6

    def test_decode_roundtrip(self, tok_b):
        ids = tok_b.encode(self.TEXT)
        assert tok_b.decode(ids) == self.TEXT

    def test_char_offsets_cover_text(self, tok_b):
        offsets = tok_b.char_offsets(self.TEXT)
        if offsets:
            assert offsets[0][0] == 0
            assert offsets[-1][1] == len(self.TEXT)

    def test_vocab_distinct_from_a(self, tok_a, tok_b):
        va = set(tok_a.vocab().keys())
        vb = set(tok_b.vocab().keys())
        # B has bigrams that A doesn't
        assert len(vb - va) > 0


# ---------------------------------------------------------------------------
# A vs B differ at the loader level
# ---------------------------------------------------------------------------


def test_mock_a_and_b_encode_differently(tok_a, tok_b):
    text = "Hello"
    assert tok_a.encode(text) != tok_b.encode(text)


def test_mock_a_and_b_have_different_vocab_sizes(tok_a, tok_b):
    assert len(tok_a.vocab()) != len(tok_b.vocab())


# ---------------------------------------------------------------------------
# Unresolved-identifier guidance (no network)
# ---------------------------------------------------------------------------


def test_unresolved_message_lists_encodings():
    msg = _unresolved_message("totally_bogus")
    assert "Cannot resolve tokenizer 'totally_bogus'" in msg
    assert "cl100k_base" in msg


def test_unresolved_message_suggests_close_tiktoken_name():
    # 'cl100k' is one '_base' away from a real encoding → should be suggested.
    msg = _unresolved_message("cl100k")
    assert "cl100k_base" in msg
    assert "Did you mean" in msg


# ---------------------------------------------------------------------------
# Loader detection (network-gated)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_typoed_encoding_raises_valueerror_with_hint():
    # A bare misspelled encoding must surface the helpful ValueError, not an
    # opaque HuggingFace 404.
    with pytest.raises(ValueError) as exc:
        TokenizerLoader.load("cl100k")
    assert "cl100k_base" in str(exc.value)


@pytest.mark.network
def test_load_tiktoken_by_name():
    tok = TokenizerLoader.load("cl100k_base")
    assert isinstance(tok, TiktokenTokenizer)


@pytest.mark.network
def test_unknown_identifier_raises():
    with pytest.raises((ValueError, Exception)):
        TokenizerLoader.load("definitely_not_a_real_tokenizer_xyz123")


@pytest.mark.network
def test_all_tiktoken_encodings_load():
    for name in TIKTOKEN_ENCODINGS:
        tok = TokenizerLoader.load(name)
        assert isinstance(tok, TiktokenTokenizer)
