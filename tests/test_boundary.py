"""Tests for tokendrift.core.boundary."""

from __future__ import annotations

import pytest

from tokendrift.core.boundary import BoundaryDetector, _whitespace_word_spans
from tokendrift.models import BoundaryViolation, ViolationType


@pytest.fixture
def detector():
    return BoundaryDetector(word_tokenizer="whitespace")


# ---------------------------------------------------------------------------
# Word span extraction
# ---------------------------------------------------------------------------


def test_whitespace_word_spans_basic():
    spans = _whitespace_word_spans("Hello world")
    words = [w for w, _, _ in spans]
    assert "Hello" in words
    assert "world" in words


def test_whitespace_word_spans_strips_punctuation():
    spans = _whitespace_word_spans("Hello, world!")
    words = [w for w, _, _ in spans]
    assert "Hello" in words
    assert "world" in words
    assert "Hello," not in words
    assert "world!" not in words


def test_whitespace_word_spans_empty():
    assert _whitespace_word_spans("") == []
    assert _whitespace_word_spans("   ") == []


def test_whitespace_word_spans_char_positions():
    text = "foo bar"
    spans = _whitespace_word_spans(text)
    for word, start, end in spans:
        assert text[start:end] == word


# ---------------------------------------------------------------------------
# Detector with mock tokenizers
# ---------------------------------------------------------------------------


def test_no_violations_same_tokenizer(detector, tok_a):
    """Same tokenizer on both sides: never a violation."""
    violations = detector.detect("Hello world foo bar", tok_a, tok_a)
    assert violations == []


def test_violations_are_boundary_violation_instances(detector, tok_a, tok_b):
    violations = detector.detect("Hello world", tok_a, tok_b)
    for v in violations:
        assert isinstance(v, BoundaryViolation)


def test_violation_word_in_original_text(detector, tok_a, tok_b):
    text = "Hello world"
    violations = detector.detect(text, tok_a, tok_b)
    for v in violations:
        assert v.word in text


def test_violation_char_span_correct(detector, tok_a, tok_b):
    text = "Hello world"
    violations = detector.detect(text, tok_a, tok_b)
    for v in violations:
        assert text[v.char_start : v.char_end] == v.word


def test_violation_tokens_nonempty(detector, tok_a, tok_b):
    violations = detector.detect("Hello world", tok_a, tok_b)
    for v in violations:
        assert len(v.tokens_a) > 0
        assert len(v.tokens_b) > 0


def test_split_has_more_b_tokens(detector, tok_a, tok_b):
    # char-level (A) vs bigram (B): even-length words get 1 token in B,
    # but each char in A. So "ab" is SPLIT from A's perspective:
    # A: ['a','b'] (2), B: ['ab'] (1) → this is actually a MERGE not SPLIT
    # For SPLIT: "abc" → A: 3 tokens, B: 2 tokens  (still MERGE for B)
    # To get a real SPLIT (1 in A, >1 in B) we'd need a 1-char "word".
    # With whitespace tokenizer, punctuation is stripped, so single chars
    # show up as their own words when surrounded by spaces.
    violations = detector.detect("a b c", tok_a, tok_b)
    for v in violations:
        if v.violation_type == ViolationType.SPLIT:
            assert len(v.tokens_b) > len(v.tokens_a)


def test_merge_has_more_a_tokens(detector, tok_a, tok_b):
    # "ab" → A: 2 tokens ('a','b'), B: 1 bigram token → MERGE
    violations = detector.detect("ab cd ef", tok_a, tok_b)
    merges = [v for v in violations if v.violation_type == ViolationType.MERGE]
    # There should be at least one MERGE (each 2-char word)
    assert len(merges) > 0
    for v in merges:
        assert len(v.tokens_a) > len(v.tokens_b)


def test_only_structural_types_emitted(detector, tok_a, tok_b):
    """Detector emits SPLIT / MERGE / RESEGMENT and nothing else."""
    violations = detector.detect("Hello world foo bar", tok_a, tok_b)
    allowed = {ViolationType.SPLIT, ViolationType.MERGE, ViolationType.RESEGMENT}
    for v in violations:
        assert v.violation_type in allowed


def test_violation_has_no_severity_attr(detector, tok_a, tok_b):
    """Severity has been removed; violations carry no impact judgement."""
    violations = detector.detect("Hello world", tok_a, tok_b)
    for v in violations:
        assert not hasattr(v, "severity")


def test_id_only_difference_is_not_reported():
    """
    A word that decodes to the same token strings in both tokenizers but with
    different integer IDs is a vocab-level remap, not a boundary change, and
    must NOT be reported (this was the dominant source of noise).
    """
    from tests.conftest import MockTokenizerA

    class ShiftedIdTokenizer(MockTokenizerA):
        """Same char-level segmentation as A, but every ID shifted by 1000."""

        def encode(self, text):
            return [ord(c) + 1000 for c in text]

        def decode_single(self, id):
            return chr(id - 1000)

        def name(self):
            return "mock_shifted_ids"

    detector = BoundaryDetector(word_tokenizer="whitespace")
    violations = detector.detect("Hello world", MockTokenizerA(), ShiftedIdTokenizer())
    assert violations == []


def test_empty_text(detector, tok_a, tok_b):
    assert detector.detect("", tok_a, tok_b) == []


def test_single_word(detector, tok_a, tok_b):
    violations = detector.detect("Hello", tok_a, tok_b)
    assert isinstance(violations, list)


def test_numeric_text(detector, tok_a, tok_b):
    violations = detector.detect("12345 67890", tok_a, tok_b)
    assert isinstance(violations, list)


def test_code_text(detector, tok_a, tok_b):
    code = "def foo(x): return x"
    violations = detector.detect(code, tok_a, tok_b)
    assert isinstance(violations, list)


@pytest.mark.network
def test_real_split_detection():
    from tokendrift.core.loader import TokenizerLoader

    ta = TokenizerLoader.load("cl100k_base")
    tb = TokenizerLoader.load("o200k_base")
    detector = BoundaryDetector(word_tokenizer="whitespace")
    violations = detector.detect("ChatGPT biostatistical significance", ta, tb)
    assert isinstance(violations, list)
