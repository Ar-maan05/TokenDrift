"""
tokendrift.core.boundary
~~~~~~~~~~~~~~~~~~~~~~~
Structural boundary-change detection (experimental).

A boundary change occurs when a word is segmented into tokens differently by
tokenizer B than by tokenizer A:

- ``SPLIT``      one token in A becomes several in B (count grew)
- ``MERGE``      several tokens in A become one in B (count shrank)
- ``RESEGMENT``  same token count, but the segmentation boundaries moved

This is a purely *structural* report. TokenDrift does not claim a boundary
change degrades model behaviour: re-segmentation is normal when tokenizers
change, and any behavioural effect is task-specific and unmeasured here.

Note this is distinct from a vocabulary ID *remapping* (see ``VocabDiffer``):
two tokenizers number their vocabularies independently, so a word encoding to
the same string in both but a different integer ID is not reported here; that
would flag almost every word and carry no structural signal.

Word tokenization
-----------------
The default strategy splits on whitespace and strips leading/trailing
punctuation.  This is intentionally conservative so the tool works without
NLTK as a hard dependency.  Pass ``word_tokenizer="nltk"`` to use NLTK's
``word_tokenize``, which handles contractions and punctuation more correctly.

Usage
-----
>>> from tokendrift.core.loader import TokenizerLoader
>>> from tokendrift.core.boundary import BoundaryDetector
>>> tok_a = TokenizerLoader.load("cl100k_base")
>>> tok_b = TokenizerLoader.load("o200k_base")
>>> text = "ChatGPT rewrites biostatistical significance tests"
>>> violations = BoundaryDetector().detect(text, tok_a, tok_b)
>>> for v in violations:
...     print(v.word, v.violation_type, v.tokens_a, "->", v.tokens_b)
"""

from __future__ import annotations

import re
from typing import Literal

from tokendrift.core.loader import UnifiedTokenizer
from tokendrift.models import BoundaryViolation, ViolationType

WordTokenizerChoice = Literal["whitespace", "nltk"]


class BoundaryDetector:
    """
    Detects word-level token boundary changes between two tokenizers.

    Parameters
    ----------
    word_tokenizer:
        ``"whitespace"`` (default, zero dependencies) or ``"nltk"``
        (more accurate, requires ``pip install nltk`` and punkt data).
    """

    def __init__(
        self,
        word_tokenizer: WordTokenizerChoice = "whitespace",
    ) -> None:
        self._word_tokenizer = word_tokenizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        text: str,
        tok_a: UnifiedTokenizer,
        tok_b: UnifiedTokenizer,
    ) -> list[BoundaryViolation]:
        """
        Return all boundary violations for *text* between *tok_a* and *tok_b*.

        Parameters
        ----------
        text:
            The raw text string to analyse.
        tok_a:
            The "before" tokenizer.
        tok_b:
            The "after" tokenizer.

        Returns
        -------
        list[BoundaryViolation]
            Sorted by character position.  Empty list if no violations.
        """
        if not text.strip():
            return []

        ids_a = tok_a.encode(text)
        ids_b = tok_b.encode(text)

        if not ids_a or not ids_b:
            return []

        offsets_a = tok_a.char_offsets(text)
        offsets_b = tok_b.char_offsets(text)

        words = self._word_spans(text)
        violations: list[BoundaryViolation] = []

        for word, w_start, w_end in words:
            # Token indices in A that overlap with [w_start, w_end)
            idx_a = _overlapping_indices(offsets_a, w_start, w_end)
            idx_b = _overlapping_indices(offsets_b, w_start, w_end)

            if not idx_a or not idx_b:
                continue

            toks_a = [ids_a[i] for i in idx_a]
            toks_b = [ids_b[i] for i in idx_b]

            str_a = [tok_a.decode_single(i) for i in toks_a]
            str_b = [tok_b.decode_single(i) for i in toks_b]

            if len(idx_a) == 1 and len(idx_b) > 1:
                vtype = ViolationType.SPLIT
            elif len(idx_a) > 1 and len(idx_b) == 1:
                vtype = ViolationType.MERGE
            elif len(idx_a) == len(idx_b) and str_a != str_b:
                # Same token count but the decoded segmentation moved.
                # Compared on decoded *strings*, not IDs: a word that encodes
                # to the same string in both tokenizers (differing only in
                # integer ID) is a vocab-level remap, not a boundary change,
                # and is intentionally not reported here.
                vtype = ViolationType.RESEGMENT
            else:
                continue

            violations.append(
                BoundaryViolation(
                    word=word,
                    char_start=w_start,
                    char_end=w_end,
                    tokens_a=str_a,
                    tokens_b=str_b,
                    ids_a=toks_a,
                    ids_b=toks_b,
                    violation_type=vtype,
                )
            )

        return violations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _word_spans(self, text: str) -> list[tuple[str, int, int]]:
        """
        Return ``(word, char_start, char_end)`` tuples for every word in
        *text*.

        Uses the configured word tokenizer.  The whitespace strategy strips
        leading/trailing ASCII punctuation so that "Hello," gives ("Hello", …)
        rather than ("Hello,", …).
        """
        if self._word_tokenizer == "nltk":
            return _nltk_word_spans(text)
        return _whitespace_word_spans(text)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _overlapping_indices(
    offsets: list[tuple[int, int]],
    start: int,
    end: int,
) -> list[int]:
    """
    Return the indices in *offsets* whose spans overlap with ``[start, end)``.

    A span ``(s, e)`` overlaps ``[start, end)`` when ``s < end and e > start``.
    """
    return [i for i, (s, e) in enumerate(offsets) if s < end and e > start]


def _whitespace_word_spans(text: str) -> list[tuple[str, int, int]]:
    """
    Tokenize *text* by splitting on whitespace and stripping outer
    ASCII punctuation.  Returns ``(word, start, end)`` tuples.
    """
    spans: list[tuple[str, int, int]] = []
    for match in re.finditer(r"\S+", text):
        raw = match.group()
        raw_start = match.start()
        # Strip leading punctuation
        stripped = raw.lstrip("\"'`([{<")
        lead_offset = len(raw) - len(stripped)
        stripped = stripped.rstrip("\"'`.,;:!?)]}>\\/")
        if not stripped:
            continue
        word_start = raw_start + lead_offset
        word_end = word_start + len(stripped)
        spans.append((stripped, word_start, word_end))
    return spans


def _nltk_word_spans(text: str) -> list[tuple[str, int, int]]:
    """
    Use NLTK's ``word_tokenize`` and map tokens back to character offsets.

    Requires ``nltk`` and the ``punkt`` / ``punkt_tab`` data.
    Falls back to whitespace tokenization if NLTK is unavailable.
    """
    try:
        import nltk  # type: ignore[import-untyped]  # noqa: PLC0415 - optional dependency

        # Download punkt data quietly if not present
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        tokens = nltk.word_tokenize(text)
        spans: list[tuple[str, int, int]] = []
        cursor = 0
        for tok in tokens:
            idx = text.find(tok, cursor)
            if idx == -1:
                continue
            spans.append((tok, idx, idx + len(tok)))
            cursor = idx + len(tok)
        return spans
    except ImportError:
        return _whitespace_word_spans(text)
