"""
tokenlens.core.differ
~~~~~~~~~~~~~~~~~~~~~
Per-entry encoding diff between two tokenizers.

``EncodingDiffer`` is the central engine.  For a given text, it:

1. Encodes with both tokenizers.
2. Computes token count delta.
3. Finds the first character position where the two encodings diverge.
4. Optionally delegates to ``BoundaryDetector`` for word-level violations.

Usage
-----
>>> from tokenlens.core.loader import TokenizerLoader
>>> from tokenlens.core.differ import EncodingDiffer
>>> tok_a = TokenizerLoader.load("cl100k_base")
>>> tok_b = TokenizerLoader.load("o200k_base")
>>> diff = EncodingDiffer().diff("Hello world", tok_a, tok_b)
>>> print(diff.count_delta, diff.first_divergence_pos)
"""

from __future__ import annotations

from tokenlens.core.boundary import BoundaryDetector
from tokenlens.core.loader import UnifiedTokenizer
from tokenlens.models import TokenDiff


class EncodingDiffer:
    """
    Computes a ``TokenDiff`` for a single text against two tokenizers.

    Parameters
    ----------
    detect_boundaries:
        If ``True``, runs ``BoundaryDetector`` for an experimental,
        structural word-level boundary report.  Defaults to ``False`` so the
        fast, fully-supported count/divergence diff is the default path.
    word_tokenizer:
        Passed through to ``BoundaryDetector``.  ``"whitespace"`` or
        ``"nltk"``.
    """

    def __init__(
        self,
        detect_boundaries: bool = False,
        word_tokenizer: str = "whitespace",
    ) -> None:
        self._detect_boundaries = detect_boundaries
        self._boundary_detector = BoundaryDetector(word_tokenizer=word_tokenizer)  # type: ignore[arg-type]

    # ------------------------------------------------------------------

    def diff(
        self,
        text: str,
        tok_a: UnifiedTokenizer,
        tok_b: UnifiedTokenizer,
        entry_id: str = "",
    ) -> TokenDiff:
        """
        Compute the full diff for *text* between *tok_a* and *tok_b*.

        Parameters
        ----------
        text:
            The raw text to analyse.
        tok_a:
            The "before" tokenizer.
        tok_b:
            The "after" tokenizer.
        entry_id:
            Optional identifier carried through to the returned ``TokenDiff``.

        Returns
        -------
        TokenDiff
            Complete diff including count delta, first divergence position,
            and (optionally) boundary violations.
        """
        ids_a = tok_a.encode(text)
        ids_b = tok_b.encode(text)

        first_div = self._first_divergence_char(text, ids_a, ids_b, tok_a, tok_b)

        violations = []
        if self._detect_boundaries:
            violations = self._boundary_detector.detect(text, tok_a, tok_b)

        return TokenDiff(
            entry_id=entry_id,
            text=text,
            token_count_a=len(ids_a),
            token_count_b=len(ids_b),
            count_delta=len(ids_b) - len(ids_a),
            first_divergence_pos=first_div,
            boundary_violations=violations,
        )

    def diff_many(
        self,
        entries: list[tuple[str, str]],
        tok_a: UnifiedTokenizer,
        tok_b: UnifiedTokenizer,
    ) -> list[TokenDiff]:
        """
        Diff multiple ``(entry_id, text)`` pairs.

        Parameters
        ----------
        entries:
            A list of ``(entry_id, text)`` tuples.
        tok_a, tok_b:
            Tokenizers.

        Returns
        -------
        list[TokenDiff]
            One ``TokenDiff`` per entry, in the same order as *entries*.
        """
        return [self.diff(text, tok_a, tok_b, entry_id=eid) for eid, text in entries]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _first_divergence_char(
        text: str,
        ids_a: list[int],
        ids_b: list[int],
        tok_a: UnifiedTokenizer,
        tok_b: UnifiedTokenizer,
    ) -> int:
        """
        Walk both token sequences in parallel and return the character offset
        of the first position where the decoded token strings diverge.

        Returns ``len(text)`` if the sequences are identical.

        The algorithm decodes one token at a time from each sequence and
        advances a shared character cursor.  At the first mismatch in decoded
        strings, we return the current cursor position.
        """
        i, j = 0, 0
        char_pos = 0

        while i < len(ids_a) and j < len(ids_b):
            str_a = tok_a.decode_single(ids_a[i])
            str_b = tok_b.decode_single(ids_b[j])

            if str_a != str_b:
                return char_pos

            char_pos += len(str_a)
            i += 1
            j += 1

        # One or both sequences exhausted without mismatch
        if i == len(ids_a) and j == len(ids_b):
            return len(text)  # fully identical

        # One sequence ended early → diverge at current position
        return char_pos
