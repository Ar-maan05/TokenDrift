"""
tokendrift.core.vocab
~~~~~~~~~~~~~~~~~~~~
Vocabulary-level diff between two tokenizers.

The most important output is ``VocabDiff.remapped``: any token string whose
integer ID changed.  Systems that store token IDs (caches keyed by ID,
classifier heads, LoRA adapter embedding indices) silently break when an ID
remapping happens, with no error at runtime.

Usage
-----
>>> from tokendrift.core.loader import TokenizerLoader
>>> from tokendrift.core.vocab import VocabDiffer
>>> tok_a = TokenizerLoader.load("cl100k_base")
>>> tok_b = TokenizerLoader.load("o200k_base")
>>> diff = VocabDiffer().diff(tok_a, tok_b)
>>> print(f"Added: {len(diff.added)}, Remapped: {len(diff.remapped)}")
"""

from __future__ import annotations

from tokendrift.core.loader import UnifiedTokenizer
from tokendrift.models import RemappedEntry, VocabDiff, VocabEntry


class VocabDiffer:
    """Computes the vocabulary diff between two ``UnifiedTokenizer`` instances."""

    def diff(
        self,
        tok_a: UnifiedTokenizer,
        tok_b: UnifiedTokenizer,
    ) -> VocabDiff:
        """
        Compare vocabularies of *tok_a* and *tok_b*.

        Parameters
        ----------
        tok_a:
            The "before" tokenizer.
        tok_b:
            The "after" tokenizer.

        Returns
        -------
        VocabDiff
            Contains added, deleted, and remapped token lists plus vocabulary
            sizes for both tokenizers.
        """
        vocab_a = tok_a.vocab()
        vocab_b = tok_b.vocab()

        keys_a = set(vocab_a)
        keys_b = set(vocab_b)

        added = [VocabEntry(token_str=k, token_id=vocab_b[k]) for k in sorted(keys_b - keys_a)]
        deleted = [VocabEntry(token_str=k, token_id=vocab_a[k]) for k in sorted(keys_a - keys_b)]
        remapped = [
            RemappedEntry(
                token_str=k,
                old_id=vocab_a[k],
                new_id=vocab_b[k],
            )
            for k in sorted(keys_a & keys_b)
            if vocab_a[k] != vocab_b[k]
        ]

        return VocabDiff(
            added=added,
            deleted=deleted,
            remapped=remapped,
            total_a=len(vocab_a),
            total_b=len(vocab_b),
        )
