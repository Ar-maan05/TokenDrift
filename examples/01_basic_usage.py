"""Basic usage example for TokenDrift.

This script demonstrates how to load tokenizers, compare encoding length
differences for a text snippet, and run a vocabulary diff.
"""

from __future__ import annotations

import sys

from tokendrift import EncodingDiffer, TokenizerLoader, VocabDiffer


def main() -> None:
    # 1. Load two tokenizers by name (using tiktoken or HuggingFace identifiers)
    print("Loading tokenizers...")
    try:
        tok_a = TokenizerLoader.load("cl100k_base")  # GPT-4 / GPT-3.5
        tok_b = TokenizerLoader.load("o200k_base")  # GPT-4o
    except Exception as exc:
        print(f"Error loading tokenizers: {exc}", file=sys.stderr)
        print("Note: Requires internet access on the first run to fetch vocabularies.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded tokenizer A: {tok_a.name()}")
    print(f"Loaded tokenizer B: {tok_b.name()}")
    print("-" * 60)

    # 2. Compare encoding differences for a text snippet
    text = "Comparing subword tokens for biostatistical significance."
    print(f"Target text: {repr(text)}")

    differ = EncodingDiffer()
    diff = differ.diff(text, tok_a, tok_b)

    print(f"Tokens in A: {diff.token_count_a}")
    print(f"Tokens in B: {diff.token_count_b}")
    print(f"Token change: {diff.count_delta:+d} ({diff.pct_change:+.1f}%)")
    print(f"First divergence at character index: {diff.first_divergence_pos}")
    print("-" * 60)

    # 3. Perform a vocabulary diff (additions, deletions, and ID remappings)
    print("Computing vocabulary diff (this may take a second)...")
    vocab_differ = VocabDiffer()
    v_diff = vocab_differ.diff(tok_a, tok_b)

    print(f"Vocab size A: {v_diff.total_a:,}")
    print(f"Vocab size B: {v_diff.total_b:,}")
    print(f"Added tokens: +{len(v_diff.added):,}")
    print(f"Deleted tokens: -{len(v_diff.deleted):,}")
    print(f"Remapped token IDs: {len(v_diff.remapped):,}")

    if v_diff.remapped:
        print("\n⚠️  WARNING: Found remapped token IDs!")
        print("This means the same text strings map to different integer IDs.")
        print("If you have stored token IDs in a database, they will point to wrong words.")
        print("Example remapped token:")
        sample = v_diff.remapped[0]
        print(f"  Token string: {repr(sample.token)} | ID in A: {sample.id_a} -> ID in B: {sample.id_b}")


if __name__ == "__main__":
    main()
