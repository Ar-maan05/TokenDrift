"""Boundary violation detection example for TokenDrift.

This script demonstrates how to detect structural boundary changes (SPLIT,
MERGE, RESEGMENT) for words across tokenizer changes.
"""

from __future__ import annotations

import sys

from tokendrift import BoundaryDetector, TokenizerLoader


def main() -> None:
    # 1. Load tokenizers
    print("Loading tokenizers...")
    try:
        tok_a = TokenizerLoader.load("cl100k_base")
        tok_b = TokenizerLoader.load("o200k_base")
    except Exception as exc:
        print(f"Error loading tokenizers: {exc}", file=sys.stderr)
        sys.exit(1)

    # 2. Define a text snippet containing words known to segment differently
    # "biostatistical" split changes, "rewrites" resegments
    text = "ChatGPT rewrites biostatistical significance tests."
    print(f"Target text: {repr(text)}")
    print("-" * 60)

    # 3. Detect boundary changes
    detector = BoundaryDetector(word_tokenizer="whitespace")
    violations = detector.detect(text, tok_a, tok_b)

    print(f"Found {len(violations)} boundary violation(s):")
    for i, v in enumerate(violations, 1):
        print(f"\nViolation {i}:")
        print(f"  Word:           {repr(v.word)}")
        print(f"  Type:           {v.violation_type.value}")
        print(f"  Tokens in A:    {v.tokens_a} (IDs: {v.ids_a})")
        print(f"  Tokens in B:    {v.tokens_b} (IDs: {v.ids_b})")
    print("=" * 60)


if __name__ == "__main__":
    main()
