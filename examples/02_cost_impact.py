"""Cost estimation example for TokenDrift.

This script demonstrates how to compute the cost impact of migrating from
tokenizer A (e.g., cl100k_base) to B (e.g., o200k_base) over a list of prompts.
"""

from __future__ import annotations

import sys

from tokendrift import CostCalculator, EncodingDiffer, TokenizerLoader


def main() -> None:
    # 1. Load tokenizers
    print("Loading tokenizers...")
    try:
        tok_a = TokenizerLoader.load("cl100k_base")
        tok_b = TokenizerLoader.load("o200k_base")
    except Exception as exc:
        print(f"Error loading tokenizers: {exc}", file=sys.stderr)
        sys.exit(1)

    # 2. Define a list of prompts (with IDs) representing your corpus
    prompts = [
        ("q001", "What is backpropagation through time (BPTT) in neural networks?"),
        ("q002", "Translate this function into idiomatic Rust with proper pattern matching."),
        ("q003", "Summarize the financial highlights from the annual fiscal statement."),
        ("q004", "ChatGPT automatically rewrites biostatistical significance tests."),
    ]

    # 3. Compute encoding diffs for the prompts
    print("Diffing prompt encodings...")
    differ = EncodingDiffer()
    diffs = differ.diff_many(prompts, tok_a, tok_b)

    # 4. Define token prices (per 1,000 tokens)
    # E.g., model A costs $0.03 / 1k tokens, model B costs $0.015 / 1k tokens
    price_a_per_1k = 0.03
    price_b_per_1k = 0.015

    # 5. Calculate the cost impact
    calculator = CostCalculator()
    report = calculator.compute(diffs, price_a=price_a_per_1k, price_b=price_b_per_1k)

    # 6. Display results
    print("=" * 60)
    print(" TOKEN MIGRATION COST REPORT")
    print("=" * 60)
    print(f"Model A Price:   ${price_a_per_1k:.5f} per 1k tokens")
    print(f"Model B Price:   ${price_b_per_1k:.5f} per 1k tokens")
    print("-" * 60)
    print(f"Total Tokens (A): {report.total_tokens_a:,}")
    print(f"Total Tokens (B): {report.total_tokens_b:,}")
    print(f"Token Delta:      {report.token_delta:+d} ({report.pct_token_change:+.2f}%)")
    print("-" * 60)
    print(f"Total Cost (A):   ${report.cost_a_usd:.6f}")
    print(f"Total Cost (B):   ${report.cost_b_usd:.6f}")
    print(f"Cost Difference:  ${report.cost_delta_usd:+.6f} ({report.pct_cost_change:+.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
