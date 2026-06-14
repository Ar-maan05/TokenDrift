"""CI regression-gate example for TokenDrift.

Mirrors what `tokendrift baseline` + `tokendrift ci` do, but from Python:

1. Snapshot per-entry token counts for a corpus under a "current" tokenizer.
2. Re-encode the same corpus under a "new" tokenizer and fail if the token
   count drifts beyond a threshold.

This is the pattern you would wire into CI to catch a provider silently
re-tokenizing a model and inflating your prompt token counts (and bill).
"""

from __future__ import annotations

import sys

from tokendrift import Baseline, CIThresholds, TokenizerLoader, build_baseline, run_ci
from tokendrift.models import CorpusEntry


def main() -> None:
    print("Loading tokenizers...")
    try:
        current = TokenizerLoader.load("cl100k_base")  # what you built the baseline with
        new = TokenizerLoader.load("o200k_base")  # the candidate change
    except Exception as exc:
        print(f"Error loading tokenizers: {exc}", file=sys.stderr)
        sys.exit(1)

    corpus = [
        CorpusEntry(id="q001", text="What is backpropagation through time in neural networks?"),
        CorpusEntry(id="q002", text="Translate this function into idiomatic Rust with pattern matching."),
        CorpusEntry(id="q003", text="Summarize the financial highlights from the annual report."),
        CorpusEntry(id="q004", text="ChatGPT rewrites biostatistical significance tests."),
    ]

    # 1. Build + persist a baseline (commit this JSON to your repo).
    baseline = build_baseline(current, corpus)
    baseline.save("tokendrift.baseline.json")
    print(f"Baseline: {baseline.total_tokens} tokens across {len(baseline.entries)} entries.")

    # 2. Gate the candidate tokenizer against the committed baseline.
    report = run_ci(
        Baseline.load("tokendrift.baseline.json"),
        new,
        corpus,
        CIThresholds(max_total_growth_pct=2, max_entry_growth_pct=10),
    )

    print("=" * 60)
    print(f" {report.baseline_tokenizer} -> {report.current_tokenizer}")
    print(f" Token delta: {report.token_delta:+d} ({report.total_pct:+.2f}%)")
    print("=" * 60)
    if report.passed:
        print("PASS: token drift within thresholds.")
    else:
        print("FAIL:")
        for reason in report.failures:
            print(f"  - {reason}")

    # Exit non-zero on failure so a pipeline step fails, just like the CLI.
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
