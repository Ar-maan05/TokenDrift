"""Cost, budget, and governance example for TokenDrift (v1.1.0).

Walks through the pre-dispatch and governance APIs built on the model registry:

1. Estimate token count, cost, and context-window fit across models.
2. Run a model migration safety report (token / cost / vocab / overflow).
3. Measure prompt-compression savings per model.
4. Forecast org-level spend across candidate models.
5. Raise a tokenizer drift alert against a committed baseline.

All numbers come from each model's own tokenizer, not a character-count
approximation. The registry ships indicative pricing: verify it before relying
on the cost figures.
"""

from __future__ import annotations

import sys

from tokendrift import (
    CostEstimator,
    ModelInfo,
    ModelRegistry,
    build_baseline,
    check_drift,
    compression_report,
    forecast,
    migrate_report,
)
from tokendrift.core.loader import TokenizerLoader
from tokendrift.models import CorpusEntry


def main() -> None:
    # The default registry maps friendly model names to tokenizer, context
    # window, and indicative input price. Add your own models (including
    # HuggingFace ones) with registry.add(...).
    registry = ModelRegistry.default()
    registry.add(ModelInfo(name="local-llama", tokenizer="cl100k_base", context_window=8_192))

    corpus = [
        CorpusEntry(id="q001", text="What is backpropagation through time in neural networks?"),
        CorpusEntry(id="q002", text="Translate this function into idiomatic Rust with pattern matching."),
        CorpusEntry(id="q003", text="Summarise the financial highlights from the annual report."),
        CorpusEntry(id="q004", text="ChatGPT rewrites biostatistical significance tests."),
    ]

    try:
        # 1. Pre-dispatch estimate across models.
        est = CostEstimator(registry)
        overlay = est.estimate(corpus[0].text, ["gpt-4o", "gpt-4-turbo"])
        print("== Estimate ==")
        for e in overlay.estimates:
            cost = "n/a" if e.cost_usd is None else f"${e.cost_usd:.5f}"
            print(f"  {e.model:14} {e.token_count:>4} tok  {cost:>10}  fits={e.fits}")
        print(f"  spread: {overlay.token_spread} tokens ({overlay.divergence_pct:.1f}%)\n")

        # 2. Migration safety report.
        mig = migrate_report(corpus, "gpt-4-turbo", "gpt-4o", registry=registry)
        print("== Migration gpt-4-turbo -> gpt-4o ==")
        print(f"  token delta: {mig.token_delta:+d} ({mig.pct_token_change:+.1f}%)")
        print(f"  cost delta:  {mig.cost_delta_usd}")
        print(f"  overflows:   {len(mig.overflows)}")
        print(f"  remapped tokens: {mig.remapped_token_count}\n")

        # 3. Compression savings per model.
        original = corpus[1].text
        compressed = "Translate to idiomatic Rust with pattern matching."
        comp = compression_report(original, compressed, ["gpt-4o", "gpt-4-turbo"], registry=registry)
        print("== Compression savings ==")
        for s in comp.savings:
            print(f"  {s.model:14} saved {s.tokens_saved:>3} tok ({s.pct_saved:.0f}%)")
        print()

        # 4. Org-level spend forecast.
        fc = forecast(corpus, ["gpt-4o", "gpt-4o-mini"], projected_requests=1_000_000, registry=registry)
        print("== Forecast (1,000,000 requests, input only) ==")
        for f in fc.forecasts:
            cost = "n/a" if f.projected_cost_usd is None else f"${f.projected_cost_usd:,.2f}"
            print(f"  {f.model:14} {f.projected_tokens:>14,} tok  {cost:>14}")
        if fc.cheapest is not None:
            print(f"  cheapest: {fc.cheapest.model}\n")

        # 5. Tokenizer drift alert against a committed baseline.
        tok_old = TokenizerLoader.load("cl100k_base")
        tok_new = TokenizerLoader.load("o200k_base")
        baseline = build_baseline(tok_old, corpus)
        alert = check_drift(baseline, tok_new, corpus, warn_pct=2, critical_pct=10)
        print("== Drift alert ==")
        print(f"  {alert.message}")
        print(f"  severity={alert.severity.value} triggered={alert.triggered}")
    except Exception as exc:  # noqa: BLE001 - example convenience
        print(f"Error (needs tokenizer downloads / network): {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
