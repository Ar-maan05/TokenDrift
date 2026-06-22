"""
tokendrift.core.compression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Prompt-compression feedback loop.

A compression step (for example middle-out compression that drops the least
salient middle of a long prompt) reduces characters, but the token saving is
tokenizer dependent: the same compressed text saves a different number of
tokens, and a different amount of money, under each model you might dispatch to.

:func:`compression_report` measures the real saving per model so a compression
decision can be model aware instead of applying one generic ratio everywhere.

Usage
-----
>>> from tokendrift.core.compression import compression_report
>>> report = compression_report(original, compressed, ["gpt-4o", "gpt-4-turbo"])
>>> for s in report.savings:
...     print(s.model, s.tokens_saved, s.pct_saved)
"""

from __future__ import annotations

from tokendrift.core.registry import ModelRegistry
from tokendrift.models import (
    CompressionReport,
    CompressionSaving,
    CorpusEntry,
)


def compression_report(
    original: str,
    compressed: str,
    models: list[str],
    registry: ModelRegistry | None = None,
) -> CompressionReport:
    """
    Measure the token and cost saving of *compressed* versus *original* across
    *models*.

    Parameters
    ----------
    original, compressed:
        The prompt before and after compression.
    models:
        Registered model names, or raw tokenizer identifiers.
    registry:
        Registry to resolve model facts and tokenizers. Defaults to
        :meth:`ModelRegistry.default`.
    """
    registry = registry or ModelRegistry.default()

    savings: list[CompressionSaving] = []
    for model in models:
        info = registry.get(model)
        tok = registry.resolve(model)
        original_tokens = len(tok.encode(original))
        compressed_tokens = len(tok.encode(compressed))

        cost_saved: float | None = None
        if info.price_per_1k_input is not None:
            cost_saved = ((original_tokens - compressed_tokens) / 1_000) * info.price_per_1k_input

        savings.append(
            CompressionSaving(
                model=info.name,
                tokenizer=info.tokenizer,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                cost_saved_usd=cost_saved,
            )
        )

    return CompressionReport(
        original_chars=len(original),
        compressed_chars=len(compressed),
        savings=savings,
    )


def compression_report_corpus(
    pairs: list[tuple[CorpusEntry, CorpusEntry]],
    models: list[str],
    registry: ModelRegistry | None = None,
) -> CompressionReport:
    """
    Aggregate compression savings across many ``(original, compressed)`` entry
    pairs, summing tokens and cost per model into one :class:`CompressionReport`.

    Each pair is ``(original_entry, compressed_entry)``. Entry ids are not
    matched; the caller is responsible for aligning the pairs.
    """
    registry = registry or ModelRegistry.default()

    orig_chars = sum(len(o.text) for o, _ in pairs)
    comp_chars = sum(len(c.text) for _, c in pairs)

    savings: list[CompressionSaving] = []
    for model in models:
        info = registry.get(model)
        tok = registry.resolve(model)
        original_tokens = sum(len(tok.encode(o.text)) for o, _ in pairs)
        compressed_tokens = sum(len(tok.encode(c.text)) for _, c in pairs)

        cost_saved: float | None = None
        if info.price_per_1k_input is not None:
            cost_saved = ((original_tokens - compressed_tokens) / 1_000) * info.price_per_1k_input

        savings.append(
            CompressionSaving(
                model=info.name,
                tokenizer=info.tokenizer,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                cost_saved_usd=cost_saved,
            )
        )

    return CompressionReport(original_chars=orig_chars, compressed_chars=comp_chars, savings=savings)
