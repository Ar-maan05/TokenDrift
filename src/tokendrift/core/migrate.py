"""
tokendrift.core.migrate
~~~~~~~~~~~~~~~~~~~~~~~~~
Model migration safety checker.

When a customer switches from one model to another, their existing prompts can
tokenize differently enough to break budget assumptions, overflow the new
context window, or shift cost. :func:`migrate_report` runs that migration as a
diff over a corpus of historical prompts and returns a single report covering:

* total and per-prompt token count delta,
* input cost delta (when both models are priced),
* vocabulary shift between the two tokenizers,
* every prompt that would exceed the target model's context window.

This is the concrete deliverable for a compliance and audit layer: a
machine-readable :meth:`MigrationReport.to_dict` and a human-readable terminal
render of the same facts.

Usage
-----
>>> from tokendrift.core.migrate import migrate_report
>>> from tokendrift.models import CorpusEntry
>>> entries = [CorpusEntry(id="p1", text="Summarise the filing.")]
>>> report = migrate_report(entries, "gpt-4-turbo", "gpt-4o")
>>> report.token_delta, report.overflows
"""

from __future__ import annotations

from tokendrift.core.differ import EncodingDiffer
from tokendrift.core.registry import ModelRegistry
from tokendrift.core.vocab import VocabDiffer
from tokendrift.models import (
    CorpusEntry,
    MigrationOverflow,
    MigrationReport,
    PromptCostDelta,
)


def migrate_report(
    entries: list[CorpusEntry],
    source_model: str,
    target_model: str,
    registry: ModelRegistry | None = None,
    reserved_output: int | None = None,
    include_vocab: bool = True,
) -> MigrationReport:
    """
    Build a :class:`MigrationReport` for moving *entries* from *source_model*
    to *target_model*.

    Parameters
    ----------
    entries:
        Sample of historical prompts to evaluate.
    source_model, target_model:
        Registered model names, or raw tokenizer identifiers.
    registry:
        Registry to resolve model facts and tokenizers. Defaults to
        :meth:`ModelRegistry.default`.
    reserved_output:
        Tokens to reserve for the completion when deciding whether a prompt
        overflows the target context window. ``None`` uses the target model's
        ``max_output_tokens`` (or 0 when unknown).
    include_vocab:
        Compute the tokenizer vocabulary diff. Set ``False`` to skip it for
        very large vocabularies when only counts and overflow matter.
    """
    registry = registry or ModelRegistry.default()

    source = registry.get(source_model)
    target = registry.get(target_model)
    tok_source = registry.resolve(source_model)
    tok_target = registry.resolve(target_model)

    differ = EncodingDiffer(detect_boundaries=False)
    pairs = [(e.id, e.text) for e in entries]
    diffs = differ.diff_many(pairs, tok_source, tok_target)

    price_source = source.price_per_1k_input
    price_target = target.price_per_1k_input

    per_prompt = [
        PromptCostDelta(
            entry_id=d.entry_id,
            tokens_a=d.token_count_a,
            tokens_b=d.token_count_b,
            delta=d.count_delta,
            cost_a_usd=(d.token_count_a / 1_000 * price_source) if price_source is not None else None,
            cost_b_usd=(d.token_count_b / 1_000 * price_target) if price_target is not None else None,
        )
        for d in diffs
    ]

    total_source = sum(d.token_count_a for d in diffs)
    total_target = sum(d.token_count_b for d in diffs)

    cost_source = (total_source / 1_000 * price_source) if price_source is not None else None
    cost_target = (total_target / 1_000 * price_target) if price_target is not None else None
    cost_delta = (cost_target - cost_source) if (cost_source is not None and cost_target is not None) else None

    # Context-window overflow against the target model.
    reserve = reserved_output if reserved_output is not None else (target.max_output_tokens or 0)
    reserve = max(0, reserve)
    overflows: list[MigrationOverflow] = []
    if target.context_window is not None:
        for d in diffs:
            if d.token_count_b + reserve > target.context_window:
                overflows.append(
                    MigrationOverflow(
                        entry_id=d.entry_id,
                        target_tokens=d.token_count_b,
                        context_window=target.context_window,
                        reserved_output=reserve,
                    )
                )

    vocab = VocabDiffer().diff(tok_source, tok_target) if include_vocab else None

    return MigrationReport(
        source_model=source.name,
        target_model=target.name,
        total_tokens_source=total_source,
        total_tokens_target=total_target,
        per_prompt=per_prompt,
        overflows=overflows,
        cost_source_usd=cost_source,
        cost_target_usd=cost_target,
        cost_delta_usd=cost_delta,
        vocab=vocab,
    )
