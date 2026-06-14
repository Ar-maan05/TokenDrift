"""
tokendrift.report.cost
~~~~~~~~~~~~~~~~~~~~~
Cost impact calculator: given a list of ``TokenDiff`` objects and per-token
pricing for two models, produces a ``CostReport``.

Usage
-----
>>> from tokendrift.report.cost import CostCalculator
>>> report = CostCalculator().compute(diffs, price_a=0.03, price_b=0.01)
>>> print(f"Delta: ${report.cost_delta_usd:.4f}")
"""

from __future__ import annotations

from tokendrift.models import CostReport, PromptCostDelta, TokenDiff


class CostCalculator:
    """Computes corpus-level and per-prompt cost impact of a tokenizer change."""

    def compute(
        self,
        diffs: list[TokenDiff],
        price_a: float | None = None,
        price_b: float | None = None,
    ) -> CostReport:
        """
        Compute cost impact across a list of ``TokenDiff`` objects.

        Parameters
        ----------
        diffs:
            Output of ``EncodingDiffer.diff_many`` or equivalent.
        price_a:
            Price per 1 000 tokens for tokenizer A (USD).  If ``None``,
            cost fields in the report will be ``None``.
        price_b:
            Price per 1 000 tokens for tokenizer B (USD).

        Returns
        -------
        CostReport
        """
        total_a = sum(d.token_count_a for d in diffs)
        total_b = sum(d.token_count_b for d in diffs)

        cost_a: float | None = None
        cost_b: float | None = None
        cost_delta: float | None = None

        if price_a is not None:
            cost_a = (total_a / 1_000) * price_a
        if price_b is not None:
            cost_b = (total_b / 1_000) * price_b
        if cost_a is not None and cost_b is not None:
            cost_delta = cost_b - cost_a

        per_prompt = [
            PromptCostDelta(
                entry_id=d.entry_id,
                tokens_a=d.token_count_a,
                tokens_b=d.token_count_b,
                delta=d.count_delta,
                cost_a_usd=(d.token_count_a / 1_000 * price_a) if price_a is not None else None,
                cost_b_usd=(d.token_count_b / 1_000 * price_b) if price_b is not None else None,
            )
            for d in diffs
        ]

        return CostReport(
            total_tokens_a=total_a,
            total_tokens_b=total_b,
            token_delta=total_b - total_a,
            per_prompt=per_prompt,
            cost_a_usd=cost_a,
            cost_b_usd=cost_b,
            cost_delta_usd=cost_delta,
        )
