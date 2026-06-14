"""Tests for tokenlens.report.cost."""

from __future__ import annotations

from tokenlens.models import TokenDiff
from tokenlens.report.cost import CostCalculator


def _diff(eid: str, a: int, b: int) -> TokenDiff:
    return TokenDiff(
        entry_id=eid,
        text="x",
        token_count_a=a,
        token_count_b=b,
        count_delta=b - a,
        first_divergence_pos=0,
    )


def test_token_totals():
    diffs = [_diff("p1", 10, 12), _diff("p2", 5, 4)]
    r = CostCalculator().compute(diffs)
    assert r.total_tokens_a == 15
    assert r.total_tokens_b == 16
    assert r.token_delta == 1


def test_cost_arithmetic():
    diffs = [_diff("p1", 1000, 2000)]
    r = CostCalculator().compute(diffs, price_a=0.03, price_b=0.01)
    assert r.cost_a_usd == 0.03
    assert r.cost_b_usd == 0.02
    assert abs(r.cost_delta_usd - (-0.01)) < 1e-9


def test_no_prices_leaves_costs_none():
    r = CostCalculator().compute([_diff("p1", 10, 12)])
    assert r.cost_a_usd is None
    assert r.cost_b_usd is None
    assert r.cost_delta_usd is None
    assert r.per_prompt[0].cost_a_usd is None


def test_zero_price_is_honoured_per_prompt():
    """
    A price of 0.0 is a valid price (free tier). It must produce 0.0 costs,
    not None, and per-prompt must agree with the corpus total.
    """
    diffs = [_diff("p1", 10, 12)]
    r = CostCalculator().compute(diffs, price_a=0.0, price_b=0.0)
    assert r.cost_a_usd == 0.0
    assert r.cost_b_usd == 0.0
    assert r.per_prompt[0].cost_a_usd == 0.0
    assert r.per_prompt[0].cost_b_usd == 0.0


def test_pct_change_zero_when_no_tokens():
    r = CostCalculator().compute([])
    assert r.pct_change == 0.0
