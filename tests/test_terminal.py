"""Tests for tokenlens.report.terminal.

These assert the renderers run without error and emit the key facts. Rich
output is captured with a recording Console so no real terminal is needed.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from tokenlens.models import (
    BoundaryViolation,
    CostReport,
    PromptCostDelta,
    RemappedEntry,
    TokenDiff,
    ViolationType,
    VocabDiff,
    VocabEntry,
)
from tokenlens.report.terminal import (
    render_cost_report,
    render_encoding_diff,
    render_entry_detail,
    render_vocab_diff,
)


@pytest.fixture
def cap():
    return Console(record=True, width=120)


def _text(console: Console) -> str:
    return console.export_text()


def _vocab_diff() -> VocabDiff:
    return VocabDiff(
        added=[VocabEntry("foo", 1)],
        deleted=[VocabEntry("bar", 2)],
        remapped=[RemappedEntry("baz", 3, 4)],
        total_a=100,
        total_b=120,
    )


def test_render_vocab_diff_summary(cap):
    render_vocab_diff(_vocab_diff(), "A", "B", console=cap, show="summary")
    out = _text(cap)
    assert "Vocab Diff" in out
    assert "100" in out and "120" in out


def test_render_vocab_diff_remap_warning(cap):
    render_vocab_diff(_vocab_diff(), "A", "B", console=cap, show="remapped")
    out = _text(cap)
    # The remap warning fires and the remapped token is listed
    assert "point elsewhere" in out
    assert "baz" in out


def test_render_vocab_diff_no_remap_no_warning(cap):
    d = VocabDiff(added=[], deleted=[], remapped=[], total_a=5, total_b=5)
    render_vocab_diff(d, "A", "B", console=cap)
    assert "point elsewhere" not in _text(cap)


def _diffs():
    return [
        TokenDiff("p1", "hello world", 3, 5, 2, 0),
        TokenDiff("p2", "identical", 4, 4, 0, 9),
    ]


def test_render_encoding_diff_runs(cap):
    render_encoding_diff(_diffs(), "A", "B", console=cap)
    out = _text(cap)
    assert "Encoding Diff" in out
    assert "Corpus entries" in out
    assert "most-affected" in out


def test_render_encoding_diff_hides_boundary_row_when_none(cap):
    """No boundary detection ran → no boundary row, no severity wording."""
    render_encoding_diff(_diffs(), "A", "B", console=cap)
    out = _text(cap)
    assert "Boundary changes" not in out
    assert "HIGH" not in out


def test_render_encoding_diff_shows_structural_boundary_row(cap):
    diffs = [
        TokenDiff(
            "p1",
            "hello",
            1,
            2,
            1,
            0,
            boundary_violations=[
                BoundaryViolation("hello", 0, 5, ["hello"], ["hel", "lo"], [1], [2, 3], ViolationType.SPLIT)
            ],
        )
    ]
    render_encoding_diff(diffs, "A", "B", console=cap)
    out = _text(cap)
    assert "Boundary changes" in out
    assert "split" in out
    assert "experimental" in out


def test_render_cost_report_with_prices(cap):
    report = CostReport(
        total_tokens_a=1000,
        total_tokens_b=2000,
        token_delta=1000,
        per_prompt=[PromptCostDelta("p1", 1000, 2000, 1000, 0.03, 0.02)],
        cost_a_usd=0.03,
        cost_b_usd=0.02,
        cost_delta_usd=-0.01,
    )
    render_cost_report(report, "A", "B", console=cap)
    out = _text(cap)
    assert "Cost Report" in out
    assert "$0.03" in out


def test_render_cost_report_without_prices(cap):
    report = CostReport(
        total_tokens_a=10,
        total_tokens_b=12,
        token_delta=2,
        per_prompt=[],
    )
    render_cost_report(report, "A", "B", console=cap)
    out = _text(cap)
    assert "Cost (A)" not in out  # no price → no cost rows


def test_render_entry_detail_no_severity_column(cap):
    diff = TokenDiff(
        "p1",
        "ChatGPT rewrites tests",
        5,
        4,
        -1,
        4,
        boundary_violations=[
            BoundaryViolation(
                "rewrites",
                8,
                16,
                [" re", "writes"],
                [" rew", "rites"],
                [1, 2],
                [3, 4],
                ViolationType.RESEGMENT,
            )
        ],
    )
    render_entry_detail(diff, "A", "B", console=cap)
    out = _text(cap)
    assert "rewrites" in out
    assert "RESEGMENT" in out
    assert "Sev" not in out  # severity column removed
    assert "experimental" in out


def test_render_entry_detail_no_violations(cap):
    diff = TokenDiff("p1", "hi", 1, 1, 0, 2)
    render_entry_detail(diff, "A", "B", console=cap)
    assert "No boundary changes" in _text(cap)
