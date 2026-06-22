"""Tests for tokendrift.report.terminal.

These assert the renderers run without error and emit the key facts. Rich
output is captured with a recording Console so no real terminal is needed.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from tokendrift.models import (
    BoundaryViolation,
    CostReport,
    PromptCostDelta,
    RemappedEntry,
    TokenDiff,
    ViolationType,
    VocabDiff,
    VocabEntry,
)
from tokendrift.report.terminal import (
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


# ---------------------------------------------------------------------------
# v1.1.0 renderers
# ---------------------------------------------------------------------------

from tokendrift.models import (  # noqa: E402
    AlertSeverity,
    CompressionReport,
    CompressionSaving,
    DriftAlert,
    ForecastReport,
    MigrationOverflow,
    MigrationReport,
    ModelEstimate,
    ModelForecast,
    MultiModelEstimate,
)
from tokendrift.report.terminal import (  # noqa: E402
    render_compression_report,
    render_drift_alert,
    render_estimate,
    render_forecast_report,
    render_migration_report,
)


def test_render_estimate(cap):
    result = MultiModelEstimate(
        text_chars=10,
        estimates=[
            ModelEstimate("gpt-4o", "o200k_base", 10, 0.001, 128_000, 16, True, 100),
            ModelEstimate("local", "x", 20, None, None, 0, None, None),
        ],
    )
    render_estimate(result, console=cap)
    out = _text(cap)
    assert "gpt-4o" in out
    assert "Tokenization spread" in out
    assert "cheapest" in out


def test_render_estimate_overflow(cap):
    result = MultiModelEstimate(
        text_chars=10,
        estimates=[ModelEstimate("m", "t", 200, None, 100, 0, False, -100)],
    )
    render_estimate(result, console=cap)
    assert "OVERFLOW" in _text(cap)


def test_render_migration_report(cap):
    report = MigrationReport(
        source_model="a",
        target_model="b",
        total_tokens_source=100,
        total_tokens_target=120,
        overflows=[MigrationOverflow("p1", 200, 128, 0)],
    )
    render_migration_report(report, console=cap)
    out = _text(cap)
    assert "Migration Report" in out
    assert "overflow" in out.lower()
    assert "p1" in out


def test_render_compression_report(cap):
    report = CompressionReport(
        original_chars=100,
        compressed_chars=50,
        savings=[CompressionSaving("m", "t", 40, 20, 0.001)],
    )
    render_compression_report(report, console=cap)
    out = _text(cap)
    assert "Compression Savings" in out
    assert "50.0%" in out


def test_render_forecast_report(cap):
    report = ForecastReport(
        projected_requests=1000,
        forecasts=[ModelForecast("m", "t", 2, 10, 1000, 1.0)],
    )
    render_forecast_report(report, console=cap)
    out = _text(cap)
    assert "Cost Forecast" in out
    assert "1,000" in out


def test_render_drift_alert(cap):
    alert = DriftAlert("a", "b", 5.0, 50, 2.0, 10.0, AlertSeverity.WARN, None, "WARN: drift")
    render_drift_alert(alert, console=cap)
    out = _text(cap)
    assert "WARN" in out
    assert "Tokenizer Drift" in out
