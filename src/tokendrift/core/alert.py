"""
tokendrift.core.alert
~~~~~~~~~~~~~~~~~~~~~~~
Tokenizer drift alerts for compliance.

A provider silently updating their tokenizer between model versions is an audit
event: a customer's prompts can suddenly cost more, or a category of content can
start tokenizing differently (which can in turn affect guardrail behaviour).
:func:`check_drift` re-encodes a committed baseline corpus under a (usually
newer) tokenizer, measures the total drift, and classifies it against
configurable warn and critical thresholds.

It is meant to run as a background job against new model versions as they
appear, and to feed an audit or compliance layer: :class:`DriftAlert` serialises
to a flat dict and JSON string for that pipeline.

This builds directly on the baseline / CI machinery; :func:`check_drift` is the
alerting view of the same comparison that :func:`tokendrift.run_ci` gates on.

Usage
-----
>>> from tokendrift.core.alert import check_drift
>>> from tokendrift.core.baseline import Baseline
>>> from tokendrift.core.loader import TokenizerLoader
>>> baseline = Baseline.load("tokendrift.baseline.json")
>>> alert = check_drift(baseline, TokenizerLoader.load("o200k_base"), entries,
...                     warn_pct=2.0, critical_pct=10.0)
>>> alert.severity, alert.triggered
"""

from __future__ import annotations

import json

from tokendrift.core.baseline import Baseline, CIReport, CIThresholds, run_ci
from tokendrift.core.loader import UnifiedTokenizer
from tokendrift.models import AlertSeverity, CorpusEntry, DriftAlert


def check_drift(
    baseline: Baseline,
    tokenizer: UnifiedTokenizer,
    entries: list[CorpusEntry],
    warn_pct: float = 2.0,
    critical_pct: float = 10.0,
    price_per_1k: float | None = None,
) -> DriftAlert:
    """
    Compare *entries* re-encoded under *tokenizer* against *baseline* and return
    a :class:`DriftAlert`.

    Parameters
    ----------
    baseline:
        A committed :class:`Baseline` snapshot.
    tokenizer:
        The (usually newer) tokenizer to test.
    entries:
        The same corpus the baseline was built from.
    warn_pct:
        Absolute total-drift percentage at or above which severity is WARN.
    critical_pct:
        Absolute total-drift percentage at or above which severity is CRITICAL.
        Must be greater than or equal to *warn_pct*.
    price_per_1k:
        USD per 1,000 tokens, used to estimate the cost delta on the alert.

    Raises
    ------
    ValueError
        If *critical_pct* is less than *warn_pct*.
    """
    if critical_pct < warn_pct:
        raise ValueError("critical_pct must be greater than or equal to warn_pct.")

    report: CIReport = run_ci(
        baseline,
        tokenizer,
        entries,
        CIThresholds(price_per_1k=price_per_1k),
    )

    total_pct = report.total_pct
    magnitude = abs(total_pct) if total_pct != float("inf") else float("inf")

    if magnitude >= critical_pct:
        severity = AlertSeverity.CRITICAL
    elif magnitude >= warn_pct:
        severity = AlertSeverity.WARN
    else:
        severity = AlertSeverity.OK

    return DriftAlert(
        baseline_tokenizer=report.baseline_tokenizer,
        current_tokenizer=report.current_tokenizer,
        total_pct=total_pct,
        token_delta=report.token_delta,
        warn_pct=warn_pct,
        critical_pct=critical_pct,
        severity=severity,
        cost_delta_usd=report.cost_delta_usd,
        message=_message(report, severity),
    )


def _message(report: CIReport, severity: AlertSeverity) -> str:
    pct = "inf" if report.total_pct == float("inf") else f"{report.total_pct:+.2f}%"
    base = (
        f"{severity.value}: token drift {pct} ({report.token_delta:+,} tokens) "
        f"from {report.baseline_tokenizer} to {report.current_tokenizer}"
    )
    if report.cost_delta_usd is not None:
        base += f"; est. cost delta ${report.cost_delta_usd:+.4f}"
    return base


def alert_to_json(alert: DriftAlert, indent: int | None = 2) -> str:
    """Serialise a :class:`DriftAlert` to a JSON string for an audit pipeline."""
    return json.dumps(alert.to_dict(), indent=indent, sort_keys=True)
