"""Tests for tokendrift.core.alert."""

from __future__ import annotations

import json

import pytest

from tokendrift.core.alert import alert_to_json, check_drift
from tokendrift.core.baseline import build_baseline
from tokendrift.models import AlertSeverity, CorpusEntry


@pytest.fixture()
def entries():
    return [CorpusEntry(id="p1", text="abcd"), CorpusEntry(id="p2", text="abcdef")]


def test_no_drift_is_ok(tok_a, entries):
    baseline = build_baseline(tok_a, entries)
    alert = check_drift(baseline, tok_a, entries, warn_pct=2.0, critical_pct=10.0)
    assert alert.severity is AlertSeverity.OK
    assert alert.triggered is False
    assert alert.token_delta == 0


def test_drift_classified_warn(tok_a, tok_b, entries):
    # char baseline (10 tokens) vs bigram (5 tokens) -> -50% drift.
    baseline = build_baseline(tok_a, entries)
    alert = check_drift(baseline, tok_b, entries, warn_pct=2.0, critical_pct=100.0)
    assert alert.severity is AlertSeverity.WARN
    assert alert.triggered is True


def test_drift_classified_critical(tok_a, tok_b, entries):
    baseline = build_baseline(tok_a, entries)
    alert = check_drift(baseline, tok_b, entries, warn_pct=2.0, critical_pct=10.0)
    assert alert.severity is AlertSeverity.CRITICAL


def test_cost_delta_included(tok_a, tok_b, entries):
    baseline = build_baseline(tok_a, entries)
    alert = check_drift(baseline, tok_b, entries, price_per_1k=1.0)
    assert alert.cost_delta_usd is not None
    assert alert.cost_delta_usd < 0  # tokens shrank


def test_critical_below_warn_raises(tok_a, entries):
    baseline = build_baseline(tok_a, entries)
    with pytest.raises(ValueError, match="greater than or equal"):
        check_drift(baseline, tok_a, entries, warn_pct=10.0, critical_pct=2.0)


def test_alert_to_json_round_trips(tok_a, tok_b, entries):
    baseline = build_baseline(tok_a, entries)
    alert = check_drift(baseline, tok_b, entries)
    payload = json.loads(alert_to_json(alert))
    assert payload["severity"] == alert.severity.value
    assert payload["triggered"] is True
    assert payload["token_delta"] == alert.token_delta
