"""CLI smoke tests for the v1.1.0 commands.

TokenizerLoader.load is patched at the source module so the registry resolves
the offline mock tokenizers: o200k_base -> bigram, everything else -> char.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tests.conftest import MockTokenizerA, MockTokenizerB
from tokendrift.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def patch_loader(monkeypatch):
    def fake_load(identifier: str):
        return MockTokenizerB() if "o200k" in identifier else MockTokenizerA()

    # Patch at the source so registry.resolve() uses the mocks too.
    monkeypatch.setattr("tokendrift.core.loader.TokenizerLoader.load", staticmethod(fake_load))
    monkeypatch.setattr("tokendrift.cli.main.TokenizerLoader.load", staticmethod(fake_load))


@pytest.fixture
def corpus(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"id": "p1", "text": "abcd"}\n{"id": "p2", "text": "abcdef"}\n', encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# estimate
# ---------------------------------------------------------------------------


def test_estimate_text():
    result = runner.invoke(app, ["estimate", "gpt-4o,gpt-4-turbo", "--text", "abcd"])
    assert result.exit_code == 0
    assert "gpt-4o" in result.stdout


def test_estimate_json():
    result = runner.invoke(app, ["estimate", "gpt-4o", "--text", "abcd", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["estimates"][0]["model"] == "gpt-4o"


def test_estimate_requires_input():
    result = runner.invoke(app, ["estimate", "gpt-4o"])
    assert result.exit_code == 2


def test_estimate_from_file(tmp_path):
    f = tmp_path / "prompt.txt"
    f.write_text("abcdef", encoding="utf-8")
    result = runner.invoke(app, ["estimate", "gpt-4o", "--file", str(f)])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# migrate
# ---------------------------------------------------------------------------


def test_migrate(corpus):
    result = runner.invoke(app, ["migrate", "gpt-4-turbo", "gpt-4o", "--corpus", str(corpus)])
    assert result.exit_code == 0
    assert "Migration Report" in result.stdout


def test_migrate_json(corpus):
    result = runner.invoke(app, ["migrate", "gpt-4-turbo", "gpt-4o", "--corpus", str(corpus), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["source_model"] == "gpt-4-turbo"
    assert payload["target_model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# compress
# ---------------------------------------------------------------------------


def test_compress(tmp_path):
    orig = tmp_path / "o.txt"
    comp = tmp_path / "c.txt"
    orig.write_text("abcdefgh", encoding="utf-8")
    comp.write_text("abcd", encoding="utf-8")
    result = runner.invoke(app, ["compress", "gpt-4o", "--original", str(orig), "--compressed", str(comp)])
    assert result.exit_code == 0
    assert "Compression Savings" in result.stdout


# ---------------------------------------------------------------------------
# forecast
# ---------------------------------------------------------------------------


def test_forecast(corpus):
    result = runner.invoke(app, ["forecast", "gpt-4o,gpt-4o-mini", "--corpus", str(corpus), "--requests", "1000"])
    assert result.exit_code == 0
    assert "Cost Forecast" in result.stdout


def test_forecast_json(corpus):
    result = runner.invoke(app, ["forecast", "gpt-4o", "--corpus", str(corpus), "--requests", "1000", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["projected_requests"] == 1000


# ---------------------------------------------------------------------------
# drift-alert
# ---------------------------------------------------------------------------


def test_drift_alert_ok_exits_zero(corpus, tmp_path):
    base = tmp_path / "base.json"
    runner.invoke(app, ["baseline", "cl100k_base", "--corpus", str(corpus), "-o", str(base)])
    # Same tokenizer -> no drift -> OK -> exit 0.
    result = runner.invoke(app, ["drift-alert", "cl100k_base", "--baseline", str(base), "--corpus", str(corpus)])
    assert result.exit_code == 0
    assert "OK" in result.stdout


def test_drift_alert_critical_exits_one(corpus, tmp_path):
    base = tmp_path / "base.json"
    runner.invoke(app, ["baseline", "cl100k_base", "--corpus", str(corpus), "-o", str(base)])
    # cl100k (char mock) -> o200k (bigram mock): big drift -> CRITICAL -> exit 1.
    result = runner.invoke(
        app,
        ["drift-alert", "o200k_base", "--baseline", str(base), "--corpus", str(corpus), "--critical-pct", "10"],
    )
    assert result.exit_code == 1
    assert "CRITICAL" in result.stdout


def test_drift_alert_json(corpus, tmp_path):
    base = tmp_path / "base.json"
    runner.invoke(app, ["baseline", "cl100k_base", "--corpus", str(corpus), "-o", str(base)])
    result = runner.invoke(
        app, ["drift-alert", "o200k_base", "--baseline", str(base), "--corpus", str(corpus), "--json"]
    )
    payload = json.loads(result.stdout)
    assert payload["severity"] in {"OK", "WARN", "CRITICAL"}


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def test_models_lists_registry():
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "gpt-4o" in result.stdout


def test_models_custom_registry(tmp_path):
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps([{"name": "custom-model", "tokenizer": "cl100k_base"}]), encoding="utf-8")
    result = runner.invoke(app, ["models", "--registry", str(reg)])
    assert result.exit_code == 0
    assert "custom-model" in result.stdout
