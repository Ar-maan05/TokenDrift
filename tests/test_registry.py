"""Tests for tokendrift.core.registry."""

from __future__ import annotations

import json

import pytest

from tokendrift.core.registry import ModelRegistry
from tokendrift.models import ModelInfo


def test_default_registry_has_known_models():
    reg = ModelRegistry.default()
    assert "gpt-4o" in reg
    assert reg.get("gpt-4o").tokenizer == "o200k_base"
    assert reg.get("gpt-4-turbo").tokenizer == "cl100k_base"
    assert len(reg) >= 4


def test_lookup_is_case_insensitive():
    reg = ModelRegistry.default()
    assert reg.get("GPT-4O").name == "gpt-4o"
    assert "GPT-4O" in reg


def test_unknown_name_falls_back_to_raw_tokenizer():
    reg = ModelRegistry.default()
    info = reg.get("cl100k_base")
    # Not a registered model, so it is wrapped as a bare tokenizer identifier.
    assert info.name == "cl100k_base"
    assert info.tokenizer == "cl100k_base"
    assert info.price_per_1k_input is None


def test_add_replaces_model(tok_a):
    reg = ModelRegistry()
    reg.add(ModelInfo(name="m", tokenizer="t", context_window=100))
    reg.register_tokenizer("t", tok_a)
    assert reg.resolve("m") is tok_a
    # Replace the model with a different tokenizer id.
    reg.add(ModelInfo(name="m", tokenizer="t2"))
    reg.register_tokenizer("t2", tok_a)
    assert reg.get("m").tokenizer == "t2"
    assert len(reg) == 1


def test_register_tokenizer_survives_later_add(tok_a):
    """A seeded tokenizer is not evicted when another model reusing it is added."""
    reg = ModelRegistry()
    reg.add(ModelInfo(name="m", tokenizer="t"))
    reg.register_tokenizer("t", tok_a)
    reg.add(ModelInfo(name="n", tokenizer="t"))  # reuses the same tokenizer id
    assert reg.resolve("n") is tok_a


def test_resolve_caches_tokenizer(tok_a):
    reg = ModelRegistry()
    reg.add(ModelInfo(name="m", tokenizer="shared"))
    reg.add(ModelInfo(name="n", tokenizer="shared"))
    reg.register_tokenizer("shared", tok_a)
    # Two models sharing a tokenizer id return the very same instance.
    assert reg.resolve("m") is reg.resolve("n")


def test_json_round_trip(tmp_path):
    reg = ModelRegistry.default()
    path = tmp_path / "reg.json"
    reg.to_json(path)
    loaded = ModelRegistry.from_json(path)
    assert loaded.names() == reg.names()
    assert loaded.get("gpt-4o").context_window == reg.get("gpt-4o").context_window


def test_from_json_accepts_object_with_models_key(tmp_path):
    path = tmp_path / "reg.json"
    path.write_text(
        json.dumps({"models": [{"name": "x", "tokenizer": "cl100k_base", "context_window": 8}]}),
        encoding="utf-8",
    )
    reg = ModelRegistry.from_json(path)
    assert reg.get("x").context_window == 8


def test_from_json_missing_file():
    with pytest.raises(FileNotFoundError):
        ModelRegistry.from_json("/nonexistent/reg.json")


def test_from_json_malformed_entry(tmp_path):
    path = tmp_path / "reg.json"
    path.write_text(json.dumps([{"name": "x"}]), encoding="utf-8")  # missing tokenizer
    with pytest.raises(ValueError, match="Malformed model entry"):
        ModelRegistry.from_json(path)


def test_from_json_rejects_non_list(tmp_path):
    path = tmp_path / "reg.json"
    path.write_text(json.dumps({"not_models": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="list of models"):
        ModelRegistry.from_json(path)
