"""
tokendrift.core.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~
A registry of model facts (tokenizer, context window, pricing) plus a cached
tokenizer resolver.

Every v1.1.0 feature that reasons about more than raw token counts (cost
estimation, context-window fit, migration safety, spend forecasting) needs to
map a friendly model name to three things:

1. which tokenizer encodes its prompts,
2. how large its context window is,
3. how much an input token costs.

:class:`ModelRegistry` is that map. It ships a small default set as a
convenience, loads from and saves to JSON, and is fully overridable at runtime.

Pricing and context windows change often and vary by provider contract. The
shipped defaults are indicative starting points, not a billing source of truth;
verify them against your provider and override them before relying on the cost
numbers for budgets or audits.

Usage
-----
>>> from tokendrift.core.registry import ModelRegistry
>>> reg = ModelRegistry.default()
>>> reg.get("gpt-4o").tokenizer
'o200k_base'
>>> reg.resolve("gpt-4o").encode("hello")        # cached UnifiedTokenizer
[24912]
"""

from __future__ import annotations

import json
from pathlib import Path

from tokendrift.core.loader import TokenizerLoader, UnifiedTokenizer
from tokendrift.models import ModelInfo

# ---------------------------------------------------------------------------
# Default model facts
# ---------------------------------------------------------------------------
#
# The tokenizer mappings below are correct and stable (gpt-4o-family models use
# o200k_base; the gpt-4 / gpt-3.5 family use cl100k_base). Context windows and
# prices are indicative and dated in ``notes``: confirm them with your provider
# before trusting the cost output. All default models use a tiktoken encoding so
# the registry works offline with no model download.

_PRICING_AS_OF = "indicative, as of 2026-06; verify with your provider"

_DEFAULT_MODELS: list[ModelInfo] = [
    ModelInfo(
        name="gpt-4o",
        tokenizer="o200k_base",
        context_window=128_000,
        price_per_1k_input=0.0025,
        price_per_1k_output=0.01,
        max_output_tokens=16_384,
        provider="openai",
        notes=_PRICING_AS_OF,
    ),
    ModelInfo(
        name="gpt-4o-mini",
        tokenizer="o200k_base",
        context_window=128_000,
        price_per_1k_input=0.00015,
        price_per_1k_output=0.0006,
        max_output_tokens=16_384,
        provider="openai",
        notes=_PRICING_AS_OF,
    ),
    ModelInfo(
        name="gpt-4-turbo",
        tokenizer="cl100k_base",
        context_window=128_000,
        price_per_1k_input=0.01,
        price_per_1k_output=0.03,
        max_output_tokens=4_096,
        provider="openai",
        notes=_PRICING_AS_OF,
    ),
    ModelInfo(
        name="gpt-3.5-turbo",
        tokenizer="cl100k_base",
        context_window=16_385,
        price_per_1k_input=0.0005,
        price_per_1k_output=0.0015,
        max_output_tokens=4_096,
        provider="openai",
        notes=_PRICING_AS_OF,
    ),
]


class ModelRegistry:
    """
    A name -> :class:`ModelInfo` map with a cached tokenizer resolver.

    Look-ups are case-insensitive on the model name. The same registry instance
    caches each loaded tokenizer, so resolving the same model (or two models
    that share a tokenizer) repeatedly does not reload it.
    """

    def __init__(self, models: list[ModelInfo] | None = None) -> None:
        self._models: dict[str, ModelInfo] = {}
        self._tok_cache: dict[str, UnifiedTokenizer] = {}
        for info in models or []:
            self.add(info)

    # -- construction --------------------------------------------------------

    @classmethod
    def default(cls) -> ModelRegistry:
        """Return a registry populated with the built-in default models."""
        return cls(list(_DEFAULT_MODELS))

    @classmethod
    def from_json(cls, path: str | Path) -> ModelRegistry:
        """
        Load a registry from a JSON file.

        The file is either a JSON list of model objects, or an object with a
        top-level ``"models"`` list. Each model object accepts the same fields
        as :class:`ModelInfo`; only ``name`` and ``tokenizer`` are required.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model registry file not found: {p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Registry file is not valid JSON: {exc}") from exc

        rows = data.get("models") if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ValueError("Registry JSON must be a list of models or an object with a 'models' list.")

        models: list[ModelInfo] = []
        for i, row in enumerate(rows):
            try:
                models.append(
                    ModelInfo(
                        name=str(row["name"]),
                        tokenizer=str(row["tokenizer"]),
                        context_window=_opt_int(row.get("context_window")),
                        price_per_1k_input=_opt_float(row.get("price_per_1k_input")),
                        price_per_1k_output=_opt_float(row.get("price_per_1k_output")),
                        max_output_tokens=_opt_int(row.get("max_output_tokens")),
                        provider=str(row.get("provider", "")),
                        notes=str(row.get("notes", "")),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Malformed model entry at index {i}: {exc}") from exc
        return cls(models)

    def to_json(self, path: str | Path) -> None:
        """Serialise the registry to a JSON file (a list of model objects)."""
        payload = [m.to_dict() for m in self._models.values()]
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # -- mutation ------------------------------------------------------------

    def add(self, info: ModelInfo) -> None:
        """
        Add or replace a model.

        The tokenizer cache is keyed by tokenizer identifier, not by model name,
        so adding or replacing a model never invalidates a cached tokenizer (a
        given identifier always resolves to the same tokenizer).
        """
        self._models[info.name.lower()] = info

    def register_tokenizer(self, identifier: str, tokenizer: UnifiedTokenizer) -> None:
        """
        Seed the tokenizer cache with a ready-made tokenizer for *identifier*.

        Use this to plug an in-memory or custom :class:`UnifiedTokenizer` into
        the registry so :meth:`resolve` returns it instead of loading by name.
        Any model whose ``tokenizer`` equals *identifier* then uses it.
        """
        self._tok_cache[identifier] = tokenizer

    # -- access --------------------------------------------------------------

    def names(self) -> list[str]:
        """All registered model names, sorted."""
        return sorted(m.name for m in self._models.values())

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._models

    def __len__(self) -> int:
        return len(self._models)

    def get(self, name: str) -> ModelInfo:
        """
        Return the :class:`ModelInfo` for *name*.

        If *name* is not a registered model but is a usable tokenizer identifier
        on its own (a tiktoken encoding, a HuggingFace id, or a local path), a
        bare ``ModelInfo`` wrapping it is returned so callers can pass raw
        tokenizer names anywhere a model name is accepted.
        """
        info = self._models.get(name.lower())
        if info is not None:
            return info
        # Fall back to treating the name as a raw tokenizer identifier.
        return ModelInfo(name=name, tokenizer=name)

    def resolve(self, name: str) -> UnifiedTokenizer:
        """Return a cached :class:`UnifiedTokenizer` for *name*."""
        info = self.get(name)
        cached = self._tok_cache.get(info.tokenizer)
        if cached is None:
            cached = TokenizerLoader.load(info.tokenizer)
            self._tok_cache[info.tokenizer] = cached
        return cached


def _opt_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        return int(value)
    raise TypeError(f"expected a number, got {type(value).__name__}")


def _opt_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"expected a number, got {type(value).__name__}")
