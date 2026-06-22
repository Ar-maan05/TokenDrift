"""
tokendrift.core.estimate
~~~~~~~~~~~~~~~~~~~~~~~~~~
Pre-dispatch token, cost, and context-window estimation across many models.

This is the engine behind two features:

* A playground cost overlay: score the same prompt for every model a user is
  comparing, before the request is sent, so the side-by-side view shows a cost
  estimate per model up front.
* A routing-engine budget check: confirm a prompt fits a target model's context
  window using that model's own tokenizer, not a generic character-count
  approximation that can silently overflow the window.

Because different models tokenize the same text differently, the estimate also
reports the spread between the lowest and highest token counts, which is what
makes cost and latency diverge between providers.

Usage
-----
>>> from tokendrift.core.estimate import CostEstimator
>>> est = CostEstimator()
>>> result = est.estimate("Summarise this contract.", ["gpt-4o", "gpt-4-turbo"])
>>> for e in result.estimates:
...     print(e.model, e.token_count, e.cost_usd, e.fits)
"""

from __future__ import annotations

from tokendrift.core.registry import ModelRegistry
from tokendrift.models import ModelEstimate, ModelInfo, MultiModelEstimate


class CostEstimator:
    """
    Estimates per-model token count, input cost, and context-window fit for a
    text, before any request is dispatched.

    Parameters
    ----------
    registry:
        The :class:`ModelRegistry` to resolve model facts and tokenizers from.
        Defaults to :meth:`ModelRegistry.default`.
    """

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self._registry = registry or ModelRegistry.default()

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    # ------------------------------------------------------------------

    def estimate_one(
        self,
        text: str,
        model: str,
        reserved_output: int | None = None,
    ) -> ModelEstimate:
        """
        Estimate token count, cost, and fit for *text* under a single *model*.

        Parameters
        ----------
        text:
            The prompt text to score.
        model:
            A registered model name, or any raw tokenizer identifier.
        reserved_output:
            Tokens to reserve for the completion when checking context-window
            fit. ``None`` uses the model's ``max_output_tokens`` (or 0 when that
            is unknown). Pass an explicit value to model a known output budget.
        """
        info = self._registry.get(model)
        tok = self._registry.resolve(model)
        token_count = len(tok.encode(text))

        cost = self._input_cost(token_count, info)
        reserve = self._resolve_reserve(reserved_output, info)
        fits, headroom = self._fit(token_count, reserve, info)

        return ModelEstimate(
            model=info.name,
            tokenizer=info.tokenizer,
            token_count=token_count,
            cost_usd=cost,
            context_window=info.context_window,
            reserved_output=reserve,
            fits=fits,
            headroom=headroom,
        )

    def estimate(
        self,
        text: str,
        models: list[str],
        reserved_output: int | None = None,
    ) -> MultiModelEstimate:
        """
        Estimate *text* across several *models* and return a side-by-side result.

        The estimates preserve the order of *models*.
        """
        estimates = [self.estimate_one(text, m, reserved_output=reserved_output) for m in models]
        return MultiModelEstimate(text_chars=len(text), estimates=estimates)

    def fits(
        self,
        text: str,
        model: str,
        reserved_output: int | None = None,
    ) -> bool:
        """
        Return ``True`` if *text* (plus reserved output) fits *model*'s context
        window. Returns ``True`` when the window is unknown (nothing to enforce).

        This is the cheap pre-dispatch guard a router calls before sending a
        request: a generic character-count estimate can be off by a wide margin
        across tokenizers, so the check uses the model's real tokenizer.
        """
        result = self.estimate_one(text, model, reserved_output=reserved_output)
        return result.fits is not False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _input_cost(token_count: int, info: ModelInfo) -> float | None:
        if info.price_per_1k_input is None:
            return None
        return (token_count / 1_000) * info.price_per_1k_input

    @staticmethod
    def _resolve_reserve(reserved_output: int | None, info: ModelInfo) -> int:
        if reserved_output is not None:
            return max(0, reserved_output)
        return info.max_output_tokens or 0

    @staticmethod
    def _fit(token_count: int, reserve: int, info: ModelInfo) -> tuple[bool | None, int | None]:
        if info.context_window is None:
            return None, None
        headroom = info.context_window - token_count - reserve
        return headroom >= 0, headroom
