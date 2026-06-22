"""
tokendrift.core.forecast
~~~~~~~~~~~~~~~~~~~~~~~~~~
Org-level cost forecasting.

For budget governance, a team needs to project monthly spend before it happens.
:func:`forecast` takes a representative sample of recent prompts, measures the
average tokens per request under each candidate model's own tokenizer, and
scales that to a target request volume. The result lets a governance layer
compare current versus proposed model spend with measured data rather than a
character-count guess.

The forecast covers input tokens only: it is a prompt-cost projection, which is
the part TokenDrift can measure from prompt text alone. Output token cost
depends on generation length, which is not knowable from the prompt.

Usage
-----
>>> from tokendrift.core.forecast import forecast
>>> report = forecast(entries, ["gpt-4o", "gpt-4o-mini"], projected_requests=1_000_000)
>>> for f in report.forecasts:
...     print(f.model, f.projected_tokens, f.projected_cost_usd)
"""

from __future__ import annotations

from tokendrift.core.registry import ModelRegistry
from tokendrift.models import CorpusEntry, ForecastReport, ModelForecast


def forecast(
    entries: list[CorpusEntry],
    models: list[str],
    projected_requests: int,
    registry: ModelRegistry | None = None,
) -> ForecastReport:
    """
    Project input-token spend across *models* for *projected_requests* requests.

    Parameters
    ----------
    entries:
        A representative sample of recent prompts. Per-request averages are
        measured from this sample.
    models:
        Registered model names, or raw tokenizer identifiers.
    projected_requests:
        The request volume to scale the sample to (for example a month of
        traffic).
    registry:
        Registry to resolve model facts and tokenizers. Defaults to
        :meth:`ModelRegistry.default`.

    Raises
    ------
    ValueError
        If *projected_requests* is negative.
    """
    if projected_requests < 0:
        raise ValueError("projected_requests must be non-negative.")

    registry = registry or ModelRegistry.default()
    sample_requests = len(entries)

    forecasts: list[ModelForecast] = []
    for model in models:
        info = registry.get(model)
        tok = registry.resolve(model)
        sample_tokens = sum(len(tok.encode(e.text)) for e in entries)
        forecasts.append(
            ModelForecast(
                model=info.name,
                tokenizer=info.tokenizer,
                sample_requests=sample_requests,
                sample_tokens=sample_tokens,
                projected_requests=projected_requests,
                price_per_1k_input=info.price_per_1k_input,
            )
        )

    return ForecastReport(projected_requests=projected_requests, forecasts=forecasts)
