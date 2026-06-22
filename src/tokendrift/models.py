"""
tokendrift.models
~~~~~~~~~~~~~~~~
All shared data classes. Kept in one file so every module imports from here
and there are no circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ViolationType(str, Enum):
    """
    Structural classification of how a word's token boundaries changed.

    These describe *what* changed structurally. They are NOT a judgement
    about whether the change affects model behaviour: TokenDrift does not
    measure behavioural impact, and a word being segmented differently is
    a normal, usually-harmless consequence of a tokenizer change.
    """

    SPLIT = "SPLIT"  # One token in A, more tokens in B (count grew)
    MERGE = "MERGE"  # More tokens in A, one token in B (count shrank)
    RESEGMENT = "RESEGMENT"  # Same token count, different segmentation


@dataclass
class BoundaryViolation:
    """
    A word whose token boundary structure differs between tokenizer A and B.

    Attributes:
        word:           The word string as it appears in the original text.
        char_start:     Start character index in the original text.
        char_end:       End character index in the original text.
        tokens_a:       Decoded token strings from tokenizer A covering this word.
        tokens_b:       Decoded token strings from tokenizer B covering this word.
        ids_a:          Token IDs from tokenizer A covering this word.
        ids_b:          Token IDs from tokenizer B covering this word.
        violation_type: SPLIT, MERGE, or RESEGMENT.
    """

    word: str
    char_start: int
    char_end: int
    tokens_a: list[str]
    tokens_b: list[str]
    ids_a: list[int]
    ids_b: list[int]
    violation_type: ViolationType


@dataclass
class TokenDiff:
    """
    The complete diff of how a single text encodes under two tokenizers.

    Attributes:
        entry_id:             Corpus entry identifier.
        text:                 The original text.
        token_count_a:        Number of tokens under tokenizer A.
        token_count_b:        Number of tokens under tokenizer B.
        count_delta:          token_count_b - token_count_a. Positive = grew.
        first_divergence_pos: Character index where encoding first diverges.
                              Equal to len(text) if sequences are identical.
        boundary_violations:  List of word-level boundary violations.
    """

    entry_id: str
    text: str
    token_count_a: int
    token_count_b: int
    count_delta: int
    first_divergence_pos: int
    boundary_violations: list[BoundaryViolation] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        """True if the encoding is different in any way."""
        return self.count_delta != 0 or bool(self.boundary_violations)

    @property
    def pct_change(self) -> float:
        """Percentage change in token count (B vs A). Positive = grew."""
        if self.token_count_a == 0:
            return 0.0
        return (self.count_delta / self.token_count_a) * 100

    @property
    def split_count(self) -> int:
        """Number of words that gained tokens (SPLIT) under tokenizer B."""
        return sum(1 for v in self.boundary_violations if v.violation_type == ViolationType.SPLIT)


@dataclass
class VocabEntry:
    """A single entry in a tokenizer vocabulary."""

    token_str: str
    token_id: int


@dataclass
class RemappedEntry:
    """A token string whose integer ID changed between tokenizers."""

    token_str: str
    old_id: int
    new_id: int


@dataclass
class VocabDiff:
    """
    Vocabulary-level diff between two tokenizers.

    Remapped entries are the most dangerous: any system that stored a token ID
    (rather than the string) now silently points to the wrong token.

    Attributes:
        added:    Tokens in B that are not in A.
        deleted:  Tokens in A that are not in B.
        remapped: Tokens present in both but with a different integer ID.
        total_a:  Vocabulary size of tokenizer A.
        total_b:  Vocabulary size of tokenizer B.
    """

    added: list[VocabEntry]
    deleted: list[VocabEntry]
    remapped: list[RemappedEntry]
    total_a: int
    total_b: int

    @property
    def has_remappings(self) -> bool:
        return len(self.remapped) > 0


@dataclass
class PromptCostDelta:
    """Per-prompt cost impact of a tokenizer change."""

    entry_id: str
    tokens_a: int
    tokens_b: int
    delta: int
    cost_a_usd: float | None = None
    cost_b_usd: float | None = None

    @property
    def cost_delta_usd(self) -> float | None:
        if self.cost_a_usd is not None and self.cost_b_usd is not None:
            return self.cost_b_usd - self.cost_a_usd
        return None


@dataclass
class CostReport:
    """
    Corpus-level cost impact report.

    Attributes:
        total_tokens_a:  Total token count across corpus under tokenizer A.
        total_tokens_b:  Total token count across corpus under tokenizer B.
        token_delta:     total_tokens_b - total_tokens_a.
        cost_a_usd:      Total cost under tokenizer A pricing (if provided).
        cost_b_usd:      Total cost under tokenizer B pricing (if provided).
        cost_delta_usd:  cost_b_usd - cost_a_usd (if pricing provided).
        per_prompt:      Per-entry breakdown.
    """

    total_tokens_a: int
    total_tokens_b: int
    token_delta: int
    per_prompt: list[PromptCostDelta]
    cost_a_usd: float | None = None
    cost_b_usd: float | None = None
    cost_delta_usd: float | None = None

    @property
    def pct_change(self) -> float:
        """Percentage change in token count. Positive = grew."""
        if self.total_tokens_a == 0:
            return 0.0
        return (self.token_delta / self.total_tokens_a) * 100

    # Explicit alias for callers that want the token/cost distinction spelled out.
    @property
    def pct_token_change(self) -> float:
        """Percentage change in token count. Alias of :attr:`pct_change`."""
        return self.pct_change

    @property
    def pct_cost_change(self) -> float:
        """Percentage change in cost. Returns 0.0 when pricing was not supplied."""
        if not self.cost_a_usd or self.cost_delta_usd is None:
            return 0.0
        return (self.cost_delta_usd / self.cost_a_usd) * 100


@dataclass
class CorpusEntry:
    """A single entry in a prompt corpus."""

    id: str
    text: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# v1.1.0: model registry and pre-dispatch estimation
# ---------------------------------------------------------------------------


@dataclass
class ModelInfo:
    """
    Static facts about a single model that TokenDrift needs to estimate cost
    and check context-window fit before a request is dispatched.

    Attributes:
        name:                Friendly model name (the registry key).
        tokenizer:           Identifier understood by :class:`TokenizerLoader`
                             (a tiktoken encoding name, a HuggingFace Hub id, or
                             a local tokenizer path).
        context_window:      Maximum total tokens the model accepts, or ``None``
                             if unknown.
        price_per_1k_input:  USD per 1,000 input tokens, or ``None`` if unpriced.
        price_per_1k_output: USD per 1,000 output tokens, or ``None``.
        max_output_tokens:   Default tokens to reserve for the completion when
                             checking context-window fit, or ``None``.
        provider:            Free-form provider label for reporting.
        notes:               Free-form notes (for example a pricing "as of" date).

    Pricing and context windows change often and vary by provider contract.
    The values shipped in :meth:`ModelRegistry.default` are indicative starting
    points; verify them against your provider and override via a JSON registry
    or :meth:`ModelRegistry.add` before relying on them for budgets or audits.
    """

    name: str
    tokenizer: str
    context_window: int | None = None
    price_per_1k_input: float | None = None
    price_per_1k_output: float | None = None
    max_output_tokens: int | None = None
    provider: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "tokenizer": self.tokenizer,
            "context_window": self.context_window,
            "price_per_1k_input": self.price_per_1k_input,
            "price_per_1k_output": self.price_per_1k_output,
            "max_output_tokens": self.max_output_tokens,
            "provider": self.provider,
            "notes": self.notes,
        }


@dataclass
class ModelEstimate:
    """
    Per-model estimate of how a single text would be billed and whether it
    fits the model's context window, computed before any request is sent.

    Attributes:
        model:          Friendly model name.
        tokenizer:      Tokenizer identifier used for the count.
        token_count:    Tokens the input text encodes to under this tokenizer.
        cost_usd:       Estimated input cost in USD, or ``None`` if unpriced.
        context_window: The model's context window, or ``None`` if unknown.
        reserved_output: Tokens reserved for the completion in the fit check.
        fits:           ``True`` if ``token_count + reserved_output`` is within
                        ``context_window``; ``None`` when the window is unknown.
        headroom:       Remaining tokens before the window is exceeded, or
                        ``None`` when the window is unknown.
    """

    model: str
    tokenizer: str
    token_count: int
    cost_usd: float | None = None
    context_window: int | None = None
    reserved_output: int = 0
    fits: bool | None = None
    headroom: int | None = None


@dataclass
class MultiModelEstimate:
    """
    Side-by-side estimate of one text across several models.

    This is what powers a playground cost overlay: the same prompt scored for
    every model the user is comparing, before they click run.

    Attributes:
        text_chars: Length of the input text in characters.
        estimates:  One :class:`ModelEstimate` per requested model, in order.
    """

    text_chars: int
    estimates: list[ModelEstimate] = field(default_factory=list)

    @property
    def min_tokens(self) -> int:
        return min((e.token_count for e in self.estimates), default=0)

    @property
    def max_tokens(self) -> int:
        return max((e.token_count for e in self.estimates), default=0)

    @property
    def token_spread(self) -> int:
        """Absolute difference between the highest and lowest token counts."""
        return self.max_tokens - self.min_tokens

    @property
    def divergence_pct(self) -> float:
        """
        Spread as a percentage of the smallest count. A high value means the
        same text tokenizes very differently across the compared models, which
        is what drives cost and latency apart between providers.
        """
        if self.min_tokens == 0:
            return 0.0
        return (self.token_spread / self.min_tokens) * 100

    @property
    def cheapest(self) -> ModelEstimate | None:
        priced = [e for e in self.estimates if e.cost_usd is not None]
        return min(priced, key=lambda e: e.cost_usd or 0.0) if priced else None

    @property
    def overflowed(self) -> list[ModelEstimate]:
        """Estimates whose input does not fit the model context window."""
        return [e for e in self.estimates if e.fits is False]


@dataclass
class MigrationOverflow:
    """A prompt that would exceed the target model's context window."""

    entry_id: str
    target_tokens: int
    context_window: int
    reserved_output: int = 0

    @property
    def overflow(self) -> int:
        """Tokens over the limit (including any reserved output)."""
        return self.target_tokens + self.reserved_output - self.context_window


@dataclass
class MigrationReport:
    """
    Safety report for switching a corpus of prompts from one model to another.

    Surfaces the four things that break when a B2B customer migrates: token
    count delta, cost delta, vocabulary shift, and any prompts that would no
    longer fit the target context window.

    Attributes:
        source_model, target_model: Friendly model names.
        total_tokens_source, total_tokens_target: Corpus totals.
        cost_source_usd, cost_target_usd, cost_delta_usd: Input cost totals.
        vocab:        Vocabulary diff between the two tokenizers, or ``None``.
        overflows:    Prompts that exceed the target context window.
        per_prompt:   Per-entry token and cost deltas.
    """

    source_model: str
    target_model: str
    total_tokens_source: int
    total_tokens_target: int
    per_prompt: list[PromptCostDelta] = field(default_factory=list)
    overflows: list[MigrationOverflow] = field(default_factory=list)
    cost_source_usd: float | None = None
    cost_target_usd: float | None = None
    cost_delta_usd: float | None = None
    vocab: VocabDiff | None = None

    @property
    def token_delta(self) -> int:
        return self.total_tokens_target - self.total_tokens_source

    @property
    def pct_token_change(self) -> float:
        if self.total_tokens_source == 0:
            return 0.0
        return (self.token_delta / self.total_tokens_source) * 100

    @property
    def pct_cost_change(self) -> float:
        if not self.cost_source_usd or self.cost_delta_usd is None:
            return 0.0
        return (self.cost_delta_usd / self.cost_source_usd) * 100

    @property
    def remapped_token_count(self) -> int:
        return len(self.vocab.remapped) if self.vocab else 0

    def to_dict(self) -> dict:
        """Machine-readable summary for an audit or compliance pipeline."""
        return {
            "source_model": self.source_model,
            "target_model": self.target_model,
            "total_tokens_source": self.total_tokens_source,
            "total_tokens_target": self.total_tokens_target,
            "token_delta": self.token_delta,
            "pct_token_change": round(self.pct_token_change, 4),
            "cost_source_usd": self.cost_source_usd,
            "cost_target_usd": self.cost_target_usd,
            "cost_delta_usd": self.cost_delta_usd,
            "pct_cost_change": round(self.pct_cost_change, 4),
            "vocab_added": len(self.vocab.added) if self.vocab else None,
            "vocab_deleted": len(self.vocab.deleted) if self.vocab else None,
            "vocab_remapped": self.remapped_token_count if self.vocab else None,
            "overflow_count": len(self.overflows),
            "overflows": [
                {
                    "entry_id": o.entry_id,
                    "target_tokens": o.target_tokens,
                    "context_window": o.context_window,
                    "overflow": o.overflow,
                }
                for o in self.overflows
            ],
        }


@dataclass
class CompressionSaving:
    """Token and cost savings of a compressed prompt under one model."""

    model: str
    tokenizer: str
    original_tokens: int
    compressed_tokens: int
    cost_saved_usd: float | None = None

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.compressed_tokens

    @property
    def pct_saved(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return (self.tokens_saved / self.original_tokens) * 100


@dataclass
class CompressionReport:
    """
    How much a prompt-compression step actually saves, per model.

    Compression savings are tokenizer dependent: middle-out compression that
    removes 30% of the characters does not remove 30% of the tokens, and the
    real saving differs across the models you might dispatch to. This report
    makes a compression decision model aware instead of generic.

    Attributes:
        original_chars, compressed_chars: Character lengths of the two texts.
        savings: One :class:`CompressionSaving` per model.
    """

    original_chars: int
    compressed_chars: int
    savings: list[CompressionSaving] = field(default_factory=list)

    @property
    def char_pct_saved(self) -> float:
        if self.original_chars == 0:
            return 0.0
        return ((self.original_chars - self.compressed_chars) / self.original_chars) * 100


@dataclass
class ModelForecast:
    """Projected spend for one model at a target request volume."""

    model: str
    tokenizer: str
    sample_requests: int
    sample_tokens: int
    projected_requests: int
    price_per_1k_input: float | None = None

    @property
    def avg_tokens_per_request(self) -> float:
        if self.sample_requests == 0:
            return 0.0
        return self.sample_tokens / self.sample_requests

    @property
    def projected_tokens(self) -> int:
        return round(self.avg_tokens_per_request * self.projected_requests)

    @property
    def projected_cost_usd(self) -> float | None:
        if self.price_per_1k_input is None:
            return None
        return (self.projected_tokens / 1_000) * self.price_per_1k_input


@dataclass
class ForecastReport:
    """
    Org-level spend forecast across candidate models.

    Takes a representative sample of recent prompts, measures the average tokens
    per request under each candidate model, and projects that to a target
    request volume so a governance team can compare current versus proposed
    model spend with measured data instead of a guess.

    Attributes:
        projected_requests: The request volume the forecast is scaled to.
        forecasts:          One :class:`ModelForecast` per candidate model.
    """

    projected_requests: int
    forecasts: list[ModelForecast] = field(default_factory=list)

    @property
    def cheapest(self) -> ModelForecast | None:
        priced = [f for f in self.forecasts if f.projected_cost_usd is not None]
        return min(priced, key=lambda f: f.projected_cost_usd or 0.0) if priced else None

    def to_dict(self) -> dict:
        return {
            "projected_requests": self.projected_requests,
            "forecasts": [
                {
                    "model": f.model,
                    "tokenizer": f.tokenizer,
                    "avg_tokens_per_request": round(f.avg_tokens_per_request, 2),
                    "projected_tokens": f.projected_tokens,
                    "projected_cost_usd": f.projected_cost_usd,
                }
                for f in self.forecasts
            ],
        }


class AlertSeverity(str, Enum):
    """Severity of a tokenizer drift alert."""

    OK = "OK"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


@dataclass
class DriftAlert:
    """
    A drift alert produced by diffing a new tokenizer against a committed
    baseline, intended to be emitted by a background job and consumed by an
    audit or compliance layer.

    A provider silently updating their tokenizer between model versions is an
    audit event: prompts can start costing more or a category of content can
    start tokenizing differently. This alert classifies the drift against
    configurable warn and critical thresholds and serialises cleanly to JSON.

    Attributes:
        baseline_tokenizer, current_tokenizer: Tokenizer identifiers compared.
        total_pct:      Percentage change in total tokens.
        token_delta:    Absolute change in total tokens.
        warn_pct:       Threshold at which severity becomes WARN.
        critical_pct:   Threshold at which severity becomes CRITICAL.
        severity:       OK, WARN, or CRITICAL.
        cost_delta_usd: Estimated cost change, if pricing was supplied.
        message:        Human-readable one-line summary.
    """

    baseline_tokenizer: str
    current_tokenizer: str
    total_pct: float
    token_delta: int
    warn_pct: float
    critical_pct: float
    severity: AlertSeverity
    cost_delta_usd: float | None = None
    message: str = ""

    @property
    def triggered(self) -> bool:
        """True when severity is WARN or CRITICAL."""
        return self.severity is not AlertSeverity.OK

    def to_dict(self) -> dict:
        return {
            "baseline_tokenizer": self.baseline_tokenizer,
            "current_tokenizer": self.current_tokenizer,
            "total_pct": round(self.total_pct, 4),
            "token_delta": self.token_delta,
            "warn_pct": self.warn_pct,
            "critical_pct": self.critical_pct,
            "severity": self.severity.value,
            "triggered": self.triggered,
            "cost_delta_usd": self.cost_delta_usd,
            "message": self.message,
        }
