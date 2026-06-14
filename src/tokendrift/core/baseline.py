"""
tokendrift.core.baseline
~~~~~~~~~~~~~~~~~~~~~~~~~~
Baseline snapshots and CI regression gating.

The workflow this module supports:

1. ``build_baseline`` encodes a corpus under one tokenizer and records the
   per-entry and total token counts. The result is serialised to JSON and
   committed to a repository.
2. ``run_ci`` re-encodes the same corpus under a (usually newer) tokenizer,
   compares the counts against the committed baseline, and reports whether the
   drift exceeds caller-supplied thresholds.

This turns TokenDrift from an interactive diff tool into a pipeline gate: a
provider that silently re-tokenizes can grow your prompt token counts (and
cost) without any code change, and a committed baseline makes that regression
visible in CI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from tokendrift.core.loader import UnifiedTokenizer
from tokendrift.models import CorpusEntry

SCHEMA_VERSION = 1


@dataclass
class Baseline:
    """
    A committed snapshot of token counts for a corpus under one tokenizer.

    Attributes:
        tokenizer:     Identifier of the tokenizer the snapshot was built with.
        total_tokens:  Sum of token counts across all entries.
        entries:       Mapping of entry id -> token count.
        created_at:    ISO-8601 UTC timestamp of when the snapshot was built.
        schema_version: Format version for forward compatibility.
    """

    tokenizer: str
    total_tokens: int
    entries: dict[str, int]
    created_at: str = ""
    schema_version: int = SCHEMA_VERSION

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "tokenizer": self.tokenizer,
            "created_at": self.created_at,
            "total_tokens": self.total_tokens,
            "entries": self.entries,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> Baseline:
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported baseline schema_version {version!r} "
                f"(this build of tokendrift writes/reads v{SCHEMA_VERSION}). "
                "Rebuild the baseline with the current tokendrift version."
            )
        try:
            entries = {str(k): int(v) for k, v in data["entries"].items()}
            return cls(
                tokenizer=str(data["tokenizer"]),
                total_tokens=int(data["total_tokens"]),
                entries=entries,
                created_at=str(data.get("created_at", "")),
                schema_version=version,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed baseline file: {exc}") from exc

    @classmethod
    def load(cls, path: str | Path) -> Baseline:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Baseline file not found: {p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Baseline file is not valid JSON: {exc}") from exc
        return cls.from_dict(data)


def build_baseline(tokenizer: UnifiedTokenizer, entries: list[CorpusEntry]) -> Baseline:
    """Encode *entries* under *tokenizer* and return a :class:`Baseline`."""
    counts: dict[str, int] = {}
    for e in entries:
        counts[e.id] = len(tokenizer.encode(e.text))
    return Baseline(
        tokenizer=tokenizer.name(),
        total_tokens=sum(counts.values()),
        entries=counts,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


@dataclass
class CIThresholds:
    """
    Failure thresholds for :func:`run_ci`.

    A threshold of ``None`` disables that check. ``max_cost_delta_usd`` is only
    evaluated when ``price_per_1k`` is also supplied.

    Attributes:
        max_total_growth_pct: Fail if total tokens grow by more than this %.
        max_entry_growth_pct: Fail if any single entry grows by more than this %.
        max_cost_delta_usd:   Fail if estimated total cost grows by more than this.
        price_per_1k:         USD per 1k tokens, used to estimate the cost delta.
        fail_on_new_entries:  Fail if the corpus contains ids absent from baseline.
        fail_on_missing_entries: Fail if baseline ids are absent from the corpus.
    """

    max_total_growth_pct: float | None = None
    max_entry_growth_pct: float | None = None
    max_cost_delta_usd: float | None = None
    price_per_1k: float | None = None
    fail_on_new_entries: bool = False
    fail_on_missing_entries: bool = False


@dataclass
class EntryDrift:
    """Per-entry token-count change between baseline and current run."""

    entry_id: str
    baseline_tokens: int
    current_tokens: int

    @property
    def delta(self) -> int:
        return self.current_tokens - self.baseline_tokens

    @property
    def pct(self) -> float:
        if self.baseline_tokens == 0:
            return 0.0 if self.delta == 0 else float("inf")
        return (self.delta / self.baseline_tokens) * 100


@dataclass
class CIReport:
    """
    Result of comparing a fresh encoding against a committed baseline.

    Attributes:
        baseline_tokenizer: Tokenizer recorded in the baseline.
        current_tokenizer:  Tokenizer used for this run.
        total_baseline:     Total baseline tokens for the compared entries.
        total_current:      Total current tokens for the compared entries.
        drifts:             Per-entry drift for every compared entry.
        new_entries:        Corpus ids not present in the baseline.
        missing_entries:    Baseline ids not present in the corpus.
        cost_delta_usd:     Estimated cost change (if pricing supplied).
        failures:           Human-readable reasons the gate failed.
    """

    baseline_tokenizer: str
    current_tokenizer: str
    total_baseline: int
    total_current: int
    drifts: list[EntryDrift]
    new_entries: list[str] = field(default_factory=list)
    missing_entries: list[str] = field(default_factory=list)
    cost_delta_usd: float | None = None
    failures: list[str] = field(default_factory=list)

    @property
    def token_delta(self) -> int:
        return self.total_current - self.total_baseline

    @property
    def total_pct(self) -> float:
        if self.total_baseline == 0:
            return 0.0 if self.token_delta == 0 else float("inf")
        return (self.token_delta / self.total_baseline) * 100

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def regressions(self) -> list[EntryDrift]:
        """Entries whose token count grew, worst first."""
        return sorted((d for d in self.drifts if d.delta > 0), key=lambda d: d.delta, reverse=True)


def run_ci(
    baseline: Baseline,
    tokenizer: UnifiedTokenizer,
    entries: list[CorpusEntry],
    thresholds: CIThresholds | None = None,
) -> CIReport:
    """
    Re-encode *entries* under *tokenizer*, compare to *baseline*, and apply
    *thresholds* to decide pass/fail.

    Only entries present in both the baseline and the corpus contribute to the
    token totals; new/missing ids are reported separately so the totals stay
    comparable.
    """
    thresholds = thresholds or CIThresholds()

    current_counts = {e.id: len(tokenizer.encode(e.text)) for e in entries}
    baseline_ids = set(baseline.entries)
    current_ids = set(current_counts)

    new_entries = sorted(current_ids - baseline_ids)
    missing_entries = sorted(baseline_ids - current_ids)
    shared = sorted(baseline_ids & current_ids)

    drifts = [
        EntryDrift(
            entry_id=eid,
            baseline_tokens=baseline.entries[eid],
            current_tokens=current_counts[eid],
        )
        for eid in shared
    ]

    total_baseline = sum(d.baseline_tokens for d in drifts)
    total_current = sum(d.current_tokens for d in drifts)
    token_delta = total_current - total_baseline

    cost_delta_usd: float | None = None
    if thresholds.price_per_1k is not None:
        cost_delta_usd = (token_delta / 1000) * thresholds.price_per_1k

    report = CIReport(
        baseline_tokenizer=baseline.tokenizer,
        current_tokenizer=tokenizer.name(),
        total_baseline=total_baseline,
        total_current=total_current,
        drifts=drifts,
        new_entries=new_entries,
        missing_entries=missing_entries,
        cost_delta_usd=cost_delta_usd,
    )

    _apply_thresholds(report, thresholds)
    return report


def _apply_thresholds(report: CIReport, thresholds: CIThresholds) -> None:
    """Populate ``report.failures`` based on *thresholds*."""
    if thresholds.max_total_growth_pct is not None and report.total_pct > thresholds.max_total_growth_pct:
        report.failures.append(
            f"total token growth {report.total_pct:+.2f}% exceeds limit {thresholds.max_total_growth_pct:+.2f}%"
        )

    if thresholds.max_entry_growth_pct is not None:
        worst = [d for d in report.regressions if d.pct > thresholds.max_entry_growth_pct]
        if worst:
            d = worst[0]
            report.failures.append(
                f"{len(worst)} entr{'y' if len(worst) == 1 else 'ies'} exceed per-entry growth limit "
                f"{thresholds.max_entry_growth_pct:+.2f}% (worst: {d.entry_id} {d.pct:+.2f}%)"
            )

    if (
        thresholds.max_cost_delta_usd is not None
        and report.cost_delta_usd is not None
        and report.cost_delta_usd > thresholds.max_cost_delta_usd
    ):
        report.failures.append(
            f"estimated cost delta ${report.cost_delta_usd:+.4f} exceeds limit ${thresholds.max_cost_delta_usd:+.4f}"
        )

    if thresholds.fail_on_new_entries and report.new_entries:
        n = len(report.new_entries)
        report.failures.append(f"{n} corpus entr{'y' if n == 1 else 'ies'} absent from baseline")

    if thresholds.fail_on_missing_entries and report.missing_entries:
        n = len(report.missing_entries)
        report.failures.append(f"{n} baseline entr{'y' if n == 1 else 'ies'} absent from corpus")
