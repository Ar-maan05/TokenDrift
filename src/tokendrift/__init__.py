"""
TokenDrift
~~~~~~~~~
Token-count, cost, and vocabulary diffing for LLM tokenizer changes.

Structural boundary-change detection is available as an experimental,
opt-in extra (``EncodingDiffer(detect_boundaries=True)``).

Public API
----------
All stable public names are importable directly from the package root::

    from tokendrift import (
        TokenizerLoader, UnifiedTokenizer,
        EncodingDiffer, VocabDiffer, BoundaryDetector,
        CostCalculator, load_corpus,
        TokenDiff, VocabDiff, CostReport,
        BoundaryViolation, ViolationType,
        CorpusEntry, VocabEntry, RemappedEntry, PromptCostDelta,
    )
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("tokendrift")
except PackageNotFoundError:  # editable install before build, or running from source
    __version__ = "unknown"

__author__ = "Armaan Sandhu"

# ---------------------------------------------------------------------------
# Re-export the full stable public API so consumers never need to know the
# internal submodule layout.  Third-party type checkers follow these imports.
# ---------------------------------------------------------------------------

from tokendrift.core.alert import alert_to_json, check_drift
from tokendrift.core.baseline import (
    Baseline,
    CIReport,
    CIThresholds,
    EntryDrift,
    build_baseline,
    run_ci,
)
from tokendrift.core.boundary import BoundaryDetector
from tokendrift.core.compression import compression_report, compression_report_corpus
from tokendrift.core.differ import EncodingDiffer
from tokendrift.core.estimate import CostEstimator
from tokendrift.core.forecast import forecast
from tokendrift.core.loader import TokenizerLoader, UnifiedTokenizer
from tokendrift.core.migrate import migrate_report
from tokendrift.core.registry import ModelRegistry
from tokendrift.core.vocab import VocabDiffer
from tokendrift.corpus.loaders import load_corpus
from tokendrift.models import (
    AlertSeverity,
    BoundaryViolation,
    CompressionReport,
    CompressionSaving,
    CorpusEntry,
    CostReport,
    DriftAlert,
    ForecastReport,
    MigrationOverflow,
    MigrationReport,
    ModelEstimate,
    ModelForecast,
    ModelInfo,
    MultiModelEstimate,
    PromptCostDelta,
    RemappedEntry,
    TokenDiff,
    ViolationType,
    VocabDiff,
    VocabEntry,
)
from tokendrift.report.cost import CostCalculator

__all__ = [
    # Tokenizer loading
    "TokenizerLoader",
    "UnifiedTokenizer",
    # Core diffing
    "EncodingDiffer",
    "VocabDiffer",
    "BoundaryDetector",
    # Reports
    "CostCalculator",
    # Baseline / CI gating
    "Baseline",
    "build_baseline",
    "run_ci",
    "CIThresholds",
    "CIReport",
    "EntryDrift",
    # Corpus
    "load_corpus",
    # Model registry
    "ModelRegistry",
    "ModelInfo",
    # Pre-dispatch estimation / budget
    "CostEstimator",
    "ModelEstimate",
    "MultiModelEstimate",
    # Migration safety
    "migrate_report",
    "MigrationReport",
    "MigrationOverflow",
    # Compression feedback
    "compression_report",
    "compression_report_corpus",
    "CompressionReport",
    "CompressionSaving",
    # Cost forecasting
    "forecast",
    "ForecastReport",
    "ModelForecast",
    # Drift alerts
    "check_drift",
    "alert_to_json",
    "DriftAlert",
    "AlertSeverity",
    # Data models
    "TokenDiff",
    "VocabDiff",
    "CostReport",
    "BoundaryViolation",
    "ViolationType",
    "CorpusEntry",
    "VocabEntry",
    "RemappedEntry",
    "PromptCostDelta",
]
