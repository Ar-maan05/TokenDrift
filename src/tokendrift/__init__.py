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

from tokendrift.core.baseline import (
    Baseline,
    CIReport,
    CIThresholds,
    EntryDrift,
    build_baseline,
    run_ci,
)
from tokendrift.core.boundary import BoundaryDetector
from tokendrift.core.differ import EncodingDiffer
from tokendrift.core.loader import TokenizerLoader, UnifiedTokenizer
from tokendrift.core.vocab import VocabDiffer
from tokendrift.corpus.loaders import load_corpus
from tokendrift.models import (
    BoundaryViolation,
    CorpusEntry,
    CostReport,
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
