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


@dataclass
class CorpusEntry:
    """A single entry in a prompt corpus."""

    id: str
    text: str
    metadata: dict = field(default_factory=dict)
