"""
conftest.py
~~~~~~~~~~~
Shared pytest fixtures.

In environments without internet access (CI sandboxes, offline dev) the real
tiktoken and HuggingFace tokenizers can't download their vocab files.

This module provides two minimal ``UnifiedTokenizer`` implementations that are
completely self-contained and require no network calls:

- ``MockTokenizerA``: character-level tokenizer where each character is one token
- ``MockTokenizerB``: bigram tokenizer where every 2 characters form one token
                      (falls back to single char for odd-length tails)

These are intentionally simple so tests can assert on exact behavior without
knowing anything about real BPE merges.  Tests that specifically need real
tiktoken behavior are marked with ``@pytest.mark.network`` and skipped when
the ``TOKENLENS_NETWORK_TESTS`` environment variable is not set.
"""

from __future__ import annotations

import os

import pytest

from tokenlens.core.loader import UnifiedTokenizer

# ---------------------------------------------------------------------------
# Mock tokenizers
# ---------------------------------------------------------------------------


class MockTokenizerA(UnifiedTokenizer):
    """
    Character-level tokenizer.
    Token ID = ord(char). Each character is exactly one token.
    """

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

    def decode_single(self, id: int) -> str:
        return chr(id)

    def vocab(self) -> dict[str, int]:
        # Printable ASCII + common extras
        return {chr(i): i for i in range(32, 127)}

    def char_offsets(self, text: str) -> list[tuple[int, int]]:
        return [(i, i + 1) for i in range(len(text))]

    def name(self) -> str:
        return "mock_char_level"


class MockTokenizerB(UnifiedTokenizer):
    """
    Bigram tokenizer.  Pairs of adjacent characters form one token.
    IDs are assigned by hashing the pair to keep them deterministic.
    Single trailing chars use their own ord() as ID.

    This deliberately differs from MockTokenizerA so diff / boundary tests
    can observe real changes.
    """

    _BIGRAM_OFFSET = 200_000  # keeps bigram IDs clearly separate from char IDs

    def _pair_id(self, a: str, b: str) -> int:
        return self._BIGRAM_OFFSET + ord(a) * 256 + ord(b)

    def encode(self, text: str) -> list[int]:
        ids = []
        i = 0
        while i < len(text):
            if i + 1 < len(text):
                ids.append(self._pair_id(text[i], text[i + 1]))
                i += 2
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def decode(self, ids: list[int]) -> str:
        parts = []
        for id_ in ids:
            parts.append(self.decode_single(id_))
        return "".join(parts)

    def decode_single(self, id: int) -> str:
        if id >= self._BIGRAM_OFFSET:
            rem = id - self._BIGRAM_OFFSET
            a = chr(rem // 256)
            b = chr(rem % 256)
            return a + b
        return chr(id)

    def vocab(self) -> dict[str, int]:
        v: dict[str, int] = {}
        # Single chars
        for i in range(32, 127):
            v[chr(i)] = i
        # Some bigrams that differ from MockTokenizerA vocab
        for a in range(32, 64):
            for b in range(32, 64):
                pair = chr(a) + chr(b)
                v[pair] = self._pair_id(chr(a), chr(b))
        return v

    def char_offsets(self, text: str) -> list[tuple[int, int]]:
        spans = []
        i = 0
        while i < len(text):
            if i + 1 < len(text):
                spans.append((i, i + 2))
                i += 2
            else:
                spans.append((i, i + 1))
                i += 1
        return spans

    def name(self) -> str:
        return "mock_bigram"


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tok_a() -> UnifiedTokenizer:
    """Character-level mock tokenizer (no network required)."""
    return MockTokenizerA()


@pytest.fixture(scope="session")
def tok_b() -> UnifiedTokenizer:
    """Bigram mock tokenizer (no network required)."""
    return MockTokenizerB()


@pytest.fixture(scope="session")
def differ():
    from tokenlens.core.differ import EncodingDiffer

    return EncodingDiffer(detect_boundaries=False)


# ---------------------------------------------------------------------------
# Network test marker
# ---------------------------------------------------------------------------

_NETWORK = bool(os.environ.get("TOKENLENS_NETWORK_TESTS"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "network: mark test as requiring internet access (tiktoken / HuggingFace Hub)",
    )


def pytest_collection_modifyitems(config, items):
    if not _NETWORK:
        skip = pytest.mark.skip(reason="Set TOKENLENS_NETWORK_TESTS=1 to run network tests")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip)
