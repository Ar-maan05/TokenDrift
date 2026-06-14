"""
tokenlens.core.loader
~~~~~~~~~~~~~~~~~~~~~
Loads any tokenizer behind a single ``UnifiedTokenizer`` interface,
regardless of whether it came from tiktoken, HuggingFace, or SentencePiece.

Detection order
---------------
1. Known tiktoken encoding name  → tiktoken backend
2. Local file / directory path   → HuggingFace tokenizers backend
3. HuggingFace Hub model ID      → HuggingFace tokenizers backend
4. Falls through to an error

Usage
-----
>>> from tokenlens.core.loader import TokenizerLoader
>>> tok = TokenizerLoader.load("cl100k_base")
>>> tok.encode("Hello world")
[9906, 1917]
>>> tok = TokenizerLoader.load("meta-llama/Llama-3.2-1B")
>>> tok.name()
'meta-llama/Llama-3.2-1B'
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

TIKTOKEN_ENCODINGS = {
    "cl100k_base",
    "o200k_base",
    "p50k_base",
    "p50k_edit",
    "r50k_base",
    "gpt2",
}


class UnifiedTokenizer(ABC):
    """
    Common interface over tiktoken, HuggingFace tokenizers, and SentencePiece.

    All methods must be implemented by every backend.  Consumers of this class
    never need to know which library is underneath.
    """

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode *text* to a list of token IDs."""

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to a string."""

    @abstractmethod
    def decode_single(self, id: int) -> str:
        """
        Decode a single token ID to its string representation.

        For byte-level or non-UTF-8 tokens, replaces undecodable bytes
        with the Unicode replacement character rather than raising.
        """

    @abstractmethod
    def vocab(self) -> dict[str, int]:
        """
        Return the full vocabulary as ``{token_string: token_id}``.

        For tiktoken, byte-level tokens that are not valid UTF-8 are
        represented as their ``repr()`` string so the dict has unique keys.
        """

    @abstractmethod
    def char_offsets(self, text: str) -> list[tuple[int, int]]:
        """
        Return ``(char_start, char_end)`` for every token in ``encode(text)``.

        The spans are over Unicode code points (i.e. ``len(text[start:end])``
        equals the number of characters in that token).  Contiguous: the end
        of token ``i`` equals the start of token ``i+1``.
        """

    @abstractmethod
    def name(self) -> str:
        """Return the canonical name / identifier for this tokenizer."""


# ---------------------------------------------------------------------------
# tiktoken backend
# ---------------------------------------------------------------------------


class TiktokenTokenizer(UnifiedTokenizer):
    """Wraps a ``tiktoken.Encoding``."""

    def __init__(self, encoding_name: str) -> None:
        import tiktoken  # lazy import so the library is optional at import time

        self._enc = tiktoken.get_encoding(encoding_name)
        self._name = encoding_name

    # ------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        # Treat the entire input as ordinary text: substrings that look like
        # special tokens (e.g. "<|endoftext|>") are encoded literally rather
        # than as control tokens. This keeps a diff over arbitrary prompt
        # text faithful to what the user actually wrote, and matches the
        # default behaviour of the HuggingFace backend.
        return self._enc.encode(text, disallowed_special=())

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    def decode_single(self, id: int) -> str:
        token_bytes = self._enc.decode_single_token_bytes(id)
        return token_bytes.decode("utf-8", errors="replace")

    def vocab(self) -> dict[str, int]:
        result: dict[str, int] = {}
        # _mergeable_ranks: bytes -> int
        for token_bytes, token_id in self._enc._mergeable_ranks.items():
            try:
                key = token_bytes.decode("utf-8")
            except UnicodeDecodeError:
                key = repr(token_bytes)
            # Avoid clobbering a real UTF-8 key with a repr key
            if key not in result:
                result[key] = token_id
        # Include special tokens
        for token_str, token_id in self._enc._special_tokens.items():
            result[token_str] = token_id
        return result

    def char_offsets(self, text: str) -> list[tuple[int, int]]:
        ids = self.encode(text)
        byte_tokens = self._enc.decode_tokens_bytes(ids)
        spans: list[tuple[int, int]] = []
        cursor = 0
        for bt in byte_tokens:
            decoded = bt.decode("utf-8", errors="replace")
            end = cursor + len(decoded)
            spans.append((cursor, end))
            cursor = end
        return spans

    def name(self) -> str:
        return self._name


# ---------------------------------------------------------------------------
# HuggingFace tokenizers backend
# ---------------------------------------------------------------------------


class HFTokenizer(UnifiedTokenizer):
    """
    Wraps a ``tokenizers.Tokenizer`` from the HuggingFace ``tokenizers``
    library (the fast Rust-backed library, not ``transformers``).
    """

    def __init__(self, identifier: str) -> None:
        from tokenizers import Tokenizer  # lazy import

        path = Path(identifier)
        if path.exists():
            if path.is_dir():
                self._tok = Tokenizer.from_pretrained(str(path))
            else:
                self._tok = Tokenizer.from_file(str(path))
        else:
            # Assume a HuggingFace Hub model ID
            self._tok = Tokenizer.from_pretrained(identifier)

        # Enable offset tracking (returns char offsets per token)
        self._tok.no_truncation()
        self._identifier = identifier

    # ------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    def decode_single(self, id: int) -> str:
        # decode() on a single-element list is correct and handles specials
        return self._tok.decode([id])

    def vocab(self) -> dict[str, int]:
        return self._tok.get_vocab(with_added_tokens=True)

    def char_offsets(self, text: str) -> list[tuple[int, int]]:
        encoding = self._tok.encode(text)
        offsets = encoding.offsets  # list[tuple[int, int]]
        if offsets:
            return list(offsets)
        # Fallback: reconstruct from token strings
        spans: list[tuple[int, int]] = []
        cursor = 0
        for token_str in encoding.tokens:
            # Strip common BPE prefix characters
            clean = token_str.lstrip("Ġ▁Ċ")
            end = cursor + len(clean) if clean else cursor
            spans.append((cursor, end))
            cursor = end
        return spans

    def name(self) -> str:
        return self._identifier


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class TokenizerLoader:
    """
    Factory that detects the appropriate backend and returns a
    ``UnifiedTokenizer``.

    Examples
    --------
    >>> tok = TokenizerLoader.load("cl100k_base")          # tiktoken
    >>> tok = TokenizerLoader.load("o200k_base")           # tiktoken
    >>> tok = TokenizerLoader.load("Qwen/Qwen3-4B")        # HuggingFace Hub
    >>> tok = TokenizerLoader.load("/path/to/tokenizer/")  # local dir
    >>> tok = TokenizerLoader.load("/path/to/tok.json")    # local file
    """

    @staticmethod
    def load(identifier: str) -> UnifiedTokenizer:
        """
        Load a tokenizer by identifier.

        Parameters
        ----------
        identifier:
            One of:
            - A tiktoken encoding name (``"cl100k_base"``, ``"o200k_base"``, …)
            - A HuggingFace Hub model ID (``"meta-llama/Llama-3.1-8B"``)
            - A local directory path containing ``tokenizer.json``
            - A path to a ``tokenizer.json`` file directly

        Raises
        ------
        ValueError
            If the identifier cannot be resolved to any known tokenizer.
        """
        # 1. tiktoken encoding name
        if identifier in TIKTOKEN_ENCODINGS:
            return TiktokenTokenizer(identifier)

        # 2. Local path
        path = Path(identifier)
        if path.exists():
            return HFTokenizer(identifier)

        # 3. HuggingFace Hub model ID  (contains "/" or looks like a repo)
        if "/" in identifier or _looks_like_hf_id(identifier):
            try:
                return HFTokenizer(identifier)
            except Exception as exc:  # noqa: BLE001 - re-raised with guidance below
                # A bare name with no "/" that fails to load from the Hub is
                # far more likely a misspelled tiktoken encoding than a real
                # single-segment repo. Surface the helpful hint instead of the
                # opaque HuggingFace 404.
                if "/" not in identifier:
                    raise ValueError(_unresolved_message(identifier)) from exc
                raise

        raise ValueError(_unresolved_message(identifier))


def _looks_like_hf_id(s: str) -> bool:
    """Heuristic: strings like 'gpt2' or 'bert-base-uncased' are HF IDs."""
    return bool(re.match(r"^[a-zA-Z0-9._-]+$", s)) and s not in TIKTOKEN_ENCODINGS


def _unresolved_message(identifier: str) -> str:
    """Build a helpful error for an identifier that did not resolve."""
    import difflib

    suggestion = ""
    close = difflib.get_close_matches(identifier, TIKTOKEN_ENCODINGS, n=1)
    if close:
        suggestion = f"  Did you mean the tiktoken encoding '{close[0]}'?\n"
    return (
        f"Cannot resolve tokenizer '{identifier}'.\n"
        f"{suggestion}"
        f"  Known tiktoken encodings: {sorted(TIKTOKEN_ENCODINGS)}\n"
        "  For HuggingFace models, use the full model ID, e.g. 'Qwen/Qwen3-4B'.\n"
        "  For local tokenizers, provide the directory or file path."
    )
