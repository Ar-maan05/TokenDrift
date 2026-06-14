"""
TokenDrift
~~~~~~~~~
Token-count, cost, and vocabulary diffing for LLM tokenizer changes.

Structural boundary-change detection is available as an experimental,
opt-in extra (``EncodingDiffer(detect_boundaries=True)``).

Quick start
-----------
>>> from tokendrift.core.loader import TokenizerLoader
>>> from tokendrift.core.differ import EncodingDiffer
>>> tok_a = TokenizerLoader.load("cl100k_base")
>>> tok_b = TokenizerLoader.load("o200k_base")
>>> diff = EncodingDiffer().diff("biostatistical", tok_a, tok_b)
>>> diff.count_delta, diff.first_divergence_pos
"""

__version__ = "0.1.0"
__author__ = "Armaan Sandhu"
