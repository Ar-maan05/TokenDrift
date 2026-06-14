# TokenDrift

[![CI](https://github.com/Ar-maan05/tokendrift/actions/workflows/ci.yml/badge.svg)](https://github.com/Ar-maan05/tokendrift/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tokendrift.svg)](https://pypi.org/project/tokendrift/)
[![Python versions](https://img.shields.io/pypi/pyversions/tokendrift.svg)](https://pypi.org/project/tokendrift/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Token-count, cost, and vocabulary diffing for LLM tokenizer changes.

When you upgrade a model, switch providers, or move to a self-hosted checkpoint, the tokenizer changes silently. Token counts shift, moving your API cost and context-window pressure. Token IDs are renumbered, breaking any system that stored raw integer IDs (cache keys, classifier heads, adapter embedding rows). None of this produces an error.

TokenDrift measures these changes against your own prompt corpus before they hit production.

```
tokendrift diff cl100k_base o200k_base --corpus prompts.jsonl --price-a 0.03 --price-b 0.01
```

```
──────────── Vocab Diff  cl100k_base → o200k_base ────────────

  Vocab size A    100,277
  Vocab size B    200,019
  Added           +11,997  tokens in B, not in A
  Deleted             -47  tokens in A, not in B
  Remapped            +19  ⚠  stored token IDs now point elsewhere

──────────── Encoding Diff  cl100k_base → o200k_base ─────────

  Corpus entries     1,247
  Entries changed    834 / 1,247  (66.9%)
  Total token Δ    +2,841  (+3.2%)
  Max Δ (single)      +47

  Top 5 most-affected entries
  ─────────────────────────────────────────────────────────────
  p041    +47  "Summarize the biostatistical significance…"
  p117    +31  "Translate the following JSON payload…"

──────────── Cost Report  cl100k_base → o200k_base ───────────

  Cost (A)    $1.24
  Cost (B)    $1.28
  Cost delta  +$0.04   (+3.2%)
```

## Installation

```bash
pip install tokendrift
```

For development:

```bash
git clone https://github.com/Ar-maan05/tokendrift
cd tokendrift
pip install -e ".[dev]"
```

## Quick start

**Single text diff:**
```bash
tokendrift diff cl100k_base o200k_base \
  --text "ChatGPT rewrites biostatistical significance tests"
```

**Corpus diff:**
```bash
tokendrift diff cl100k_base o200k_base --corpus prompts.jsonl
```

**Vocabulary diff only (no corpus needed):**
```bash
tokendrift vocab-diff cl100k_base o200k_base --show remapped
```

**Cost impact:**
```bash
tokendrift cost cl100k_base o200k_base \
  --corpus prompts.jsonl \
  --price-a 0.03 \
  --price-b 0.01
```

**Inspect how a single text re-segments** (experimental boundary detection):
```bash
tokendrift entry cl100k_base o200k_base \
  --text "ChatGPT rewrites biostatistical significance tests"
```

## Corpus format

TokenDrift accepts JSONL (recommended), CSV, or plain text.

**JSONL**: one object per line, must have a `text` key:
```jsonl
{"id": "p001", "text": "What is the capital of France?"}
{"id": "p002", "text": "Summarize the following document:"}
```

`id` and `metadata` are optional. Everything else in the object is stored as metadata.

## What TokenDrift detects

### Vocabulary changes

- **Added tokens**: present in B, not in A.
- **Deleted tokens**: present in A, not in B.
- **Remapped tokens**: same string, different integer ID. This is the change that breaks silently; any system that stored a raw token ID rather than the string now points to a different entry.

### Encoding changes (the core of the tool)

- **Token count delta** per prompt and across the corpus. Positive = more tokens = higher cost and more context pressure.
- **Cost delta**: the count delta priced out, per prompt and corpus-wide.
- **First divergence position**: the character offset where the two encodings first differ.

These are exact, fully-supported, and the reason to use TokenDrift.

### Boundary changes (experimental)

Enabled with `--boundaries` on `diff`, or shown by the `entry` command. This is a **structural** report of how individual words are segmented differently, nothing more:

| Type | Meaning |
|------|---------|
| SPLIT | a word gains tokens (1 → 2+) |
| MERGE | a word loses tokens (2+ → 1) |
| RESEGMENT | same token count, but the segmentation boundaries moved |

**This is not a quality judgement.** TokenDrift does not claim a boundary change degrades model behaviour: re-segmentation is a normal consequence of a tokenizer change, and any behavioural effect is task-specific and not measured here. The feature is off by default and reports structure only, without severity ranking. (Pure ID renumbering, a word that encodes to the same strings in both tokenizers but with different IDs, is reported at the vocabulary level, not here, where it would flag almost every word.)

## Python API

```python
from tokendrift.core.loader import TokenizerLoader
from tokendrift.core.differ import EncodingDiffer
from tokendrift.core.vocab import VocabDiffer
from tokendrift.corpus.loaders import load_corpus

# Load tokenizers
tok_a = TokenizerLoader.load("cl100k_base")    # tiktoken
tok_b = TokenizerLoader.load("o200k_base")     # tiktoken
# tok_b = TokenizerLoader.load("Qwen/Qwen3-4B")  # HuggingFace Hub

# Vocab diff
from tokendrift.core.vocab import VocabDiffer
v_diff = VocabDiffer().diff(tok_a, tok_b)
print(f"Added: {len(v_diff.added)}, Remapped: {len(v_diff.remapped)}")

# Single text diff (count/divergence only: the default, fully-supported path)
differ = EncodingDiffer()
d = differ.diff("biostatistical significance", tok_a, tok_b)
print(f"Token delta: {d.count_delta}, first divergence at char {d.first_divergence_pos}")

# Opt into experimental structural boundary detection
boundary_differ = EncodingDiffer(detect_boundaries=True)
d = boundary_differ.diff("biostatistical significance", tok_a, tok_b)
for v in d.boundary_violations:  # SPLIT / MERGE / RESEGMENT, structural only
    print(f"  {v.word}: {v.tokens_a} → {v.tokens_b} ({v.violation_type.value})")

# Corpus diff
entries = load_corpus("prompts.jsonl")
pairs = [(e.id, e.text) for e in entries]
diffs = differ.diff_many(pairs, tok_a, tok_b)

# Cost report
from tokendrift.report.cost import CostCalculator
report = CostCalculator().compute(diffs, price_a=0.03, price_b=0.01)
print(f"Cost delta: ${report.cost_delta_usd:.4f}")
```

## Supported tokenizers

| Source | Example identifier | Notes |
|--------|-------------------|-------|
| tiktoken | `cl100k_base`, `o200k_base`, `p50k_base` | All OpenAI encodings |
| HuggingFace Hub | `Qwen/Qwen3-4B`, `meta-llama/Llama-3.2-1B` | Any model with `tokenizer.json` |
| Local directory | `/path/to/tokenizer/` | Loaded via HuggingFace `tokenizers` |
| Local file | `/path/to/tokenizer.json` | Direct file load |

## Running tests

```bash
# Offline tests (no network required)
pytest

# Full suite including real tiktoken / HuggingFace tokenizers
TOKENDRIFT_NETWORK_TESTS=1 pytest
```

## Project structure

```
tokendrift/
├── core/
│   ├── loader.py       # UnifiedTokenizer + backends (tiktoken, HuggingFace)
│   ├── vocab.py        # VocabDiffer
│   ├── differ.py       # EncodingDiffer
│   └── boundary.py     # BoundaryDetector
├── corpus/
│   └── loaders.py      # JSONL / CSV / plain-text corpus loading
├── report/
│   ├── terminal.py     # Rich terminal renderer
│   └── cost.py         # CostCalculator
└── cli/
    └── main.py         # Typer CLI
```

## Roadmap

The next milestone is turning the diff into a gate:

- [ ] `ci` command: pin a corpus's token counts in a baseline and exit non-zero when a tokenizer change moves them (the feature that makes this CI infrastructure rather than a one-off diagnostic)
- [ ] `gen-tests` command: generate a pytest regression suite pinning current behavior

Later:

- [ ] DuckDB corpus persistence (`corpus/store.py`)
- [ ] HTML report output
- [ ] Validate (or drop) the behavioural significance of boundary changes against a task benchmark; promote out of "experimental" only if it holds up
- [ ] Rust batch encoder for large corpora (100k+ entries)
- [ ] SentencePiece backend

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev
setup and the lint/format/type/test checks CI runs. Notable changes are recorded
in [CHANGELOG.md](CHANGELOG.md).

## License

MIT, see [LICENSE](LICENSE).
