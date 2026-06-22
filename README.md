# TokenDrift

[![CI](https://github.com/Ar-maan05/tokendrift/actions/workflows/ci.yml/badge.svg)](https://github.com/Ar-maan05/tokendrift/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tokendrift.svg?v=2)](https://pypi.org/project/tokendrift/)
[![Python versions](https://img.shields.io/pypi/pyversions/tokendrift.svg?v=2)](https://pypi.org/project/tokendrift/)
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

## CI gating: catch tokenizer regressions automatically

A provider can silently re-tokenize a model and inflate your prompt token
counts (and bill) with no change on your side. TokenDrift turns that into a
build failure.

**1. Snapshot a baseline** against your current tokenizer and commit it:
```bash
tokendrift baseline cl100k_base --corpus prompts.jsonl -o tokendrift.baseline.json
git add tokendrift.baseline.json
```

**2. Gate every build** by re-encoding the same corpus and comparing:
```bash
tokendrift ci o200k_base \
  --baseline tokendrift.baseline.json \
  --corpus prompts.jsonl \
  --max-total-growth-pct 2 \
  --max-entry-growth-pct 10
```

`ci` exits non-zero when any threshold is breached, so it fails the pipeline:

| Flag | Fails the build when |
|------|----------------------|
| `--max-total-growth-pct` | total tokens grow by more than N% |
| `--max-entry-growth-pct` | any single entry grows by more than N% |
| `--price-per-1k` + `--max-cost-delta` | estimated cost grows by more than $X |
| `--fail-on-new` | the corpus has entries missing from the baseline |
| `--fail-on-missing` | the baseline has entries missing from the corpus |

Exit codes: `0` pass, `1` drift exceeded a threshold, `2` usage/IO error
(missing baseline, bad flags) so config mistakes are distinguishable from real
regressions.

**GitHub Actions:**
```yaml
- name: Guard against tokenizer drift
  run: |
    pip install tokendrift
    tokendrift ci o200k_base \
      --baseline tokendrift.baseline.json \
      --corpus prompts.jsonl \
      --max-total-growth-pct 2
```

**pre-commit** (`.pre-commit-config.yaml`):
```yaml
- repo: local
  hooks:
    - id: tokendrift-ci
      name: tokendrift token-drift gate
      entry: tokendrift ci o200k_base --baseline tokendrift.baseline.json --corpus prompts.jsonl --max-total-growth-pct 2
      language: system
      pass_filenames: false
```

## Cost, budget, and governance (v1.1.0)

The commands above diff two tokenizers. The commands below add cost, budget, and
governance on top, by mapping a model name to its tokenizer, context window, and
input price through a model registry.

The registry ships an indicative default set of models. Pricing and context
windows change often and vary by provider contract, so treat the shipped numbers
as starting points: verify them and override with your own registry JSON before
trusting the cost output for budgets or audits.

```bash
# Inspect the registry
tokendrift models

# Use your own registry
tokendrift models --registry my_models.json
```

A registry JSON file is a list of model objects (only `name` and `tokenizer` are
required):
```json
[
  {
    "name": "gpt-4o",
    "tokenizer": "o200k_base",
    "context_window": 128000,
    "price_per_1k_input": 0.0025,
    "max_output_tokens": 16384,
    "provider": "openai"
  },
  { "name": "my-hf-model", "tokenizer": "Qwen/Qwen3-4B", "context_window": 32768 }
]
```

Because `tokenizer` accepts any identifier `TokenizerLoader` understands, a
single registry (and a single comparison) can mix HuggingFace Hub models with
API-provider models.

### Estimate cost and context-window fit before dispatch

Score one prompt across several models for token count, input cost, and whether
it fits each context window, before any request is sent. This is the data behind
a playground cost overlay and a router's pre-dispatch budget check.

```bash
tokendrift estimate gpt-4o,gpt-4-turbo --text "Summarise this contract."
```

```
──────────────────── Cost & Budget Estimate ────────────────────
  Model         Tokens   Cost (in)            Context   Headroom   Fit
  gpt-4o             6    $0.0000 (cheapest)   128,000    111,610   ok
  gpt-4-turbo        6    $0.0001              128,000    123,898   ok

  Tokenization spread: 0 tokens (0.0% between lowest and highest).
```

A higher spread means the same text tokenizes very differently across models,
which is what drives cost and latency apart between providers. Add `--json` for
machine consumption, `--file prompt.txt` to read the prompt from a file, and
`--reserved-output N` to model a known completion budget in the fit check.

### Model migration safety check

Before switching a corpus of prompts from one model to another, see the token
delta, cost delta, vocabulary shift, and every prompt that would overflow the
target context window:

```bash
tokendrift migrate gpt-4-turbo gpt-4o --corpus prompts.jsonl --json
```

The `--json` form emits a flat report suitable for an audit or compliance layer,
including the list of overflowing prompts with how far over the limit each is.

### Prompt-compression savings, per model

A compression step removes characters, but the token saving is tokenizer
dependent. Measure the real saving (and cost saving) under each candidate model
so a compression decision is model aware:

```bash
tokendrift compress gpt-4o,gpt-4-turbo --original raw.txt --compressed small.txt
```

### Org-level spend forecast

Project input-token spend across candidate models for a target request volume,
using measured per-request token averages from a representative prompt sample:

```bash
tokendrift forecast gpt-4o,gpt-4o-mini --corpus sample.jsonl --requests 1000000
```

The forecast covers input tokens only (the part measurable from prompt text);
output cost depends on generation length.

### Tokenizer drift alerts for compliance

A provider silently updating their tokenizer between model versions is an audit
event. Run this as a background job against new versions and classify the drift
against a committed baseline as OK / WARN / CRITICAL:

```bash
tokendrift drift-alert o200k_base \
  --baseline tokendrift.baseline.json \
  --corpus prompts.jsonl \
  --warn-pct 2 --critical-pct 10 --json
```

It exits non-zero on CRITICAL (or on WARN with `--fail-on-warn`) and emits a JSON
alert for an audit pipeline. This is the alerting view of the same comparison the
`ci` command gates on.

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

# Baseline + CI gate (programmatic equivalent of `tokendrift ci`)
from tokendrift import build_baseline, run_ci, CIThresholds, Baseline

snapshot = build_baseline(tok_a, entries)
snapshot.save("tokendrift.baseline.json")

result = run_ci(
    Baseline.load("tokendrift.baseline.json"),
    tok_b,
    entries,
    CIThresholds(max_total_growth_pct=2),
)
print("passed" if result.passed else f"failed: {result.failures}")
```

### Cost, budget, and governance APIs (v1.1.0)

```python
from tokendrift import (
    ModelRegistry, ModelInfo, CostEstimator,
    migrate_report, compression_report, forecast,
    check_drift, build_baseline,
)

# Built-in registry, or load your own with ModelRegistry.from_json("models.json"),
# and add models (including HuggingFace ones) at runtime.
registry = ModelRegistry.default()
registry.add(ModelInfo(name="my-hf", tokenizer="Qwen/Qwen3-4B", context_window=32768))

# Pre-dispatch estimate across models (playground overlay / router budget check)
est = CostEstimator(registry)
result = est.estimate("Summarise this contract.", ["gpt-4o", "gpt-4-turbo"])
for e in result.estimates:
    print(e.model, e.token_count, e.cost_usd, e.fits)
print("spread:", result.token_spread, f"({result.divergence_pct:.1f}%)")

# A cheap boolean guard for a router
if est.fits(prompt, "gpt-4o", reserved_output=1000):
    ...  # safe to dispatch

# Migration safety report (serialises for audit via .to_dict())
report = migrate_report(entries, "gpt-4-turbo", "gpt-4o", registry=registry)
print(report.token_delta, report.cost_delta_usd, len(report.overflows))

# Compression savings, per model
csav = compression_report(original_text, compressed_text, ["gpt-4o"], registry=registry)

# Org-level spend forecast
fc = forecast(entries, ["gpt-4o", "gpt-4o-mini"], projected_requests=1_000_000, registry=registry)
print(fc.cheapest.model, fc.cheapest.projected_cost_usd)

# Tokenizer drift alert against a committed baseline
baseline = build_baseline(tok_a, entries)
alert = check_drift(baseline, tok_b, entries, warn_pct=2, critical_pct=10)
print(alert.severity.value, alert.triggered, alert.to_dict())
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
src/tokendrift/
├── core/
│   ├── loader.py       # UnifiedTokenizer + backends (tiktoken, HuggingFace)
│   ├── vocab.py        # VocabDiffer
│   ├── differ.py       # EncodingDiffer
│   ├── boundary.py     # BoundaryDetector
│   ├── baseline.py     # Baseline snapshots + CI gating (build_baseline, run_ci)
│   ├── registry.py     # ModelRegistry + ModelInfo (tokenizer, window, price)
│   ├── estimate.py     # CostEstimator (pre-dispatch cost + budget fit)
│   ├── migrate.py      # migrate_report (model migration safety)
│   ├── compression.py  # compression_report (per-model savings)
│   ├── forecast.py     # forecast (org-level spend projection)
│   └── alert.py        # check_drift (tokenizer drift alerts)
├── corpus/
│   └── loaders.py      # JSONL / CSV / plain-text corpus loading
├── report/
│   ├── terminal.py     # Rich terminal renderer
│   └── cost.py         # CostCalculator
└── cli/
    └── main.py         # Typer CLI
```

## Roadmap

- [x] `baseline` + `ci` commands: pin a corpus's token counts in a baseline and exit non-zero when a tokenizer change moves them (the feature that makes this CI infrastructure rather than a one-off diagnostic)
- [x] cost, budget, and governance toolkit (v1.1.0): model registry plus `estimate`, `migrate`, `compress`, `forecast`, and `drift-alert` commands for pre-dispatch cost and context-window checks, migration safety, and compliance drift alerts
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
