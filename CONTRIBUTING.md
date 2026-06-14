# Contributing to TokenLens

Thanks for your interest in improving `tokenlens`. This project diffs two
tokenizers (token counts, cost, vocabulary, and experimental structural
boundary changes) against a prompt corpus.

## Development setup

Python 3.10+ is required. Install the package with the dev tooling:

```bash
git clone https://github.com/Ar-maan05/tokenlens
cd tokenlens
pip install -e ".[dev]"
```

## Running the checks

CI runs these four checks on Python 3.10 through 3.13, and they must pass before
a PR can merge. Run them locally first:

```bash
ruff check .            # lint
ruff format --check .   # formatting (run `ruff format .` to fix)
pyright tokenlens/      # static type checking
pytest                  # tests
```

Code style is enforced by `ruff format`; please run it rather than hand-tuning
whitespace.

## Tests and network access

Most of the suite runs against two self-contained mock tokenizers (a
character-level one and a bigram one), so no network is needed for normal
development:

```bash
pytest
```

Tests that load real tokenizers (tiktoken encodings, HuggingFace Hub models)
are marked `@pytest.mark.network` and skipped by default. Enable them with an
environment variable; the first run downloads vocab files:

```bash
TOKENLENS_NETWORK_TESTS=1 pytest
```

When adding behavior, please cover it with a test. Prefer the mock tokenizers
so the test stays offline; reserve `@pytest.mark.network` for cases that
genuinely need real BPE behavior.

## A note on boundary detection

Structural boundary-change detection (`SPLIT` / `MERGE` / `RESEGMENT`) is
experimental and intentionally reports structure only, with no claim about
behavioural impact. Please keep that framing: do not reintroduce severity
ranking or language implying a boundary change degrades model quality unless it
is backed by a task benchmark.

## Submitting changes

1. Open a pull request against `main`.
2. CI must be green (lint, format, types, and tests across all four Python
   versions) before it can merge.
3. Keep PRs focused: one logical change per PR makes review faster.
