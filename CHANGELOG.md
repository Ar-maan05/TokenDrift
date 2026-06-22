# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-06-22

This release extends TokenDrift from a tokenizer diff tool into a pre-dispatch
cost, budget, and governance toolkit. Every new feature reasons about more than
raw token counts by mapping a model name to its tokenizer, context window, and
input price through a new registry.

### Added
- **Model registry** (`ModelRegistry`, `ModelInfo`): maps a friendly model name
  to its tokenizer, context window, and input price. Ships an indicative default
  set (verify pricing before relying on it), loads from and saves to JSON, caches
  resolved tokenizers, and accepts custom in-memory tokenizers via
  `register_tokenizer`. New `tokendrift models` command lists the registry.
- **Pre-dispatch estimation** (`CostEstimator`, `estimate` command): scores one
  prompt across many models for token count, input cost, and context-window fit
  before any request is sent. Reports the tokenization spread between models,
  which is what drives cost and latency apart between providers. Powers a
  playground cost overlay and a routing-engine budget check (`CostEstimator.fits`
  uses the correct tokenizer instead of a character-count approximation).
- **Migration safety checker** (`migrate_report`, `migrate` command): for a
  corpus of prompts, reports token delta, cost delta, vocabulary shift, and every
  prompt that would overflow the target model's context window. Serialises to a
  machine-readable dict for an audit or compliance layer.
- **Prompt-compression feedback** (`compression_report`,
  `compression_report_corpus`, `compress` command): measures the real token and
  cost saving of a compressed prompt per model, since the saving is tokenizer
  dependent.
- **Cost forecasting** (`forecast`, `forecast` command): projects input-token
  spend across candidate models for a target request volume using measured
  per-request token averages from a prompt sample.
- **Tokenizer drift alerts** (`check_drift`, `alert_to_json`, `drift-alert`
  command): classifies drift against a committed baseline as OK / WARN /
  CRITICAL with configurable thresholds, for a compliance background job. Emits
  JSON and exits non-zero on CRITICAL (or on WARN with `--fail-on-warn`).
- `--json` output on `estimate`, `migrate`, `forecast`, and `drift-alert` for
  pipeline consumption.
- `examples/05_cost_governance.py` demonstrating the new library APIs.

### Notes
- HuggingFace Hub tokenizers continue to work anywhere a tokenizer identifier is
  accepted, including registry entries, so a comparison can mix a HuggingFace
  model and an API-provider model in a single estimate or migration.

## [1.0.0] - 2026-06-14

First stable release. The CI regression-gate workflow that the project was
built toward is now in place and the public API is considered stable.

### Added
- **`baseline` command** and `build_baseline()`: snapshot per-entry and total
  token counts for a corpus under one tokenizer into a versioned JSON file you
  commit to your repository.
- **`ci` command** and `run_ci()`: re-encode a corpus under a candidate
  tokenizer, compare against a committed baseline, and exit non-zero when the
  drift breaks a threshold. Thresholds: `--max-total-growth-pct`,
  `--max-entry-growth-pct`, `--price-per-1k` + `--max-cost-delta`,
  `--fail-on-new`, and `--fail-on-missing`. Exit codes distinguish a real
  regression (`1`) from a usage/IO error (`2`).
- `CIThresholds`, `CIReport`, and `EntryDrift` are exported from the package
  root; a `render_ci_report` Rich renderer shows totals, worst regressions, and
  the reasons a gate failed.
- README sections plus a GitHub Actions and pre-commit recipe for the gate, and
  `examples/04_ci_gate.py`.
- `TokenDiff.pct_change`, `CostReport.pct_token_change`, and
  `CostReport.pct_cost_change` convenience properties.

### Changed
- Moved the package to a `src/` layout (`src/tokendrift/`). No import changes
  for consumers; `import tokendrift` is unchanged.

## [0.1.0] - 2026-06-14

Initial release.

### Added
- **`TokenizerLoader`**: a unified interface over tiktoken encodings and
  HuggingFace `tokenizers` (by Hub model ID, local directory, or `tokenizer.json`
  file). Detection picks the backend automatically; an unresolved bare name
  surfaces a helpful error with the closest tiktoken encoding suggested.
- **`VocabDiffer`**: vocabulary diff at the string and ID level, reporting added,
  deleted, and remapped tokens. ID remappings are highlighted because any system
  that stored a raw token ID now points to a different entry.
- **`EncodingDiffer`**: per-entry encoding diff with token-count delta and first
  divergence position, plus a `diff_many` batch helper.
- **`CostCalculator`**: per-prompt and corpus-level cost impact from per-1k-token
  prices. A price of `0.0` is honoured (free tier) rather than dropped.
- **`BoundaryDetector`** (experimental, opt-in): structural word-level boundary
  changes typed as `SPLIT`, `MERGE`, or `RESEGMENT`. This is a structural report
  only, with no severity ranking and no claim of behavioural impact. Pure
  integer-ID renumbering is reported at the vocabulary level, not here, so it
  does not flood the output.
- **Corpus loaders**: JSONL, CSV/TSV, and plain-text ingestion into
  `CorpusEntry` objects, with auto-assigned IDs when absent.
- **Rich terminal renderers** for vocab diff, encoding diff (with a top-N
  most-affected table), cost report, and a single-entry detail view.
- **CLI** (`tokendrift`): `diff`, `vocab-diff`, `cost`, and `entry` commands.
  Boundary detection is opt-in via `--boundaries` on `diff`.
- Ships type information (`py.typed`); tested on Python 3.10 through 3.13.

[Unreleased]: https://github.com/Ar-maan05/tokendrift/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Ar-maan05/tokendrift/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/Ar-maan05/tokendrift/releases/tag/v0.1.0
