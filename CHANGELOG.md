# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/Ar-maan05/tokendrift/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Ar-maan05/tokendrift/releases/tag/v0.1.0
