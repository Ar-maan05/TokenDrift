## What does this change?

<!-- A short description of the change and why it's needed. -->

## Checklist

- [ ] Ran the checks locally: `ruff check .`, `ruff format --check .`, `pyright tokendrift/`, `pytest`
- [ ] Added or updated tests for the change (offline mock tokenizers where possible)
- [ ] For changes touching real-tokenizer behavior: ran `TOKENDRIFT_NETWORK_TESTS=1 pytest`
- [ ] Updated the CHANGELOG (`Unreleased` section) if user-facing
