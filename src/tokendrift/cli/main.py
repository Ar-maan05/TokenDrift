"""
tokendrift.cli.main
~~~~~~~~~~~~~~~~~~
Command-line interface.

Commands
--------
diff         Diff two tokenizers against a corpus or a single text snippet.
vocab-diff   Compare vocabularies only (no corpus required).
entry        Inspect a single corpus entry in detail.
cost         Generate a cost impact report.
baseline     Snapshot token counts for a corpus under one tokenizer.
ci           Gate a build on token drift against a committed baseline.
estimate     Pre-dispatch token + cost + context-window estimate across models.
migrate      Model migration safety report (token, cost, vocab, overflow).
compress     Measure prompt-compression savings per model.
forecast     Project org-level spend across candidate models.
drift-alert  Classify tokenizer drift against a baseline for compliance.
models       List the model registry.
"""

from __future__ import annotations

import json as _json
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from tokendrift.core.alert import alert_to_json, check_drift
from tokendrift.core.baseline import (
    Baseline,
    CIThresholds,
    build_baseline,
    run_ci,
)
from tokendrift.core.compression import compression_report
from tokendrift.core.differ import EncodingDiffer
from tokendrift.core.estimate import CostEstimator
from tokendrift.core.forecast import forecast as forecast_fn
from tokendrift.core.loader import TokenizerLoader
from tokendrift.core.migrate import migrate_report
from tokendrift.core.registry import ModelRegistry
from tokendrift.core.vocab import VocabDiffer
from tokendrift.corpus.loaders import load_corpus
from tokendrift.models import AlertSeverity
from tokendrift.report.cost import CostCalculator
from tokendrift.report.terminal import (
    render_ci_report,
    render_compression_report,
    render_cost_report,
    render_drift_alert,
    render_encoding_diff,
    render_entry_detail,
    render_estimate,
    render_forecast_report,
    render_migration_report,
    render_vocab_diff,
)

app = typer.Typer(
    name="tokendrift",
    help=(
        "Token-count, cost, and vocabulary diffing for LLM tokenizer changes.\n\n"
        "Examples:\n\n"
        "  tokendrift diff cl100k_base o200k_base --text 'biostatistical'\n\n"
        "  tokendrift diff cl100k_base o200k_base --corpus prompts.jsonl\n\n"
        "  tokendrift vocab-diff cl100k_base o200k_base --show remapped"
    ),
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

_console = Console()


def version_callback(value: bool) -> None:
    if value:
        import importlib.metadata

        try:
            version_str = importlib.metadata.version("tokendrift")
        except importlib.metadata.PackageNotFoundError:
            version_str = "0.0.0-dev"
        _console.print(f"tokendrift version {version_str}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool | None = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    pass


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


@app.command()
def diff(
    model_a: str = typer.Argument(..., help="Tokenizer A (before). E.g. 'cl100k_base' or 'Qwen/Qwen3-4B'."),
    model_b: str = typer.Argument(..., help="Tokenizer B (after)."),
    corpus: Path | None = typer.Option(
        None,
        "--corpus",
        "-c",
        help="Path to a JSONL / CSV / plain-text corpus file.",
        exists=False,
    ),
    text: str | None = typer.Option(
        None,
        "--text",
        "-t",
        help="Inline text to diff (skip corpus).",
    ),
    top_n: int = typer.Option(
        10,
        "--top-n",
        help="Number of worst-affected entries to show.",
    ),
    boundaries: bool = typer.Option(
        False,
        "--boundaries",
        help="Enable experimental structural boundary-change detection (SPLIT / MERGE / RESEGMENT). Off by default.",
    ),
    word_tok: str = typer.Option(
        "whitespace",
        "--word-tok",
        help="Word tokenizer for boundary detection: whitespace | nltk.",
    ),
    price_a: float | None = typer.Option(
        None,
        "--price-a",
        help="Price per 1k tokens for model A (USD). Enables cost report.",
    ),
    price_b: float | None = typer.Option(
        None,
        "--price-b",
        help="Price per 1k tokens for model B (USD).",
    ),
    vocab: bool = typer.Option(
        True,
        "--vocab/--no-vocab",
        help="Include vocabulary diff in output.",
    ),
) -> None:
    """
    Diff two tokenizers against a corpus or a single text snippet.

    At least one of --corpus or --text must be provided.
    """
    if corpus is None and text is None:
        _console.print("[red]Error:[/] provide --corpus or --text.")
        raise typer.Exit(1)

    # Load tokenizers
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=_console) as p:
        t = p.add_task("Loading tokenizers…", total=None)
        try:
            tok_a = TokenizerLoader.load(model_a)
            tok_b = TokenizerLoader.load(model_b)
        except Exception as exc:
            _console.print(f"[red]Failed to load tokenizer:[/] {exc}")
            raise typer.Exit(1) from exc
        p.remove_task(t)

    # Vocabulary diff
    if vocab:
        v_diff = VocabDiffer().diff(tok_a, tok_b)
        render_vocab_diff(v_diff, model_a, model_b)

    # Build entries list
    if text is not None:
        from tokendrift.models import CorpusEntry

        entries = [CorpusEntry(id="inline", text=text)]
    else:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=_console) as p:
            t = p.add_task(f"Loading corpus {corpus}…", total=None)
            try:
                entries = load_corpus(corpus)  # type: ignore[arg-type]
            except Exception as exc:
                p.remove_task(t)
                _console.print(f"[red]Error loading corpus:[/] {exc}")
                raise typer.Exit(1)
            p.remove_task(t)
        _console.print(f"[dim]Loaded {len(entries):,} corpus entries.[/]")

    # Encoding diff
    differ = EncodingDiffer(
        detect_boundaries=boundaries,
        word_tokenizer=word_tok,  # type: ignore[arg-type]
    )
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=_console) as p:
        t = p.add_task("Diffing encodings…", total=None)
        pairs = [(e.id, e.text) for e in entries]
        diffs = differ.diff_many(pairs, tok_a, tok_b)
        p.remove_task(t)

    if text is not None:
        # Single-entry detail view
        render_entry_detail(diffs[0], model_a, model_b)
    else:
        render_encoding_diff(diffs, model_a, model_b, top_n=top_n)

    # Cost report
    if price_a is not None or price_b is not None:
        cost_report = CostCalculator().compute(diffs, price_a=price_a, price_b=price_b)
        render_cost_report(cost_report, model_a, model_b)


# ---------------------------------------------------------------------------
# vocab-diff
# ---------------------------------------------------------------------------


@app.command(name="vocab-diff")
def vocab_diff(
    model_a: str = typer.Argument(..., help="Tokenizer A (before)."),
    model_b: str = typer.Argument(..., help="Tokenizer B (after)."),
    show: str = typer.Option(
        "summary",
        "--show",
        help="What to display: summary | added | deleted | remapped | all.",
    ),
) -> None:
    """
    Compare tokenizer vocabularies: additions, deletions, and ID remappings.

    Remapped tokens are the most dangerous; any system that stored a token ID
    rather than the string now points to the wrong token silently.

    Example:

      tokendrift vocab-diff cl100k_base o200k_base --show remapped
    """
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=_console) as p:
        t = p.add_task("Loading tokenizers…", total=None)
        try:
            tok_a = TokenizerLoader.load(model_a)
            tok_b = TokenizerLoader.load(model_b)
        except Exception as exc:
            _console.print(f"[red]Failed to load tokenizer:[/] {exc}")
            raise typer.Exit(1) from exc
        p.remove_task(t)

    v_diff = VocabDiffer().diff(tok_a, tok_b)
    render_vocab_diff(v_diff, model_a, model_b, show=show)


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------


@app.command()
def cost(
    model_a: str = typer.Argument(..., help="Tokenizer A (before)."),
    model_b: str = typer.Argument(..., help="Tokenizer B (after)."),
    corpus: Path = typer.Option(..., "--corpus", "-c", help="Path to corpus file."),
    price_a: float = typer.Option(..., "--price-a", help="Price per 1k tokens for A (USD)."),
    price_b: float = typer.Option(..., "--price-b", help="Price per 1k tokens for B (USD)."),
) -> None:
    """
    Generate a cost impact report for a tokenizer change.

    Example:

      tokendrift cost cl100k_base o200k_base --corpus prompts.jsonl \\
          --price-a 0.03 --price-b 0.01
    """
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=_console) as p:
        t = p.add_task("Loading…", total=None)
        try:
            tok_a = TokenizerLoader.load(model_a)
            tok_b = TokenizerLoader.load(model_b)
            entries = load_corpus(corpus)
        except Exception as exc:
            p.remove_task(t)
            _console.print(f"[red]Error:[/] {exc}")
            raise typer.Exit(1)
        p.remove_task(t)

    differ = EncodingDiffer(detect_boundaries=False)
    pairs = [(e.id, e.text) for e in entries]
    diffs = differ.diff_many(pairs, tok_a, tok_b)

    report = CostCalculator().compute(diffs, price_a=price_a, price_b=price_b)
    render_cost_report(report, model_a, model_b)


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------


@app.command()
def entry(
    model_a: str = typer.Argument(..., help="Tokenizer A (before)."),
    model_b: str = typer.Argument(..., help="Tokenizer B (after)."),
    text: str = typer.Option(..., "--text", "-t", help="Text to inspect."),
    word_tok: str = typer.Option("whitespace", "--word-tok", help="Word tokenizer."),
) -> None:
    """
    Inspect a single text in detail: token counts, first divergence, and all
    boundary violations with their token strings.

    Example:

      tokendrift entry cl100k_base o200k_base \\
          --text "ChatGPT rewrites biostatistical significance tests"
    """
    try:
        tok_a = TokenizerLoader.load(model_a)
        tok_b = TokenizerLoader.load(model_b)
    except Exception as exc:
        _console.print(f"[red]Failed to load tokenizer:[/] {exc}")
        raise typer.Exit(1)
    differ = EncodingDiffer(detect_boundaries=True, word_tokenizer=word_tok)  # type: ignore[arg-type]
    d = differ.diff(text, tok_a, tok_b, entry_id="inline")
    render_entry_detail(d, model_a, model_b)


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------


@app.command()
def baseline(
    model: str = typer.Argument(..., help="Tokenizer to snapshot. E.g. 'cl100k_base' or 'Qwen/Qwen3-4B'."),
    corpus: Path = typer.Option(..., "--corpus", "-c", help="Path to a JSONL / CSV / plain-text corpus file."),
    output: Path = typer.Option(
        Path("tokendrift.baseline.json"),
        "--output",
        "-o",
        help="Where to write the baseline JSON.",
    ),
) -> None:
    """
    Snapshot per-entry token counts for a corpus under one tokenizer and write
    them to a JSON file you commit to your repository.

    Run this once against your current tokenizer, then gate future builds with
    `tokendrift ci`.

    Example:

      tokendrift baseline cl100k_base --corpus prompts.jsonl -o tokendrift.baseline.json
    """
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=_console) as p:
        t = p.add_task("Loading…", total=None)
        try:
            tok = TokenizerLoader.load(model)
            entries = load_corpus(corpus)
        except Exception as exc:
            p.remove_task(t)
            _console.print(f"[red]Error:[/] {exc}")
            raise typer.Exit(1) from exc
        p.remove_task(t)

    snapshot = build_baseline(tok, entries)
    snapshot.save(output)
    _console.print(
        f"[green]Wrote baseline[/] [bold]{output}[/]  "
        f"({len(snapshot.entries):,} entries, {snapshot.total_tokens:,} tokens, "
        f"tokenizer [bold]{snapshot.tokenizer}[/])"
    )


# ---------------------------------------------------------------------------
# ci
# ---------------------------------------------------------------------------


@app.command()
def ci(
    model: str = typer.Argument(..., help="Tokenizer to test against the baseline (usually the new one)."),
    baseline_path: Path = typer.Option(
        Path("tokendrift.baseline.json"),
        "--baseline",
        "-b",
        help="Path to the committed baseline JSON.",
    ),
    corpus: Path = typer.Option(
        ..., "--corpus", "-c", help="Path to the corpus file (same one used for the baseline)."
    ),
    max_total_growth_pct: float | None = typer.Option(
        None,
        "--max-total-growth-pct",
        help="Fail if total tokens grow by more than this percent.",
    ),
    max_entry_growth_pct: float | None = typer.Option(
        None,
        "--max-entry-growth-pct",
        help="Fail if any single entry grows by more than this percent.",
    ),
    price_per_1k: float | None = typer.Option(
        None,
        "--price-per-1k",
        help="USD per 1k tokens, used to estimate the cost delta.",
    ),
    max_cost_delta: float | None = typer.Option(
        None,
        "--max-cost-delta",
        help="Fail if the estimated total cost grows by more than this many USD. Requires --price-per-1k.",
    ),
    fail_on_new: bool = typer.Option(
        False,
        "--fail-on-new",
        help="Fail if the corpus has entries absent from the baseline.",
    ),
    fail_on_missing: bool = typer.Option(
        False,
        "--fail-on-missing",
        help="Fail if the baseline has entries absent from the corpus.",
    ),
    top_n: int = typer.Option(10, "--top-n", help="Number of worst regressions to show."),
) -> None:
    """
    Compare a corpus encoded under MODEL against a committed baseline and exit
    non-zero if the token drift breaks any threshold.

    Designed for CI pipelines and pre-commit hooks: a provider that silently
    re-tokenizes can inflate your prompt token counts (and cost) with no code
    change, and this command turns that into a build failure.

    Example:

      tokendrift ci o200k_base --baseline tokendrift.baseline.json \\
          --corpus prompts.jsonl --max-total-growth-pct 2
    """
    if max_cost_delta is not None and price_per_1k is None:
        _console.print("[red]Error:[/] --max-cost-delta requires --price-per-1k.")
        raise typer.Exit(2)

    try:
        snapshot = Baseline.load(baseline_path)
    except (FileNotFoundError, ValueError) as exc:
        _console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(2) from exc

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=_console) as p:
        t = p.add_task("Loading…", total=None)
        try:
            tok = TokenizerLoader.load(model)
            entries = load_corpus(corpus)
        except Exception as exc:
            p.remove_task(t)
            _console.print(f"[red]Error:[/] {exc}")
            raise typer.Exit(2) from exc
        p.remove_task(t)

    thresholds = CIThresholds(
        max_total_growth_pct=max_total_growth_pct,
        max_entry_growth_pct=max_entry_growth_pct,
        max_cost_delta_usd=max_cost_delta,
        price_per_1k=price_per_1k,
        fail_on_new_entries=fail_on_new,
        fail_on_missing_entries=fail_on_missing,
    )

    report = run_ci(snapshot, tok, entries, thresholds)
    render_ci_report(report, top_n=top_n)

    raise typer.Exit(0 if report.passed else 1)


# ---------------------------------------------------------------------------
# v1.1.0 helpers
# ---------------------------------------------------------------------------


def _parse_models(models: str) -> list[str]:
    """Split a comma-separated --models value into a clean list."""
    names = [m.strip() for m in models.split(",") if m.strip()]
    if not names:
        _console.print("[red]Error:[/] --models must list at least one model.")
        raise typer.Exit(2)
    return names


def _load_registry(registry_path: Path | None) -> ModelRegistry:
    """Load a registry from JSON, or return the built-in default."""
    if registry_path is None:
        return ModelRegistry.default()
    try:
        return ModelRegistry.from_json(registry_path)
    except (FileNotFoundError, ValueError) as exc:
        _console.print(f"[red]Error loading registry:[/] {exc}")
        raise typer.Exit(2) from exc


# ---------------------------------------------------------------------------
# estimate
# ---------------------------------------------------------------------------


@app.command()
def estimate(
    models: str = typer.Argument(..., help="Comma-separated model names or tokenizer ids. E.g. 'gpt-4o,gpt-4-turbo'."),
    text: str | None = typer.Option(None, "--text", "-t", help="Inline text to estimate."),
    file: Path | None = typer.Option(None, "--file", "-f", help="Read the prompt text from this file instead."),
    reserved_output: int | None = typer.Option(
        None,
        "--reserved-output",
        help="Tokens to reserve for the completion in the fit check. Defaults to each model's max output.",
    ),
    registry_path: Path | None = typer.Option(
        None, "--registry", help="Path to a model registry JSON file (defaults to the built-in registry)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the estimate as JSON instead of a table."),
) -> None:
    """
    Estimate token count, input cost, and context-window fit for a prompt across
    several models, before any request is sent.

    Example:

      tokendrift estimate gpt-4o,gpt-4-turbo --text "Summarise this contract."
    """
    if text is None and file is None:
        _console.print("[red]Error:[/] provide --text or --file.")
        raise typer.Exit(2)
    if file is not None:
        try:
            text = file.read_text(encoding="utf-8")
        except OSError as exc:
            _console.print(f"[red]Error reading file:[/] {exc}")
            raise typer.Exit(2) from exc

    registry = _load_registry(registry_path)
    estimator = CostEstimator(registry)
    try:
        result = estimator.estimate(text or "", _parse_models(models), reserved_output=reserved_output)
    except Exception as exc:  # noqa: BLE001 - surface a clean message
        _console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(2) from exc

    if as_json:
        payload = {
            "text_chars": result.text_chars,
            "token_spread": result.token_spread,
            "divergence_pct": round(result.divergence_pct, 4),
            "estimates": [
                {
                    "model": e.model,
                    "tokenizer": e.tokenizer,
                    "token_count": e.token_count,
                    "cost_usd": e.cost_usd,
                    "context_window": e.context_window,
                    "fits": e.fits,
                    "headroom": e.headroom,
                }
                for e in result.estimates
            ],
        }
        _console.print_json(_json.dumps(payload))
        return
    render_estimate(result)


# ---------------------------------------------------------------------------
# migrate
# ---------------------------------------------------------------------------


@app.command()
def migrate(
    source: str = typer.Argument(..., help="Source model name or tokenizer id."),
    target: str = typer.Argument(..., help="Target model name or tokenizer id."),
    corpus: Path = typer.Option(..., "--corpus", "-c", help="Path to a sample of historical prompts."),
    reserved_output: int | None = typer.Option(
        None, "--reserved-output", help="Tokens to reserve for the completion in the overflow check."
    ),
    no_vocab: bool = typer.Option(False, "--no-vocab", help="Skip the vocabulary diff."),
    registry_path: Path | None = typer.Option(None, "--registry", help="Path to a model registry JSON file."),
    as_json: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
    top_n: int = typer.Option(10, "--top-n", help="Number of overflowing prompts to show."),
) -> None:
    """
    Report what changes when a corpus of prompts moves from SOURCE to TARGET:
    token delta, cost delta, vocabulary shift, and prompts that would overflow
    the target context window.

    Example:

      tokendrift migrate gpt-4-turbo gpt-4o --corpus prompts.jsonl
    """
    registry = _load_registry(registry_path)
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=_console) as p:
        t = p.add_task("Loading…", total=None)
        try:
            entries = load_corpus(corpus)
        except Exception as exc:
            p.remove_task(t)
            _console.print(f"[red]Error:[/] {exc}")
            raise typer.Exit(2) from exc
        p.remove_task(t)

    try:
        report = migrate_report(
            entries,
            source,
            target,
            registry=registry,
            reserved_output=reserved_output,
            include_vocab=not no_vocab,
        )
    except Exception as exc:  # noqa: BLE001
        _console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(2) from exc

    if as_json:
        _console.print_json(_json.dumps(report.to_dict()))
        return
    render_migration_report(report, top_n=top_n)


# ---------------------------------------------------------------------------
# compress
# ---------------------------------------------------------------------------


@app.command()
def compress(
    models: str = typer.Argument(..., help="Comma-separated model names or tokenizer ids."),
    original: Path = typer.Option(..., "--original", help="File with the original (uncompressed) prompt."),
    compressed: Path = typer.Option(..., "--compressed", help="File with the compressed prompt."),
    registry_path: Path | None = typer.Option(None, "--registry", help="Path to a model registry JSON file."),
) -> None:
    """
    Measure how many tokens (and how much money) a compression step actually
    saves under each model, since the saving is tokenizer dependent.

    Example:

      tokendrift compress gpt-4o,gpt-4-turbo --original raw.txt --compressed small.txt
    """
    try:
        original_text = original.read_text(encoding="utf-8")
        compressed_text = compressed.read_text(encoding="utf-8")
    except OSError as exc:
        _console.print(f"[red]Error reading file:[/] {exc}")
        raise typer.Exit(2) from exc

    registry = _load_registry(registry_path)
    try:
        report = compression_report(original_text, compressed_text, _parse_models(models), registry=registry)
    except Exception as exc:  # noqa: BLE001
        _console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(2) from exc
    render_compression_report(report)


# ---------------------------------------------------------------------------
# forecast
# ---------------------------------------------------------------------------


@app.command()
def forecast(
    models: str = typer.Argument(..., help="Comma-separated model names or tokenizer ids."),
    corpus: Path = typer.Option(..., "--corpus", "-c", help="Representative sample of recent prompts."),
    requests: int = typer.Option(..., "--requests", "-r", help="Projected number of requests (e.g. a month)."),
    registry_path: Path | None = typer.Option(None, "--registry", help="Path to a model registry JSON file."),
    as_json: bool = typer.Option(False, "--json", help="Emit the forecast as JSON."),
) -> None:
    """
    Project input-token spend across candidate models for a target request
    volume, using measured per-request token averages from a prompt sample.

    Example:

      tokendrift forecast gpt-4o,gpt-4o-mini --corpus sample.jsonl --requests 1000000
    """
    registry = _load_registry(registry_path)
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=_console) as p:
        t = p.add_task("Loading…", total=None)
        try:
            entries = load_corpus(corpus)
        except Exception as exc:
            p.remove_task(t)
            _console.print(f"[red]Error:[/] {exc}")
            raise typer.Exit(2) from exc
        p.remove_task(t)

    try:
        report = forecast_fn(entries, _parse_models(models), requests, registry=registry)
    except Exception as exc:  # noqa: BLE001
        _console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(2) from exc

    if as_json:
        _console.print_json(_json.dumps(report.to_dict()))
        return
    render_forecast_report(report)


# ---------------------------------------------------------------------------
# drift-alert
# ---------------------------------------------------------------------------


@app.command(name="drift-alert")
def drift_alert(
    model: str = typer.Argument(..., help="Tokenizer to test against the baseline (usually the new one)."),
    baseline_path: Path = typer.Option(
        Path("tokendrift.baseline.json"), "--baseline", "-b", help="Path to the committed baseline JSON."
    ),
    corpus: Path = typer.Option(..., "--corpus", "-c", help="Corpus the baseline was built from."),
    warn_pct: float = typer.Option(2.0, "--warn-pct", help="Total-drift percent at which severity becomes WARN."),
    critical_pct: float = typer.Option(
        10.0, "--critical-pct", help="Total-drift percent at which severity becomes CRITICAL."
    ),
    price_per_1k: float | None = typer.Option(
        None, "--price-per-1k", help="USD per 1k tokens, used to estimate the cost delta."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the alert as JSON (for an audit pipeline)."),
    fail_on_warn: bool = typer.Option(
        False, "--fail-on-warn", help="Exit non-zero on WARN as well as CRITICAL (default: CRITICAL only)."
    ),
) -> None:
    """
    Classify tokenizer drift against a committed baseline as OK / WARN /
    CRITICAL, for a compliance background job.

    Exits 1 on CRITICAL (or on WARN with --fail-on-warn), else 0.

    Example:

      tokendrift drift-alert o200k_base --baseline base.json --corpus prompts.jsonl \\
          --warn-pct 2 --critical-pct 10 --json
    """
    try:
        snapshot = Baseline.load(baseline_path)
    except (FileNotFoundError, ValueError) as exc:
        _console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(2) from exc

    try:
        tok = TokenizerLoader.load(model)
        entries = load_corpus(corpus)
    except Exception as exc:  # noqa: BLE001
        _console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(2) from exc

    try:
        alert = check_drift(
            snapshot, tok, entries, warn_pct=warn_pct, critical_pct=critical_pct, price_per_1k=price_per_1k
        )
    except ValueError as exc:
        _console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(2) from exc

    if as_json:
        _console.print_json(alert_to_json(alert))
    else:
        render_drift_alert(alert)

    failed = alert.severity is AlertSeverity.CRITICAL or (fail_on_warn and alert.triggered)
    raise typer.Exit(1 if failed else 0)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


@app.command()
def models(
    registry_path: Path | None = typer.Option(None, "--registry", help="Path to a model registry JSON file."),
) -> None:
    """
    List the models in the registry with their tokenizer, context window, and
    input price.

    Example:

      tokendrift models
    """
    from rich import box as _box
    from rich.table import Table as _Table

    registry = _load_registry(registry_path)
    tbl = _Table(box=_box.SIMPLE_HEAD, show_header=True)
    tbl.add_column("Model", style="cyan")
    tbl.add_column("Tokenizer")
    tbl.add_column("Context", justify="right")
    tbl.add_column("$/1k in", justify="right")
    tbl.add_column("Provider", style="dim")
    for name in registry.names():
        info = registry.get(name)
        window = f"{info.context_window:,}" if info.context_window is not None else "[dim]?[/]"
        price = f"{info.price_per_1k_input:g}" if info.price_per_1k_input is not None else "[dim]?[/]"
        tbl.add_row(info.name, info.tokenizer, window, price, info.provider)
    _console.print(tbl)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    app()


if __name__ == "__main__":
    main()
