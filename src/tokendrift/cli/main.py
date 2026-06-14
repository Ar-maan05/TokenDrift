"""
tokendrift.cli.main
~~~~~~~~~~~~~~~~~~
Command-line interface.

Commands
--------
diff        Diff two tokenizers against a corpus or a single text snippet.
vocab-diff  Compare vocabularies only (no corpus required).
entry       Inspect a single corpus entry in detail.
cost        Generate a cost impact report.
baseline    Snapshot token counts for a corpus under one tokenizer.
ci          Gate a build on token drift against a committed baseline.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from tokendrift.core.baseline import (
    Baseline,
    CIThresholds,
    build_baseline,
    run_ci,
)
from tokendrift.core.differ import EncodingDiffer
from tokendrift.core.loader import TokenizerLoader
from tokendrift.core.vocab import VocabDiffer
from tokendrift.corpus.loaders import load_corpus
from tokendrift.report.cost import CostCalculator
from tokendrift.report.terminal import (
    render_ci_report,
    render_cost_report,
    render_encoding_diff,
    render_entry_detail,
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
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    app()


if __name__ == "__main__":
    main()
