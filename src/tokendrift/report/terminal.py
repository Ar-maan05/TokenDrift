"""
tokendrift.report.terminal
~~~~~~~~~~~~~~~~~~~~~~~~~
Rich-based terminal renderer for diff results.

Produces human-readable output for:
- ``VocabDiff``     via ``render_vocab_diff``
- ``list[TokenDiff]``  via ``render_encoding_diff``
- ``CostReport``    via ``render_cost_report``

All functions write to stdout by default but accept a Rich ``Console``
so callers can redirect to a file or capture for testing.
"""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tokendrift.core.baseline import CIReport
from tokendrift.models import (
    CostReport,
    TokenDiff,
    ViolationType,
    VocabDiff,
)

_console = Console()


# ---------------------------------------------------------------------------
# Vocab diff
# ---------------------------------------------------------------------------


def render_vocab_diff(
    diff: VocabDiff,
    model_a: str,
    model_b: str,
    console: Console | None = None,
    show: str = "summary",
) -> None:
    """
    Render a ``VocabDiff`` to the terminal.

    Parameters
    ----------
    diff:
        The vocabulary diff to render.
    model_a, model_b:
        Names displayed in the header.
    console:
        Rich console (defaults to stdout).
    show:
        ``"summary"`` (default): counts only.
        ``"added"``:    list added tokens.
        ``"deleted"``:  list deleted tokens.
        ``"remapped"``: list remapped tokens.
        ``"all"``:      all three lists.
    """
    c = console or _console

    c.print()
    c.rule(f"[bold cyan]Vocab Diff[/]  [dim]{model_a}[/] → [dim]{model_b}[/]")
    c.print()

    tbl = Table(box=box.SIMPLE_HEAD, show_header=False, padding=(0, 2))
    tbl.add_column(style="dim")
    tbl.add_column()

    tbl.add_row("Vocab size A", f"{diff.total_a:,}")
    tbl.add_row("Vocab size B", f"{diff.total_b:,}")
    tbl.add_row("", "")
    tbl.add_row(
        "Added",
        f"[green]+{len(diff.added):,}[/]  tokens in B, not in A",
    )
    tbl.add_row(
        "Deleted",
        f"[red]-{len(diff.deleted):,}[/]  tokens in A, not in B",
    )

    remap_style = "bold red" if diff.has_remappings else "dim"
    remap_suffix = "  [bold red]⚠  stored token IDs now point elsewhere[/]" if diff.has_remappings else ""
    tbl.add_row(
        "Remapped",
        f"[{remap_style}]{len(diff.remapped):,}[/]{remap_suffix}",
    )

    c.print(tbl)

    if show in {"remapped", "all"} and diff.remapped:
        c.print()
        c.print("[bold]Remapped tokens (same string, different ID):[/]")
        rem_tbl = Table(box=box.SIMPLE, show_header=True)
        rem_tbl.add_column("Token", style="cyan")
        rem_tbl.add_column("Old ID", justify="right")
        rem_tbl.add_column("New ID", justify="right")
        for r in diff.remapped[:50]:  # cap to 50 rows
            rem_tbl.add_row(repr(r.token_str), str(r.old_id), str(r.new_id))
        if len(diff.remapped) > 50:
            rem_tbl.add_row(f"… and {len(diff.remapped) - 50} more", "", "")
        c.print(rem_tbl)

    if show in {"added", "all"} and diff.added:
        c.print()
        c.print("[bold]Added tokens:[/]")
        add_tbl = Table(box=box.SIMPLE, show_header=True)
        add_tbl.add_column("Token", style="green")
        add_tbl.add_column("ID", justify="right")
        for entry in diff.added[:50]:
            add_tbl.add_row(repr(entry.token_str), str(entry.token_id))
        if len(diff.added) > 50:
            add_tbl.add_row(f"… and {len(diff.added) - 50} more", "")
        c.print(add_tbl)

    if show in {"deleted", "all"} and diff.deleted:
        c.print()
        c.print("[bold]Deleted tokens:[/]")
        del_tbl = Table(box=box.SIMPLE, show_header=True)
        del_tbl.add_column("Token", style="red")
        del_tbl.add_column("ID", justify="right")
        for entry in diff.deleted[:50]:
            del_tbl.add_row(repr(entry.token_str), str(entry.token_id))
        if len(diff.deleted) > 50:
            del_tbl.add_row(f"… and {len(diff.deleted) - 50} more", "")
        c.print(del_tbl)

    c.print()


# ---------------------------------------------------------------------------
# Encoding diff
# ---------------------------------------------------------------------------


def render_encoding_diff(
    diffs: list[TokenDiff],
    model_a: str,
    model_b: str,
    top_n: int = 10,
    console: Console | None = None,
) -> None:
    """
    Render corpus-level encoding diff statistics and the top-N worst entries.

    Parameters
    ----------
    diffs:
        All ``TokenDiff`` objects (one per corpus entry).
    model_a, model_b:
        Names displayed in the header.
    top_n:
        Number of worst-affected entries to show.
    console:
        Rich console (defaults to stdout).
    """
    c = console or _console

    c.print()
    c.rule(f"[bold cyan]Encoding Diff[/]  [dim]{model_a}[/] → [dim]{model_b}[/]")
    c.print()

    changed = [d for d in diffs if d.changed]
    deltas = [d.count_delta for d in diffs]
    n = len(diffs)

    # Summary table
    tbl = Table(box=box.SIMPLE_HEAD, show_header=False, padding=(0, 2))
    tbl.add_column(style="dim")
    tbl.add_column()

    changed_pct = (len(changed) / n * 100) if n else 0
    tbl.add_row("Corpus entries", f"{n:,}")
    tbl.add_row(
        "Entries changed",
        f"[yellow]{len(changed):,}[/] / {n:,}  ({changed_pct:.1f}%)",
    )

    total_delta = sum(deltas)
    total_a = sum(d.token_count_a for d in diffs)
    pct_str = f" ({total_delta / total_a * 100:+.1f}%)" if total_a else ""
    delta_color = "green" if total_delta <= 0 else "red"
    tbl.add_row(
        "Total token Δ",
        f"[{delta_color}]{total_delta:+,}[/]{pct_str}",
    )

    if deltas:
        tbl.add_row("Max Δ (single)", f"{max(deltas):+,}")
        tbl.add_row("Min Δ (single)", f"{min(deltas):+,}")

    all_violations = [v for d in diffs for v in d.boundary_violations]
    boundaries_run = any(d.boundary_violations for d in diffs)
    splits = sum(1 for v in all_violations if v.violation_type == ViolationType.SPLIT)
    merges = sum(1 for v in all_violations if v.violation_type == ViolationType.MERGE)
    reseg = sum(1 for v in all_violations if v.violation_type == ViolationType.RESEGMENT)

    if boundaries_run:
        tbl.add_row("", "")
        tbl.add_row(
            "Boundary changes",
            f"[yellow]{len(all_violations):,}[/] words  "
            f"({splits} split, {merges} merged, {reseg} resegmented)  "
            "[dim](experimental, structural only)[/]",
        )

    c.print(tbl)

    # Top-N worst entries
    if changed:
        c.print()
        c.print(f"[bold]Top {min(top_n, len(changed))} most-affected entries[/]")
        worst = sorted(
            changed,
            key=lambda d: (abs(d.count_delta), d.split_count),
            reverse=True,
        )[:top_n]

        worst_tbl = Table(box=box.SIMPLE, show_header=True)
        worst_tbl.add_column("ID", style="dim", no_wrap=True)
        worst_tbl.add_column("Δ tokens", justify="right")
        if boundaries_run:
            worst_tbl.add_column("Splits", justify="right")
        worst_tbl.add_column("Preview", no_wrap=True)

        for d in worst:
            delta_str = f"[red]{d.count_delta:+}[/]" if d.count_delta > 0 else f"[green]{d.count_delta:+}[/]"
            preview = d.text[:60].replace("\n", " ")
            if len(d.text) > 60:
                preview += "…"
            row = [d.entry_id, delta_str]
            if boundaries_run:
                row.append(str(d.split_count) if d.split_count else "[dim]0[/]")
            row.append(f"[dim]{preview}[/]")
            worst_tbl.add_row(*row)

        c.print(worst_tbl)

    c.print()


# ---------------------------------------------------------------------------
# Cost report
# ---------------------------------------------------------------------------


def render_cost_report(
    report: CostReport,
    model_a: str,
    model_b: str,
    console: Console | None = None,
) -> None:
    """
    Render a ``CostReport`` to the terminal.

    Parameters
    ----------
    report:
        The cost report to render.
    model_a, model_b:
        Names displayed in the header.
    console:
        Rich console (defaults to stdout).
    """
    c = console or _console

    c.print()
    c.rule(f"[bold cyan]Cost Report[/]  [dim]{model_a}[/] → [dim]{model_b}[/]")
    c.print()

    tbl = Table(box=box.SIMPLE_HEAD, show_header=False, padding=(0, 2))
    tbl.add_column(style="dim")
    tbl.add_column()
    tbl.add_column(justify="right")

    tbl.add_row("Total tokens (A)", "", f"{report.total_tokens_a:,}")
    tbl.add_row("Total tokens (B)", "", f"{report.total_tokens_b:,}")
    delta_col = "red" if report.token_delta > 0 else "green"
    tbl.add_row(
        "Token delta",
        "",
        f"[{delta_col}]{report.token_delta:+,} ({report.pct_change:+.1f}%)[/]",
    )

    if report.cost_a_usd is not None:
        tbl.add_row("", "", "")
        tbl.add_row("Cost (A)", "", f"${report.cost_a_usd:.4f}")
        if report.cost_b_usd is not None:
            tbl.add_row("Cost (B)", "", f"${report.cost_b_usd:.4f}")
        if report.cost_delta_usd is not None:
            dcol = "red" if report.cost_delta_usd > 0 else "green"
            tbl.add_row(
                "Cost delta",
                "",
                f"[{dcol}]{report.cost_delta_usd:+.4f}[/]",
            )

    c.print(tbl)
    c.print()


# ---------------------------------------------------------------------------
# Single-entry detail view
# ---------------------------------------------------------------------------


def render_entry_detail(
    diff: TokenDiff,
    model_a: str,
    model_b: str,
    console: Console | None = None,
) -> None:
    """
    Render a detailed view of a single ``TokenDiff``, including all boundary
    violations with their token strings.
    """
    c = console or _console

    c.print()
    header = f"Entry [bold]{diff.entry_id}[/]  |  Δ {diff.count_delta:+} tokens  |  {model_a} → {model_b}"
    c.print(Panel(header, box=box.ROUNDED))

    c.print(f"\n[bold]Text[/] ({len(diff.text)} chars):")
    c.print(f"[dim]{diff.text[:300]}[/]")
    if len(diff.text) > 300:
        c.print("[dim]…[/]")

    c.print(f"\n[bold]Token counts:[/] {diff.token_count_a} → {diff.token_count_b}")
    c.print(f"[bold]First divergence:[/] char {diff.first_divergence_pos}")

    if diff.boundary_violations:
        c.print(
            f"\n[bold]Boundary changes ({len(diff.boundary_violations)})[/] [dim](experimental, structural only)[/]:"
        )
        vtbl = Table(box=box.SIMPLE, show_header=True)
        vtbl.add_column("Word", style="cyan")
        vtbl.add_column("Type")
        vtbl.add_column("Tokens A")
        vtbl.add_column("Tokens B")

        for v in diff.boundary_violations:
            vtype_color = {"SPLIT": "yellow", "MERGE": "cyan", "RESEGMENT": "dim"}[v.violation_type.value]
            vtbl.add_row(
                v.word,
                f"[{vtype_color}]{v.violation_type.value}[/]",
                " | ".join(repr(t) for t in v.tokens_a),
                " | ".join(repr(t) for t in v.tokens_b),
            )
        c.print(vtbl)
    else:
        c.print("\n[dim]No boundary changes detected.[/]")

    c.print()


# ---------------------------------------------------------------------------
# CI gate report
# ---------------------------------------------------------------------------


def render_ci_report(
    report: CIReport,
    top_n: int = 10,
    console: Console | None = None,
) -> None:
    """
    Render a :class:`CIReport` for a CI / pre-commit run.

    Shows the totals, the worst per-entry regressions, any new/missing entries,
    and a PASS / FAIL verdict with the reasons the gate failed.
    """
    c = console or _console

    c.print()
    verdict = "[bold green]PASS[/]" if report.passed else "[bold red]FAIL[/]"
    header = (
        f"TokenDrift CI  {verdict}  |  baseline [bold]{report.baseline_tokenizer}[/] "
        f"→ current [bold]{report.current_tokenizer}[/]"
    )
    c.print(Panel(header, box=box.ROUNDED))

    tbl = Table(box=box.SIMPLE, show_header=False)
    tbl.add_column("Metric", style="bold")
    tbl.add_column("Value", justify="right")
    tbl.add_row("Baseline tokens", f"{report.total_baseline:,}")
    tbl.add_row("Current tokens", f"{report.total_current:,}")
    dcol = "red" if report.token_delta > 0 else "green"
    pct = "inf" if report.total_pct == float("inf") else f"{report.total_pct:+.2f}%"
    tbl.add_row("Token delta", f"[{dcol}]{report.token_delta:+,} ({pct})[/]")
    if report.cost_delta_usd is not None:
        ccol = "red" if report.cost_delta_usd > 0 else "green"
        tbl.add_row("Est. cost delta", f"[{ccol}]${report.cost_delta_usd:+.4f}[/]")
    if report.new_entries:
        tbl.add_row("New entries", f"{len(report.new_entries):,}")
    if report.missing_entries:
        tbl.add_row("Missing entries", f"{len(report.missing_entries):,}")
    c.print(tbl)

    regressions = report.regressions
    if regressions:
        c.print(f"\n[bold]Top regressions (worst {min(top_n, len(regressions))} of {len(regressions)}):[/]")
        rtbl = Table(box=box.SIMPLE, show_header=True)
        rtbl.add_column("Entry", style="cyan")
        rtbl.add_column("Baseline", justify="right")
        rtbl.add_column("Current", justify="right")
        rtbl.add_column("Delta", justify="right")
        for d in regressions[:top_n]:
            epct = "inf" if d.pct == float("inf") else f"{d.pct:+.1f}%"
            rtbl.add_row(
                d.entry_id,
                str(d.baseline_tokens),
                str(d.current_tokens),
                f"[red]{d.delta:+} ({epct})[/]",
            )
        c.print(rtbl)

    if not report.passed:
        c.print("\n[bold red]Gate failed:[/]")
        for reason in report.failures:
            c.print(f"  [red]✗[/] {reason}")

    c.print()
