# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Terminal output: header panel, progress bars with ETA, summary tables.

Uses `rich`. When stdout is not a terminal (CI logs, `nohup`), progress bars
fall back to plain line output; `--quiet` disables them entirely.
"""
from __future__ import annotations

from contextlib import contextmanager

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskProgressColumn,
                           TextColumn, TimeElapsedColumn, TimeRemainingColumn)
from rich.table import Table

console = Console(highlight=False)
QUIET = False


def set_quiet(q: bool) -> None:
    global QUIET
    QUIET = q


def header(title: str, rows: dict[str, str]) -> None:
    if QUIET:
        return
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold")
    t.add_column()
    for k, v in rows.items():
        t.add_row(escape(k), escape(v))
    console.print(Panel(t, title=f"[bold]{title}[/bold]", expand=False))


@contextmanager
def progress():
    """A progress display with spinner, bar, M/N, percentage, elapsed and ETA."""
    p = Progress(SpinnerColumn(), TextColumn("[bold]{task.description:<28}"), BarColumn(bar_width=36),
                 MofNCompleteColumn(), TaskProgressColumn(), TextColumn("elapsed"), TimeElapsedColumn(),
                 TextColumn("eta"), TimeRemainingColumn(), TextColumn("[dim]{task.fields[detail]}"),
                 console=console, disable=QUIET, transient=False)
    with p:
        yield p


def table(title: str, columns: list[str], rows: list[list], justify_right_from: int = 1) -> None:
    if QUIET:
        return
    t = Table(title=title, title_justify="left", header_style="bold")
    for i, c in enumerate(columns):
        t.add_column(c, justify="right" if i >= justify_right_from else "left")
    for r in rows:
        t.add_row(*[escape(str(x)) for x in r])
    console.print(t)


def info(msg: str) -> None:
    if not QUIET:
        console.print(msg)


def warn(msg: str) -> None:
    console.print(f"[bold yellow]warning[/bold yellow] {msg}")
