"""Terminal presentation for the CLI: progress bars and result tables.

This is the only module besides ``cli.py`` that writes to the terminal.
"""

from __future__ import annotations

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from shortgen.app import ProjectStatus, RunSummary
from shortgen.providers.health import HealthCheck, HealthStatus

_STATUS_STYLE = {
    HealthStatus.OK: "[green]ok[/green]",
    HealthStatus.WARN: "[yellow]warn[/yellow]",
    HealthStatus.FAIL: "[red]FAIL[/red]",
}


class RichReporter:
    """Shows one progress bar per stage that has work to do."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._progress: Progress | None = None
        self._task: TaskID | None = None

    def stage_started(self, stage: str, total: int, pending: int) -> None:
        if pending == 0:
            self._console.print(f"[dim]{stage:<10} up to date ({total})[/dim]")
            return
        self._progress = Progress(
            TextColumn("[bold]{task.description:<10}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self._console,
        )
        self._progress.start()
        self._task = self._progress.add_task(stage, total=pending)

    def unit_finished(self, stage: str, label: str, ok: bool) -> None:
        if self._progress is not None and self._task is not None:
            self._progress.advance(self._task)
        if not ok:
            self._console.print(f"[red]{stage} failed for {label}[/red] (details in the log above)")

    def stage_finished(self, stage: str) -> None:
        if self._progress is not None:
            self._progress.stop()
        self._progress = None
        self._task = None


def print_summary(console: Console, summary: RunSummary) -> None:
    """One line per stage, then where the video is."""
    for stage in summary.stages:
        if stage.completed or stage.failed:
            console.print(
                f"{stage.stage:<10} {stage.completed} done, {stage.skipped} skipped"
                + (f", [red]{len(stage.failed)} failed[/red]" if stage.failed else "")
            )
    if summary.final_video:
        console.print(f"[bold green]Video:[/bold green] {summary.final_video}", soft_wrap=True)


def print_status(console: Console, status: ProjectStatus) -> None:
    """A table of every stage's freshness."""
    table = Table(title=f"{status.name}: {status.scenes} scene(s)", title_justify="left")
    for column in ("stage", "done", "to do", "failed", "note"):
        table.add_column(column)
    for stage in status.stages:
        table.add_row(
            stage.stage,
            f"{stage.fresh}/{stage.total}",
            str(stage.pending),
            f"[red]{stage.failed}[/red]" if stage.failed else "0",
            _status_note(stage.blocked_reason, stage.note),
        )
    console.print(table)
    console.print(f"Video: {status.final_video or 'not rendered yet'}", soft_wrap=True)


def _status_note(blocked_reason: str | None, note: str | None) -> str:
    if blocked_reason:
        return f"[yellow]blocked: {blocked_reason}[/yellow]"
    return f"[yellow]{note}[/yellow]" if note else ""


def print_doctor(console: Console, checks: list[HealthCheck]) -> None:
    """A table of health checks with fix hints underneath each problem."""
    table = Table(show_header=True)
    for column in ("check", "status", "detail"):
        table.add_column(column, overflow="fold")
    for check in checks:
        detail = check.detail
        if check.hint and check.status is not HealthStatus.OK:
            detail += f"\n[dim]{check.hint}[/dim]"
        table.add_row(check.name, _STATUS_STYLE[check.status], detail)
    console.print(table)
