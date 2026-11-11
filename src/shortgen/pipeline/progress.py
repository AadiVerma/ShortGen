"""Progress reporting, decoupled from any terminal.

Library code reports through this protocol; the CLI plugs in progress bars
and everything else gets the logging-only default.
"""

from __future__ import annotations

import logging
from typing import Protocol

log = logging.getLogger(__name__)


class Reporter(Protocol):
    """Receives pipeline progress events."""

    def stage_started(self, stage: str, total: int, pending: int) -> None:
        """A stage begins: ``total`` units exist, ``pending`` of them need work."""
        ...

    def unit_finished(self, stage: str, label: str, ok: bool) -> None:
        """One unit of work ended, successfully or not."""
        ...

    def stage_finished(self, stage: str) -> None:
        """The stage is over (also called when it was interrupted)."""
        ...


class LogReporter:
    """Default reporter: structured log lines, no terminal control."""

    def stage_started(self, stage: str, total: int, pending: int) -> None:
        log.info(
            "stage %s: %d of %d up to date, %d to do",
            stage,
            total - pending,
            total,
            pending,
            extra={"stage": stage, "total": total, "pending": pending},
        )

    def unit_finished(self, stage: str, label: str, ok: bool) -> None:
        log.log(
            logging.INFO if ok else logging.ERROR,
            "%s %s %s",
            stage,
            label,
            "done" if ok else "FAILED",
            extra={"stage": stage, "unit": label, "ok": ok},
        )

    def stage_finished(self, stage: str) -> None:
        log.debug("stage %s finished", stage, extra={"stage": stage})
