"""Running external commands. Argument lists only: nothing ever goes through a shell."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from shortgen.errors import CommandError, RenderError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandResult:
    """Captured output of a finished command."""

    stdout: str
    stderr: str


class CommandRunner(Protocol):
    """Anything that can execute a command; tests substitute a recorder."""

    def run(self, args: Sequence[str], *, cwd: Path | None = None) -> CommandResult:
        """Run ``args`` to completion or raise :class:`CommandError`."""
        ...


class SubprocessRunner:
    """Runs commands as child processes and never leaves one behind."""

    def run(self, args: Sequence[str], *, cwd: Path | None = None) -> CommandResult:
        """Run ``args``; on Ctrl+C the child is stopped before the interrupt propagates."""
        command = [str(arg) for arg in args]
        log.debug("running command", extra={"command": command})
        try:
            process = subprocess.Popen(  # noqa: S603 - argument list, shell is never used
                command,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RenderError(f"executable not found: {command[0]}") from exc
        try:
            stdout, stderr = process.communicate()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        if process.returncode != 0:
            raise CommandError(command, process.returncode, stderr)
        return CommandResult(stdout=stdout, stderr=stderr)
