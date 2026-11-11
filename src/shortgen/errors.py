"""The exception hierarchy.

Everything shortgen raises on purpose derives from :class:`ShortgenError`, so
callers (the CLI, the MCP server) can turn expected failures into clean
messages while letting real bugs surface with a traceback.
"""

from __future__ import annotations

from collections.abc import Sequence

# How much of a failed command's stderr is worth showing in the message.
_STDERR_TAIL_LINES = 12


class ShortgenError(Exception):
    """Base class for every expected shortgen failure."""


class ConfigError(ShortgenError):
    """The configuration or model registry is missing, malformed or inconsistent."""


class ManifestError(ShortgenError):
    """The project manifest is missing, unreadable or from an unsupported schema."""


class PathValidationError(ShortgenError):
    """A user-supplied path escapes its allowed root or is not a usable file."""


class PlanningError(ShortgenError):
    """A script could not be turned into a valid scene plan."""


class ProviderError(ShortgenError):
    """A model adapter failed."""


class ProviderUnavailableError(ProviderError):
    """An adapter cannot run here: a dependency, binary or service is missing."""


class UnsupportedFeatureError(ProviderError):
    """The request asks for something this adapter does not implement."""


class TransientError(ProviderError):
    """A failure that is worth retrying (timeouts, a busy local service)."""


class CommandError(ShortgenError):
    """An external command exited with a non-zero status."""

    def __init__(self, args: Sequence[str], returncode: int, stderr: str) -> None:
        self.command = list(args)
        self.returncode = returncode
        self.stderr = stderr
        tail = "\n".join(stderr.strip().splitlines()[-_STDERR_TAIL_LINES:])
        super().__init__(f"{self.command[0]} exited with status {returncode}:\n{tail}")


class RenderError(ShortgenError):
    """ffmpeg produced something unusable, or could not be asked to render."""


class StageError(ShortgenError):
    """A pipeline stage cannot run, usually because an upstream stage has not."""


class StageFailedError(StageError):
    """One or more units of a stage failed; the manifest records each error."""

    def __init__(self, stage: str, failures: Sequence[str]) -> None:
        self.stage = stage
        self.failures = list(failures)
        super().__init__(f"stage '{stage}' failed for: {', '.join(self.failures)}")
