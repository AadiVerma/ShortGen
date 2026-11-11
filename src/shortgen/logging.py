"""Logging setup. Library modules only ever call ``logging.getLogger(__name__)``."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

ROOT_LOGGER = "shortgen"
JSON_FORMAT = "json"
_STANDARD_ATTRIBUTES = frozenset(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, including any ``extra=`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key not in _STANDARD_ATTRIBUTES
            }
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str, log_format: str, console: Console | None = None) -> None:
    """Attach a single handler to the ``shortgen`` logger (idempotent).

    Logs go to stderr so that stdout stays clean for command output such as
    ``shortgen status --json`` and for the MCP stdio transport. Pass the
    ``console`` that progress bars use so the two never overwrite each other.
    """
    handler: logging.Handler
    if log_format == JSON_FORMAT:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
    else:
        handler = RichHandler(
            console=console or Console(stderr=True),
            show_path=False,
            rich_tracebacks=False,
            markup=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
    logger = logging.getLogger(ROOT_LOGGER)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
