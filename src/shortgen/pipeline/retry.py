"""Bounded retry with exponential backoff, for transient failures only."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from shortgen.errors import TransientError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """How often to try and how long to wait in between."""

    max_attempts: int
    backoff_base_s: float
    backoff_max_s: float

    def delay(self, failed_attempts: int) -> float:
        """Seconds to wait after ``failed_attempts`` consecutive failures."""
        return float(min(self.backoff_max_s, self.backoff_base_s * 2 ** (failed_attempts - 1)))


def call_with_retry(
    action: Callable[[], T],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, TransientError, float], None] | None = None,
) -> T:
    """Run ``action``, retrying only :class:`TransientError`.

    Any other exception propagates immediately: retrying a deterministic
    failure (a bad prompt, out of memory) would only waste hours.
    """
    attempt = 1
    while True:
        try:
            return action()
        except TransientError as error:
            if attempt >= policy.max_attempts:
                raise
            wait = policy.delay(attempt)
            if on_retry is not None:
                on_retry(attempt, error, wait)
            sleep(wait)
            attempt += 1
