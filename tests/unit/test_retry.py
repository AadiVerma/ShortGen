from __future__ import annotations

import pytest

from shortgen.errors import ProviderError, TransientError
from shortgen.pipeline.retry import RetryPolicy, call_with_retry

POLICY = RetryPolicy(max_attempts=3, backoff_base_s=2.0, backoff_max_s=5.0)


class Flaky:
    """Fails ``failures`` times with ``error``, then returns ``"ok"``."""

    def __init__(self, failures: int, error: Exception) -> None:
        self.calls = 0
        self._failures = failures
        self._error = error

    def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self._failures:
            raise self._error
        return "ok"


def test_backoff_doubles_and_is_capped() -> None:
    assert [POLICY.delay(n) for n in (1, 2, 3)] == [2.0, 4.0, 5.0]


def test_transient_failures_are_retried_with_backoff() -> None:
    action = Flaky(2, TransientError("busy"))
    waits: list[float] = []
    assert call_with_retry(action, POLICY, sleep=waits.append) == "ok"
    assert action.calls == 3
    assert waits == [2.0, 4.0]


def test_attempts_are_bounded() -> None:
    action = Flaky(5, TransientError("busy"))
    with pytest.raises(TransientError):
        call_with_retry(action, POLICY, sleep=lambda _: None)
    assert action.calls == POLICY.max_attempts


def test_permanent_failures_are_not_retried() -> None:
    action = Flaky(1, ProviderError("bad prompt"))
    with pytest.raises(ProviderError):
        call_with_retry(action, POLICY, sleep=lambda _: None)
    assert action.calls == 1
