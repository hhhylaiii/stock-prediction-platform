import logging

import pytest

from apps.etl.ratelimit import RateLimit, RateLimiter, RetryableError, ThrottledError

LIMIT = RateLimit(
    min_interval=3.0, max_retries=3, backoff_base=2.0, backoff_max=5.0, cooldown=600.0
)


class FakeClock:
    """Monotonic clock that only moves when the code under test sleeps or a test advances it."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _limiter(clock: FakeClock, limit: RateLimit = LIMIT) -> RateLimiter:
    return RateLimiter(limit, clock=clock, sleep=clock.sleep)


def test_first_call_does_not_wait_and_next_waits_the_interval() -> None:
    clock = FakeClock()
    limiter = _limiter(clock)
    limiter.call(lambda: 1)
    clock.now += 1.0
    limiter.call(lambda: 2)
    assert clock.sleeps == [pytest.approx(2.0)]


def test_no_wait_when_interval_already_elapsed() -> None:
    clock = FakeClock()
    limiter = _limiter(clock)
    limiter.call(lambda: 1)
    clock.now += 10.0
    limiter.call(lambda: 2)
    assert clock.sleeps == []


def test_retryable_errors_back_off_exponentially_with_a_cap() -> None:
    clock = FakeClock()
    attempts: list[int] = []

    def flaky() -> str:
        attempts.append(1)
        if len(attempts) < 4:
            raise RetryableError("timeout")
        return "ok"

    assert _limiter(clock).call(flaky) == "ok"
    assert clock.sleeps == [2.0, pytest.approx(1.0), 4.0, 5.0]


def test_throttled_waits_for_cooldown() -> None:
    clock = FakeClock()
    calls: list[int] = []

    def throttled_once() -> str:
        calls.append(1)
        if len(calls) == 1:
            raise ThrottledError("429")
        return "ok"

    assert _limiter(clock).call(throttled_once) == "ok"
    assert clock.sleeps == [600.0]


def test_gives_up_after_max_retries() -> None:
    clock = FakeClock()
    calls: list[int] = []

    def always_failing() -> None:
        calls.append(1)
        raise RetryableError("down")

    with pytest.raises(RetryableError):
        _limiter(clock).call(always_failing)
    assert len(calls) == LIMIT.max_retries + 1


def test_non_retryable_errors_propagate_immediately() -> None:
    clock = FakeClock()
    calls: list[int] = []

    def broken() -> None:
        calls.append(1)
        raise KeyError("bad")

    with pytest.raises(KeyError):
        _limiter(clock).call(broken)
    assert calls == [1]
    assert clock.sleeps == []


def test_from_settings_reads_every_field() -> None:
    limit = RateLimit.from_settings(
        {"min_interval": 3, "max_retries": 2, "backoff_base": 1, "backoff_max": 8, "cooldown": 60}
    )
    assert limit == RateLimit(3.0, 2, 1.0, 8.0, 60.0)
    assert [limit.backoff(n) for n in range(5)] == [1.0, 2.0, 4.0, 8.0, 8.0]


def test_interval_counts_from_when_the_previous_request_finished() -> None:
    clock = FakeClock()
    limiter = _limiter(clock)

    def slow() -> int:
        clock.now += 5.0
        return 1

    limiter.call(slow)
    limiter.call(lambda: 2)
    assert clock.sleeps == [pytest.approx(3.0)]


def test_failed_request_also_starts_the_interval() -> None:
    clock = FakeClock()
    limiter = _limiter(clock)

    def broken() -> int:
        clock.now += 5.0
        raise ValueError("bad payload")

    with pytest.raises(ValueError, match="bad payload"):
        limiter.call(broken)
    limiter.call(lambda: 2)
    assert clock.sleeps == [pytest.approx(3.0)]


def test_retries_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    clock = FakeClock()
    limiter = _limiter(clock)
    outcomes: list[Exception | int] = [ThrottledError("busy"), 7]

    def flaky() -> int:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    with caplog.at_level(logging.WARNING, logger="apps.etl.ratelimit"):
        assert limiter.call(flaky) == 7
    assert "busy" in caplog.text
    assert "retrying in" in caplog.text
