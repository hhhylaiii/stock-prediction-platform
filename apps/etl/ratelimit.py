"""Request pacing with exponential backoff; limits come from ETL_RATE_LIMITS (SPEC §8.10)."""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


class RetryableError(Exception):
    """A transient failure such as a timeout, a dropped connection or an HTTP 5xx."""


class ThrottledError(RetryableError):
    """The source says we are over its rate limit; wait for the cooldown before retrying."""


@dataclass(frozen=True)
class RateLimit:
    min_interval: float
    max_retries: int
    backoff_base: float
    backoff_max: float
    cooldown: float

    @classmethod
    def from_settings(cls, values: Mapping[str, float]) -> "RateLimit":
        return cls(
            min_interval=float(values["min_interval"]),
            max_retries=int(values["max_retries"]),
            backoff_base=float(values["backoff_base"]),
            backoff_max=float(values["backoff_max"]),
            cooldown=float(values["cooldown"]),
        )

    def backoff(self, attempt: int) -> float:
        return float(min(self.backoff_max, self.backoff_base * 2.0**attempt))


class RateLimiter:
    """Spaces calls at least min_interval apart and retries retryable failures."""

    def __init__(
        self,
        limit: RateLimit,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.limit = limit
        self._clock = clock if clock is not None else time.monotonic
        self._sleep = sleep if sleep is not None else time.sleep
        self._last_call: float | None = None

    def wait(self) -> None:
        if self._last_call is not None:
            remaining = self.limit.min_interval - (self._clock() - self._last_call)
            if remaining > 0:
                self._sleep(remaining)
        self._last_call = self._clock()

    def call(self, fn: Callable[[], T]) -> T:
        attempt = 0
        while True:
            self.wait()
            try:
                return fn()
            except RetryableError as exc:
                if attempt >= self.limit.max_retries:
                    raise
                throttled = isinstance(exc, ThrottledError)
                self._sleep(self.limit.cooldown if throttled else self.limit.backoff(attempt))
                attempt += 1
