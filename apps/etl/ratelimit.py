"""Request pacing with exponential backoff; limits come from ETL_RATE_LIMITS (SPEC §8.10)."""

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeVar

logger = logging.getLogger(__name__)

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


def _monotonic() -> float:
    return time.monotonic()


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


class RateLimiter:
    """Keeps min_interval between one request ending and the next starting; retries failures."""

    def __init__(
        self,
        limit: RateLimit,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.limit = limit
        self._clock = clock if clock is not None else _monotonic
        self._sleep = sleep if sleep is not None else _sleep
        self._last_finished: float | None = None

    def wait(self) -> None:
        """Sleep until min_interval has passed since the previous request finished."""
        if self._last_finished is not None:
            remaining = self.limit.min_interval - (self._clock() - self._last_finished)
            if remaining > 0:
                self._sleep(remaining)

    def call(self, fn: Callable[[], T]) -> T:
        attempt = 0
        while True:
            self.wait()
            try:
                result = fn()
            except RetryableError as exc:
                self._last_finished = self._clock()
                if attempt >= self.limit.max_retries:
                    raise
                throttled = isinstance(exc, ThrottledError)
                delay = self.limit.cooldown if throttled else self.limit.backoff(attempt)
                logger.warning(
                    "%s; retrying in %.0f s (retry %d of %d)",
                    exc,
                    delay,
                    attempt + 1,
                    self.limit.max_retries,
                )
                self._sleep(delay)
                attempt += 1
                continue
            except BaseException:
                self._last_finished = self._clock()
                raise
            self._last_finished = self._clock()
            return result
