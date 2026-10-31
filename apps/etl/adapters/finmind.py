"""FinMind REST client for the one-off history backfill (SPEC §3, §5)."""

import time
from collections.abc import Callable
from datetime import datetime, timedelta
from datetime import time as clock_time
from typing import Any

import requests
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils import timezone

from apps.etl.adapters.base import SchemaChangedError, SourceRequestError
from apps.etl.client import SourceClient
from apps.etl.models import RawFinmind
from apps.etl.ratelimit import RateLimiter, ThrottledError

DATA_PATH = "/api/v4/data"
HOURLY_QUOTA_WITH_TOKEN = 600
HOURLY_QUOTA_ANONYMOUS = 300
QUOTA_EXHAUSTED = 402
MAINTENANCE_WEEKDAY = 6
MAINTENANCE_END = clock_time(7, 0)


def maintenance_wait(moment: datetime) -> float:
    """Seconds until FinMind's Sunday 00:00–07:00 Taipei maintenance ends, else 0."""
    local = timezone.localtime(moment)
    if local.weekday() != MAINTENANCE_WEEKDAY or local.time() >= MAINTENANCE_END:
        return 0.0
    reopen = local.replace(hour=MAINTENANCE_END.hour, minute=0, second=0, microsecond=0)
    return (reopen - local) / timedelta(seconds=1)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


class FinmindClient(SourceClient):
    source = "finmind"
    label = "FinMind"
    raw_model = RawFinmind
    throttle_statuses = frozenset({QUOTA_EXHAUSTED, 429})

    def __init__(
        self,
        limiter: RateLimiter,
        base_url: str,
        timeout: float,
        token: str,
        session: requests.Session | None = None,
        now: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        quota = HOURLY_QUOTA_WITH_TOKEN if token else HOURLY_QUOTA_ANONYMOUS
        if limiter.limit.min_interval * quota < 3600:
            raise ImproperlyConfigured(
                f"FinMind allows {quota} requests per hour; "
                f"min_interval must be at least {3600 / quota:g} seconds"
            )
        super().__init__(limiter, base_url, timeout, session)
        self.token = token
        self._now = now if now is not None else timezone.now
        self._sleep = sleep if sleep is not None else _sleep

    @classmethod
    def build(
        cls, limiter: RateLimiter, session: requests.Session | None = None
    ) -> "FinmindClient":
        return cls(
            limiter,
            str(settings.FINMIND_BASE_URL),
            float(settings.FINMIND_TIMEOUT),
            str(settings.FINMIND_API_TOKEN),
            session,
        )

    def data(self, dataset: str, finmind_dataset: str, **params: str) -> list[dict[str, Any]]:
        """Rows of one FinMind dataset; the whole response is kept in raw_finmind."""
        query = {"dataset": finmind_dataset, **params}
        payload = self.get_json(DATA_PATH, query)
        self.save_raw(dataset, query, payload)
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise SchemaChangedError(f"FinMind {finmind_dataset} data is not a list")
        return rows

    def _send(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Every attempt, retries included, first waits out the Sunday maintenance window."""
        wait = maintenance_wait(self._now())
        if wait > 0:
            self._sleep(wait)
        return super()._send(method, path, **kwargs)

    def headers(self) -> dict[str, str]:
        headers = super().headers()
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def check_payload(self, payload: dict[str, Any]) -> None:
        status = payload.get("status")
        if status == QUOTA_EXHAUSTED:
            raise ThrottledError(f"FinMind quota exhausted: {payload.get('msg')}")
        if status != 200:
            raise SourceRequestError(f"FinMind answered {status}: {payload.get('msg')}")
