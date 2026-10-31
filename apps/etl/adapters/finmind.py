"""FinMind REST client for the one-off history backfill (SPEC §3, §5)."""

import logging
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from datetime import time as clock_time
from typing import Any

import pandas as pd
import requests
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils import timezone

from apps.etl.adapters.base import BaseAdapter
from apps.etl.adapters.finmind_datasets import (
    dividend_events,
    price_bars,
    share_counts,
    valuations,
)
from apps.etl.client import SourceClient
from apps.etl.errors import SchemaChangedError, SourceRequestError
from apps.etl.models import RawFinmind
from apps.etl.ratelimit import RateLimiter, ThrottledError
from apps.etl.upsert import upsert
from apps.market.adjust import refresh_adj_factor
from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice, DailyValuation, DividendEvent, SharesOutstanding

logger = logging.getLogger(__name__)
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
            logger.info("FinMind maintenance until 07:00 Taipei; waiting %.0f s", wait)
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


FINMIND_DATASETS = {
    "daily_price": ("TaiwanStockPrice",),
    "dividend_event": ("TaiwanStockDividend", "TaiwanStockDividendResult"),
    "daily_valuation": ("TaiwanStockPER",),
    "shares_outstanding": ("TaiwanStockShareholding",),
}
RAW_NAMES = {
    "TaiwanStockDividend": "dividend_policy",
    "TaiwanStockDividendResult": "dividend_result",
}
PRICE_FIELDS = ("open", "high", "low", "close", "volume", "turnover")
EVENT_FIELDS = ("cash_dividend", "stock_dividend", "adj_factor", "announce_date")


class FinmindAdapter(BaseAdapter):
    """Per-stock history backfill; the raw frame keeps which FinMind dataset each row came from."""

    source = "finmind"
    raw_model = RawFinmind
    per_stock = True
    datasets = frozenset(FINMIND_DATASETS)

    def __init__(
        self,
        calendar: TradingDays,
        limiter: RateLimiter,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(calendar, limiter)
        self.client = FinmindClient.build(limiter, session)

    def fetch(
        self, dataset: str, start: date, end: date, stock_id: str | None = None
    ) -> pd.DataFrame:
        self.require_dataset(dataset)
        if stock_id is None:
            raise ValueError("FinMind datasets are fetched one stock at a time")
        params = {"data_id": stock_id, "start_date": start.isoformat(), "end_date": end.isoformat()}
        rows = []
        for finmind_dataset in FINMIND_DATASETS[dataset]:
            raw_name = RAW_NAMES.get(finmind_dataset, dataset)
            for row in self.client.data(raw_name, finmind_dataset, **params):
                rows.append({**row, FROM_COLUMN: finmind_dataset})
        frame = pd.DataFrame(rows)
        frame.attrs["stock_id"] = stock_id
        return frame

    def normalize(self, dataset: str, raw_df: pd.DataFrame) -> pd.DataFrame:
        stock_id = str(raw_df.attrs["stock_id"])
        parts = {name: _rows_from(raw_df, name) for name in FINMIND_DATASETS[dataset]}
        if dataset == "dividend_event":
            return dividend_events(
                parts["TaiwanStockDividend"], parts["TaiwanStockDividendResult"], stock_id
            )
        normalizer = {
            "daily_price": price_bars,
            "daily_valuation": valuations,
            "shares_outstanding": share_counts,
        }[dataset]
        return normalizer(parts[FINMIND_DATASETS[dataset][0]], stock_id)

    def upsert(self, dataset: str, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        if dataset == "daily_price":
            rows = upsert(DailyPrice, frame, ("stock_id", "date"), PRICE_FIELDS)
        elif dataset == "dividend_event":
            rows = upsert(DividendEvent, frame, ("stock_id", "ex_date"), EVENT_FIELDS)
        elif dataset == "daily_valuation":
            return upsert(
                DailyValuation, frame, ("stock_id", "date"), ("pe", "pb", "dividend_yield")
            )
        else:
            return upsert(
                SharesOutstanding,
                _without_repeat_of_stored(frame),
                ("stock_id", "effective_date"),
                ("shares", "reason"),
            )
        for stock_id in frame["stock_id"].unique():
            refresh_adj_factor(str(stock_id))
        return rows


FROM_COLUMN = "_finmind_dataset"


def _rows_from(raw_df: pd.DataFrame, finmind_dataset: str) -> list[dict[str, Any]]:
    if raw_df.empty:
        return []
    picked = raw_df.loc[raw_df[FROM_COLUMN] == finmind_dataset].drop(columns=FROM_COLUMN)
    return [
        {str(key): value for key, value in row.items() if not _is_missing(value)}
        for row in picked.to_dict("records")
    ]


def _is_missing(value: Any) -> bool:
    return isinstance(value, float) and value != value


def _without_repeat_of_stored(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop a leading row that only repeats the count already stored before it."""
    first = frame.iloc[0]
    stored = (
        SharesOutstanding.objects.filter(
            stock_id=first["stock_id"], effective_date__lt=first["effective_date"]
        )
        .order_by("-effective_date")
        .values_list("shares", flat=True)
        .first()
    )
    return frame.iloc[1:] if stored == first["shares"] else frame
