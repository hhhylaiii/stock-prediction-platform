"""Yahoo chart API for US and FX daily closes into index_daily (SPEC §3, §8.9)."""

import logging
import math
from collections.abc import Iterator, Mapping
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any, NamedTuple
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import requests
from django.conf import settings
from django.utils import timezone

from apps.etl.adapters.base import BaseAdapter
from apps.etl.adapters.exchange_tables import upsert_index_rows
from apps.etl.client import SourceClient
from apps.etl.errors import SchemaChangedError
from apps.etl.models import RawYahoo
from apps.etl.ratelimit import RateLimiter
from apps.market.calendar import OutsideCalendarError, TradingDays
from apps.market.models import IndexId

logger = logging.getLogger(__name__)

YAHOO_MIN_INTERVAL = 1.0
CHART_PATH = "/v8/finance/chart/"
INDEX_COLUMNS = ["index_id", "date", "close", "volume"]
BAR_COLUMNS = ["index_id", "date", "close", "volume", "price_hint"]
REQUEST_MARGIN = timedelta(days=1)
RECHECK_SESSIONS = 5


class Series(NamedTuple):
    symbol: str
    index_id: str
    keeps_volume: bool


SERIES = {
    "sox": Series("^SOX", IndexId.SOX.value, keeps_volume=False),
    "tsm_adr": Series("TSM", IndexId.TSM_ADR.value, keeps_volume=True),
    "usdtwd": Series("TWD=X", IndexId.USDTWD.value, keeps_volume=False),
}


class YahooClient(SourceClient):
    source = "yahoo"
    label = "Yahoo"
    raw_model = RawYahoo
    min_interval = YAHOO_MIN_INTERVAL

    @classmethod
    def build(cls, limiter: RateLimiter, session: requests.Session | None = None) -> "YahooClient":
        return cls(limiter, str(settings.YAHOO_BASE_URL), float(settings.YAHOO_TIMEOUT), session)

    def chart(self, symbol: str, params: Mapping[str, str]) -> dict[str, Any]:
        return self.get_json(f"{CHART_PATH}{quote(symbol, safe='')}", params)


class YahooAdapter(BaseAdapter):
    source = "yahoo"
    raw_model = RawYahoo
    datasets = frozenset(SERIES)

    def __init__(
        self,
        calendar: TradingDays,
        limiter: RateLimiter,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(calendar, limiter)
        self.client = YahooClient.build(limiter, session)

    def units(self, dataset: str, start: date, end: date) -> Iterator[tuple[date, date]]:
        """One request for the whole range; the source serves any span of daily bars at once."""
        days = self.trading_days(start, end)
        yield days[0], days[-1]

    def fetch(
        self, dataset: str, start: date, end: date, stock_id: str | None = None
    ) -> pd.DataFrame:
        self.require_dataset(dataset)
        series = SERIES[dataset]
        window = self._window(start, end)
        params = {
            "interval": "1d",
            "period1": str(_midnight(window[0] - REQUEST_MARGIN)),
            "period2": str(_midnight(window[1] + 2 * REQUEST_MARGIN)),
        }
        payload = self.client.chart(series.symbol, params)
        self.save_raw(dataset, {"symbol": series.symbol, **params}, payload)
        bars = final_bars(payload, series, window, timezone.now().timestamp())
        return pd.DataFrame(bars, columns=BAR_COLUMNS, dtype=object)

    def normalize(self, dataset: str, raw_df: pd.DataFrame) -> pd.DataFrame:
        self.require_dataset(dataset)
        keeps_volume = SERIES[dataset].keeps_volume
        rows = [
            {
                "index_id": row["index_id"],
                "date": row["date"],
                "close": _close(row["close"], row["price_hint"]),
                "volume": _volume(row["volume"]) if keeps_volume else None,
            }
            for row in raw_df.to_dict("records")
        ]
        return pd.DataFrame(rows, columns=INDEX_COLUMNS, dtype=object)

    def upsert(self, dataset: str, frame: pd.DataFrame) -> int:
        self.require_dataset(dataset)
        return upsert_index_rows(frame)

    def _window(self, start: date, end: date) -> tuple[date, date]:
        """Exchange dates back to a few Taiwan sessions, so bars left open by earlier runs land."""
        try:
            return self.calendar.shift(start, -RECHECK_SESSIONS) + timedelta(days=1), end
        except OutsideCalendarError:
            return min(self.calendar.start, start), end


def _midnight(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())


def final_bars(
    payload: dict[str, Any], series: Series, window: tuple[date, date], now: float
) -> list[dict[str, Any]]:
    """Bars dated inside the window in the exchange's own zone, leaving out an open session."""
    result = _result(payload, series.symbol)
    meta = result["meta"]
    zone = _zone(meta)
    session_start, session_end = _session(meta)
    hint = meta.get("priceHint")
    if not isinstance(hint, int) or isinstance(hint, bool):
        raise SchemaChangedError(f"Yahoo {series.symbol} price hint is {hint!r}")
    stamps = result.get("timestamp", [])
    quote_values = (result.get("indicators", {}).get("quote") or [{}])[0]
    closes = quote_values.get("close", [])
    volumes = quote_values.get("volume", [])
    if not (isinstance(stamps, list) and len(stamps) == len(closes) == len(volumes)):
        raise SchemaChangedError(f"Yahoo {series.symbol} bars and closes do not line up")
    bars = []
    for stamp, close, volume in zip(stamps, closes, volumes, strict=True):
        day = datetime.fromtimestamp(stamp, zone).date()
        if not window[0] <= day <= window[1]:
            continue
        if stamp >= session_start and now < session_end:
            logger.info("Yahoo %s %s: session still open, left for a later run", series.symbol, day)
            continue
        if close is None:
            logger.warning("Yahoo %s %s: bar has no close, skipped", series.symbol, day)
            continue
        bars.append(
            {
                "index_id": series.index_id,
                "date": day,
                "close": close,
                "volume": volume,
                "price_hint": hint,
            }
        )
    days = [bar["date"] for bar in bars]
    if len(days) != len(set(days)):
        raise SchemaChangedError(f"Yahoo {series.symbol} repeats dates in {days}")
    return bars


def _result(payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    chart = payload.get("chart")
    if not isinstance(chart, dict) or chart.get("error") is not None:
        raise SchemaChangedError(f"Yahoo {symbol} answered with error {payload!r:.200}")
    results = chart.get("result")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise SchemaChangedError(f"Yahoo {symbol} sent no single result")
    result: dict[str, Any] = results[0]
    meta = result.get("meta")
    answered = meta.get("symbol") if isinstance(meta, dict) else None
    if answered != symbol:
        raise SchemaChangedError(f"Yahoo answered {symbol} with symbol {answered!r}")
    return result


def _zone(meta: Mapping[str, Any]) -> ZoneInfo:
    name = meta.get("exchangeTimezoneName")
    try:
        return ZoneInfo(str(name))
    except (ZoneInfoNotFoundError, ValueError):
        raise SchemaChangedError(f"Yahoo exchange time zone {name!r} is unknown") from None


def _session(meta: Mapping[str, Any]) -> tuple[float, float]:
    regular = (meta.get("currentTradingPeriod") or {}).get("regular") or {}
    start, end = regular.get("start"), regular.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        raise SchemaChangedError(f"Yahoo current trading period is {regular!r}")
    return start, end


def _close(value: Any, hint: int) -> Decimal:
    number = isinstance(value, int | float) and not isinstance(value, bool)
    if not number or not math.isfinite(value) or value <= 0:
        raise SchemaChangedError(f"unexpected Yahoo close {value!r}")
    return Decimal(str(round(value, hint)))


def _volume(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SchemaChangedError(f"unexpected Yahoo volume {value!r}")
    return value
