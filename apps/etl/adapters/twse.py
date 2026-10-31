"""TWSE listed-market adapter using whole-market daily endpoints (SPEC §3, §8.6)."""

from collections.abc import Callable
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

import pandas as pd
import requests
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from apps.etl.adapters.base import (
    BaseAdapter,
    SchemaChangedError,
    SourceNotReadyError,
    SourceRequestError,
)
from apps.etl.models import RawTwse
from apps.etl.ratelimit import RateLimiter, RetryableError, ThrottledError
from apps.etl.upsert import upsert
from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice

TWSE_MIN_INTERVAL = 3.0
DAILY_QUOTES_PATH = "/rwd/zh/afterTrading/MI_INDEX"
DAILY_QUOTES_TITLE = "每日收盤行情"
DAILY_QUOTES_UNIT_HINT = "單位：元、股"
DAILY_QUOTES_COLUMNS = {
    "證券代號": "stock_id",
    "開盤價": "open",
    "最高價": "high",
    "最低價": "low",
    "收盤價": "close",
    "成交股數": "volume",
    "成交金額": "turnover",
}
PRICE_COLUMNS = ("open", "high", "low", "close")
COUNT_COLUMNS = ("volume", "turnover")
NO_TRADE_MARKERS = frozenset({"--", "---", ""})
NO_DATA_STAT = "沒有符合條件的資料"
FOREIGN_CURRENCY_CODE_SUFFIXES = frozenset("KMSC")


def is_foreign_currency_etf(code: str) -> bool:
    """TWSE: an ETF code whose sixth character is K, M, S or C trades in a foreign currency."""
    return len(code) == 6 and code.startswith("00") and code[5] in FOREIGN_CURRENCY_CODE_SUFFIXES


def parse_price(text: str) -> Decimal | None:
    cleaned = str(text).replace(",", "").strip()
    if cleaned in NO_TRADE_MARKERS:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        raise SchemaChangedError(f"unexpected price value {text!r}") from None


def parse_count(text: str) -> int:
    cleaned = str(text).replace(",", "").strip()
    if not cleaned.isdigit():
        raise SchemaChangedError(f"unexpected count value {text!r}")
    return int(cleaned)


def find_table(payload: dict[str, Any], title_part: str) -> dict[str, Any]:
    for table in payload.get("tables", []):
        if title_part in (table.get("title") or ""):
            return dict(table)
    raise SchemaChangedError(f"no table titled *{title_part}* in TWSE response")


def quotes_frame(payload: dict[str, Any], day: date) -> pd.DataFrame:
    """Rows of the daily quotes table in a stored MI_INDEX payload, for fetch or a rerun."""
    table = find_table(payload, DAILY_QUOTES_TITLE)
    if table.get("hints") != DAILY_QUOTES_UNIT_HINT:
        raise SchemaChangedError(f"TWSE daily quotes units changed: {table.get('hints')!r}")
    try:
        frame = pd.DataFrame(table["data"], columns=table["fields"])
    except (KeyError, ValueError) as exc:
        raise SchemaChangedError(f"TWSE daily quotes table is malformed: {exc}") from None
    return frame.assign(date=day)


class TwseAdapter(BaseAdapter):
    source = "twse"
    raw_model = RawTwse
    datasets = frozenset({"daily_price"})

    def __init__(
        self,
        calendar: TradingDays,
        limiter: RateLimiter,
        session: requests.Session | None = None,
    ) -> None:
        if limiter.limit.min_interval < TWSE_MIN_INTERVAL:
            raise ImproperlyConfigured("TWSE requests must be at least 3 seconds apart (SPEC §3)")
        super().__init__(calendar, limiter)
        self.session = session if session is not None else requests.Session()
        self.base_url = str(settings.TWSE_BASE_URL).rstrip("/")
        self.timeout = float(settings.TWSE_TIMEOUT)

    def fetch(
        self, dataset: str, start: date, end: date, stock_id: str | None = None
    ) -> pd.DataFrame:
        self.require_dataset(dataset)
        frames = [self._fetch_daily_quotes(day) for day in self.trading_days(start, end)]
        return pd.concat(frames, ignore_index=True)

    def normalize(self, dataset: str, raw_df: pd.DataFrame) -> pd.DataFrame:
        self.require_dataset(dataset)
        missing = set(DAILY_QUOTES_COLUMNS) - set(raw_df.columns)
        if missing:
            raise SchemaChangedError(f"TWSE daily quotes lost columns {sorted(missing)}")
        codes = raw_df["證券代號"].astype(str).str.strip()
        local = raw_df.loc[~codes.map(is_foreign_currency_etf)]
        frame = local[list(DAILY_QUOTES_COLUMNS)].rename(columns=DAILY_QUOTES_COLUMNS)
        converters: dict[str, Callable[[str], object]] = {
            **dict.fromkeys(PRICE_COLUMNS, parse_price),
            **dict.fromkeys(COUNT_COLUMNS, parse_count),
        }
        return frame.assign(
            stock_id=frame["stock_id"].astype(str).str.strip(),
            date=local["date"],
            **{col: frame[col].map(convert) for col, convert in converters.items()},
        )

    def upsert(self, dataset: str, frame: pd.DataFrame) -> int:
        self.require_dataset(dataset)
        return upsert(
            DailyPrice,
            frame,
            unique_fields=("stock_id", "date"),
            update_fields=(*PRICE_COLUMNS, *COUNT_COLUMNS),
        )

    def _fetch_daily_quotes(self, day: date) -> pd.DataFrame:
        params = {"date": day.strftime("%Y%m%d"), "type": "ALLBUT0999", "response": "json"}
        payload = self.limiter.call(lambda: self._get_json(DAILY_QUOTES_PATH, params))
        stat = str(payload.get("stat", ""))
        if NO_DATA_STAT in stat:
            raise SourceNotReadyError(f"TWSE daily quotes for {day} not available: {stat}")
        self.save_raw("daily_price", params, payload)
        if stat != "OK":
            raise SchemaChangedError(f"TWSE answered with unexpected stat {stat!r}")
        answered = str(payload.get("date", ""))
        if answered != params["date"]:
            if answered.isdigit() and answered < params["date"]:
                raise SourceNotReadyError(f"TWSE still serves {answered} when asked for {day}")
            raise SchemaChangedError(f"TWSE answered for {answered!r} instead of {params['date']}")
        return quotes_frame(payload, day)

    def _get_json(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        try:
            response = self.session.get(
                f"{self.base_url}{path}",
                params=params,
                timeout=self.timeout,
                headers={"User-Agent": settings.ETL_USER_AGENT},
            )
        except requests.RequestException as exc:
            raise RetryableError(f"TWSE request failed: {exc}") from exc
        if response.status_code == 429:
            raise ThrottledError("TWSE returned 429")
        if response.status_code >= 500:
            raise RetryableError(f"TWSE returned {response.status_code}")
        if response.status_code >= 400:
            raise SourceRequestError(f"TWSE refused the request with {response.status_code}")
        try:
            data = response.json()
        except ValueError:
            raise SchemaChangedError(f"TWSE returned non-JSON: {response.text[:200]!r}") from None
        if not isinstance(data, dict):
            raise SchemaChangedError("TWSE returned JSON that is not an object")
        return data
