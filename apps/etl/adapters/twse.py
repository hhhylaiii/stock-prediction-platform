"""TWSE listed-market adapter using whole-market daily endpoints (SPEC §3, §8.6)."""

import logging
from datetime import date
from typing import Any

import pandas as pd
import requests
from django.conf import settings

from apps.etl.adapters.base import BaseAdapter
from apps.etl.adapters.exchange_tables import (
    NO_DATA_STAT,
    check_published,
    find_table,
    is_foreign_currency_etf,
    parsed_quotes,
    quoted_only,
    share_changes,
    source_rows,
    standard_rows,
    upsert_daily_prices,
    upsert_index_rows,
)
from apps.etl.adapters.twse_tables import (
    TABLES,
    check_answer,
    taiex_frame,
    taiex_rows,
)
from apps.etl.client import SourceClient
from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.exchange_rows import TWSE_LAYOUT, OfficialDay, official_days
from apps.etl.models import RawTwse
from apps.etl.ratelimit import RateLimiter
from apps.etl.share_counts import drop_repeated_counts
from apps.etl.upsert import upsert
from apps.market.calendar import TradingDays
from apps.market.models import DailyChip, SharesOutstanding

logger = logging.getLogger(__name__)
TWSE_MIN_INTERVAL = 3.0
DAILY_QUOTES_PATH = "/rwd/zh/afterTrading/MI_INDEX"
SUSPEND_LISTING_PATH = "/rwd/zh/company/suspendListing"
HOLIDAY_SCHEDULE_PATH = "/rwd/zh/holidaySchedule/holidaySchedule"
MONTHLY_TRADING_PATH = "/rwd/zh/afterTrading/FMTQIK"
STOCK_DAY_PATH = "/rwd/zh/afterTrading/STOCK_DAY"
STOCK_DAY_FIRST_MONTH = date(2010, 1, 1)
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
DAILY_PRICE = "daily_price"
INDEX_DAILY = "index_daily"
INSTITUTIONAL = "institutional"
SHARES_OUTSTANDING = "shares_outstanding"


def quotes_frame(payload: dict[str, Any], day: date) -> pd.DataFrame:
    """Rows of the daily quotes table in a stored MI_INDEX payload, for fetch or a rerun."""
    check_answer(payload, day)
    table = find_table(payload, DAILY_QUOTES_TITLE)
    if table.get("hints") != DAILY_QUOTES_UNIT_HINT:
        raise SchemaChangedError(f"TWSE daily quotes units changed: {table.get('hints')!r}")
    try:
        frame = pd.DataFrame(table["data"], columns=table["fields"])
    except (KeyError, ValueError) as exc:
        raise SchemaChangedError(f"TWSE daily quotes table is malformed: {exc}") from None
    if frame.empty:
        raise SchemaChangedError(f"TWSE daily quotes table for {day} has no rows")
    return frame.assign(date=day)


class TwseClient(SourceClient):
    source = "twse"
    label = "TWSE"
    raw_model = RawTwse
    min_interval = TWSE_MIN_INTERVAL

    @classmethod
    def build(cls, limiter: RateLimiter, session: requests.Session | None = None) -> "TwseClient":
        return cls(limiter, str(settings.TWSE_BASE_URL), float(settings.TWSE_TIMEOUT), session)

    def holiday_schedule(self, year: int) -> dict[str, Any]:
        """The 市場開休市日期 table of one year (2021 onwards); empty until TWSE publishes it."""
        params = {"date": f"{year}0101", "response": "json"}
        payload = self.get_json(HOLIDAY_SCHEDULE_PATH, params)
        self.save_raw("holiday_schedule", params, payload)
        return payload

    def monthly_trading(self, month: date) -> dict[str, Any]:
        """The FMTQIK market totals of one month, one row per trading day."""
        params = {"date": f"{month:%Y%m}01", "response": "json"}
        payload = self.get_json(MONTHLY_TRADING_PATH, params)
        self.save_raw("trading_days", params, payload)
        return payload

    def stock_month(self, stock_id: str, month: date) -> list[date] | None:
        """Days one stock traded on TWSE in a month; None before 2010, which STOCK_DAY refuses."""
        days = self.stock_days(stock_id, month)
        return None if days is None else [official.day for official in days]

    def stock_days(self, stock_id: str, month: date) -> list[OfficialDay] | None:
        """STOCK_DAY rows of one stock-month in shares; empty before listing, None before 2010."""
        if month < STOCK_DAY_FIRST_MONTH:
            return None
        params = {"date": f"{month:%Y%m}01", "stockNo": stock_id, "response": "json"}
        payload = self.get_json(STOCK_DAY_PATH, params)
        self.save_raw("stock_day", params, payload)
        stat = str(payload.get("stat", ""))
        if NO_DATA_STAT in stat:
            return []
        if stat != "OK":
            raise SchemaChangedError(f"TWSE stock month stat is {stat!r}")
        return official_days(
            payload.get("data"), payload.get("fields"), month, "TWSE stock month", TWSE_LAYOUT
        )

    def suspend_listing(self) -> dict[str, Any]:
        """The 終止上市公司 list (2001 onwards); kept raw as dataset 'delisted'."""
        params = {"response": "json"}
        payload = self.get_json(SUSPEND_LISTING_PATH, params)
        self.save_raw("delisted", params, payload)
        return payload


class TwseAdapter(BaseAdapter):
    source = "twse"
    raw_model = RawTwse
    datasets = frozenset({DAILY_PRICE, INDEX_DAILY, *TABLES})

    def __init__(
        self,
        calendar: TradingDays,
        limiter: RateLimiter,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(calendar, limiter)
        self.client = TwseClient.build(limiter, session)

    def fetch(
        self, dataset: str, start: date, end: date, stock_id: str | None = None
    ) -> pd.DataFrame:
        self.require_dataset(dataset)
        frames = [self._fetch_day(dataset, day) for day in self.trading_days(start, end)]
        return pd.concat(frames, ignore_index=True)

    def normalize(self, dataset: str, raw_df: pd.DataFrame) -> pd.DataFrame:
        self.require_dataset(dataset)
        if dataset == DAILY_PRICE:
            return self._normalize_quotes(raw_df)
        if dataset == INDEX_DAILY:
            return taiex_frame(raw_df)
        table = TABLES[dataset]
        frame = standard_rows(raw_df, table)
        if table.model is DailyChip:
            quoted = {day: self._quoted_codes(day) for day in sorted(set(raw_df["date"]))}
            zeros = tuple(table.columns) if dataset == INSTITUTIONAL else ()
            return quoted_only(frame, quoted, f"TWSE {table.label}", zeros)
        if dataset == SHARES_OUTSTANDING:
            return share_changes(frame, self._latest_shares(min(raw_df["date"])))
        return frame

    def upsert(self, dataset: str, frame: pd.DataFrame) -> int:
        self.require_dataset(dataset)
        if dataset == DAILY_PRICE:
            return upsert_daily_prices(frame)
        if dataset == INDEX_DAILY:
            return upsert_index_rows(frame)
        table = TABLES[dataset]
        rows = upsert(
            table.model, frame, unique_fields=table.keys, update_fields=tuple(table.columns)
        )
        if dataset == SHARES_OUTSTANDING:
            drop_repeated_counts(frame["stock_id"])
        return rows

    def _fetch_day(self, dataset: str, day: date) -> pd.DataFrame:
        if dataset == DAILY_PRICE:
            return self._fetch_daily_quotes(day)
        if dataset == INDEX_DAILY:
            params = {"date": f"{day:%Y%m}01", "response": "json"}
            payload = self.client.get_json(MONTHLY_TRADING_PATH, params)
            check_published(payload, "TWSE FMTQIK", day)
            self.save_raw(dataset, params, payload)
            return taiex_rows(payload, day)
        table = TABLES[dataset]
        params = {"date": day.strftime("%Y%m%d"), **table.params, "response": "json"}
        payload = self.client.get_json(table.path, params)
        check_published(payload, table.where, day)
        self.save_raw(dataset, params, payload)
        return source_rows(payload, day, table, check_answer)

    def _normalize_quotes(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        missing = set(DAILY_QUOTES_COLUMNS) - set(raw_df.columns)
        if missing:
            raise SchemaChangedError(f"TWSE daily quotes lost columns {sorted(missing)}")
        codes = raw_df["證券代號"].astype(str).str.strip()
        local = raw_df.loc[~codes.map(is_foreign_currency_etf)]
        frame = local[list(DAILY_QUOTES_COLUMNS)].rename(columns=DAILY_QUOTES_COLUMNS)
        return parsed_quotes(
            frame.assign(stock_id=frame["stock_id"].astype(str).str.strip(), date=local["date"])
        )

    def _fetch_daily_quotes(self, day: date) -> pd.DataFrame:
        params = {"date": day.strftime("%Y%m%d"), "type": "ALLBUT0999", "response": "json"}
        payload = self.client.get_json(DAILY_QUOTES_PATH, params)
        stat = str(payload.get("stat", ""))
        if NO_DATA_STAT in stat:
            raise SourceNotReadyError(f"TWSE daily quotes for {day} not available: {stat}")
        self.save_raw(DAILY_PRICE, params, payload)
        return quotes_frame(payload, day)

    def _latest_shares(self, before: date) -> dict[str, int]:
        """The last shares_outstanding count of each stock effective before the given day."""
        latest = (
            SharesOutstanding.objects.filter(effective_date__lt=before)
            .order_by("stock_id", "-effective_date")
            .distinct("stock_id")
            .values_list("stock_id", "shares")
        )
        return dict(latest)

    def _quoted_codes(self, day: date) -> set[str]:
        """Local-currency codes of the latest stored MI_INDEX quotes answered for the day."""
        stored = (
            RawTwse.objects.filter(dataset=DAILY_PRICE, payload__date=f"{day:%Y%m%d}")
            .order_by("-fetched_at")
            .first()
        )
        if stored is None:
            raise SourceNotReadyError(
                f"TWSE daily_price for {day} is not loaded; chip rows need its quotes"
            )
        codes = quotes_frame(stored.payload, day)["證券代號"].astype(str).str.strip()
        return {code for code in codes if not is_foreign_currency_etf(code)}
