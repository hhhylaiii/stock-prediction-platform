"""TWSE listed-market adapter using whole-market daily endpoints (SPEC §3, §8.6)."""

import logging
from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from typing import Any, ClassVar

import pandas as pd
import requests
from django.conf import settings

from apps.etl.adapters.base import BaseAdapter
from apps.etl.adapters.corporate_actions import (
    DIVIDEND_EVENT,
    EX_RIGHTS,
    EX_RIGHTS_DETAIL,
    action_frame,
    events_from,
    write_events,
)
from apps.etl.adapters.dispositions import (
    ATTENTION,
    DISPOSITION,
    FULL_DELIVERY,
    announced_events,
    day_events,
    event_frame,
    list_rows,
    listed_code,
    period_events,
    previous_session,
    standard_events,
    upsert_events,
)
from apps.etl.adapters.exchange_actions import (
    CASH_ONLY,
    TWSE_RESTARTS,
    ExRight,
    restart_actions,
    twse_action,
    twse_detail,
    twse_ex_rights,
)
from apps.etl.adapters.exchange_tables import (
    NO_DATA_STAT,
    check_published,
    find_table,
    is_foreign_currency_etf,
    parsed_quotes,
    priced_codes,
    quoted_only,
    require_kept,
    share_changes,
    source_rows,
    standard_rows,
    upsert_daily_prices,
    upsert_index_rows,
)
from apps.etl.adapters.twse_tables import (
    TABLES,
    check_answer,
    return_index_frame,
    return_index_rows,
    taiex_frame,
    taiex_rows,
)
from apps.etl.client import SourceClient
from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.exchange_rows import TWSE_LAYOUT, OfficialDay, official_days
from apps.etl.models import RawTwse
from apps.etl.ratelimit import RateLimiter, RetryableError
from apps.etl.share_counts import drop_repeated_counts, latest_counts
from apps.etl.upsert import upsert
from apps.market.calendar import TradingDays
from apps.market.models import DailyChip

logger = logging.getLogger(__name__)
TWSE_MIN_INTERVAL = 3.0
DAILY_QUOTES_PATH = "/rwd/zh/afterTrading/MI_INDEX"
SUSPEND_LISTING_PATH = "/rwd/zh/company/suspendListing"
HOLIDAY_SCHEDULE_PATH = "/rwd/zh/holidaySchedule/holidaySchedule"
MONTHLY_TRADING_PATH = "/rwd/zh/afterTrading/FMTQIK"
RETURN_INDEX_PATH = "/rwd/zh/TAIEX/MFI94U"
STOCK_DAY_PATH = "/rwd/zh/afterTrading/STOCK_DAY"
NOTICE_PATH = "/rwd/zh/announcement/notice"
PUNISH_PATH = "/rwd/zh/announcement/punish"
FULL_DELIVERY_PATH = "/exchangeReport/TWT85U"
BUSY_STAT = "系統忙碌"
EX_RIGHTS_PATH = "/rwd/zh/exRight/TWT49U"
EX_RIGHTS_DETAIL_PATH = "/rwd/zh/exRight/TWT49UDetail"
NOTICE_FIELDS = ("證券代號", "日期")
PUNISH_FIELDS = ("證券代號", "公布日期", "處置起迄時間")
CODE_FIELD = "證券代號"
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
RETURN_INDEX = "return_index"
INSTITUTIONAL = "institutional"
SHARES_OUTSTANDING = "shares_outstanding"
DAY_TRADE = "day_trade"
EVENT_DATASETS = frozenset({ATTENTION, DISPOSITION, FULL_DELIVERY})


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
    datasets = frozenset(
        {DAILY_PRICE, INDEX_DAILY, RETURN_INDEX, DIVIDEND_EVENT, *TABLES, *EVENT_DATASETS}
    )
    monthly_datasets = frozenset({INDEX_DAILY, RETURN_INDEX, ATTENTION, DISPOSITION})
    raw_pages: ClassVar[Mapping[str, Mapping[str, str]]] = {
        DIVIDEND_EVENT: {
            EX_RIGHTS_PATH: EX_RIGHTS,
            EX_RIGHTS_DETAIL_PATH: EX_RIGHTS_DETAIL,
            **{table.path: table.raw for table in TWSE_RESTARTS},
        }
    }

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
        spans = self.units(dataset, start, end)
        frames = [self._fetch_span(dataset, first, last) for first, last in spans]
        frame = pd.concat(frames, ignore_index=True)
        frame.attrs["day"] = start
        return frame

    def normalize(self, dataset: str, raw_df: pd.DataFrame) -> pd.DataFrame:
        self.require_dataset(dataset)
        if dataset == DIVIDEND_EVENT:
            events = events_from(raw_df)
            events.attrs["day"] = raw_df.attrs["day"]
            return events
        if dataset in EVENT_DATASETS:
            return standard_events(raw_df)
        if dataset == DAILY_PRICE:
            return self._normalize_quotes(raw_df)
        if dataset == INDEX_DAILY:
            return taiex_frame(raw_df)
        if dataset == RETURN_INDEX:
            return return_index_frame(raw_df)
        table = TABLES[dataset]
        frame = standard_rows(raw_df, table)
        if table.model is DailyChip:
            quoted = {day: self._quoted_codes(day, dataset) for day in sorted(set(raw_df["date"]))}
            zeros = tuple(table.columns) if dataset == INSTITUTIONAL else ()
            kept = quoted_only(frame, quoted, f"TWSE {table.label}", zeros)
            if dataset == DAY_TRADE:
                require_kept(frame, kept, f"TWSE {table.label}")
            return kept
        if dataset == SHARES_OUTSTANDING:
            return share_changes(frame, latest_counts(min(raw_df["date"])))
        return frame

    def upsert(self, dataset: str, frame: pd.DataFrame) -> int:
        self.require_dataset(dataset)
        if dataset == DIVIDEND_EVENT:
            return write_events(frame, self.source, frame.attrs["day"])
        if dataset in EVENT_DATASETS:
            return upsert_events(frame)
        if dataset == DAILY_PRICE:
            return upsert_daily_prices(frame)
        if dataset in (INDEX_DAILY, RETURN_INDEX):
            return upsert_index_rows(frame)
        table = TABLES[dataset]
        rows = upsert(
            table.model, frame, unique_fields=table.keys, update_fields=tuple(table.columns)
        )
        if dataset == SHARES_OUTSTANDING:
            drop_repeated_counts(frame["stock_id"])
        return rows

    def _fetch_span(self, dataset: str, first: date, last: date) -> pd.DataFrame:
        if dataset == INDEX_DAILY:
            return self._fetch_taiex(first, last)
        if dataset == RETURN_INDEX:
            return self._fetch_return_index(first, last)
        if dataset == ATTENTION:
            return self._fetch_attention(first, last)
        if dataset == DISPOSITION:
            return self._fetch_disposition(first, last)
        if dataset == DIVIDEND_EVENT:
            return self._fetch_actions(first)
        return self._fetch_day(dataset, first)

    def _fetch_actions(self, day: date) -> pd.DataFrame:
        """Ex-rights and restart days of this session, whose tables are out before it opens."""
        span = (day, day)
        params = {
            "startDate": f"{span[0]:%Y%m%d}",
            "endDate": f"{span[1]:%Y%m%d}",
            "response": "json",
        }
        body = self._listed(EX_RIGHTS, EX_RIGHTS_PATH, params)
        rows = twse_ex_rights(body, span) if body is not None else []
        actions = [twse_action(row, self._detail(row)) for row in rows]
        for table in TWSE_RESTARTS:
            body = self._listed(table.raw, table.path, params)
            actions.extend(restart_actions(body, table, span) if body is not None else [])
        return action_frame(actions)

    def _detail(self, row: ExRight) -> tuple[Decimal, Decimal] | None:
        """Cash per share and bonus shares per thousand from TWT49UDetail, for non-cash rows."""
        if row.kind == CASH_ONLY:
            return None
        params = {"STK_NO": row.stock_id, "T1": f"{row.ex_date:%Y%m%d}", "response": "json"}
        body = self._listed(EX_RIGHTS_DETAIL, EX_RIGHTS_DETAIL_PATH, params)
        if body is None:
            raise SchemaChangedError(
                f"TWSE TWT49UDetail has nothing for {row.stock_id} {row.ex_date}"
            )
        return twse_detail(body, row.stock_id)

    def _fetch_day(self, dataset: str, day: date) -> pd.DataFrame:
        if dataset == DAILY_PRICE:
            return self._fetch_daily_quotes(day)
        if dataset == FULL_DELIVERY:
            return self._fetch_full_delivery(day)
        table = TABLES[dataset]
        params = {"date": day.strftime("%Y%m%d"), **table.params, "response": "json"}
        payload = self.client.get_json(table.path, params)
        check_published(payload, table.where, day)
        self.save_raw(dataset, params, payload)
        return source_rows(payload, day, table, check_answer)

    def _fetch_taiex(self, first: date, last: date) -> pd.DataFrame:
        """FMTQIK rows of the span's trading days from one monthly request."""
        params = {"date": f"{first:%Y%m}01", "response": "json"}
        payload = self.client.get_json(MONTHLY_TRADING_PATH, params)
        check_published(payload, "TWSE FMTQIK", last)
        self.save_raw(INDEX_DAILY, params, payload)
        days = self.calendar.between(first, last)
        return pd.concat([taiex_rows(payload, day) for day in days], ignore_index=True)

    def _fetch_return_index(self, first: date, last: date) -> pd.DataFrame:
        """MFI94U rows of the span's trading days from one monthly request."""
        params = {"date": f"{first:%Y%m}01", "response": "json"}
        payload = self.client.get_json(RETURN_INDEX_PATH, params)
        check_published(payload, "TWSE MFI94U", last)
        self.save_raw(RETURN_INDEX, params, payload)
        days = self.calendar.between(first, last)
        return pd.concat([return_index_rows(payload, day) for day in days], ignore_index=True)

    def _fetch_attention(self, first: date, last: date) -> pd.DataFrame:
        """Attention notices announced in the span or on the session before it."""
        window = (previous_session(self.calendar, first), last)
        params = {
            "querytype": "1",
            "startDate": f"{window[0]:%Y%m%d}",
            "endDate": f"{last:%Y%m%d}",
            "response": "json",
        }
        payload = self._listed(ATTENTION, NOTICE_PATH, params)
        if payload is None:
            return event_frame([])
        records = list_rows(payload, "TWSE notice", NOTICE_FIELDS)
        sessions = self.calendar.between(*window)
        return event_frame(announced_events(records, NOTICE_FIELDS, "TWSE notice", sessions))

    def _fetch_disposition(self, first: date, last: date) -> pd.DataFrame:
        """Disposition periods overlapping the span; TWSE matches on the period itself."""
        params = {"startDate": f"{first:%Y%m%d}", "endDate": f"{last:%Y%m%d}", "response": "json"}
        payload = self._listed(DISPOSITION, PUNISH_PATH, params)
        if payload is None:
            return event_frame([])
        records = list_rows(payload, "TWSE punish", PUNISH_FIELDS)
        return event_frame(period_events(records, PUNISH_FIELDS, "TWSE punish", (first, last)))

    def _fetch_full_delivery(self, day: date) -> pd.DataFrame:
        """The day's TWT85U 變更交易 list, published before the session opens."""
        params = {"date": f"{day:%Y%m%d}", "response": "json"}
        payload = self._listed(FULL_DELIVERY, FULL_DELIVERY_PATH, params)
        if payload is None:
            raise SourceNotReadyError(
                f"TWSE TWT85U for {day} lists nothing; the list is never empty"
            )
        check_answer(payload, day)
        records = list_rows(payload, "TWSE TWT85U", (CODE_FIELD,))
        codes = [listed_code(record, CODE_FIELD, "TWSE TWT85U") for record in records]
        return event_frame(day_events([code for code in codes if code], day, FULL_DELIVERY))

    def _listed(self, dataset: str, path: str, params: dict[str, str]) -> dict[str, Any] | None:
        """An answered list kept raw; None when nothing matched, retried later when busy."""
        payload = self.client.get_json(path, params)
        stat = str(payload.get("stat", ""))
        if BUSY_STAT in stat:
            raise RetryableError(f"TWSE {path} is busy: {stat}")
        self.save_raw(dataset, params, payload)
        if NO_DATA_STAT in stat:
            return None
        if stat.upper() != "OK":
            raise SchemaChangedError(f"TWSE {path} answered with stat {stat!r}")
        return payload

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

    def _quotes_kept_from(self) -> str | None:
        """The first day whose quote page was kept; from then on each day waits for its own."""
        kept = RawTwse.objects.filter(dataset=DAILY_PRICE).order_by("payload__date")
        return kept.values_list("payload__date", flat=True).first()

    def _quoted_codes(self, day: date, dataset: str) -> set[str]:
        """Local-currency codes of the latest stored MI_INDEX quotes answered for the day."""
        stored = (
            RawTwse.objects.filter(dataset=DAILY_PRICE, payload__date=f"{day:%Y%m%d}")
            .order_by("-fetched_at")
            .first()
        )
        if stored is None:
            priced = priced_codes(day, self._quotes_kept_from()) if dataset == DAY_TRADE else set()
            if priced:
                return priced
            raise SourceNotReadyError(
                f"TWSE daily_price for {day} is not loaded; chip rows need its quotes"
            )
        codes = quotes_frame(stored.payload, day)["證券代號"].astype(str).str.strip()
        return {code for code in codes if not is_foreign_currency_etf(code)}
