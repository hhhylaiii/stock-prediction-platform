"""TPEx OTC-market client and adapter using whole-market daily endpoints (SPEC §3, §8.6)."""

from datetime import date
from typing import Any

import pandas as pd
import requests
from django.conf import settings

from apps.etl.adapters.base import BaseAdapter
from apps.etl.adapters.dispositions import (
    ATTENTION,
    DISPOSITION,
    FULL_DELIVERY,
    day_events,
    event_frame,
    previous_session,
    standard_events,
    upsert_events,
)
from apps.etl.adapters.exchange_tables import (
    check_published,
    parsed_quotes,
    quoted_only,
    share_changes,
    source_rows,
    standard_rows,
    upsert_daily_prices,
    upsert_index_rows,
)
from apps.etl.adapters.tpex_tables import (
    OK_STAT,
    TABLES,
    attention_events,
    check_answer,
    disposal_events,
    full_delivery_codes,
    index_frame,
    index_rows,
    quote_rows,
    quoted_codes,
)
from apps.etl.client import SourceClient
from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.exchange_rows import TPEX_LAYOUT, OfficialDay, official_days
from apps.etl.models import RawTpex
from apps.etl.ratelimit import RateLimiter
from apps.etl.share_counts import drop_repeated_counts, latest_counts
from apps.etl.upsert import upsert
from apps.market.calendar import TradingDays
from apps.market.models import DailyChip

TPEX_MIN_INTERVAL = 3.0
DELISTED_PATH = "/www/zh-tw/company/deListed"
TRADING_STOCK_PATH = "/www/zh-tw/afterTrading/tradingStock"
DEFAULT_PAGE_SIZE = 1000
OTC_QUOTES_PATH = "/www/zh-tw/afterTrading/otc"
DAILY_QUOTES_PATH = "/www/zh-tw/afterTrading/dailyQuotes"
INDEX_PATH = "/www/zh-tw/indexInfo/inx"
ATTENTION_PATH = "/www/zh-tw/bulletin/attention"
DISPOSAL_PATH = "/www/zh-tw/bulletin/disposal"
ALTERED_PATH = "/www/zh-tw/afterTrading/chtm"
WHOLE_MARKET = {"type": "all", "code": "", "cate": ""}
QUOTED_TYPE = "EW"
DAILY_PRICE = "daily_price"
INDEX_DAILY = "index_daily"
INSTITUTIONAL = "institutional"
SHARES_OUTSTANDING = "shares_outstanding"
EVENT_DATASETS = frozenset({ATTENTION, DISPOSITION, FULL_DELIVERY})


class TpexClient(SourceClient):
    source = "tpex"
    label = "TPEx"
    raw_model = RawTpex
    min_interval = TPEX_MIN_INTERVAL

    def __init__(
        self,
        limiter: RateLimiter,
        base_url: str,
        timeout: float,
        session: requests.Session | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> None:
        super().__init__(limiter, base_url, timeout, session)
        self.page_size = page_size

    @classmethod
    def build(cls, limiter: RateLimiter, session: requests.Session | None = None) -> "TpexClient":
        return cls(limiter, str(settings.TPEX_BASE_URL), float(settings.TPEX_TIMEOUT), session)

    def delisted(self) -> dict[str, Any]:
        """Every 終止上櫃 row across pages, as one table with the source's field names."""
        rows: list[Any] = []
        while True:
            form = {
                "code": "",
                "date": "ALL",
                "reason": "-1",
                "response": "json",
                "paging-offset": str(len(rows)),
                "paging-size": str(self.page_size),
            }
            payload = self.post_json(DELISTED_PATH, form)
            self.save_raw("delisted", form, payload)
            table = self._first_table(payload)
            total = table.get("totalCount")
            if not isinstance(total, int):
                raise SchemaChangedError(f"TPEx delisting list totalCount is {total!r}")
            page = table.get("data") or []
            rows.extend(page)
            if len(rows) >= total:
                return {**table, "data": rows}
            if not page:
                raise SchemaChangedError(
                    f"TPEx delisting list ended at {len(rows)} of {total} rows"
                )

    def stock_month(self, stock_id: str, month: date) -> list[date]:
        """Days one stock traded on TPEx in a month (個股日成交資訊); empty before listing."""
        return [official.day for official in self.stock_days(stock_id, month)]

    def stock_days(self, stock_id: str, month: date) -> list[OfficialDay]:
        """個股日成交資訊 rows of one stock and month; volume arrives in thousands of shares."""
        form = {"code": stock_id, "date": f"{month:%Y/%m}/01", "response": "json"}
        payload = self.post_json(TRADING_STOCK_PATH, form)
        self.save_raw("trading_stock", form, payload)
        if payload.get("stat") != "ok":
            raise SchemaChangedError(f"TPEx stock month stat is {payload.get('stat')!r}")
        tables = payload.get("tables")
        if not isinstance(tables, list) or not tables or not isinstance(tables[0], dict):
            raise SchemaChangedError("TPEx stock month has no table")
        table = tables[0]
        return official_days(
            table.get("data"), table.get("fields"), month, "TPEx stock month", TPEX_LAYOUT
        )

    @staticmethod
    def _first_table(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("stat") != "ok":
            raise SchemaChangedError(f"TPEx delisting list stat is {payload.get('stat')!r}")
        tables = payload.get("tables")
        if not isinstance(tables, list) or not tables or not isinstance(tables[0], dict):
            raise SchemaChangedError("TPEx delisting list has no table")
        return dict(tables[0])


def _day_form(day: date, **params: str) -> dict[str, str]:
    return {"date": f"{day:%Y/%m/%d}", **params, "response": "json"}


def _range_form(first: date, last: date) -> dict[str, str]:
    return {"startDate": f"{first:%Y/%m/%d}", "endDate": f"{last:%Y/%m/%d}"}


class TpexAdapter(BaseAdapter):
    source = "tpex"
    raw_model = RawTpex
    datasets = frozenset({DAILY_PRICE, INDEX_DAILY, *TABLES, *EVENT_DATASETS})
    monthly_datasets = frozenset({INDEX_DAILY, ATTENTION, DISPOSITION})

    def __init__(
        self,
        calendar: TradingDays,
        limiter: RateLimiter,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(calendar, limiter)
        self.client = TpexClient.build(limiter, session)

    def fetch(
        self, dataset: str, start: date, end: date, stock_id: str | None = None
    ) -> pd.DataFrame:
        self.require_dataset(dataset)
        spans = self.units(dataset, start, end)
        frames = [self._fetch_span(dataset, first, last) for first, last in spans]
        return pd.concat(frames, ignore_index=True)

    def normalize(self, dataset: str, raw_df: pd.DataFrame) -> pd.DataFrame:
        self.require_dataset(dataset)
        if dataset in EVENT_DATASETS:
            return standard_events(raw_df)
        if dataset == DAILY_PRICE:
            return parsed_quotes(raw_df)
        if dataset == INDEX_DAILY:
            return index_frame(raw_df)
        table = TABLES[dataset]
        frame = standard_rows(raw_df, table)
        if table.model is DailyChip:
            quoted = {day: self._quoted_codes(day) for day in sorted(set(raw_df["date"]))}
            zeros = tuple(table.columns) if dataset == INSTITUTIONAL else ()
            return quoted_only(frame, quoted, table.where, zeros)
        if dataset == SHARES_OUTSTANDING:
            return share_changes(frame, latest_counts(min(raw_df["date"])))
        return frame

    def upsert(self, dataset: str, frame: pd.DataFrame) -> int:
        self.require_dataset(dataset)
        if dataset in EVENT_DATASETS:
            return upsert_events(frame)
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

    def _fetch_span(self, dataset: str, first: date, last: date) -> pd.DataFrame:
        if dataset == INDEX_DAILY:
            form = {"date": f"{first:%Y/%m}/01", "response": "json"}
            payload = self._published(dataset, INDEX_PATH, form, last)
            days = self.calendar.between(first, last)
            return pd.concat([index_rows(payload, day) for day in days], ignore_index=True)
        if dataset == ATTENTION:
            window = (previous_session(self.calendar, first), last)
            form = {**_range_form(*window), **WHOLE_MARKET, "order": "date", "response": "json"}
            payload = self._answered(dataset, ATTENTION_PATH, form)
            return event_frame(attention_events(payload, self.calendar.between(*window)))
        if dataset == DISPOSITION:
            form = {
                **_range_form(first, last),
                **WHOLE_MARKET,
                "reason": "-1",
                "measure": "-1",
                "order": "date",
                "response": "json",
            }
            payload = self._answered(dataset, DISPOSAL_PATH, form)
            return event_frame(disposal_events(payload, (first, last)))
        return self._fetch_day(dataset, first)

    def _fetch_day(self, dataset: str, day: date) -> pd.DataFrame:
        if dataset == FULL_DELIVERY:
            payload = self._answered(dataset, ALTERED_PATH, _day_form(day))
            return event_frame(day_events(full_delivery_codes(payload, day), day, FULL_DELIVERY))
        if dataset == DAILY_PRICE:
            otc_form = _day_form(day, type=QUOTED_TYPE)
            otc = self._published(DAILY_PRICE, OTC_QUOTES_PATH, otc_form, day)
            quotes = self._published(DAILY_PRICE, DAILY_QUOTES_PATH, _day_form(day), day)
            return quote_rows(otc, quotes, day)
        table = TABLES[dataset]
        payload = self._published(dataset, table.path, _day_form(day, **table.params), day)
        return source_rows(payload, day, table, check_answer)

    def _published(
        self, dataset: str, path: str, form: dict[str, str], day: date
    ) -> dict[str, Any]:
        """The answer to one request, kept raw once it carries rows."""
        payload = self.client.post_json(path, form)
        check_published(payload, f"TPEx {path.removeprefix('/www/zh-tw/')}", day)
        self.save_raw(dataset, form, payload)
        return payload

    def _answered(self, dataset: str, path: str, form: dict[str, str]) -> dict[str, Any]:
        """The answer to a list request, kept raw; an empty list is a valid answer."""
        payload = self.client.post_json(path, form)
        self.save_raw(dataset, form, payload)
        if payload.get("stat") != OK_STAT:
            raise SchemaChangedError(f"TPEx {path} answered with stat {payload.get('stat')!r}")
        return payload

    def _quoted_codes(self, day: date) -> set[str]:
        """Local-currency codes of the latest stored EW quotes answered for the day."""
        stored = (
            RawTpex.objects.filter(
                dataset=DAILY_PRICE,
                request_params__type=QUOTED_TYPE,
                payload__date=f"{day:%Y%m%d}",
            )
            .order_by("-fetched_at")
            .first()
        )
        if stored is None:
            raise SourceNotReadyError(
                f"TPEx daily_price for {day} is not loaded; chip rows need its quotes"
            )
        return quoted_codes(stored.payload, day)
