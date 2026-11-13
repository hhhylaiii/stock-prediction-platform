"""TPEx OTC-market client and adapter using whole-market daily endpoints (SPEC §3, §8.6)."""

from collections.abc import Iterator, Mapping
from datetime import date
from typing import Any, ClassVar

import pandas as pd
import requests
from django.conf import settings
from django.utils import timezone

from apps.etl import dividend_notices
from apps.etl.adapters.base import BaseAdapter
from apps.etl.adapters.corporate_actions import (
    DIVIDEND_EVENT,
    EX_RIGHTS,
    action_frame,
    events_from,
    write_events,
)
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
from apps.etl.adapters.exchange_actions import (
    TPEX_RESTARTS,
    restart_actions,
    tpex_ex_rights,
    tpex_table,
)
from apps.etl.adapters.exchange_notices import (
    DIVIDEND_NOTICE,
    NOTICE_FORM,
    TPEX_LABEL,
    TPEX_NOTICE_PATH,
    kept_by_session,
    one_day,
    seen_at,
    tpex_notices,
)
from apps.etl.adapters.exchange_tables import (
    check_published,
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
EX_RIGHTS_PATH = "/www/zh-tw/bulletin/exDailyQ"
WHOLE_MARKET = {"type": "all", "code": "", "cate": ""}
QUOTED_TYPE = "EW"
DAILY_PRICE = "daily_price"
INDEX_DAILY = "index_daily"
INSTITUTIONAL = "institutional"
SHARES_OUTSTANDING = "shares_outstanding"
DAY_TRADE = "day_trade"
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
    datasets = frozenset(
        {DAILY_PRICE, INDEX_DAILY, DIVIDEND_EVENT, DIVIDEND_NOTICE, *TABLES, *EVENT_DATASETS}
    )
    monthly_datasets = frozenset({INDEX_DAILY, ATTENTION, DISPOSITION})
    raw_pages: ClassVar[Mapping[str, Mapping[str, str]]] = {
        DIVIDEND_EVENT: {
            EX_RIGHTS_PATH: EX_RIGHTS,
            **{table.path: table.raw for table in TPEX_RESTARTS},
        }
    }

    def __init__(
        self,
        calendar: TradingDays,
        limiter: RateLimiter,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(calendar, limiter)
        self.client = TpexClient.build(limiter, session)

    def units(self, dataset: str, start: date, end: date) -> Iterator[tuple[date, date]]:
        """The notice list has no history, so even a run may ask for one day only."""
        if dataset == DIVIDEND_NOTICE:
            one_day(start, end)
        return super().units(dataset, start, end)

    def fetch(
        self, dataset: str, start: date, end: date, stock_id: str | None = None
    ) -> pd.DataFrame:
        self.require_dataset(dataset)
        if dataset == DIVIDEND_NOTICE:
            return self._fetch_notices(start, end)
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
        if dataset == DIVIDEND_NOTICE:
            return raw_df
        if dataset in EVENT_DATASETS:
            return standard_events(raw_df)
        if dataset == DAILY_PRICE:
            return parsed_quotes(raw_df)
        if dataset == INDEX_DAILY:
            return index_frame(raw_df)
        table = TABLES[dataset]
        frame = standard_rows(raw_df, table)
        if table.model is DailyChip:
            quoted = {day: self._quoted_codes(day, dataset) for day in sorted(set(raw_df["date"]))}
            zeros = tuple(table.columns) if dataset == INSTITUTIONAL else ()
            kept = quoted_only(frame, quoted, table.where, zeros)
            if dataset == DAY_TRADE:
                require_kept(frame, kept, table.where)
            return kept
        if dataset == SHARES_OUTSTANDING:
            return share_changes(frame, latest_counts(min(raw_df["date"])))
        return frame

    def upsert(self, dataset: str, frame: pd.DataFrame) -> int:
        self.require_dataset(dataset)
        if dataset == DIVIDEND_EVENT:
            return write_events(frame, self.source, frame.attrs["day"])
        if dataset == DIVIDEND_NOTICE:
            return dividend_notices.write_notices(self.source, frame, self.calendar)
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
        if dataset == DIVIDEND_EVENT:
            return self._fetch_actions(first)
        return self._fetch_day(dataset, first)

    def _fetch_actions(self, day: date) -> pd.DataFrame:
        """Ex-rights and restart days of this session, whose tables are out before it opens."""
        span = (day, day)
        form = {**_range_form(*span), "response": "json"}
        payload = self._answered(EX_RIGHTS, EX_RIGHTS_PATH, form)
        actions = tpex_ex_rights(payload, span)
        for table in TPEX_RESTARTS:
            payload = self._answered(table.raw, table.path, form)
            actions.extend(restart_actions(tpex_table(payload, table.label), table, span))
        return action_frame(actions)

    def _fetch_notices(self, start: date, end: date) -> pd.DataFrame:
        """prePost as it reads now; it takes no date, so the day asked only names the run."""
        one_day(start, end)
        self.trading_days(start, end)
        payload = self.client.post_json(TPEX_NOTICE_PATH, NOTICE_FORM)
        seen = timezone.now()
        kept = kept_by_session(self.calendar, seen, payload)
        self.save_raw(DIVIDEND_NOTICE, NOTICE_FORM, payload, kept)
        return seen_at(tpex_notices(payload), seen, TPEX_LABEL)

    def _fetch_day(self, dataset: str, day: date) -> pd.DataFrame:
        if dataset == FULL_DELIVERY:
            payload = self._answered(dataset, ALTERED_PATH, _day_form(day))
            return event_frame(day_events(full_delivery_codes(payload, day), day, FULL_DELIVERY))
        if dataset == DAILY_PRICE:
            return self._quotes(day, DAILY_PRICE)
        table = TABLES[dataset]
        payload = self._published(dataset, table.path, _day_form(day, **table.params), day)
        return source_rows(payload, day, table, check_answer)

    def _quotes(self, day: date, kept_as: str) -> pd.DataFrame:
        """Both quote tables of the day as rows, their answers kept under the given raw dataset."""
        otc = self._published(kept_as, OTC_QUOTES_PATH, _day_form(day, type=QUOTED_TYPE), day)
        quotes = self._published(kept_as, DAILY_QUOTES_PATH, _day_form(day), day)
        return quote_rows(otc, quotes, day)

    def require_before_kept_quotes(self, day: date) -> None:
        """A day from the first kept quote page on gets its bars from the usual fetch."""
        kept_from = self._quotes_kept_from()
        if kept_from is not None and f"{day:%Y%m%d}" >= kept_from:
            raise ValueError(
                f"TPEx quote pages are kept from {kept_from}; fetch {DAILY_PRICE} for {day}"
            )

    def day_quotes(self, day: date, kept_as: str) -> pd.DataFrame:
        """Parsed quotes of a day before the first kept quote page, kept under another dataset.

        Kept as daily_price they would become that first page and strand the days after them.
        """
        if kept_as == DAILY_PRICE:
            raise ValueError(f"these answers must not be kept as {DAILY_PRICE}")
        self.require_before_kept_quotes(day)
        return parsed_quotes(self._quotes(day, kept_as))

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

    def _quotes_kept_from(self) -> str | None:
        """The first day whose quote page was kept; from then on each day waits for its own."""
        kept = RawTpex.objects.filter(dataset=DAILY_PRICE, request_params__type=QUOTED_TYPE)
        return kept.order_by("payload__date").values_list("payload__date", flat=True).first()

    def _quoted_codes(self, day: date, dataset: str) -> set[str]:
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
            priced = priced_codes(day, self._quotes_kept_from()) if dataset == DAY_TRADE else set()
            if priced:
                return priced
            raise SourceNotReadyError(
                f"TPEx daily_price for {day} is not loaded; chip rows need its quotes"
            )
        return quoted_codes(stored.payload, day)
