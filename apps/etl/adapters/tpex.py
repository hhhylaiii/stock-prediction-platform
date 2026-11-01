"""TPEx OTC-market client; daily datasets arrive in Phase 2 (SPEC §3, §8.6)."""

from datetime import date
from typing import Any

import requests
from django.conf import settings

from apps.etl.client import SourceClient
from apps.etl.errors import SchemaChangedError
from apps.etl.exchange_rows import TPEX_LAYOUT, OfficialDay, official_days
from apps.etl.models import RawTpex
from apps.etl.ratelimit import RateLimiter

TPEX_MIN_INTERVAL = 3.0
DELISTED_PATH = "/www/zh-tw/company/deListed"
TRADING_STOCK_PATH = "/www/zh-tw/afterTrading/tradingStock"
DEFAULT_PAGE_SIZE = 1000


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
