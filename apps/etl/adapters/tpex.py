"""TPEx OTC-market client; daily datasets arrive in Phase 2 (SPEC §3, §8.6)."""

from typing import Any

import requests
from django.conf import settings

from apps.etl.adapters.base import SchemaChangedError
from apps.etl.client import SourceClient
from apps.etl.models import RawTpex
from apps.etl.ratelimit import RateLimiter

TPEX_MIN_INTERVAL = 3.0
DELISTED_PATH = "/www/zh-tw/company/deListed"
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
            page = table.get("data") or []
            rows.extend(page)
            if not page or len(rows) >= int(table.get("totalCount", 0)):
                return {**table, "data": rows}

    @staticmethod
    def _first_table(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("stat") != "ok":
            raise SchemaChangedError(f"TPEx delisting list stat is {payload.get('stat')!r}")
        tables = payload.get("tables")
        if not isinstance(tables, list) or not tables or not isinstance(tables[0], dict):
            raise SchemaChangedError("TPEx delisting list has no table")
        return dict(tables[0])
