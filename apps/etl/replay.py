"""Rebuild standard rows from stored raw responses without calling the source (SPEC §4, §5)."""

import json
from collections.abc import Mapping
from datetime import date
from typing import Any

import requests
from django.conf import settings
from django.db import transaction

from apps.etl.adapters import ADAPTERS
from apps.etl.adapters.base import BaseAdapter
from apps.etl.errors import SourceNotReadyError
from apps.etl.models import EtlRun, RawRecord
from apps.etl.ratelimit import RateLimit, RateLimiter
from apps.etl.runs import tracked_run
from apps.market.calendar import TradingDays
from apps.market.models import DispositionEvent, Market, SharesOutstanding, StockMaster

REPLAYABLE = {"twse": Market.TWSE.value, "tpex": Market.TPEX.value}
SHARES = "shares_outstanding"
DAY_EVENTS = frozenset({"attention", "full_delivery"})


class ReplaySession(requests.Session):
    """Answers each request with the latest raw stored for the same dataset and parameters."""

    def __init__(self, raw_model: type[RawRecord], dataset: str) -> None:
        super().__init__()
        self.raw_model = raw_model
        self.dataset = dataset

    def request(  # type: ignore[override]
        self, method: str, url: str, **kwargs: Any
    ) -> requests.Response:
        params: Mapping[str, Any] = kwargs.get("params") or kwargs.get("data") or {}
        stored = (
            self.raw_model._default_manager.filter(
                dataset=self.dataset, request_params=dict(params)
            )
            .order_by("-fetched_at")
            .first()
        )
        if stored is None:
            raise SourceNotReadyError(f"no stored raw for {self.dataset} {dict(params)}")
        response = requests.Response()
        response.status_code = 200
        response.url = url
        response._content = json.dumps(stored.payload, ensure_ascii=False).encode()
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response


def replay_adapter(source: str, dataset: str, calendar: TradingDays) -> BaseAdapter:
    """The source's adapter wired to stored raw and a limiter that never waits."""
    if source not in REPLAYABLE:
        raise ValueError(
            f"cannot reprocess {source!r}: only {sorted(REPLAYABLE)} key raw by request date"
        )
    adapter_class = ADAPTERS[source]
    limit = RateLimit.from_settings(settings.ETL_RATE_LIMITS[source])
    limiter = RateLimiter(limit, sleep=lambda seconds: None)
    session = ReplaySession(adapter_class.raw_model, dataset)
    return adapter_class(calendar, limiter, session)  # type: ignore[call-arg]


def reprocess(source: str, dataset: str, start: date, end: date, calendar: TradingDays) -> EtlRun:
    """Span by span in date order; rows that only exist on change are cleared before rebuilding."""
    adapter = replay_adapter(source, dataset, calendar)
    adapter.require_dataset(dataset)
    spans = list(adapter.units(dataset, start, end))
    with tracked_run(source, dataset, start, end) as run:
        run.checkpoint = {"kind": "reprocess"}
        for first, last in spans:
            raw = adapter.fetch(dataset, first, last)
            frame = adapter.normalize(dataset, raw)
            with transaction.atomic():
                _clear(source, dataset, first, last)
                run.rows += adapter.upsert(dataset, frame)
            run.checkpoint = {"kind": "reprocess", "last_date": last.isoformat()}
            run.save(update_fields=["rows", "checkpoint"])
    return run


def _clear(source: str, dataset: str, first: date, last: date) -> None:
    codes = StockMaster.objects.filter(market=REPLAYABLE[source]).values("stock_id")
    if dataset == SHARES:
        SharesOutstanding.objects.filter(
            stock_id__in=codes, effective_date__range=(first, last)
        ).delete()
    elif dataset in DAY_EVENTS:
        DispositionEvent.objects.filter(
            stock_id__in=codes, type=dataset, start_date__range=(first, last)
        ).delete()
