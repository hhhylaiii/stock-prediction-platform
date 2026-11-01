"""Rebuild standard rows from stored raw responses without calling the source (SPEC §4, §5)."""

import json
from collections.abc import Iterator, Mapping
from datetime import date
from itertools import groupby
from typing import Any, NamedTuple

import pandas as pd
import requests
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from apps.etl.adapters import ADAPTERS
from apps.etl.adapters.base import BaseAdapter
from apps.etl.adapters.tpex_tables import TABLES as TPEX_TABLES
from apps.etl.adapters.twse_tables import TABLES as TWSE_TABLES
from apps.etl.errors import SourceNotReadyError
from apps.etl.models import EtlRun, RawRecord
from apps.etl.ratelimit import RateLimit, RateLimiter
from apps.etl.runs import covered_days, tracked_run
from apps.market.calendar import TradingDays
from apps.market.models import DispositionEvent, Market, SharesOutstanding, StockMaster

REPLAYABLE = {"twse": Market.TWSE.value, "tpex": Market.TPEX.value}
SHARES = "shares_outstanding"
SHARE_CODE_FIELDS = {"twse": TWSE_TABLES[SHARES].code, "tpex": TPEX_TABLES[SHARES].code}
FULL_DELIVERY = "full_delivery"


class MissingRawError(SourceNotReadyError):
    """No stored raw answers the request with exactly these parameters."""


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
            raise MissingRawError(f"no stored raw for {self.dataset} {dict(params)}")
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


class Reprocessed(NamedTuple):
    run: EtlRun
    rebuilt: list[date]


def reprocess(
    source: str, dataset: str, start: date, end: date, calendar: TradingDays
) -> Reprocessed:
    """Replay the range; share counts also replay every later fetched day after it."""
    run = replay_range(source, dataset, start, end, calendar)
    rebuilt = rebuild_after(source, dataset, end, calendar) if dataset == SHARES else []
    return Reprocessed(run, rebuilt)


def replay_range(
    source: str, dataset: str, start: date, end: date, calendar: TradingDays
) -> EtlRun:
    """Span by span in date order; rows that only exist on change are cleared before rebuilding."""
    adapter = replay_adapter(source, dataset, calendar)
    adapter.require_dataset(dataset)
    spans = list(adapter.units(dataset, start, end))
    with tracked_run(source, dataset, start, end) as run:
        run.checkpoint = {"kind": "reprocess"}
        for span in spans:
            for first, last, raw, frame in _replayed(adapter, dataset, *span):
                with transaction.atomic():
                    _clear(source, dataset, first, last, raw)
                    run.rows += adapter.upsert(dataset, frame)
                run.checkpoint = {"kind": "reprocess", "last_date": last.isoformat()}
                run.save(update_fields=["rows", "checkpoint"])
    return run


def rebuild_after(source: str, dataset: str, day: date, calendar: TradingDays) -> list[date]:
    """Replay every fetched day after the given one in order; days never fetched are skipped."""
    end = min(calendar.end, timezone.localdate())
    if day >= end:
        return []
    later = [other for other in calendar.between(day, end) if other > day]
    fetched = covered_days(source, dataset, later)
    rebuilt: list[date] = []
    for is_fetched, group in groupby(later, key=lambda other: other in fetched):
        span = list(group)
        if is_fetched:
            replay_range(source, dataset, span[0], span[-1], calendar)
            rebuilt.extend(span)
    return rebuilt


def _replayed(
    adapter: BaseAdapter, dataset: str, first: date, last: date
) -> Iterator[tuple[date, date, pd.DataFrame, pd.DataFrame]]:
    """The span as one stored answer, else day by day as daily runs stored a monthly dataset."""
    try:
        raw = adapter.fetch(dataset, first, last)
    except MissingRawError:
        if first == last:
            raise
        for day in adapter.trading_days(first, last):
            daily = adapter.fetch(dataset, day, day)
            yield day, day, daily, adapter.normalize(dataset, daily)
        return
    yield first, last, raw, adapter.normalize(dataset, raw)


def _clear(source: str, dataset: str, first: date, last: date, raw: pd.DataFrame) -> None:
    """Share rows of the codes the answer lists; full delivery rows of the market's codes."""
    if dataset == SHARES:
        listed = raw[SHARE_CODE_FIELDS[source]].astype(str).str.strip()
        SharesOutstanding.objects.filter(
            stock_id__in=list(listed), effective_date__range=(first, last)
        ).delete()
    elif dataset == FULL_DELIVERY:
        codes = StockMaster.objects.filter(market=REPLAYABLE[source]).values("stock_id")
        DispositionEvent.objects.filter(
            stock_id__in=codes, type=dataset, start_date__range=(first, last)
        ).delete()
