"""Adapter contract: fetch to raw, normalize to the standard schema, upsert (SPEC §5)."""

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from datetime import date
from typing import Any, ClassVar

import pandas as pd
from django.utils import timezone

from apps.etl.models import EtlRun, RawRecord
from apps.etl.ratelimit import RateLimiter
from apps.etl.runs import tracked_run
from apps.market.calendar import CalendarError, NotATradingDayError, TradingDays


class SourceError(Exception):
    """A source answered, but not with usable data."""


class SourceNotReadyError(SourceError):
    """The source has not published data for a trading day yet; retry later."""


class SchemaChangedError(SourceError):
    """The source response no longer matches the expected layout; raw data is kept."""


class SourceRequestError(SourceError):
    """The source refused the request (a 4xx other than 429); retrying will not help."""


def payload_hash(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def store_raw(
    raw_model: type[RawRecord], dataset: str, params: Mapping[str, Any], payload: Any
) -> RawRecord:
    """Keep one row per distinct response; refetching the same payload adds nothing."""
    record, _ = raw_model._default_manager.get_or_create(
        dataset=dataset,
        source_hash=payload_hash(payload),
        defaults={"request_params": dict(params), "payload": payload},
    )
    return record


class BaseAdapter(ABC):
    """One adapter per source; each dataset maps to one standard table."""

    source: ClassVar[str]
    raw_model: ClassVar[type[RawRecord]]
    datasets: ClassVar[frozenset[str]]

    def __init__(self, calendar: TradingDays, limiter: RateLimiter) -> None:
        self.calendar = calendar
        self.limiter = limiter

    @abstractmethod
    def fetch(
        self, dataset: str, start: date, end: date, stock_id: str | None = None
    ) -> pd.DataFrame:
        """Call the source, store each response in the raw table and return it as rows."""

    @abstractmethod
    def normalize(self, dataset: str, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Convert raw rows to the standard schema: column names, units and dates."""

    @abstractmethod
    def upsert(self, dataset: str, frame: pd.DataFrame) -> int:
        """Write standard rows by primary key and return how many were written."""

    def save_raw(self, dataset: str, params: Mapping[str, Any], payload: Any) -> RawRecord:
        return store_raw(self.raw_model, dataset, params, payload)

    def require_dataset(self, dataset: str) -> None:
        if dataset not in self.datasets:
            raise ValueError(
                f"{self.source} has no dataset {dataset!r}; choose from {sorted(self.datasets)}"
            )

    def trading_days(self, start: date, end: date) -> tuple[date, ...]:
        """Trading days to process; a single requested date must itself be a trading day."""
        today = timezone.localdate()
        if end > today:
            raise CalendarError(f"{end} is after today ({today}); nothing has been published")
        if start == end and not self.calendar.is_trading_day(start):
            raise NotATradingDayError(f"{start} is not a trading day")
        days = self.calendar.between(start, end)
        if not days:
            raise CalendarError(f"no trading days between {start} and {end}")
        return days

    def units(self, start: date, end: date) -> Iterator[date]:
        yield from self.trading_days(start, end)

    def run(self, dataset: str, start: date, end: date, stock_id: str | None = None) -> EtlRun:
        """Fetch, normalize and upsert day by day, recording progress in one etl_run row."""
        self.require_dataset(dataset)
        days = list(self.units(start, end))
        with tracked_run(self.source, dataset, start, end) as run:
            for day in days:
                raw = self.fetch(dataset, day, day, stock_id)
                run.rows += self.upsert(dataset, self.normalize(dataset, raw))
                run.checkpoint = {"last_date": day.isoformat()}
                run.save(update_fields=["rows", "checkpoint"])
        return run

    def backfill(
        self,
        dataset: str,
        stock_ids: Sequence[str],
        start: date,
        end: date,
        resume_after: str | None = None,
    ) -> EtlRun:
        """Fetch, normalize and upsert stock by stock; the checkpoint is the last finished stock."""
        self.require_dataset(dataset)
        pending = [
            code for code in sorted(set(stock_ids)) if resume_after is None or code > resume_after
        ]
        with tracked_run(self.source, dataset, start, end) as run:
            if resume_after is not None:
                run.checkpoint = {"last_stock_id": resume_after}
            for stock_id in pending:
                raw = self.fetch(dataset, start, end, stock_id)
                run.rows += self.upsert(dataset, self.normalize(dataset, raw))
                run.checkpoint = {"last_stock_id": stock_id}
                run.save(update_fields=["rows", "checkpoint"])
        return run
