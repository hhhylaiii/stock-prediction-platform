"""Adapter contract: fetch to raw, normalize to the standard schema, upsert (SPEC §5)."""

import hashlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import date
from itertools import groupby
from typing import Any, ClassVar

import pandas as pd
from django.db import transaction
from django.utils import timezone

from apps.etl.errors import RejectedStocksError, SourceError
from apps.etl.models import EtlRun, RawRecord
from apps.etl.ratelimit import RateLimiter
from apps.etl.raw import store_raw
from apps.etl.runs import fail_on_rejections, stop_on_streak, tracked_run
from apps.market.calendar import CalendarError, NotATradingDayError, TradingDays

logger = logging.getLogger(__name__)


def stock_list_digest(stock_ids: Iterable[str]) -> str:
    return hashlib.sha256(",".join(sorted(set(stock_ids))).encode()).hexdigest()


def monthly_spans(days: Iterable[date]) -> Iterator[tuple[date, date]]:
    """First and last trading day of each calendar month the days touch."""
    for _, group in groupby(days, key=lambda day: (day.year, day.month)):
        month = list(group)
        yield month[0], month[-1]


class BaseAdapter(ABC):
    """One adapter per source; each dataset maps to one standard table."""

    source: ClassVar[str]
    raw_model: ClassVar[type[RawRecord]]
    datasets: ClassVar[frozenset[str]]
    monthly_datasets: ClassVar[frozenset[str]] = frozenset()
    per_stock: ClassVar[bool] = False

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

    def units(self, dataset: str, start: date, end: date) -> Iterator[tuple[date, date]]:
        """Spans fetched in one go: a trading day each, or a month for monthly datasets."""
        days = self.trading_days(start, end)
        if dataset in self.monthly_datasets:
            yield from monthly_spans(days)
        else:
            yield from ((day, day) for day in days)

    def run(self, dataset: str, start: date, end: date, stock_id: str | None = None) -> EtlRun:
        """Fetch, normalize and upsert span by span, recording progress in one etl_run row."""
        self.require_dataset(dataset)
        spans = list(self.units(dataset, start, end))
        with tracked_run(self.source, dataset, start, end) as run:
            for first, last in spans:
                raw = self.fetch(dataset, first, last, stock_id)
                run.rows += self.upsert(dataset, self.normalize(dataset, raw))
                run.checkpoint = {"last_date": last.isoformat()}
                run.save(update_fields=["rows", "checkpoint"])
        return run

    def backfill(
        self,
        dataset: str,
        stock_ids: Sequence[str],
        start: date,
        end: date,
        resume_after: str | None = None,
        selection: str | list[str] = "all",
        rejected: Mapping[str, str] | None = None,
        guard_streaks: bool = True,
    ) -> EtlRun:
        """Stock by stock; a stock with unusable source data is skipped and recorded."""
        self.require_dataset(dataset)
        if not self.per_stock:
            raise ValueError(f"{self.source} cannot fetch one stock at a time")
        pending = [
            code for code in sorted(set(stock_ids)) if resume_after is None or code > resume_after
        ]
        with tracked_run(self.source, dataset, start, end) as run:
            checkpoint = {
                "kind": "backfill",
                "selection": selection,
                "stocks": len(set(stock_ids)),
                "digest": stock_list_digest(stock_ids),
                "last_stock_id": resume_after,
            }
            found = dict(rejected or {})
            run.checkpoint = {**checkpoint, "rejected": found}
            run.save(update_fields=["checkpoint"])
            streak: list[str] = []
            for stock_id in pending:
                try:
                    run.rows += self._load_stock(dataset, start, end, stock_id)
                    streak = []
                except SourceError as exc:
                    found[stock_id] = f"{type(exc).__name__}: {exc}"
                    streak.append(stock_id)
                    logger.warning("%s %s: rejected %s", self.source, dataset, found[stock_id])
                run.checkpoint = {**checkpoint, "last_stock_id": stock_id, "rejected": found}
                run.save(update_fields=["rows", "checkpoint"])
                try:
                    stop_on_streak(streak if guard_streaks else [], found)
                except RejectedStocksError:
                    run.checkpoint = _before_streak(checkpoint, pending, streak, found)
                    raise
            run.checkpoint = {**run.checkpoint, "finished": True}
            fail_on_rejections(found)
        return run

    def _load_stock(self, dataset: str, start: date, end: date, stock_id: str) -> int:
        raw = self.fetch(dataset, start, end, stock_id)
        frame = self.normalize(dataset, raw)
        with transaction.atomic():
            return self.upsert(dataset, frame)


def _before_streak(
    checkpoint: dict[str, Any], pending: list[str], streak: list[str], found: dict[str, str]
) -> dict[str, Any]:
    """Rewind past a streak of rejections, likely an outage, so --resume fetches them again."""
    first = pending.index(streak[0])
    last = pending[first - 1] if first else checkpoint["last_stock_id"]
    kept = {code: reason for code, reason in found.items() if code not in streak}
    return {**checkpoint, "last_stock_id": last, "rejected": kept}
