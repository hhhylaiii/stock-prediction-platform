"""Rows the exchanges' whole-market tables hold and the standard tables lack, filled (plan K17c)."""

from collections.abc import Callable, Mapping
from datetime import date
from typing import Any

import pandas as pd
from django.db import transaction
from django.db.models import Min

from apps.etl.adapters.base import BaseAdapter
from apps.etl.adapters.exchange_tables import DailyTable, standard_rows
from apps.etl.adapters.tpex_tables import TABLES as TPEX_TABLES
from apps.etl.adapters.twse_tables import TABLES as TWSE_TABLES
from apps.etl.models import DataQualityLog, EtlRun
from apps.etl.runs import failing_on_termination, tracked_run
from apps.etl.share_counts import latest_counts
from apps.market.models import (
    DailyChip,
    DailyPrice,
    DailyValuation,
    ShareChangeReason,
    SharesOutstanding,
    StockMaster,
)

FILL_CHECK = "exchange_fill"
FILLS: dict[str, tuple[str, tuple[str, ...]]] = {
    "holdings": ("foreign_holding", ("foreign_holding", "shares_outstanding")),
    "valuation": ("daily_valuation", ("daily_valuation",)),
}
TABLES: dict[str, dict[str, DailyTable]] = {"twse": TWSE_TABLES, "tpex": TPEX_TABLES}
Filled = dict[str, dict[str, list[Any]]]


class Shares:
    """Counts are stored only on change, and only before a stock's first stored count."""

    def __init__(self, windows: Mapping[str, str], day: date) -> None:
        self.windows = dict(windows)
        self.last = latest_counts(day)

    def fill(self, frame: pd.DataFrame, day: date) -> list[str]:
        rows = []
        for code, count in zip(frame["stock_id"], frame["shares"], strict=True):
            window = self.windows.get(code)
            if count is None or (window is not None and day.isoformat() >= window):
                continue
            if self.last.get(code) == count:
                continue
            self.last[code] = count
            rows.append(
                SharesOutstanding(
                    stock_id=code,
                    effective_date=day,
                    shares=count,
                    reason=ShareChangeReason.UNKNOWN.value,
                )
            )
        SharesOutstanding.objects.bulk_create(rows)
        return [row.stock_id for row in rows]


def fill_missing(
    adapter: BaseAdapter, fill: str, start: date, end: date, resume: EtlRun | None = None
) -> EtlRun:
    """Each trading day once: priced codes the market's table lists get the values they lack."""
    fetched, targets = FILLS[fill]
    tables = TABLES[adapter.source]
    windows = resume.checkpoint["windows"] if resume else first_counts()
    filled: Filled = resume.checkpoint["filled"] if resume else {}
    first = _after(adapter, resume) if resume else start
    days = adapter.trading_days(first, end)
    with failing_on_termination(), tracked_run(adapter.source, f"fill:{fill}", first, end) as run:
        shares = Shares(windows, first)
        for day in days:
            raw = adapter.fetch(fetched, day, day)
            codes = _priced(day)
            with transaction.atomic():
                for target in targets:
                    frame = standard_rows(raw, tables[target])
                    frame = frame.loc[frame["stock_id"].isin(codes)]
                    written = _write(target, frame, day, shares)
                    _tally(filled.setdefault(target, {}), written, day)
                    run.rows += len(written)
            run.checkpoint = {"last_date": day.isoformat(), "windows": windows, "filled": filled}
            run.save(update_fields=["rows", "checkpoint"])
        _log(adapter.source, fill, first, end, filled)
    return run


def first_counts() -> dict[str, str]:
    """Each stock's first stored count; fills stop there, and a resumed run keeps these."""
    firsts = SharesOutstanding.objects.values("stock_id").annotate(first=Min("effective_date"))
    return {row["stock_id"]: row["first"].isoformat() for row in firsts}


def _write(target: str, frame: pd.DataFrame, day: date, shares: Shares) -> list[str]:
    writers: dict[str, Callable[[], list[str]]] = {
        "foreign_holding": lambda: _fill_chip(frame, day, "fi_holding_pct"),
        "daily_valuation": lambda: _add_valuations(frame, day),
        "shares_outstanding": lambda: shares.fill(frame, day),
    }
    return writers[target]()


def _fill_chip(frame: pd.DataFrame, day: date, column: str) -> list[str]:
    """New rows for codes without one; a stored row gets the value only where it has none."""
    values = {
        code: value
        for code, value in zip(frame["stock_id"], frame[column], strict=True)
        if value is not None
    }
    stored = {row.stock_id: row for row in DailyChip.objects.filter(date=day, stock_id__in=values)}
    blank = [row for code, row in stored.items() if getattr(row, column) is None]
    for row in blank:
        setattr(row, column, values[row.stock_id])
    DailyChip.objects.bulk_update(blank, [column])
    new = [
        DailyChip(stock_id=code, date=day, **{column: value})
        for code, value in values.items()
        if code not in stored
    ]
    DailyChip.objects.bulk_create(new)
    return sorted([row.stock_id for row in blank] + [row.stock_id for row in new])


def _add_valuations(frame: pd.DataFrame, day: date) -> list[str]:
    """Rows only for codes without one that day, and only with some figure."""
    stored = set(
        DailyValuation.objects.filter(date=day, stock_id__in=list(frame["stock_id"])).values_list(
            "stock_id", flat=True
        )
    )
    rows = [
        DailyValuation(
            stock_id=record["stock_id"],
            date=day,
            pe=record["pe"],
            pb=record["pb"],
            dividend_yield=record["dividend_yield"],
        )
        for record in frame.to_dict("records")
        if record["stock_id"] not in stored
        and any(record[name] is not None for name in ("pe", "pb", "dividend_yield"))
    ]
    DailyValuation.objects.bulk_create(rows)
    return [row.stock_id for row in rows]


def _priced(day: date) -> set[str]:
    return set(DailyPrice.objects.filter(date=day).values_list("stock_id", flat=True))


def _after(adapter: BaseAdapter, run: EtlRun) -> date:
    last = date.fromisoformat(run.checkpoint["last_date"])
    return adapter.calendar.shift(last, 1) if adapter.calendar.is_trading_day(last) else last


def _tally(filled: dict[str, list[Any]], codes: list[str], day: date) -> None:
    for code in codes:
        first, _, count = filled.get(code, [day.isoformat(), None, 0])
        filled[code] = [first, day.isoformat(), count + 1]


def _log(source: str, fill: str, start: date, end: date, filled: Filled) -> None:
    """One info row per run: what was filled, split by whether the stock is delisted by now."""
    delisted = set(
        StockMaster.objects.filter(delist_date__isnull=False).values_list("stock_id", flat=True)
    )
    totals = {
        target: {
            "listed": sum(n for code, (_, _, n) in codes.items() if code not in delisted),
            "delisted": sum(n for code, (_, _, n) in codes.items() if code in delisted),
        }
        for target, codes in filled.items()
    }
    DataQualityLog.objects.create(
        date=end,
        check_name=FILL_CHECK,
        severity=DataQualityLog.Severity.INFO,
        detail={
            "source": source,
            "fill": fill,
            "span": [start.isoformat(), end.isoformat()],
            "totals": totals,
            "filled": filled,
        },
    )
