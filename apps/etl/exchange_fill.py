"""Rows the exchanges' whole-market tables hold and the standard tables lack, filled (plan K17c)."""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from datetime import date
from typing import Any, NamedTuple

import pandas as pd
from django.db import transaction
from django.db.models import Min

from apps.etl.adapters.base import BaseAdapter
from apps.etl.adapters.exchange_tables import DailyTable, standard_rows
from apps.etl.adapters.tpex_tables import TABLES as TPEX_TABLES
from apps.etl.adapters.twse_tables import TABLES as TWSE_TABLES
from apps.etl.errors import SourceNotReadyError
from apps.etl.models import DataQualityLog, EtlRun
from apps.etl.runs import failing_on_termination, tracked_run
from apps.etl.share_counts import drop_repeated_counts, latest_counts
from apps.market.models import (
    DailyChip,
    DailyPrice,
    DailyValuation,
    ShareChangeReason,
    SharesOutstanding,
    StockMaster,
)

FILL_CHECK = "exchange_fill"
HOLDINGS = "holdings"
FILLS: dict[str, tuple[str, tuple[str, ...]]] = {
    HOLDINGS: ("foreign_holding", ("foreign_holding", "shares_outstanding")),
    "valuation": ("daily_valuation", ("daily_valuation",)),
}
TABLES: dict[str, dict[str, DailyTable]] = {"twse": TWSE_TABLES, "tpex": TPEX_TABLES}
FIGURES = ("pe", "pb", "dividend_yield")
DIGITS = {"fi_holding_pct": 0.01, "pe": 0.01, "pb": 0.01, "dividend_yield": 1e-4}
SAMPLES = 20
Filled = dict[str, dict[str, list[Any]]]


class Written(NamedTuple):
    """Codes given a value on a stored row, codes given a new row, and stored values that differ."""

    filled: list[str]
    created: list[str]
    differ: list[list[Any]]


class Shares:
    """Counts are stored only on change, and only before a stock's first stored count."""

    def __init__(self, windows: Mapping[str, str], day: date) -> None:
        self.windows = dict(windows)
        self.last = latest_counts(day)

    def fill(self, frame: pd.DataFrame, day: date) -> Written:
        today = day.isoformat()
        firsts = self._firsts(today)
        rows, differ = [], []
        for code, count in zip(frame["stock_id"], frame["shares"], strict=True):
            if count is None:
                continue
            if code in firsts and firsts[code] != count:
                differ.append([code, today, "shares", firsts[code], count])
            window = self.windows.get(code)
            if (window is not None and today >= window) or self.last.get(code) == count:
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
        taken = SharesOutstanding.objects.filter(
            effective_date=day, stock_id__in=[row.stock_id for row in rows]
        ).values_list("stock_id", flat=True)
        new = [row for row in rows if row.stock_id not in set(taken)]
        SharesOutstanding.objects.bulk_create(new)
        return Written([], [row.stock_id for row in new], differ)

    def drop_repeats(self, codes: Iterable[str]) -> list[list[Any]]:
        """Once the bounds are passed, stored counts the fill now repeats go; each is listed."""
        bounded = sorted(code for code in set(codes) if code in self.windows)
        rows = SharesOutstanding.objects.filter(stock_id__in=bounded).order_by(
            "stock_id", "effective_date"
        )
        dropped: list[list[Any]] = []
        previous: SharesOutstanding | None = None
        for row in rows:
            if (
                previous is not None
                and previous.stock_id == row.stock_id
                and previous.shares == row.shares
                and row.reason == ShareChangeReason.UNKNOWN
            ):
                dropped.append([row.stock_id, row.effective_date.isoformat(), row.shares])
            previous = row
        drop_repeated_counts(bounded)
        return dropped

    def _firsts(self, today: str) -> dict[str, int]:
        """First stored counts taking effect today, to set beside the exchange's."""
        codes = [code for code, window in self.windows.items() if window == today]
        if not codes:
            return {}
        stored = SharesOutstanding.objects.filter(effective_date=today, stock_id__in=codes)
        return dict(stored.values_list("stock_id", "shares"))


def fill_missing(
    adapter: BaseAdapter, fill: str, start: date, end: date, resume: EtlRun | None = None
) -> EtlRun:
    """Each trading day once: priced codes the market's table lists get the values they lack."""
    fetched, targets = FILLS[fill]
    tables = TABLES[adapter.source]
    windows = share_windows() if fill == HOLDINGS else {}
    kept: dict[str, Any] = dict(resume.checkpoint) if resume else {}
    span = str(kept.get("start", start.isoformat()))
    first = _after(adapter, resume) if resume else start
    days = adapter.trading_days(first, end) if first <= end else ()
    with failing_on_termination(), tracked_run(adapter.source, f"fill:{fill}", first, end) as run:
        state: dict[str, Any] = {"filled": {}, "created": {}, "differences": {}, **kept}
        run.checkpoint = {**state, "start": span, "windows": windows}
        run.save(update_fields=["checkpoint"])
        shares = Shares(windows, first)
        for day in days:
            raw = adapter.fetch(fetched, day, day)
            codes = _priced(day)
            rows = run.rows
            with transaction.atomic():
                for target in targets:
                    frame = _kept(standard_rows(raw, tables[target]), codes, adapter.source, day)
                    written = _write(target, frame, day, shares)
                    state = _tallied(state, target, written, day)
                    rows += len(written.filled) + len(written.created)
                checkpoint = {**state, "start": span, "windows": windows, "last_date": str(day)}
                EtlRun.objects.filter(pk=run.pk).update(rows=rows, checkpoint=checkpoint)
            run.rows, run.checkpoint = rows, checkpoint
        with transaction.atomic():
            written_counts = state["created"].get("shares_outstanding", {})
            state = {
                **state,
                "dropped": state.get("dropped", []) + shares.drop_repeats(written_counts),
            }
            _log(adapter.source, fill, date.fromisoformat(span), end, state)
    return run


def share_windows() -> dict[str, str]:
    """The bounds the first holdings fill saw, so later and resumed fills keep the same ones."""
    earlier = (
        EtlRun.objects.filter(dataset=f"fill:{HOLDINGS}", checkpoint__has_key="windows")
        .order_by("started_at", "run_id")
        .first()
    )
    return dict(earlier.checkpoint["windows"]) if earlier else first_counts()


def first_counts() -> dict[str, str]:
    """Each stock's first stored count; fills stop there."""
    firsts = SharesOutstanding.objects.values("stock_id").annotate(first=Min("effective_date"))
    return {row["stock_id"]: row["first"].isoformat() for row in firsts}


def _write(target: str, frame: pd.DataFrame, day: date, shares: Shares) -> Written:
    writers: dict[str, Callable[[], Written]] = {
        "foreign_holding": lambda: _fill_chip(frame, day, "fi_holding_pct"),
        "daily_valuation": lambda: _add_valuations(frame, day),
        "shares_outstanding": lambda: shares.fill(frame, day),
    }
    return writers[target]()


def _fill_chip(frame: pd.DataFrame, day: date, column: str) -> Written:
    """New rows for codes without one; a stored row gets the value only where it has none."""
    values = {
        code: value
        for code, value in zip(frame["stock_id"], frame[column], strict=True)
        if value is not None
    }
    stored = {row.stock_id: row for row in DailyChip.objects.filter(date=day, stock_id__in=values)}
    differ = [
        [code, str(day), column, getattr(row, column), values[code]]
        for code, row in sorted(stored.items())
        if _differs(column, getattr(row, column), values[code])
    ]
    blank = [row for row in stored.values() if getattr(row, column) is None]
    for row in blank:
        setattr(row, column, values[row.stock_id])
    DailyChip.objects.bulk_update(blank, [column])
    new = [
        DailyChip(stock_id=code, date=day, **{column: value})
        for code, value in values.items()
        if code not in stored
    ]
    DailyChip.objects.bulk_create(new, ignore_conflicts=True)
    return Written(
        sorted(row.stock_id for row in blank), sorted(row.stock_id for row in new), differ
    )


def _add_valuations(frame: pd.DataFrame, day: date) -> Written:
    """Rows only for codes without one that day, and only with some figure."""
    records = {record["stock_id"]: record for record in frame.to_dict("records")}
    stored = {
        row.stock_id: row
        for row in DailyValuation.objects.filter(date=day, stock_id__in=list(records))
    }
    rows = [
        DailyValuation(stock_id=code, date=day, **{name: record[name] for name in FIGURES})
        for code, record in records.items()
        if code not in stored and any(record[name] is not None for name in FIGURES)
    ]
    DailyValuation.objects.bulk_create(rows, ignore_conflicts=True)
    differ = [
        [code, str(day), name, getattr(row, name), records[code][name]]
        for code, row in sorted(stored.items())
        for name in FIGURES
        if _differs(name, getattr(row, name), records[code][name])
    ]
    return Written([], [row.stock_id for row in rows], differ)


def _differs(name: str, stored: float | None, exchange: float | None) -> bool:
    """Apart by the exchange's last published digit or more."""
    if stored is None or exchange is None:
        return False
    return abs(stored - exchange) > DIGITS[name] / 2


def _kept(frame: pd.DataFrame, codes: set[str], source: str, day: date) -> pd.DataFrame:
    """The listed codes priced that day; a table listing codes none of which is priced waits."""
    kept = frame.loc[frame["stock_id"].isin(codes)]
    if kept.empty and not frame.empty:
        raise SourceNotReadyError(
            f"{source} {day}: none of the {len(frame)} listed codes is priced"
        )
    return kept


def _priced(day: date) -> set[str]:
    return set(DailyPrice.objects.filter(date=day).values_list("stock_id", flat=True))


def _after(adapter: BaseAdapter, run: EtlRun) -> date:
    """The day after the last one done; a run stopped before its first day starts over."""
    last = run.checkpoint.get("last_date")
    if last is None:
        return run.start_date
    done = date.fromisoformat(last)
    return adapter.calendar.shift(done, 1) if adapter.calendar.is_trading_day(done) else done


def _tallied(state: Mapping[str, Any], target: str, written: Written, day: date) -> dict[str, Any]:
    """The run's tallies with one day of one target added."""
    found = state["differences"].get(target, {"count": 0, "sample": [], "codes": {}})
    room = max(SAMPLES - len(found["sample"]), 0)
    codes = Counter(found.get("codes", {})) + Counter(entry[0] for entry in written.differ)
    return {
        **state,
        "filled": _counted(state["filled"], target, written.filled + written.created, day),
        "created": _counted(state["created"], target, written.created, day),
        "differences": {
            **state["differences"],
            target: {
                "count": found["count"] + len(written.differ),
                "sample": found["sample"] + written.differ[:room],
                "codes": dict(codes),
            },
        },
    }


def _counted(tally: Filled, target: str, codes: list[str], day: date) -> Filled:
    """Each code's first day, last day and number of days, with this day counted."""
    before = tally.get(target, {})
    today = day.isoformat()
    after = {
        code: [before[code][0], today, before[code][2] + 1] if code in before else [today, today, 1]
        for code in codes
    }
    return {**tally, target: {**before, **after}}


def _log(source: str, fill: str, start: date, end: date, state: Mapping[str, Any]) -> None:
    """One info row per run: what was filled, split by whether the stock is delisted by now."""
    delisted = set(
        StockMaster.objects.filter(delist_date__isnull=False).values_list("stock_id", flat=True)
    )
    totals = {
        target: {
            "listed": sum(n for code, (_, _, n) in codes.items() if code not in delisted),
            "delisted": sum(n for code, (_, _, n) in codes.items() if code in delisted),
        }
        for target, codes in state["filled"].items()
    }
    differing = {
        target: {
            "listed": sum(n for code, n in found["codes"].items() if code not in delisted),
            "delisted": sum(n for code, n in found["codes"].items() if code in delisted),
        }
        for target, found in state["differences"].items()
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
            "filled": state["filled"],
            "created": state["created"],
            "differences": {
                target: {"count": found["count"], "sample": found["sample"], **differing[target]}
                for target, found in state["differences"].items()
            },
            "dropped": state["dropped"],
        },
    )
