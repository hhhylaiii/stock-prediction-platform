"""FinMind's zero foreign holdings that the exchanges' stored daily pages contradict (plan K21)."""

import logging
from collections import defaultdict
from collections.abc import Callable, Mapping
from datetime import date
from typing import Any, NamedTuple

from django.db import connection, transaction

from apps.etl.adapters import tpex_tables, twse_tables
from apps.etl.adapters.exchange_tables import DailyTable, source_rows, standard_rows
from apps.etl.errors import SourceError
from apps.etl.models import DataQualityLog, RawRecord, RawTpex, RawTwse
from apps.market.common_stock import (
    COMMON_STOCK_PATTERN,
    DEPOSITARY_INDUSTRY,
    DEPOSITARY_PREFIX,
)
from apps.market.models import DailyChip

logger = logging.getLogger(__name__)
REPLACED_CHECK = "fi_holding_zero_replaced"
HOLDING = "fi_holding_pct"
LONG_RUN = 20
WRONG_TENTHS = 9
COVERED_TENTHS = 9
WHOLE_RUN = "whole_run"
SHORT_RUN = "short_run"
MIXED_RUN = "mixed_run"
THIN_RUN = "thin_run"
REPLACED_KINDS = (WHOLE_RUN, SHORT_RUN)
NO_PAGE = "no_page"
UNREADABLE = "unreadable"
NOT_LISTED = "not_listed"
ZERO_RUNS_SQL = """
select stock_id, date, run from (
    select stock_id, date, fi_holding_pct = 0 as zero,
        row_number() over (partition by stock_id order by date)
        - row_number() over (partition by stock_id, fi_holding_pct = 0 order by date) as run
    from daily_chip
    where fi_holding_pct is not null and stock_id ~ %s and stock_id not like %s
        and stock_id not in (select stock_id from stock_master where industry = %s)
) ordered
where zero
order by stock_id, date
"""


class Page(NamedTuple):
    """Where one exchange's stored daily holdings page is kept and how its date is keyed."""

    source: str
    raw_model: type[RawRecord]
    table: DailyTable
    check: Callable[[dict[str, Any], date], None]
    key: str


class Seen(NamedTuple):
    value: float
    source: str
    raw_id: int


class Holdings(NamedTuple):
    """What the stored pages say, the days a market's page is missing, and unreadable days."""

    seen: dict[tuple[str, date], Seen]
    missing: set[date]
    unreadable: set[date]


class Replacement(NamedTuple):
    stock_id: str
    date: date
    value: float
    source: str
    raw_id: int
    kind: str


PAGES = (
    Page(
        "twse", RawTwse, twse_tables.TABLES["foreign_holding"], twse_tables.check_answer, "%Y%m%d"
    ),
    Page(
        "tpex", RawTpex, tpex_tables.TABLES["foreign_holding"], tpex_tables.check_answer, "%Y/%m/%d"
    ),
)


def replace_zero_holdings(dry_run: bool = False) -> dict[str, Any]:
    """Wrong zero runs and short zeros take the exchange's value; mixed and thin runs stay."""
    runs = zero_runs()
    wanted: dict[date, set[str]] = defaultdict(set)
    for (code, _), days in runs.items():
        for day in days:
            wanted[day].add(code)
    holdings = exchange_holdings(wanted)
    found, kept = classify(runs, holdings.seen)
    written = found if dry_run else _write(found)
    left = _tally(kept, lambda row: row.kind)
    return {
        "dry_run": dry_run,
        "replaced": _tally(written, lambda row: f"{row.source}/{row.kind}"),
        "left": {
            MIXED_RUN: left.get(MIXED_RUN, {}),
            THIN_RUN: left.get(THIN_RUN, {}),
            "unchecked": _unchecked(runs, holdings),
        },
    }


def zero_runs() -> dict[tuple[str, int], list[date]]:
    """Each common stock's runs of consecutive stored zeros; depositary receipts are left out."""
    runs: dict[tuple[str, int], list[date]] = defaultdict(list)
    with connection.cursor() as cursor:
        cursor.execute(
            ZERO_RUNS_SQL, [COMMON_STOCK_PATTERN, DEPOSITARY_PREFIX, DEPOSITARY_INDUSTRY]
        )
        for stock_id, day, run in cursor.fetchall():
            runs[(stock_id, run)].append(day)
    return dict(runs)


def exchange_holdings(wanted: Mapping[date, set[str]]) -> Holdings:
    """The holdings each day's latest stored exchange page gives the codes asked for that day."""
    seen: dict[tuple[str, date], Seen] = {}
    missing: set[date] = set()
    unreadable: set[date] = set()
    for day in sorted(wanted):
        for page in PAGES:
            stored = (
                page.raw_model._default_manager.filter(
                    dataset="foreign_holding", request_params__date=f"{day:{page.key}}"
                )
                .order_by("-fetched_at")
                .first()
            )
            if stored is None:
                missing.add(day)
                continue
            try:
                frame = standard_rows(
                    source_rows(stored.payload, day, page.table, page.check), page.table
                )
            except SourceError as exc:
                logger.warning(
                    "%s raw %s for %s is unreadable: %s", page.source, stored.pk, day, exc
                )
                unreadable.add(day)
                continue
            for code, value in zip(frame["stock_id"], frame[HOLDING], strict=True):
                if code in wanted[day] and value is not None:
                    seen[(code, day)] = Seen(float(value), page.source, stored.pk)
    return Holdings(seen, missing, unreadable)


def classify(
    runs: Mapping[tuple[str, int], list[date]], seen: Mapping[tuple[str, date], Seen]
) -> tuple[list[Replacement], list[Replacement]]:
    """Replacements, and the contradicted days of mixed and thin runs that stay zero.

    A run of 20 or more zeros is wrong when pages cover 90% of its days and the exchange is above
    zero on 90% of those; a shorter run takes every day the exchange is above zero.
    """
    found: list[Replacement] = []
    kept: list[Replacement] = []
    for (code, _), days in runs.items():
        checked = [(day, seen[(code, day)]) for day in days if (code, day) in seen]
        wrong = [(day, hit) for day, hit in checked if hit.value > 0]
        if not wrong:
            continue
        if len(days) < LONG_RUN:
            kind = SHORT_RUN
        elif 10 * len(wrong) < WRONG_TENTHS * len(checked):
            kind = MIXED_RUN
        elif 10 * len(checked) < COVERED_TENTHS * len(days):
            kind = THIN_RUN
        else:
            kind = WHOLE_RUN
        rows = [
            Replacement(code, day, hit.value, hit.source, hit.raw_id, kind) for day, hit in wrong
        ]
        (found if kind in REPLACED_KINDS else kept).extend(rows)
    return found, kept


def _unchecked(runs: Mapping[tuple[str, int], list[date]], holdings: Holdings) -> dict[str, int]:
    """Why the zeros without an exchange value have none."""
    counts = dict.fromkeys((NO_PAGE, UNREADABLE, NOT_LISTED), 0)
    for (code, _), days in runs.items():
        for day in days:
            if (code, day) in holdings.seen:
                continue
            if day in holdings.unreadable:
                counts[UNREADABLE] += 1
            elif day in holdings.missing:
                counts[NO_PAGE] += 1
            else:
                counts[NOT_LISTED] += 1
    return counts


def _write(found: list[Replacement]) -> list[Replacement]:
    by_code: dict[str, dict[date, Replacement]] = defaultdict(dict)
    for row in found:
        by_code[row.stock_id][row.date] = row
    with transaction.atomic():
        written: list[Replacement] = []
        for code, rows in by_code.items():
            stored = list(
                DailyChip.objects.filter(stock_id=code, date__in=list(rows), fi_holding_pct=0)
            )
            for chip in stored:
                chip.fi_holding_pct = rows[chip.date].value
                written.append(rows[chip.date])
            DailyChip.objects.bulk_update(stored, [HOLDING])
        DataQualityLog.objects.bulk_create(_logs(written))
    return written


def _logs(written: list[Replacement]) -> list[DataQualityLog]:
    """One log per replaced row, unless that stock and day were logged before."""
    logged = set(
        DataQualityLog.objects.filter(
            check_name=REPLACED_CHECK, date__in={row.date for row in written}
        ).values_list("date", "detail__stock_id")
    )
    return [
        DataQualityLog(
            date=row.date,
            check_name=REPLACED_CHECK,
            severity=DataQualityLog.Severity.INFO,
            detail={
                "stock_id": row.stock_id,
                "kind": row.kind,
                "old": 0.0,
                "new": row.value,
                "source": row.source,
                "raw_id": row.raw_id,
            },
        )
        for row in written
        if (row.date, row.stock_id) not in logged
    ]


def _tally(rows: list[Replacement], key: Callable[[Replacement], str]) -> dict[str, dict[str, int]]:
    stocks: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        stocks[key(row)].add(row.stock_id)
        counts[key(row)] += 1
    return {name: {"stocks": len(stocks[name]), "rows": counts[name]} for name in sorted(counts)}
