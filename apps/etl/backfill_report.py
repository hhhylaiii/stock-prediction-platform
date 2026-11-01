"""Numbers for the Phase 1 acceptance report: coverage, delisted stocks, price jumps, runs."""

from datetime import date
from typing import Any, NamedTuple

from django.db import connection, models
from django.db.models import Count, Max, Min

from apps.etl.models import EtlRun
from apps.market.models import (
    DailyChip,
    DailyPrice,
    DailyValuation,
    DispositionEvent,
    DividendEvent,
    FinancialStatement,
    IndexDaily,
    MacroIndicator,
    MonthlyRevenue,
    SharesOutstanding,
    StockMaster,
    TradingCalendar,
    WeeklyHolding,
)

COVERED: tuple[tuple[type[models.Model], str, str | None], ...] = (
    (StockMaster, "list_date", "stock_id"),
    (TradingCalendar, "date", None),
    (DailyPrice, "date", "stock_id"),
    (DailyValuation, "date", "stock_id"),
    (DailyChip, "date", "stock_id"),
    (SharesOutstanding, "effective_date", "stock_id"),
    (DividendEvent, "ex_date", "stock_id"),
    (DispositionEvent, "start_date", "stock_id"),
    (MonthlyRevenue, "period", "stock_id"),
    (FinancialStatement, "period", "stock_id"),
    (WeeklyHolding, "data_date", "stock_id"),
    (IndexDaily, "date", None),
    (MacroIndicator, "period", None),
)
LIMIT_CHANGE = date(2015, 6, 1)
OLD_LIMIT = 0.07
LIMIT = 0.10
FUND_PREFIX = "00"
JUMPS_SQL = """
WITH bars AS (
    SELECT stock_id, date, close * adj_factor AS adjusted,
           LAG(close * adj_factor) OVER w AS previous,
           ROW_NUMBER() OVER w AS nth,
           MIN(date) OVER (PARTITION BY stock_id) AS first_day
    FROM daily_price
    WHERE close IS NOT NULL
    WINDOW w AS (PARTITION BY stock_id ORDER BY date)
)
SELECT b.stock_id, b.date, b.previous, b.adjusted, b.adjusted / b.previous - 1 AS change,
       EXISTS (
           SELECT 1 FROM dividend_event e WHERE e.stock_id = b.stock_id AND e.ex_date = b.date
       ) AS event
FROM bars b
WHERE b.date >= %(since)s AND b.previous > 0
  AND NOT (b.nth <= %(free_days)s AND b.first_day > (SELECT MIN(date) FROM daily_price))
  AND ABS(b.adjusted / b.previous - 1)
      > CASE WHEN b.date < %(limit_change)s THEN %(old_limit)s ELSE %(limit)s END + %(tolerance)s
ORDER BY ABS(b.adjusted / b.previous - 1) DESC, b.stock_id, b.date
"""


class Coverage(NamedTuple):
    table: str
    rows: int
    stocks: int | None
    first: Any
    last: Any


class Jump(NamedTuple):
    stock_id: str
    day: date
    previous: float
    adjusted: float
    change: float
    event: bool

    @property
    def fund(self) -> bool:
        return self.stock_id.startswith(FUND_PREFIX)


class Backfill(NamedTuple):
    source: str
    dataset: str
    start: date
    end: date
    run_ids: list[int]
    rows: int
    latest: EtlRun
    outstanding: dict[str, str]


def coverage() -> list[Coverage]:
    found = []
    for model, field, stock in COVERED:
        numbers = model._default_manager.aggregate(
            rows=Count("pk"),
            first=Min(field),
            last=Max(field),
            **({"stocks": Count(stock, distinct=True)} if stock else {}),
        )
        found.append(
            Coverage(
                model._meta.db_table,
                numbers["rows"],
                numbers.get("stocks"),
                numbers["first"],
                numbers["last"],
            )
        )
    return found


def delisted_counts() -> tuple[int, int]:
    """Delisted stocks in stock_master, and how many of them have daily_price rows."""
    delisted = StockMaster.objects.filter(delist_date__isnull=False)
    priced = delisted.filter(stock_id__in=DailyPrice.objects.values("stock_id")).count()
    return delisted.count(), priced


def price_jumps(since: date, free_days: int, tolerance: float) -> list[Jump]:
    """Adjusted daily moves past the price limit; on an event day the factor itself is suspect."""
    params: dict[str, date | int | float] = {
        "since": since,
        "free_days": free_days,
        "limit_change": LIMIT_CHANGE,
        "old_limit": OLD_LIMIT,
        "limit": LIMIT,
        "tolerance": tolerance,
    }
    with connection.cursor() as cursor:
        cursor.execute(JUMPS_SQL, params)
        return [
            Jump(row[0], row[1], float(row[2]), float(row[3]), float(row[4]), bool(row[5]))
            for row in cursor.fetchall()
        ]


def backfills() -> list[Backfill]:
    """Whole-market backfills per range, rows summed over resumes, rejections net of retries."""
    runs = EtlRun.objects.filter(checkpoint__kind="backfill").order_by("run_id")
    groups: dict[tuple[str, str, date, date], list[EtlRun]] = {}
    for run in runs:
        groups.setdefault((run.source, run.dataset, run.start_date, run.end_date), []).append(run)
    found = []
    for key, members in sorted(groups.items()):
        whole = [run for run in members if run.checkpoint.get("selection") == "all"]
        if not whole:
            continue
        latest = whole[-1]
        later = [run for run in members if run.run_id > latest.run_id and run not in whole]
        found.append(
            Backfill(
                *key,
                run_ids=[run.run_id for run in whole],
                rows=sum(run.rows for run in whole),
                latest=latest,
                outstanding=_outstanding(latest, later),
            )
        )
    return found


def _outstanding(latest: EtlRun, later: list[EtlRun]) -> dict[str, str]:
    """Rejections left after later runs over chosen stocks that reached their last stock."""
    left = dict(latest.checkpoint.get("rejected", {}))
    for run in later:
        if not (run.status == EtlRun.Status.SUCCESS or run.checkpoint.get("finished")):
            continue
        rejected = run.checkpoint.get("rejected", {})
        for code in run.checkpoint.get("selection", []):
            if code in rejected:
                left[code] = rejected[code]
            else:
                left.pop(code, None)
    return left
