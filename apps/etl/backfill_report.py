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
    (MonthlyRevenue, "period", "stock_id"),
    (FinancialStatement, "period", "stock_id"),
    (WeeklyHolding, "data_date", "stock_id"),
    (IndexDaily, "date", None),
    (MacroIndicator, "period", None),
)
LIMIT_CHANGE = date(2015, 6, 1)
OLD_LIMIT = 0.07
LIMIT = 0.10
JUMPS_SQL = """
WITH bars AS (
    SELECT stock_id, date, close * adj_factor AS adjusted,
           LAG(close * adj_factor) OVER w AS previous,
           ROW_NUMBER() OVER w AS nth
    FROM daily_price
    WHERE close IS NOT NULL
    WINDOW w AS (PARTITION BY stock_id ORDER BY date)
)
SELECT b.stock_id, b.date, b.previous, b.adjusted, b.adjusted / b.previous - 1 AS change
FROM bars b
WHERE b.date >= %(since)s AND b.nth > %(free_days)s AND b.previous > 0
  AND ABS(b.adjusted / b.previous - 1)
      > CASE WHEN b.date < %(limit_change)s THEN %(old_limit)s ELSE %(limit)s END + %(tolerance)s
  AND NOT EXISTS (
      SELECT 1 FROM dividend_event e WHERE e.stock_id = b.stock_id AND e.ex_date = b.date
  )
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
    """Adjusted daily moves past the price limit with no event that day (SPEC §8.2)."""
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
        return [Jump(row[0], row[1], *map(float, row[2:])) for row in cursor.fetchall()]


def latest_backfills() -> list[EtlRun]:
    """The newest whole-market backfill of each dataset."""
    runs = EtlRun.objects.filter(checkpoint__kind="backfill", checkpoint__selection="all")
    newest: dict[tuple[str, str], EtlRun] = {}
    for run in runs.order_by("run_id"):
        newest[run.source, run.dataset] = run
    return [newest[key] for key in sorted(newest)]
