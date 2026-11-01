"""Daily data-quality checks of SPEC §5, written to data_quality_log."""

from collections.abc import Sequence
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from django.db import models, transaction

from apps.etl.models import DataQualityLog
from apps.market.calendar import OutsideCalendarError, TradingDays
from apps.market.models import DailyChip, DailyPrice, DailyValuation, DividendEvent, StockMaster

Severity = DataQualityLog.Severity
ROW_COUNT = "row_count"
BIG_MOVE = "big_move"
MISSING = "missing_values"
CALENDAR = "trading_calendar"
CHECKS = (ROW_COUNT, BIG_MOVE, MISSING, CALENDAR)
ROW_CHANGE_LIMIT = 0.05
BIG_MOVE_LIMIT = Decimal("0.10")
MISSING_LIMIT = 0.01
RATIO_DIGITS = 4
ROW = "row"
TABLES: dict[str, type[models.Model]] = {
    "daily_price": DailyPrice,
    "daily_chip": DailyChip,
    "daily_valuation": DailyValuation,
}
MISSING_COLUMNS: dict[str, tuple[str, ...]] = {
    "daily_price": (ROW,),
    "daily_chip": (
        "fi_net",
        "it_net",
        "dealer_self_net",
        "dealer_hedge_net",
        "margin_balance",
        "margin_limit",
        "short_balance",
        "sbl_short_balance",
        "fi_holding_pct",
        "day_trade_volume",
    ),
    "daily_valuation": ("pb", "dividend_yield"),
}


def run_checks(day: date, calendar: TradingDays) -> list[DataQualityLog]:
    """Run the four checks for a trading day, replacing their earlier results for it."""
    previous = _previous_session(calendar, day)
    found = [
        *row_count_results(day, previous),
        big_move_result(day, previous),
        *missing_results(day, previous),
        calendar_result(day, previous, calendar),
    ]
    with transaction.atomic():
        DataQualityLog.objects.filter(date=day, check_name__in=CHECKS).delete()
        return DataQualityLog.objects.bulk_create(found)


def should_run_prediction(day: date) -> bool:
    """True once every check ran for the day and nothing that day is an error."""
    ran = set(
        DataQualityLog.objects.filter(date=day, check_name__in=CHECKS).values_list(
            "check_name", flat=True
        )
    )
    errors = DataQualityLog.objects.filter(date=day, severity=Severity.ERROR).exists()
    return ran == set(CHECKS) and not errors


def row_count_results(day: date, previous: date | None) -> list[DataQualityLog]:
    """Rows per table against the previous session: 5% apart warns, an emptied table is an error."""
    results = []
    for table, model in TABLES.items():
        count = model._default_manager.filter(date=day).count()
        before = None if previous is None else model._default_manager.filter(date=previous).count()
        if not before:
            severity = Severity.INFO
        elif count == 0:
            severity = Severity.ERROR
        elif abs(count - before) / before > ROW_CHANGE_LIMIT:
            severity = Severity.WARNING
        else:
            severity = Severity.INFO
        detail = {"table": table, "count": count, "previous": before}
        results.append(_result(day, ROW_COUNT, severity, detail))
    return results


def big_move_result(day: date, previous: date | None) -> DataQualityLog:
    """Closes more than 10% off the previous session's with no ex-dividend that day, for review."""
    stocks: dict[str, float] = {}
    if previous is not None:
        before = _closes(previous)
        paid = set(DividendEvent.objects.filter(ex_date=day).values_list("stock_id", flat=True))
        for code, close in _closes(day).items():
            prior = before.get(code)
            if prior is None or code in paid:
                continue
            change = close / prior - 1
            if abs(change) > BIG_MOVE_LIMIT:
                stocks[code] = round(float(change), RATIO_DIGITS)
    detail = {"table": "daily_price", "count": len(stocks), "stocks": stocks}
    return _result(day, BIG_MOVE, Severity.INFO, detail)


def missing_results(day: date, previous: date | None) -> list[DataQualityLog]:
    """Per column, stocks that had a value the previous session and none today."""
    delisted = set(
        StockMaster.objects.filter(delist_date__lte=day).values_list("stock_id", flat=True)
    )
    results = []
    for table, columns in MISSING_COLUMNS.items():
        model = TABLES[table]
        gaps = {column: _gap(model, column, day, previous, delisted) for column in columns}
        over = any(
            gap["population"] and gap["missing"] / gap["population"] > MISSING_LIMIT
            for gap in gaps.values()
        )
        severity = Severity.WARNING if over else Severity.INFO
        results.append(_result(day, MISSING, severity, {"table": table, "columns": gaps}))
    return results


def calendar_result(day: date, previous: date | None, calendar: TradingDays) -> DataQualityLog:
    """Rows dated on closed days since the previous session reject the batch."""
    first = calendar.start if previous is None else previous + timedelta(days=1)
    closed = [first + timedelta(days=n) for n in range((day - first).days)]
    rows: dict[str, dict[str, int]] = {}
    for table, model in TABLES.items():
        counts = (
            model._default_manager.filter(date__in=closed)
            .values("date")
            .annotate(rows=models.Count("pk"))
        )
        found = {entry["date"].isoformat(): entry["rows"] for entry in counts}
        if found:
            rows[table] = found
    severity = Severity.ERROR if rows else Severity.INFO
    return _result(day, CALENDAR, severity, {"rows": rows})


def _previous_session(calendar: TradingDays, day: date) -> date | None:
    try:
        return calendar.shift(day, -1)
    except OutsideCalendarError:
        return None


def _closes(day: date) -> dict[str, Decimal]:
    rows = DailyPrice.objects.filter(date=day).values_list("stock_id", "close")
    return {code: close for code, close in rows if close}


def _gap(
    model: type[models.Model],
    column: str,
    day: date,
    previous: date | None,
    delisted: set[str],
) -> dict[str, int]:
    if previous is None:
        return {"missing": 0, "population": 0}
    before = _codes_with(model, column, previous) - delisted
    missing = before - _codes_with(model, column, day)
    return {"missing": len(missing), "population": len(before)}


def _codes_with(model: type[models.Model], column: str, day: date) -> set[str]:
    rows = model._default_manager.filter(date=day)
    if column != ROW:
        rows = rows.filter(**{f"{column}__isnull": False})
    return set(rows.values_list("stock_id", flat=True))


def _result(day: date, check: str, severity: str, detail: dict[str, Any]) -> DataQualityLog:
    return DataQualityLog(date=day, check_name=check, severity=severity, detail=detail)


def alerting(results: Sequence[DataQualityLog]) -> list[DataQualityLog]:
    return [row for row in results if row.severity != Severity.INFO]
