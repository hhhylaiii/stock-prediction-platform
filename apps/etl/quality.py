"""Daily data-quality checks of SPEC §5, written to data_quality_log."""

from collections.abc import Sequence
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from django.db import models, transaction

from apps.etl.models import DataQualityLog
from apps.market.calendar import NotATradingDayError, OutsideCalendarError, TradingDays
from apps.market.models import (
    DailyChip,
    DailyPrice,
    DailyValuation,
    DividendEvent,
    IndexDaily,
    IndexId,
    StockMaster,
)

Severity = DataQualityLog.Severity
ROW_COUNT = "row_count"
BIG_MOVE = "big_move"
MISSING = "missing_values"
CALENDAR = "trading_calendar"
FRESHNESS = "index_freshness"
CHECKS = (ROW_COUNT, BIG_MOVE, MISSING, CALENDAR, FRESHNESS)
ROW_CHANGE_LIMIT = 0.05
BIG_MOVE_LIMIT = Decimal("0.10")
MISSING_LIMIT = 0.01
MISSING_ERROR_LIMIT = 0.5
MIN_MARKET_POPULATION = 20
UNKNOWN_MARKET = "unknown"
NO_TRADE_LIMIT = 0.05
BASELINE_DAYS = timedelta(days=31)
STALE_AFTER_DAYS = {
    IndexId.TAIEX.value: 0,
    IndexId.TPEX.value: 0,
    IndexId.SOX.value: 4,
    IndexId.TSM_ADR.value: 4,
    IndexId.USDTWD.value: 4,
}
RATIO_DIGITS = 4
ROW = "row"
TABLES: dict[str, type[models.Model]] = {
    "daily_price": DailyPrice,
    "daily_chip": DailyChip,
    "daily_valuation": DailyValuation,
}
MISSING_COLUMNS: dict[str, tuple[str, ...]] = {
    "daily_price": (ROW, "close"),
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
COLUMN_LIMITS = {("daily_price", "close"): NO_TRADE_LIMIT}


def run_checks(day: date, calendar: TradingDays) -> list[DataQualityLog]:
    """Run every check for a trading day, replacing their earlier results for it."""
    if not calendar.is_trading_day(day):
        raise NotATradingDayError(f"{day} is not a trading day")
    previous = _previous_session(calendar, day)
    found = [
        *row_count_results(day, previous),
        big_move_result(day, previous),
        *missing_results(day),
        calendar_result(day, previous, calendar),
        freshness_result(day),
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
        if count == 0 and model._default_manager.filter(date__lt=day).exists():
            severity = Severity.ERROR
        elif not before:
            severity = Severity.INFO
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


def missing_results(day: date) -> list[DataQualityLog]:
    """Per column, stocks valued on its fullest recent day but not today, by market too."""
    delisted = set(
        StockMaster.objects.filter(delist_date__lte=day).values_list("stock_id", flat=True)
    )
    markets = dict(StockMaster.objects.values_list("stock_id", "market"))
    results = []
    for table, columns in MISSING_COLUMNS.items():
        model = TABLES[table]
        gaps = {column: _gap(model, column, day, delisted, markets) for column in columns}
        shares = {column: _share(gap) for column, gap in gaps.items()}
        if any(_market_failed(gap) for gap in gaps.values()):
            severity = Severity.ERROR
        elif any(
            share > COLUMN_LIMITS.get((table, column), MISSING_LIMIT)
            for column, share in shares.items()
        ):
            severity = Severity.WARNING
        else:
            severity = Severity.INFO
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


def freshness_result(day: date) -> DataQualityLog:
    """Market series whose latest close is older than allowed; US series trail by a day or more."""
    recent = (
        IndexDaily.objects.filter(date__lte=day, date__gt=day - BASELINE_DAYS)
        .values("index_id")
        .annotate(latest=models.Max("date"))
    )
    latest = {entry["index_id"]: entry["latest"] for entry in recent}
    stale = {
        index_id: latest[index_id].isoformat() if index_id in latest else None
        for index_id, allowed in STALE_AFTER_DAYS.items()
        if index_id not in latest or (day - latest[index_id]).days > allowed
    }
    severity = Severity.WARNING if stale else Severity.INFO
    return _result(day, FRESHNESS, severity, {"stale": stale})


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
    delisted: set[str],
    markets: dict[str, str],
) -> dict[str, Any]:
    baseline = _baseline(model, column, day)
    before = set() if baseline is None else _codes_with(model, column, baseline) - delisted
    missing = before - _codes_with(model, column, day)
    by_market: dict[str, dict[str, int]] = {}
    for code in before:
        group = by_market.setdefault(
            markets.get(code, UNKNOWN_MARKET), {"missing": 0, "population": 0}
        )
        group["population"] += 1
        group["missing"] += code in missing
    return {"missing": len(missing), "population": len(before), "markets": by_market}


def _share(gap: dict[str, Any]) -> float:
    return int(gap["missing"]) / int(gap["population"]) if gap["population"] else 0.0


def _market_failed(gap: dict[str, Any]) -> bool:
    """Either exchange's adapter failing leaves most of that market's stocks without values."""
    groups = [gap, *gap["markets"].values()]
    return any(
        group["population"] >= MIN_MARKET_POPULATION and _share(group) > MISSING_ERROR_LIMIT
        for group in groups
    )


def _baseline(model: type[models.Model], column: str, day: date) -> date | None:
    """The day within the past month with the most values in the column, the latest on ties."""
    rows = model._default_manager.filter(date__lt=day, date__gte=day - BASELINE_DAYS)
    if column != ROW:
        rows = rows.filter(**{f"{column}__isnull": False})
    fullest = rows.values("date").annotate(n=models.Count("pk")).order_by("-n", "-date").first()
    return None if fullest is None else fullest["date"]


def _codes_with(model: type[models.Model], column: str, day: date) -> set[str]:
    rows = model._default_manager.filter(date=day)
    if column != ROW:
        rows = rows.filter(**{f"{column}__isnull": False})
    return set(rows.values_list("stock_id", flat=True))


def _result(day: date, check: str, severity: str, detail: dict[str, Any]) -> DataQualityLog:
    return DataQualityLog(date=day, check_name=check, severity=severity, detail=detail)


def replace_day_log(day: date, check: str, severity: str, detail: dict[str, Any] | None) -> None:
    """One row per day and check: a rerun replaces it, and nothing to report removes it."""
    with transaction.atomic():
        DataQualityLog.objects.filter(date=day, check_name=check).delete()
        if detail is not None:
            DataQualityLog.objects.create(
                date=day, check_name=check, severity=severity, detail=detail
            )


def alerting(results: Sequence[DataQualityLog]) -> list[DataQualityLog]:
    return [row for row in results if row.severity != Severity.INFO]
