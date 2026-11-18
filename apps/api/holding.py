"""The main shadow book's latest list and fill, and its run beside the universe and the index."""

import logging
import math
from datetime import date
from typing import Any

from django.db.models import Case, F, IntegerField, OuterRef, QuerySet, Subquery, Value, When
from rest_framework.request import Request
from rest_framework.response import Response

from apps.api.envelope import envelope
from apps.api.serving import latest_served_date
from apps.api.views import ReadOnlyView
from apps.market.models import IndexDaily, IndexId, StockMaster
from apps.research.holding_series import Fill, Point, cumulative
from apps.research.models import (
    OUTPUT_HORIZONS,
    BookAction,
    BookPhase,
    HoldingBookAction,
    HoldingBookDay,
)

logger = logging.getLogger(__name__)

MAIN_BOOK = "H70N30-C0"
RULE_NAME = "H70/N30"
BOOK_START = date(2026, 11, 16)
SHORT = min(OUTPUT_HORIZONS)
DIGITS = 6
LEDGER_INVALID = "ledger_invalid"
LIST_ORDER = (BookAction.SELL, BookAction.KEEP, BookAction.BUY, BookAction.BACKUP)
LIST_FIELDS = ("signal_date", "trade_date", "status", "backfilled", "held_before", "sells")
LIST_COUNTS = ("buys", "backups")
FILL_FIELDS = ("signal_date", "trade_date", "status", "value", "cash", "held", "paid", "traded")


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, DIGITS)


def named(lines: QuerySet[HoldingBookAction], *fields: str) -> list[dict[str, Any]]:
    names = StockMaster.objects.filter(stock_id=OuterRef("stock_id")).values("name")[:1]
    return list(lines.annotate(name=Subquery(names)).values("stock_id", "name", *fields))


def main_days(phase: str) -> QuerySet[HoldingBookDay]:
    return HoldingBookDay.objects.filter(book=MAIN_BOOK, phase=phase).order_by("-signal_date")


def list_part(day: HoldingBookDay | None) -> dict[str, Any] | None:
    """The latest list with each line in sell, keep, buy, backup order, highest percentile first."""
    if day is None:
        return None
    rank = Case(
        *(When(action=action, then=Value(k)) for k, action in enumerate(LIST_ORDER)),
        output_field=IntegerField(),
    )
    lines = HoldingBookAction.objects.filter(
        book=MAIN_BOOK, signal_date=day.signal_date, phase=BookPhase.LIST
    ).order_by(rank, "priority", F("pct").desc(nulls_last=True), "stock_id")
    return {
        **{field: getattr(day, field) for field in (*LIST_FIELDS, *LIST_COUNTS)},
        "actions": [
            {**line, "pct": rounded(line["pct"])}
            for line in named(lines, "action", "priority", "pct")
        ],
    }


def fill_part(day: HoldingBookDay | None) -> dict[str, Any] | None:
    """The latest fill's book figures and the stocks it could not sell."""
    if day is None:
        return None
    stuck = HoldingBookAction.objects.filter(
        book=MAIN_BOOK, signal_date=day.signal_date, phase=BookPhase.FILL, action=BookAction.STUCK
    ).order_by("stock_id")
    figures = {field: getattr(day, field) for field in FILL_FIELDS}
    return {
        **figures,
        **{field: rounded(figures[field]) for field in ("value", "cash", "paid", "traded")},
        "backfilled": day.backfilled,
        "stuck": named(stuck),
    }


def since_part() -> dict[str, Any]:
    """Every fill as cumulative points; a ledger the series refuses is logged and left blank.

    A fill without a value is handed on as NaN, which the series refuses like any bad value.
    """
    rows = list(
        main_days(BookPhase.FILL).values("signal_date", "trade_date", "value", "universe_return")
    )
    closes = {
        row["date"]: float(row["close"])
        for row in IndexDaily.objects.filter(
            index_id=IndexId.TAIEX_TR, date__in=[row["trade_date"] for row in rows]
        ).values("date", "close")
    }
    fills = [
        Fill(
            row["signal_date"],
            row["trade_date"],
            math.nan if row["value"] is None else row["value"],
            row["universe_return"],
            closes.get(row["trade_date"]),
        )
        for row in rows
    ]
    try:
        points = cumulative(fills)
    except ValueError as error:
        logger.error("holding book %s cannot be compared: %s", MAIN_BOOK, error)
        return {"points": [], "no_prediction_days": 0, "error": LEDGER_INVALID}
    return {
        "points": [point_part(point) for point in points],
        "no_prediction_days": no_prediction_days(fills),
        "error": None,
    }


def point_part(point: Point) -> dict[str, Any]:
    return {
        "trade_date": point.trade_date,
        "book": rounded(point.book),
        "universe": rounded(point.universe),
        "index": rounded(point.index),
    }


def no_prediction_days(fills: list[Fill]) -> int:
    """Fills after the first whose list had an empty universe, whose universe return is so 0."""
    later = sorted(fills, key=lambda fill: fill.trade_date)[1:]
    if not later:
        return 0
    empty = set(
        main_days(BookPhase.LIST)
        .filter(universe=[], signal_date__in=[fill.signal_date for fill in later])
        .values_list("signal_date", flat=True)
    )
    return sum(fill.signal_date in empty for fill in later)


class HoldingView(ReadOnlyView):
    def get(self, request: Request) -> Response:
        data = {
            "book": MAIN_BOOK,
            "rule": RULE_NAME,
            "start": BOOK_START,
            "list": list_part(main_days(BookPhase.LIST).first()),
            "fill": fill_part(main_days(BookPhase.FILL).first()),
            "since": since_part(),
        }
        return envelope(data, {"predictions_date": latest_served_date(SHORT)})
