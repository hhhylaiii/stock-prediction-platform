"""Adjusted prices are raw price × adj_factor, computed on read and never stored (SPEC §4, §8.2)."""

from collections.abc import Iterable
from datetime import date
from decimal import Decimal
from typing import Any

import pandas as pd
from django.db import connection, transaction
from django.utils import timezone

from apps.etl.models import DataQualityLog
from apps.market.models import DividendEvent

PRICE_COLUMNS = ("open", "high", "low", "close")
ADJ_CHECK = "adj_factor_changed"
REBUILT = "rebuilt"
ADJ_STOCK_LIMIT = 200
RATIO_TOLERANCE = 1e-9
REFRESH_SQL = """
WITH factors AS (
    SELECT p.date, p.adj_factor AS old, COALESCE(EXP(SUM(LN(e.adj_factor))), 1.0) AS adj
    FROM daily_price p
    LEFT JOIN dividend_event e
        ON e.stock_id = p.stock_id AND e.ex_date > p.date AND e.adj_factor IS NOT NULL
    WHERE p.stock_id = %(stock)s AND p.date >= %(since)s
    GROUP BY p.date, p.adj_factor
),
changed AS (
    UPDATE daily_price p
    SET adj_factor = factors.adj
    FROM factors
    WHERE p.stock_id = %(stock)s AND p.date = factors.date AND p.adj_factor <> factors.adj
    RETURNING p.date
),
ratios AS (
    SELECT date, adj / old AS ratio, LAG(adj / old) OVER (ORDER BY date) AS before
    FROM factors
)
SELECT
    (SELECT count(*) FROM changed),
    (
        SELECT min(date) FROM ratios
        WHERE before IS NOT NULL AND abs(ratio - before) > %(tolerance)s * abs(before)
    )
"""


def adjusted(price: Decimal | float | None, adj_factor: float) -> float | None:
    return None if price is None else float(price) * adj_factor


def adjust_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """New frame on the same index holding adj_<col> = <col> × adj_factor for each OHLC column."""
    if "adj_factor" not in frame.columns:
        raise KeyError("adj_factor column is required")
    factor = frame["adj_factor"].astype(float)
    return pd.DataFrame(
        {f"adj_{col}": frame[col].astype(float) * factor for col in PRICE_COLUMNS if col in frame},
        index=frame.index,
    )


def refresh_adj_factor(stock_id: str, since: date = date.min, note: bool = True) -> int:
    """Set adj_factor from since on to the product of later ex-date factors; returns changes.

    Rows before the first session whose factor moved apart from the day before's only scale
    as one, so their ratios stand; that session is noted for the research layer to rebuild.
    """
    params: dict[str, str | date | float] = {
        "stock": stock_id,
        "since": since,
        "tolerance": RATIO_TOLERANCE,
    }
    with connection.cursor() as cursor:
        cursor.execute(REFRESH_SQL, params)
        changed, earliest = cursor.fetchone()
    if note and earliest is not None:
        note_adj_change(stock_id, earliest)
    return int(changed)


def refresh_where_later_events(stock_ids: Iterable[str], since: date) -> int:
    """Give new price rows the factor of later events; their stocks' history is left as it was."""
    stocks = (
        DividendEvent.objects.filter(
            stock_id__in=list(stock_ids), ex_date__gt=since, adj_factor__isnull=False
        )
        .values_list("stock_id", flat=True)
        .distinct()
    )
    return sum(refresh_adj_factor(stock_id, since, note=False) for stock_id in stocks)


def _merged(detail: dict[str, Any], stock_id: str, earliest: date) -> dict[str, Any]:
    day = earliest.isoformat()
    if "earliest" in detail:
        return {"earliest": min(detail["earliest"], day), "count": detail["count"] + 1}
    stocks = dict(detail.get("stocks", {}))
    stocks[stock_id] = min(stocks.get(stock_id, day), day)
    if len(stocks) > ADJ_STOCK_LIMIT:
        return {"earliest": min(stocks.values()), "count": len(stocks)}
    return {"stocks": stocks}


def note_adj_change(stock_id: str, earliest: date) -> None:
    """Add the stock's first moved session to today's open adj_factor_changed row."""
    today = timezone.localdate()
    with transaction.atomic():
        row = (
            DataQualityLog.objects.select_for_update()
            .filter(date=today, check_name=ADJ_CHECK)
            .exclude(detail__has_key=REBUILT)
            .order_by("pk")
            .first()
        )
        if row is None:
            DataQualityLog.objects.create(
                date=today,
                check_name=ADJ_CHECK,
                severity=DataQualityLog.Severity.INFO,
                detail=_merged({}, stock_id, earliest),
            )
            return
        row.detail = _merged(row.detail, stock_id, earliest)
        row.save(update_fields=["detail"])
