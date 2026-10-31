"""Adjusted prices are raw price × adj_factor, computed on read and never stored (SPEC §4, §8.2)."""

from collections.abc import Iterable
from datetime import date
from decimal import Decimal

import pandas as pd
from django.db import connection

from apps.market.models import DividendEvent

PRICE_COLUMNS = ("open", "high", "low", "close")
REFRESH_SQL = """
WITH factors AS (
    SELECT p.date, COALESCE(EXP(SUM(LN(e.adj_factor))), 1.0) AS adj
    FROM daily_price p
    LEFT JOIN dividend_event e
        ON e.stock_id = p.stock_id AND e.ex_date > p.date AND e.adj_factor IS NOT NULL
    WHERE p.stock_id = %(stock)s AND p.date >= %(since)s
    GROUP BY p.date
)
UPDATE daily_price p
SET adj_factor = factors.adj
FROM factors
WHERE p.stock_id = %(stock)s AND p.date = factors.date AND p.adj_factor <> factors.adj
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


def refresh_adj_factor(stock_id: str, since: date = date.min) -> int:
    """Set adj_factor from since on to the product of later ex-date factors; returns changes."""
    with connection.cursor() as cursor:
        cursor.execute(REFRESH_SQL, {"stock": stock_id, "since": since})
        return int(cursor.rowcount)


def refresh_where_later_events(stock_ids: Iterable[str], since: date) -> int:
    """Refresh only stocks whose events after since carry a factor, e.g. after new prices."""
    stocks = (
        DividendEvent.objects.filter(
            stock_id__in=list(stock_ids), ex_date__gt=since, adj_factor__isnull=False
        )
        .values_list("stock_id", flat=True)
        .distinct()
    )
    return sum(refresh_adj_factor(stock_id, since) for stock_id in stocks)
