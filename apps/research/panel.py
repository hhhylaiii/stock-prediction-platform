"""Research price panel: common stocks on a trading-day grid per listing, adjusted on read."""

from collections.abc import Sequence
from datetime import date, timedelta
from itertools import groupby
from operator import itemgetter
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from django.db import connection

from apps.market.calendar import NotATradingDayError, TradingDays
from apps.market.universe import COMMON_STOCK_PATTERN, DEPOSITARY_INDUSTRY, DEPOSITARY_PREFIX

REUSE_GAP = 250
PRICE_COLUMNS = ("open", "high", "low", "close")
PANEL_COLUMNS = (
    "stock_id",
    "date",
    "segment",
    "age",
    "market",
    "industry",
    "has_bar",
    *PRICE_COLUMNS,
    "volume",
    "turnover",
    "adj_factor",
    *(f"adj_{column}" for column in PRICE_COLUMNS),
)
BREAKS_SQL = """
WITH bars AS (
    SELECT p.stock_id, p.date, m.market, m.industry, m.list_date, m.delist_date,
           LAG(p.date) OVER (PARTITION BY p.stock_id ORDER BY p.date) AS previous,
           MAX(p.date) OVER (PARTITION BY p.stock_id) AS last_day
    FROM daily_price p
    JOIN stock_master m ON m.stock_id = p.stock_id
    WHERE m.stock_id ~ %(pattern)s
      AND m.stock_id NOT LIKE %(depositary)s
      AND m.industry <> %(depositary_industry)s
      AND (m.list_date IS NULL OR p.date >= m.list_date)
      AND (m.delist_date IS NULL OR p.date < m.delist_date)
      AND (%(stocks)s::text[] IS NULL OR p.stock_id = ANY(%(stocks)s::text[]))
)
SELECT stock_id, previous, date, last_day, market, industry, list_date, delist_date
FROM bars
WHERE previous IS NULL OR date - previous > %(gap)s
ORDER BY stock_id, date
"""
BARS_SQL = """
SELECT p.stock_id, p.date, p.open::float8 AS open, p.high::float8 AS high,
       p.low::float8 AS low, p.close::float8 AS close, p.volume::float8 AS volume,
       p.turnover::float8 AS turnover, p.adj_factor
FROM daily_price p
JOIN stock_master m ON m.stock_id = p.stock_id
WHERE p.date BETWEEN %(start)s AND %(end)s
  AND m.stock_id ~ %(pattern)s
  AND m.stock_id NOT LIKE %(depositary)s
  AND m.industry <> %(depositary_industry)s
  AND (m.list_date IS NULL OR p.date >= m.list_date)
  AND (m.delist_date IS NULL OR p.date < m.delist_date)
  AND (%(stocks)s::text[] IS NULL OR p.stock_id = ANY(%(stocks)s::text[]))
"""


class Segment(NamedTuple):
    """Trading days one listing spans; listing_day is None when it predates the data.

    Industry comes from stock_master, which describes only the code's latest listing.
    """

    stock_id: str
    segment: int
    first_day: date
    last_day: date
    listing_day: date | None
    market: str
    industry: str


def _fetch(sql: str, params: dict[str, Any]) -> tuple[list[str], list[tuple[Any, ...]]]:
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        return [column.name for column in cursor.description], cursor.fetchall()


def _filters(stock_ids: Sequence[str] | None) -> dict[str, Any]:
    return {
        "pattern": COMMON_STOCK_PATTERN,
        "depositary": DEPOSITARY_PREFIX,
        "depositary_industry": DEPOSITARY_INDUSTRY,
        "stocks": None if stock_ids is None else list(stock_ids),
    }


def _data_range() -> tuple[date | None, date | None]:
    with connection.cursor() as cursor:
        cursor.execute("SELECT MIN(date), MAX(date) FROM daily_price")
        first, last = cursor.fetchone()
        return first, last


def require_sorted(panel: pd.DataFrame) -> None:
    """Refuse panels whose rows are not in date order within each stock or repeat a day."""
    ascending = panel.groupby("stock_id", sort=False)["date"].is_monotonic_increasing.all()
    if not ascending or panel.duplicated(["stock_id", "date"]).any():
        raise ValueError("panel rows must be unique and sorted by date within each stock")


def _split(calendar: TradingDays, rows: list[tuple[Any, ...]]) -> list[tuple[date, date]]:
    starts: list[date] = [rows[0][2]]
    ends: list[date] = []
    for _, previous, current, *_ in rows[1:]:
        if len(calendar.between(previous, current)) - 2 > REUSE_GAP:
            ends.append(previous)
            starts.append(current)
    ends.append(rows[0][3])
    return list(zip(starts, ends, strict=True))


def _span(
    number: int, bars: list[tuple[date, date]], row: tuple[Any, ...], data: tuple[date, date]
) -> tuple[date, date, date | None]:
    first_bar, last_bar = bars[number - 1]
    list_date, delist_date = row[6], row[7]
    first_day, last_day = first_bar, last_bar
    listing_day: date | None = first_bar
    if number == 1 and list_date is not None:
        first_day, listing_day = max(list_date, data[0]), list_date
    elif number == 1 and first_bar == data[0]:
        listing_day = None
    if number == len(bars):
        last_day = data[1] if delist_date is None else delist_date - timedelta(days=1)
    return first_day, last_day, listing_day


def _segments(calendar: TradingDays, stock_ids: Sequence[str] | None) -> list[Segment]:
    _, breaks = _fetch(BREAKS_SQL, {**_filters(stock_ids), "gap": REUSE_GAP})
    data = _data_range()
    if data[0] is None or data[1] is None:
        return []
    segments = []
    for stock_id, group in groupby(breaks, key=itemgetter(0)):
        rows = list(group)
        bars = _split(calendar, rows)
        for number in range(1, len(bars) + 1):
            first_day, last_day, listing_day = _span(number, bars, rows[0], (data[0], data[1]))
            market = rows[0][4]
            industry = rows[0][5] if number == len(bars) else ""
            segments.append(
                Segment(stock_id, number, first_day, last_day, listing_day, market, industry)
            )
    return segments


def listing_segments(calendar: TradingDays, stock_ids: Sequence[str] | None = None) -> pd.DataFrame:
    """Listings per common stock, split where bars stop for more than REUSE_GAP trading days."""
    return pd.DataFrame(_segments(calendar, stock_ids), columns=list(Segment._fields))


def _grid(calendar: TradingDays, segments: list[Segment], start: date, end: date) -> pd.DataFrame:
    ordinals = np.array([day.toordinal() for day in calendar.between(calendar.start, calendar.end)])
    parts = []
    for segment in segments:
        low, high = max(segment.first_day, start), min(segment.last_day, end)
        if low > high:
            continue
        window = calendar.between(low, high)
        positions = np.searchsorted(ordinals, [day.toordinal() for day in window])
        listing = segment.listing_day
        origin = 0 if listing is None else int(np.searchsorted(ordinals, listing.toordinal()))
        parts.append(
            pd.DataFrame(
                {
                    "stock_id": segment.stock_id,
                    "date": window,
                    "segment": segment.segment,
                    "age": positions - origin,
                    "market": segment.market,
                    "industry": segment.industry,
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=["stock_id", "date", "segment", "age", "market", "industry"])
    return pd.concat(parts, ignore_index=True)


def _require_trading_days(bars: pd.DataFrame, calendar: TradingDays) -> None:
    for day in bars["date"].unique():
        if not calendar.is_trading_day(day):
            raise NotATradingDayError(f"daily_price has {day}, which is not a trading day")


def load_price_panel(
    calendar: TradingDays, start: date, end: date, stock_ids: Sequence[str] | None = None
) -> pd.DataFrame:
    """Common-stock bars on each listing's trading days; days without a close carry no prices."""
    start, end = max(start, calendar.start), min(end, calendar.end)
    segments = _segments(calendar, stock_ids)
    columns, rows = _fetch(BARS_SQL, {**_filters(stock_ids), "start": start, "end": end})
    bars = pd.DataFrame(rows, columns=columns)
    _require_trading_days(bars, calendar)
    grid = _grid(calendar, segments, start, end)
    panel = grid.merge(bars, on=["stock_id", "date"], how="left", indicator="has_bar")
    panel["has_bar"] = panel["has_bar"].eq("both")
    for column in (*PRICE_COLUMNS, "volume", "turnover", "adj_factor"):
        panel[column] = panel[column].astype("float64")
    for column in ("open", "high", "low"):
        panel[column] = panel[column].where(panel["close"].notna())
    for column in PRICE_COLUMNS:
        panel[f"adj_{column}"] = panel[column] * panel["adj_factor"]
    panel["segment"] = panel["segment"].astype("int64")
    panel["age"] = panel["age"].astype("int64")
    ordered = panel.sort_values(["stock_id", "date"], ignore_index=True)
    return ordered.loc[:, list(PANEL_COLUMNS)]
