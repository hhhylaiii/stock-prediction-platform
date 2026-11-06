"""Second-batch dividend feature of SPEC §6: trading days to the next announced cash ex-date."""

from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd

from apps.market.calendar import TradingDays
from apps.market.models import DividendEvent

EVENT_COLUMNS = ["stock_id", "announce_date", "ex_date"]


def load_ex_dates(start: date, end: date, stock_ids: Sequence[str] | None = None) -> pd.DataFrame:
    """Cash dividends announced by end that go ex after start; unannounced ones are unknown."""
    rows = DividendEvent.objects.filter(
        cash_dividend__gt=0, announce_date__lte=end, ex_date__gt=start
    ).order_by("stock_id", "ex_date")
    if stock_ids is not None:
        rows = rows.filter(stock_id__in=list(stock_ids))
    return pd.DataFrame(list(rows.values_list(*EVENT_COLUMNS)), columns=EVENT_COLUMNS)


def _active_days(events: pd.DataFrame, calendar: TradingDays) -> pd.DataFrame:
    """One row per event and trading day from its announcement to the day before it goes ex.

    An ex-date after the calendar's last trading day has no known trading-day count.
    """
    days = np.array(calendar.between(calendar.start, calendar.end), dtype="datetime64[D]")
    exits = np.searchsorted(days, events["ex_date"].to_numpy(dtype="datetime64[D]"))
    inside = events[exits < len(days)]
    first = np.searchsorted(days, inside["announce_date"].to_numpy(dtype="datetime64[D]"))
    ex = exits[exits < len(days)]
    spans = np.clip(ex - first, 0, None)
    positions = np.repeat(first, spans) + _offsets(spans)
    return pd.DataFrame(
        {
            "stock_id": np.repeat(inside["stock_id"].to_numpy(), spans),
            "date": days[positions].astype(object),
            "days": (np.repeat(ex, spans) - positions).astype("float64"),
        }
    )


def _offsets(spans: np.ndarray) -> np.ndarray:
    """0..span−1 for each span, concatenated."""
    starts = np.repeat(np.cumsum(spans) - spans, spans)
    return np.arange(int(spans.sum()), dtype=np.int64) - np.asarray(starts, dtype=np.int64)


def days_to_ex_date(panel: pd.DataFrame, events: pd.DataFrame, calendar: TradingDays) -> pd.Series:
    """Trading days from each row's day to the nearest announced ex-date after it; else NaN."""
    active = _active_days(events, calendar)
    nearest = active.groupby(["stock_id", "date"])["days"].min()
    keys = pd.MultiIndex.from_arrays([panel["stock_id"], panel["date"]])
    return pd.Series(nearest.reindex(keys).to_numpy(dtype=np.float64), index=panel.index)


def attach_ex_dates(
    panel: pd.DataFrame, calendar: TradingDays, start: date, end: date
) -> pd.DataFrame:
    """New panel with the trading days to the next announced cash ex-date."""
    distance = days_to_ex_date(panel, load_ex_dates(start, end), calendar)
    return panel.assign(fund_days_to_ex_date=distance)
