"""The up label against the market: a 20-day return beyond TAIEX_TR's move by 5% (SPEC §7).

TAIEX_TR reinvests dividends like the adjusted returns the labels use (runbook).
"""

import hashlib
import math
from collections.abc import Iterable
from datetime import date

import pandas as pd

from apps.market.calendar import OutsideCalendarError, TradingDays
from apps.market.models import IndexDaily, IndexId
from apps.research.labels import UP_THRESHOLD, UP_TOLERANCE, VOL_DAYS

MARKET = IndexId.TAIEX_TR
RETURN = "fwd_ret_open"
RELATIVE_LABEL = "fwd_up_5pct_vs_taiex_tr"


def _later(calendar: TradingDays, day: date) -> date | None:
    try:
        return calendar.shift(day, VOL_DAYS)
    except OutsideCalendarError:
        return None


def market_returns(calendar: TradingDays, days: Iterable[date]) -> pd.Series:
    """TAIEX_TR close VOL_DAYS sessions after each day over its close that day; NaN without both."""
    chosen = sorted(set(days))
    if not chosen:
        return pd.Series(dtype="float64")
    ends = {day: _later(calendar, day) for day in chosen}
    reach = max([day for day in ends.values() if day is not None], default=chosen[-1])
    rows = IndexDaily.objects.filter(index_id=MARKET, date__range=(chosen[0], reach))
    closes = {day: float(close) for day, close in rows.values_list("date", "close")}
    values = []
    for day in chosen:
        end = ends[day]
        first, last = closes.get(day), None if end is None else closes.get(end)
        values.append(math.nan if first is None or last is None else last / first - 1)
    return pd.Series(values, index=chosen, dtype="float64")


def relative_up(labels: pd.DataFrame, calendar: TradingDays) -> pd.Series:
    """Whether each row's 20-day open return beats TAIEX_TR's 20 days by UP_THRESHOLD."""
    market = labels["date"].map(market_returns(calendar, labels["date"]))
    excess = labels[RETURN] - market
    flags = (excess > UP_THRESHOLD + UP_TOLERANCE).astype(object)
    return flags.where(excess.notna(), None)


def market_digest(start: date, end: date) -> str:
    """SHA-256 of the TAIEX_TR closes dated in [start, end]."""
    rows = IndexDaily.objects.filter(index_id=MARKET, date__range=(start, end)).order_by("date")
    listed = ";".join(f"{day}={close}" for day, close in rows.values_list("date", "close"))
    return hashlib.sha256(f"{start}..{end} {listed}".encode()).hexdigest()
