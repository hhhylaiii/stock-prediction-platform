from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd

from tests.research.conftest import weekday_calendar

CALENDAR = weekday_calendar(date(2014, 1, 1), date(2016, 12, 31))


def listing(
    stock_id: str,
    closes: Sequence[float],
    start: date = date(2014, 1, 2),
    segment: int = 1,
    **columns: object,
) -> pd.DataFrame:
    """One listing on consecutive weekdays; a NaN close is a halted day without a bar."""
    days = CALENDAR.between(start, CALENDAR.end)[: len(closes)]
    close = pd.Series(closes, dtype="float64")
    has_bar = close.notna()
    base: dict[str, object] = {
        "stock_id": stock_id,
        "date": days,
        "segment": segment,
        "market": "TWSE",
        "industry": "半導體業",
        "has_bar": has_bar,
        "close": close,
        "adj_close": close,
        "adj_high": close,
        "adj_low": close,
        "volume": np.where(has_bar, 1_000.0, np.nan),
        "turnover": close * 1_000.0,
        "shares": 1_000_000.0,
        "limit_up_close": False,
        "limit_down_close": False,
        "attention": False,
        "in_universe": True,
        "turnover_med_20d": 20_000_000.0,
        "market_ret": 0.0,
    }
    return pd.DataFrame({**base, **columns})


def rising(n: int, first: float = 100.0) -> list[float]:
    return [first + k for k in range(n)]


def at(features: pd.DataFrame, stock_id: str, row: int, column: str) -> float:
    """Feature value on the listing's row-th day."""
    rows = features[features["stock_id"] == stock_id].reset_index(drop=True)
    return float(rows[column].iloc[row])
