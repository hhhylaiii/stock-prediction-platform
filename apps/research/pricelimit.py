"""Daily price limits by era, rounded to the tick of the limit price (SPEC §6 技術面)."""

from collections.abc import Iterable
from datetime import date

import numpy as np
import numpy.typing as npt
import pandas as pd

from apps.research.panel import REUSE_GAP, require_sorted

LIMIT_CHANGE = date(2015, 6, 1)
OLD_LIMIT = 0.07
LIMIT = 0.10
NO_LIMIT_DAYS = 5
TICK_BANDS = ((10.0, 0.01), (50.0, 0.05), (100.0, 0.1), (500.0, 0.5), (1000.0, 1.0))
TOP_TICK = 5.0
ROUNDING_SLACK = 1e-9
MATCH_TOLERANCE = 1e-6

Prices = npt.NDArray[np.float64]


def tick_size(prices: Prices) -> Prices:
    """Exchange tick for each price: 0.01 under 10 up to 5 from 1,000."""
    conditions = [prices < upper for upper, _ in TICK_BANDS]
    return np.select(conditions, [tick for _, tick in TICK_BANDS], default=TOP_TICK)


def _rates(days: Iterable[date]) -> Prices:
    return np.array([OLD_LIMIT if day < LIMIT_CHANGE else LIMIT for day in days])


def limit_up_price(reference: Prices, days: Iterable[date]) -> Prices:
    """Reference × (1 + era limit), rounded down to the tick of the result."""
    raw = reference * (1 + _rates(days))
    tick = tick_size(raw)
    return np.round(np.floor(raw / tick + ROUNDING_SLACK) * tick, 2)


def limit_down_price(reference: Prices, days: Iterable[date]) -> Prices:
    """Reference × (1 − era limit), rounded up to the tick of the result."""
    raw = reference * (1 - _rates(days))
    tick = tick_size(raw)
    return np.round(np.ceil(raw / tick - ROUNDING_SLACK) * tick, 2)


def apply_price_limits(panel: pd.DataFrame) -> pd.DataFrame:
    """New panel with reference, limit prices and flags; closes over REUSE_GAP rows back expire."""
    require_sorted(panel)
    keys = [panel["stock_id"], panel["segment"]]
    previous = panel["adj_close"].groupby(keys).shift(1).groupby(keys).ffill(limit=REUSE_GAP)
    reference = (previous / panel["adj_factor"]).mask(panel["age"] < NO_LIMIT_DAYS)
    values = reference.to_numpy(dtype=np.float64)
    up = pd.Series(limit_up_price(values, panel["date"]), index=panel.index)
    down = pd.Series(limit_down_price(values, panel["date"]), index=panel.index)
    return panel.assign(
        reference=reference,
        limit_up=up,
        limit_down=down,
        limit_up_close=panel["close"] >= up - MATCH_TOLERANCE,
        limit_down_close=panel["close"] <= down + MATCH_TOLERANCE,
        limit_up_open=panel["open"] >= up - MATCH_TOLERANCE,
    )
