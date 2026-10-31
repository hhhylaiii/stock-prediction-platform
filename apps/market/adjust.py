"""Adjusted prices are raw price × adj_factor, computed on read and never stored (SPEC §4, §8.2)."""

from decimal import Decimal

import pandas as pd

PRICE_COLUMNS = ("open", "high", "low", "close")


def adjusted(price: Decimal | float | None, adj_factor: float) -> float | None:
    return None if price is None else float(price) * adj_factor


def adjust_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with adj_<col> = <col> × adj_factor for every OHLC column present."""
    if "adj_factor" not in frame.columns:
        raise KeyError("adj_factor column is required")
    factor = frame["adj_factor"].astype(float)
    adjusted_columns = {
        f"adj_{col}": frame[col].astype(float) * factor for col in PRICE_COLUMNS if col in frame
    }
    return frame.assign(**adjusted_columns)
