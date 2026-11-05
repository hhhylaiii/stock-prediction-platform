"""Return-to-price conversion in the service layer, each stock on its own close (SPEC §2, §7)."""

import math
from dataclasses import dataclass

Z_80 = 1.28
TRADING_DAYS_PER_YEAR = 252
QUANTILE = "quantile"
VOLATILITY = "volatility"


@dataclass(frozen=True)
class Interval:
    low: float
    high: float
    ret_low: float
    ret_high: float
    source: str


def price_at(close: float, ret: float) -> float:
    return close * (1 + ret)


def quantile_interval(close: float, q10: float, q90: float) -> Interval:
    return Interval(price_at(close, q10), price_at(close, q90), q10, q90, QUANTILE)


def volatility_interval(close: float, vol: float, horizon: int) -> Interval:
    """close × exp(±1.28 × annualised vol × √(horizon / 252))."""
    spread = Z_80 * vol * math.sqrt(horizon / TRADING_DAYS_PER_YEAR)
    down, up = math.exp(-spread), math.exp(spread)
    return Interval(close * down, close * up, down - 1, up - 1, VOLATILITY)


def choose_interval(
    close: float,
    horizon: int,
    *,
    q10: float | None,
    q90: float | None,
    vol: float | None,
    quantile_alert: bool,
) -> Interval | None:
    """Quantiles first; the volatility band while their model is on a coverage alert or missing."""
    if q10 is not None and q90 is not None and not quantile_alert:
        return quantile_interval(close, q10, q90)
    if vol is not None:
        return volatility_interval(close, vol, horizon)
    return None
