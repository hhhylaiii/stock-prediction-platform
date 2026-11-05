"""Return-to-price conversion in the service layer, each stock on its own close (SPEC §2, §7)."""

import math
from dataclasses import dataclass

Z_80 = 1.28
TRADING_DAYS_PER_YEAR = 252
QUANTILE = "quantile"
VOLATILITY = "volatility"
TECH_VOL = "tech_vol_20d"
LOWEST_RETURN = -1.0


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


def volatility_interval(
    close: float, ret: float, vol: float, horizon: int, source: str = VOLATILITY
) -> Interval:
    """close × (1 + ret ± 1.28 × annualised vol × √(horizon / 252)), never below a zero price."""
    spread = Z_80 * vol * math.sqrt(horizon / TRADING_DAYS_PER_YEAR)
    low, high = max(ret - spread, LOWEST_RETURN), ret + spread
    return Interval(price_at(close, low), price_at(close, high), low, high, source)


def choose_interval(
    close: float,
    horizon: int,
    *,
    ret: float | None,
    q10: float | None,
    q90: float | None,
    vol: float | None,
    tech_vol: float | None,
    quantile_alert: bool,
) -> Interval | None:
    """Quantiles first; else a band around the predicted return from vol_pred, then tech_vol_20d."""
    if q10 is not None and q90 is not None and not quantile_alert:
        return quantile_interval(close, q10, q90)
    centre = 0.0 if ret is None else ret
    if vol is not None:
        return volatility_interval(close, centre, vol, horizon)
    if tech_vol is not None:
        return volatility_interval(close, centre, tech_vol, horizon, TECH_VOL)
    return None
