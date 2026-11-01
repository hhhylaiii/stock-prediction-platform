"""Size, liquidity, risk and industry features of SPEC §6 from the universe panel."""

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import pandas as pd

from apps.research.features.common import (
    ANNUAL_ROOT,
    carried,
    feature_frame,
    filled_close,
    flat_returns,
    rolling,
    shifted,
)
from apps.research.features.registry import Group, feature_names
from apps.research.panel import require_sorted

AMIHUD_DAYS = 20
BETA_DAYS, MIN_BETA_PAIRS = 250, 60
IDIO_DAYS, MIN_IDIO_PAIRS = 60, 20
SECTOR_DAYS = 20
INDUSTRY_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "其他電子類": "其他電子業",
        "居家生活類": "居家生活",
        "數位雲端類": "數位雲端",
        "綠能環保類": "綠能環保",
        "運動休閒類": "運動休閒",
        "觀光事業": "觀光餐旅",
        "金融業": "金融保險",
    }
)


def canonical_industry(industry: pd.Series) -> pd.Series:
    """TWSE names for TPEx industries; empty names become None (missing)."""
    names = industry.fillna("").astype(str).str.strip().replace(dict(INDUSTRY_ALIASES))
    return names.astype(object).where(names != "", None)


def size_liquidity_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Market cap from the raw close, the universe's median turnover and Amihud illiquidity."""
    require_sorted(panel)
    raw_close = carried(panel["close"], panel)
    cap = raw_close * panel["shares"]
    turnover = panel["turnover"].where(panel["has_bar"].astype(bool) & panel["close"].notna())
    impact = flat_returns(filled_close(panel), panel).abs() / turnover.where(turnover > 0)
    columns = {
        "size_log_mcap": pd.Series(np.log(cap.where(cap > 0)), index=panel.index),
        "liq_dollar_vol_med_20d": panel["turnover_med_20d"],
        "liq_amihud_20d": rolling(impact, panel, AMIHUD_DAYS, "mean", 1),
    }
    return feature_frame(panel, columns, feature_names(Group.SIZE_LIQUIDITY))


def _regression(
    stock: pd.Series, market: pd.Series, panel: pd.DataFrame, window: int, minimum: int
) -> tuple[pd.Series, pd.Series]:
    """Rolling OLS slope and annualized residual std over days where both returns exist."""
    pair = stock.notna() & market.notna()
    x, y = market.where(pair), stock.where(pair)
    sums = {
        name: rolling(values, panel, window, "sum", 1)
        for name, values in {"n": pair.astype("float64"), "x": x, "y": y}.items()
    }
    n = sums["n"]
    xx = rolling(x * x, panel, window, "sum", 1) - sums["x"] ** 2 / n
    xy = rolling(x * y, panel, window, "sum", 1) - sums["x"] * sums["y"] / n
    yy = rolling(y * y, panel, window, "sum", 1) - sums["y"] ** 2 / n
    fitted = (n >= minimum) & (xx > 0)
    residual = ((yy - xy * xy / xx).clip(lower=0) / (n - 2)).pow(0.5) * ANNUAL_ROOT
    return (xy / xx).where(fitted), residual.where(fitted)


def industry_risk_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Beta and residual volatility against the market index, and industry returns.

    Unknown industries take the day's universe average, so a blank never marks a later delisting.
    Industry averages need the day's whole cross-section, so split loads by date, not by stock.
    """
    require_sorted(panel)
    stock = panel["adj_close"] / shifted(panel["adj_close"], panel, 1) - 1
    beta, _ = _regression(stock, panel["market_ret"], panel, BETA_DAYS, MIN_BETA_PAIRS)
    _, idio = _regression(stock, panel["market_ret"], panel, IDIO_DAYS, MIN_IDIO_PAIRS)
    close = filled_close(panel)
    own = close / shifted(close, panel, SECTOR_DAYS) - 1
    industry = canonical_industry(panel["industry"])
    counted = own.where(panel["in_universe"].astype(bool))
    whole = counted.groupby(panel["date"]).transform("mean")
    sector = counted.groupby([panel["date"], industry]).transform("mean").fillna(whole)
    columns = {
        "risk_beta_250d": beta,
        "risk_idio_vol_60d": idio,
        "ind_sector_ret_20d": sector,
        "ind_rel_ret_20d": own - sector,
    }
    return feature_frame(panel, columns, feature_names(Group.INDUSTRY_RISK))
