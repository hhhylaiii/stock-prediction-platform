"""Batch-1 technical features of SPEC §6 from the universe panel, per listing."""

import pandas as pd

from apps.research.features.common import (
    ANNUAL_ROOT,
    closes_seen,
    ewm_mean,
    feature_frame,
    filled_close,
    flat_returns,
    ratio,
    rolling,
    shifted,
    traded,
)
from apps.research.features.registry import Group, feature_names
from apps.research.panel import require_sorted

RETURN_DAYS = (1, 5, 20, 60)
MOMENTUM_SKIP, MOMENTUM_DAYS = 20, 250
AVERAGE_DAYS = (5, 20, 60)
VOLATILITY_DAYS = (20, 60)
VOLUME_SHORT, VOLUME_LONG = 5, 20
RSI_DAYS = 14
KD_DAYS, KD_SMOOTHING = 9, 3
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
YEAR_DAYS = 250
LIMIT_DAYS = 20

Columns = dict[str, pd.Series]


def _returns(panel: pd.DataFrame, close: pd.Series) -> Columns:
    columns = {f"tech_ret_{days}d": close / shifted(close, panel, days) - 1 for days in RETURN_DAYS}
    far, near = shifted(close, panel, MOMENTUM_DAYS), shifted(close, panel, MOMENTUM_SKIP)
    return {**columns, "tech_mom_250_20": near / far - 1}


def _trend(panel: pd.DataFrame, close: pd.Series) -> Columns:
    steps = flat_returns(close, panel)
    gaps = {
        f"tech_ma_gap_{days}d": close / rolling(close, panel, days, "mean", days) - 1
        for days in AVERAGE_DAYS
    }
    vols = {
        f"tech_vol_{days}d": rolling(steps, panel, days, "std", days) * ANNUAL_ROOT
        for days in VOLATILITY_DAYS
    }
    return {**gaps, **vols}


def _volume(panel: pd.DataFrame) -> Columns:
    volume = traded(panel, "volume")
    short = rolling(volume, panel, VOLUME_SHORT, "mean", VOLUME_SHORT)
    long = rolling(volume, panel, VOLUME_LONG, "mean", VOLUME_LONG)
    return {
        "tech_volume_ratio_5_20": ratio(short, long),
        "tech_turnover_rate_5d": ratio(short, panel["shares"]),
    }


def _oscillators(panel: pd.DataFrame, close: pd.Series) -> Columns:
    seen = closes_seen(close, panel)
    change = close - shifted(close, panel, 1)
    gain = ewm_mean(change.clip(lower=0), panel, 1 / RSI_DAYS)
    loss = ewm_mean((-change).clip(lower=0), panel, 1 / RSI_DAYS)
    high = rolling(panel["adj_high"], panel, KD_DAYS, "max", 1)
    low = rolling(panel["adj_low"], panel, KD_DAYS, "min", 1)
    stochastic = (100 * ratio(close - low, high - low)).where(seen >= KD_DAYS)
    k_line = ewm_mean(stochastic, panel, 1 / KD_SMOOTHING)
    line = ewm_mean(close, panel, 2 / (MACD_FAST + 1)) - ewm_mean(close, panel, 2 / (MACD_SLOW + 1))
    histogram = line - ewm_mean(line, panel, 2 / (MACD_SIGNAL + 1))
    return {
        "tech_rsi_14": (100 * ratio(gain, gain + loss)).where(seen > RSI_DAYS),
        "tech_kd_k": k_line,
        "tech_kd_d": ewm_mean(k_line, panel, 1 / KD_SMOOTHING),
        "tech_macd_hist_pct": (histogram / close).where(seen >= MACD_SLOW + MACD_SIGNAL - 1),
    }


def _range(panel: pd.DataFrame, close: pd.Series) -> Columns:
    high = rolling(panel["adj_high"], panel, YEAR_DAYS, "max", 1)
    low = rolling(panel["adj_low"], panel, YEAR_DAYS, "min", 1)
    return {"tech_high_52w_gap": close / high - 1, "tech_low_52w_gap": close / low - 1}


def _limits(panel: pd.DataFrame) -> Columns:
    up = panel["limit_up_close"].astype(bool).astype("float64")
    down = panel["limit_down_close"].astype(bool).astype("float64")
    return {
        "tech_limit_up_t": up,
        "tech_limit_up_cnt_20d": rolling(up, panel, LIMIT_DAYS, "sum", 1),
        "tech_limit_down_cnt_20d": rolling(down, panel, LIMIT_DAYS, "sum", 1),
        "tech_attention_t": panel["attention"].astype(bool).astype("float64"),
    }


def technical_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Technical features of every panel row; needs the universe panel plus a shares column."""
    require_sorted(panel)
    close = filled_close(panel)
    columns = {
        **_returns(panel, close),
        **_trend(panel, close),
        **_volume(panel),
        **_oscillators(panel, close),
        **_range(panel, close),
        **_limits(panel),
    }
    return feature_frame(panel, columns, feature_names(Group.TECHNICAL))
