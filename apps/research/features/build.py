"""Every stored feature of a date range, both batches, assembled from the groups (SPEC §6)."""

from datetime import date

import numpy as np
import pandas as pd

from apps.market.calendar import TradingDays
from apps.research.features.chip import (
    attach_chips,
    attach_holding_changes,
    chip_features,
    load_chips,
    load_holding_changes,
    second_batch_chip_features,
)
from apps.research.features.fundamental import attach_fundamentals, fundamental_features
from apps.research.features.inputs import load_feature_panel
from apps.research.features.market import (
    MARKET_NAMES,
    attach_market_features,
    load_market_features,
)
from apps.research.features.registry import feature_names
from apps.research.features.size_risk import industry_risk_features, size_liquidity_features
from apps.research.features.technical import (
    second_batch_technical_features,
    technical_features,
)

FEATURE_COLUMNS = [*feature_names(batch=1), *feature_names(batch=2)]
PENDING = ["ind_sector_id", "ind_mom_neutral", "ind_ep_neutral", "fund_days_to_ex_date"]
FRAME_COLUMNS = ["stock_id", "date", "in_universe", *FEATURE_COLUMNS]
FRAME_TYPES: dict[str, str] = {"in_universe": "bool", **dict.fromkeys(FEATURE_COLUMNS, "float64")}


def _with_inputs(panel: pd.DataFrame, calendar: TradingDays, end: date) -> pd.DataFrame:
    """Panel with chips, holdings, fundamentals and market features from its first day on."""
    first = panel["date"].min()
    chips = attach_chips(panel, load_chips(first, end))
    holdings = attach_holding_changes(chips, load_holding_changes(calendar, first, end))
    funds = attach_fundamentals(holdings, calendar, first, end)
    return attach_market_features(funds, load_market_features(calendar, first, end))


def compute_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Stock, date, universe flag and every stored feature of each panel row.

    The second batch is stored for later evaluation; screening and training read the first only.
    """
    groups = [
        technical_features(panel),
        second_batch_technical_features(panel),
        size_liquidity_features(panel),
        industry_risk_features(panel),
        chip_features(panel),
        second_batch_chip_features(panel),
        fundamental_features(panel),
        panel[MARKET_NAMES],
        pd.DataFrame(dict.fromkeys(PENDING, np.nan), index=panel.index),
    ]
    values = pd.concat(
        [group.drop(columns=["stock_id", "date"], errors="ignore") for group in groups], axis=1
    )
    keys = panel[["stock_id", "date"]].assign(in_universe=panel["in_universe"].astype(bool))
    return pd.concat([keys, values[FEATURE_COLUMNS].astype("float64")], axis=1)


def build_feature_frame(calendar: TradingDays, start: date, end: date) -> pd.DataFrame:
    """Features of every listed stock on the trading days in [start, end], after the warm-up.

    Always the whole market: industry averages need the full cross-section of each day.
    """
    low, high = max(start, calendar.start), min(end, calendar.end)
    if low > high or not calendar.between(low, high):
        return pd.DataFrame(columns=FRAME_COLUMNS).astype(FRAME_TYPES)
    panel = load_feature_panel(calendar, start, end)
    if panel.empty:
        return pd.DataFrame(columns=FRAME_COLUMNS).astype(FRAME_TYPES)
    features = compute_features(_with_inputs(panel, calendar, end))
    inside = features[features["date"].between(start, end)]
    return inside.sort_values(["date", "stock_id"], ignore_index=True)[FRAME_COLUMNS]
