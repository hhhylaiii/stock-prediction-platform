"""Every first-batch feature of a date range, assembled from the feature groups (SPEC §6)."""

from datetime import date

import pandas as pd

from apps.market.calendar import TradingDays
from apps.research.features.chip import (
    attach_chips,
    attach_holding_changes,
    chip_features,
    load_chips,
    load_holding_changes,
)
from apps.research.features.fundamental import attach_fundamentals, fundamental_features
from apps.research.features.inputs import load_feature_panel
from apps.research.features.market import attach_market_features, load_market_features
from apps.research.features.registry import Group, feature_names
from apps.research.features.size_risk import industry_risk_features, size_liquidity_features
from apps.research.features.technical import technical_features

FEATURE_COLUMNS = feature_names(batch=1)
FRAME_COLUMNS = ["stock_id", "date", "in_universe", *FEATURE_COLUMNS]


def _with_inputs(panel: pd.DataFrame, calendar: TradingDays, end: date) -> pd.DataFrame:
    """Panel with chips, holdings, fundamentals and market features from its first day on."""
    first = panel["date"].min()
    chips = attach_chips(panel, load_chips(first, end))
    holdings = attach_holding_changes(chips, load_holding_changes(calendar, first, end))
    funds = attach_fundamentals(holdings, calendar, first, end)
    return attach_market_features(funds, load_market_features(calendar, first, end))


def compute_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Stock, date, universe flag and every first-batch feature of each panel row."""
    groups = [
        technical_features(panel),
        size_liquidity_features(panel),
        industry_risk_features(panel),
        chip_features(panel),
        fundamental_features(panel),
        panel[feature_names(Group.MARKET)],
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
    panel = load_feature_panel(calendar, start, end)
    if panel.empty:
        return pd.DataFrame(columns=FRAME_COLUMNS)
    features = compute_features(_with_inputs(panel, calendar, end))
    inside = features[features["date"].between(start, end)]
    return inside.sort_values(["date", "stock_id"], ignore_index=True)[FRAME_COLUMNS]
