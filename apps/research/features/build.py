"""Every stored feature of a date range, assembled from the groups (SPEC §6)."""

from collections.abc import Mapping
from datetime import date

import pandas as pd

from apps.market.calendar import TradingDays
from apps.research.features.chip import (
    SOURCED_CHIPS,
    attach_chips,
    attach_holding_changes,
    chip_features,
    chip_sourcing,
    lending_features,
    load_chip_starts,
    load_chips,
    load_holding_changes,
    second_batch_chip_features,
)
from apps.research.features.dividend import attach_ex_dates
from apps.research.features.fundamental import (
    BASIS_SIGNALS,
    attach_fundamentals,
    crossed_name,
    fundamental_features,
)
from apps.research.features.inputs import load_feature_panel
from apps.research.features.market import (
    MARKET_NAMES,
    attach_market_features,
    load_market_features,
)
from apps.research.features.registry import FEATURES, Group, feature_names
from apps.research.features.residual import (
    MASKED_MONTHS,
    OBSERVABLE,
    apply_fallbacks,
    mask_months,
    observable_gaps,
    observable_name,
)
from apps.research.features.size_risk import (
    industry_risk_features,
    second_batch_industry_features,
    size_liquidity_features,
)
from apps.research.features.technical import (
    second_batch_technical_features,
    technical_features,
)

FEATURE_COLUMNS = [spec.name for spec in FEATURES]
CURRENT = "_now"
CHECKED = [
    name for name in OBSERVABLE if name in {*feature_names(Group.FUNDAMENTAL), *SOURCED_CHIPS}
]
MODEL_COLUMNS = [*feature_names(batch=1), *feature_names(batch=2)]
FRAME_COLUMNS = ["stock_id", "date", "in_universe", *FEATURE_COLUMNS]
FIRST_BATCH_COLUMNS = ["stock_id", "date", "in_universe", *feature_names(batch=1)]
FRAME_TYPES: dict[str, str] = {"in_universe": "bool", **dict.fromkeys(FEATURE_COLUMNS, "float64")}


def _with_inputs(panel: pd.DataFrame, calendar: TradingDays, end: date) -> pd.DataFrame:
    """Panel with chips, holdings, fundamentals, ex-dates and market features from its first day."""
    first = panel["date"].min()
    chips = attach_chips(panel, load_chips(first, end))
    holdings = attach_holding_changes(chips, load_holding_changes(calendar, first, end))
    funds = attach_ex_dates(
        attach_fundamentals(holdings, calendar, first, end), calendar, first, end
    )
    return attach_market_features(funds, load_market_features(calendar, first, end))


def compute_features(
    panel: pd.DataFrame, masked: Mapping[str, frozenset[str]] = MASKED_MONTHS
) -> pd.DataFrame:
    """Stock, date, universe flag and every stored feature of each panel row, source-gap rules on.

    The second batch and unused features are stored too; screening and training read the first.
    """
    technical = technical_features(panel)
    fundamental = apply_fallbacks(fundamental_features(panel), panel)
    groups = [
        technical,
        second_batch_technical_features(panel),
        size_liquidity_features(panel),
        industry_risk_features(panel),
        second_batch_industry_features(panel, technical, fundamental),
        chip_features(panel),
        second_batch_chip_features(panel),
        fundamental,
        panel[[*MARKET_NAMES, "fund_days_to_ex_date"]],
    ]
    values = pd.concat(
        [group.drop(columns=["stock_id", "date"], errors="ignore") for group in groups], axis=1
    )
    keys = panel[["stock_id", "date"]].assign(in_universe=panel["in_universe"].astype(bool))
    return mask_months(pd.concat([keys, values[FEATURE_COLUMNS].astype("float64")], axis=1), masked)


def build_feature_frame(
    calendar: TradingDays,
    start: date,
    end: date,
    masked: Mapping[str, frozenset[str]] = MASKED_MONTHS,
) -> pd.DataFrame:
    """Features of every listed stock on the trading days in [start, end], after the warm-up.

    Always the whole market: industry averages need the full cross-section of each day.
    """
    low, high = max(start, calendar.start), min(end, calendar.end)
    if low > high or not calendar.between(low, high):
        return pd.DataFrame(columns=FRAME_COLUMNS).astype(FRAME_TYPES)
    panel = load_feature_panel(calendar, start, end)
    if panel.empty:
        return pd.DataFrame(columns=FRAME_COLUMNS).astype(FRAME_TYPES)
    features = compute_features(_with_inputs(panel, calendar, end), masked)
    inside = features[features["date"].between(start, end)]
    return inside.sort_values(["date", "stock_id"], ignore_index=True)[FRAME_COLUMNS]


def current_name(name: str) -> str:
    """Column of a feature as today's database gives it, beside the value stored earlier."""
    return f"{name}{CURRENT}"


def observable_frame(
    calendar: TradingDays,
    start: date,
    end: date,
    current: bool = False,
    masked: Mapping[str, frozenset[str]] = MASKED_MONTHS,
) -> pd.DataFrame:
    """Per listed stock and day in [start, end], which OBSERVABLE gaps have a known cause.

    With current, also each CHECKED feature as today's database gives it, masked.
    """
    keys = [
        "stock_id",
        "date",
        *map(observable_name, OBSERVABLE),
        *map(crossed_name, BASIS_SIGNALS),
    ]
    columns = [*keys, *(map(current_name, CHECKED) if current else ())]
    low, high = max(start, calendar.start), min(end, calendar.end)
    panel = load_feature_panel(calendar, start, end) if low <= high else pd.DataFrame()
    if panel.empty:
        return pd.DataFrame(columns=columns)
    first = panel["date"].min()
    chips = attach_chips(attach_fundamentals(panel, calendar, first, end), load_chips(first, end))
    flags = chip_sourcing(chips, load_chip_starts()).drop(columns=["stock_id", "date"])
    attached = chips.assign(**flags)
    gaps = observable_gaps(attached)
    if current:
        fundamental = apply_fallbacks(fundamental_features(attached), attached)
        lending = lending_features(attached).drop(columns=["stock_id", "date"])
        values = mask_months(fundamental.join(lending), masked)
        gaps = gaps.assign(**{current_name(name): values[name].to_numpy() for name in CHECKED})
    return gaps[gaps["date"].between(start, end)].reset_index(drop=True)[columns]
