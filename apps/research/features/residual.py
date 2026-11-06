"""Rules for source gaps left after backfill, so missing values never mark a stock to be delisted.

C: a feature whose source is missing takes a second source for every stock alike.
A: a feature without a second source is masked for every stock in the months its source covers
only stocks that are still listed (delisted coverage under COVERAGE_FLOOR of the listed one).
"""

from collections.abc import Mapping
from types import MappingProxyType

import pandas as pd

COVERAGE_FLOOR = 0.9
FALLBACKS: Mapping[str, str] = MappingProxyType(
    {"fund_rev_yoy": "fund_rev_yoy_q", "fund_rev_3m_yoy": "fund_rev_yoy_q"}
)
MASKED_MONTHS: Mapping[str, frozenset[str]] = MappingProxyType({"fund_rev_mom": frozenset()})
GROUPS = {True: "delisted", False: "current"}


def _months(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates).dt.strftime("%Y-%m")


def apply_residual_rules(
    features: pd.DataFrame,
    fallbacks: Mapping[str, str] = FALLBACKS,
    masked: Mapping[str, frozenset[str]] = MASKED_MONTHS,
) -> pd.DataFrame:
    """New frame with C fallbacks filled, then A months masked, the same way for every stock."""
    filled = {name: features[name].fillna(features[source]) for name, source in fallbacks.items()}
    ruled = features.assign(**filled)
    if not any(masked.values()):
        return ruled
    months = _months(ruled["date"])
    return ruled.assign(
        **{name: ruled[name].mask(months.isin(list(hidden))) for name, hidden in masked.items()}
    )


def monthly_coverage(rows: pd.DataFrame, column: str, delisted: pd.Series) -> pd.DataFrame:
    """Share of universe rows with a value per month, for stocks later delisted and the rest."""
    inside = rows[rows["in_universe"].astype(bool)]
    group = inside["stock_id"].map(delisted).fillna(False).astype(bool).map(GROUPS)
    known = inside[column].notna().astype("float64")
    shares = known.groupby([_months(inside["date"]), group]).mean().unstack()
    return shares.reindex(columns=list(GROUPS.values())).rename_axis(index=None, columns=None)


def affected_months(coverage: pd.DataFrame, floor: float = COVERAGE_FLOOR) -> list[str]:
    """Months whose delisted coverage is under floor times the listed coverage."""
    if not 0 < floor <= 1:
        raise ValueError(f"floor {floor} must be in (0, 1]")
    short = coverage["delisted"] < floor * coverage["current"]
    return sorted(str(month) for month in coverage.index[short.fillna(False)])
