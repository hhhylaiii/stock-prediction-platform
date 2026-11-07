"""Source-gap rules applied alike to every stock, so missing values never mark a later delisting."""

import json
from collections.abc import Mapping
from types import MappingProxyType

import pandas as pd

from apps.research.features.fundamental import sourced_name

ACCEPT_RATIO = 1.5
ACCEPT_GAP = 0.05
MAX_MASKED_SHARE = 0.2
SHARE_DIGITS = 12
FALLBACKS: Mapping[str, str] = MappingProxyType(
    {"fund_rev_yoy": "fund_rev_yoy_q", "fund_rev_3m_yoy": "fund_rev_yoy_q"}
)
MASKED_MONTHS: Mapping[str, frozenset[str]] = MappingProxyType({"fund_rev_mom": frozenset()})
GROUPS = {True: "delisted", False: "current"}
RULES_KEY = b"residual_rules"


def _months(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates).dt.strftime("%Y-%m")


def apply_fallbacks(
    features: pd.DataFrame, sourcing: pd.DataFrame, fallbacks: Mapping[str, str] = FALLBACKS
) -> pd.DataFrame:
    """New frame where a missing feature whose source months are off record takes its fallback."""
    filled = {
        name: features[name].fillna(features[source].where(sourcing[sourced_name(name)].ne(1.0)))
        for name, source in fallbacks.items()
    }
    return features.assign(**filled)


def mask_months(
    frame: pd.DataFrame, masked: Mapping[str, frozenset[str]] = MASKED_MONTHS
) -> pd.DataFrame:
    """New frame with each listed column missing for every stock in its affected months."""
    hidden = {name: months for name, months in masked.items() if months}
    if not hidden:
        return frame
    months = _months(frame["date"])
    return frame.assign(
        **{name: frame[name].mask(months.isin(sorted(chosen))) for name, chosen in hidden.items()}
    )


def monthly_coverage(rows: pd.DataFrame, column: str, delisted: pd.Series) -> pd.DataFrame:
    """Share of universe rows with a value per month, for stocks later delisted and the rest."""
    inside = rows[rows["in_universe"].astype(bool)]
    group = inside["stock_id"].map(delisted).eq(True).map(GROUPS)
    known = inside[column].notna().astype("float64")
    shares = known.groupby([_months(inside["date"]), group]).mean().unstack()
    return shares.reindex(columns=list(GROUPS.values())).rename_axis(index=None, columns=None)


def affected_months(coverage: pd.DataFrame) -> list[str]:
    """Months failing the acceptance alone: delisted missing share over 1.5x and 5 points above."""
    current, delisted = 1 - coverage["current"], 1 - coverage["delisted"]
    gap = (delisted - current).round(SHARE_DIGITS) > ACCEPT_GAP
    ratio = (delisted - ACCEPT_RATIO * current).round(SHARE_DIGITS) > 0
    return sorted(str(month) for month in coverage.index[gap & ratio])


def rules_signature(
    fallbacks: Mapping[str, str] = FALLBACKS,
    masked: Mapping[str, frozenset[str]] = MASKED_MONTHS,
) -> str:
    """The rules as stable JSON, stamped on each exported partition."""
    rules = {
        "fallbacks": dict(sorted(fallbacks.items())),
        "masked": {name: sorted(months) for name, months in sorted(masked.items())},
        "fills": "off_record",
    }
    return json.dumps(rules, sort_keys=True)
