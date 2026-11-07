"""Source-gap rules applied alike to every stock, so missing values never mark a later delisting."""

import json
from collections.abc import Mapping
from types import MappingProxyType

import pandas as pd

from apps.research.features.fundamental import (
    BASIS_SIGNALS,
    STATEMENT_INPUTS,
    crossed_name,
    sourced_name,
)
from apps.research.features.size_risk import canonical_industry

ACCEPT_RATIO = 1.5
ACCEPT_GAP = 0.05
MAX_MASKED_SHARE = 0.2
BASIS_GAP = 0.01
SHARE_DIGITS = 12
FALLBACKS: Mapping[str, str] = MappingProxyType(
    {"fund_rev_yoy": "fund_rev_yoy_q", "fund_rev_3m_yoy": "fund_rev_yoy_q"}
)
MASKED_MONTHS: Mapping[str, frozenset[str]] = MappingProxyType({})
GROUPS = {True: "delisted", False: "current"}
OBSERVED = "_observable"
OBSERVABLE: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "fund_ep": ("eps_ttm",),
        "ind_ep_neutral": ("eps_ttm",),
        **{name: (name,) for name in STATEMENT_INPUTS if name != "eps_ttm"},
        "fund_rev_yoy": ("fund_rev_yoy", "fund_rev_yoy_q"),
        "fund_rev_3m_yoy": ("fund_rev_3m_yoy", "fund_rev_yoy_q"),
    }
)
NO_MARGINS = MappingProxyType(
    dict.fromkeys(
        ["fund_gross_margin", "fund_op_margin", "fund_gm_yoy_chg", "fund_opm_yoy_chg"],
        frozenset({"金融保險"}),
    )
)
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


def observable_name(name: str) -> str:
    """Column that is True when a missing value of the feature has a cause known on its day."""
    return f"{name}{OBSERVED}"


def observable_gaps(
    panel: pd.DataFrame, exempt: Mapping[str, frozenset[str]] = NO_MARGINS
) -> pd.DataFrame:
    """Per row, whether each OBSERVABLE gap reads only inputs on record or its industry lacks it.

    A value read from inputs all on record is missing by its formula, not by the source.
    """
    industry = canonical_industry(panel["industry"])
    observed = {}
    for name, signals in OBSERVABLE.items():
        recorded = panel[[sourced_name(signal) for signal in signals]].eq(1.0).any(axis=1)
        observed[observable_name(name)] = recorded | industry.isin(exempt.get(name, frozenset()))
    crossed = {crossed_name(name): panel[crossed_name(name)].eq(1.0) for name in BASIS_SIGNALS}
    return panel[["stock_id", "date"]].assign(**observed, **crossed)


def basis_counts(rows: pd.DataFrame, column: str, delisted: pd.Series) -> pd.Series:
    """Universe rows missing a value read across report bases, and all universe rows, by fate."""
    inside = rows[rows["in_universe"].astype(bool)]
    fate = inside["stock_id"].map(delisted).eq(True)
    hit = inside[column].isna() & inside[crossed_name(column)].eq(True)
    return pd.Series(
        {
            "current": int(hit[~fate].sum()),
            "delisted": int(hit[fate].sum()),
            "rows_current": int((~fate).sum()),
            "rows_delisted": int(fate.sum()),
        }
    )


def masked_as_observable(
    frame: pd.DataFrame, masked: Mapping[str, frozenset[str]] = MASKED_MONTHS
) -> pd.DataFrame:
    """New frame where gaps in a column's masked months also count as having a known cause."""
    months = _months(frame["date"])
    marked = {
        observable_name(name): frame.get(observable_name(name), pd.Series(False, index=frame.index))
        .eq(True)
        .astype(bool)
        | months.isin(sorted(chosen))
        for name, chosen in masked.items()
        if chosen
    }
    return frame.assign(**marked)


def monthly_coverage(rows: pd.DataFrame, column: str, delisted: pd.Series) -> pd.DataFrame:
    """Share of universe rows with a value, or an observable gap, per month and later fate."""
    inside = rows[rows["in_universe"].astype(bool)]
    group = inside["stock_id"].map(delisted).eq(True).map(GROUPS)
    known = inside[column].notna()
    observed = observable_name(column)
    if observed in inside:
        known = known | inside[observed].eq(True)
    shares = known.astype("float64").groupby([_months(inside["date"]), group]).mean().unstack()
    return shares.reindex(columns=list(GROUPS.values())).rename_axis(index=None, columns=None)


def _apart(more: pd.Series, less: pd.Series) -> pd.Series:
    """Whether one share is over ACCEPT_RATIO times the other and more than ACCEPT_GAP above it."""
    gap = (more - less).round(SHARE_DIGITS) > ACCEPT_GAP
    return gap & ((more - ACCEPT_RATIO * less).round(SHARE_DIGITS) > 0)


def affected_months(coverage: pd.DataFrame) -> list[str]:
    """Months where, on their own, the delisted group is missing or present far more often."""
    current, delisted = coverage["current"], coverage["delisted"]
    affected = _apart(1 - delisted, 1 - current) | _apart(delisted, current)
    return sorted(str(month) for month in coverage.index[affected])


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
