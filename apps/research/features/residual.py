"""Source-gap rules applied alike to every stock, so missing values never mark a later delisting."""

import json
import math
from collections.abc import Iterable, Mapping
from types import MappingProxyType

import pandas as pd

from apps.research.features.chip import SOURCED_CHIPS
from apps.research.features.fundamental import (
    BASIS_SIGNALS,
    STATEMENT_INPUTS,
    crossed_name,
    sourced_name,
)
from apps.research.features.size_risk import canonical_industry
from apps.research.features.technical import MOMENTUM_DAYS

ACCEPT_RATIO = 1.5
ACCEPT_GAP = 0.05
MAX_MASKED_SHARE = 0.2
BASIS_GAP = 0.01
MAX_FILL_RATIO = 3.0
SHARE_DIGITS = 12
FALLBACKS: Mapping[str, str] = MappingProxyType(
    {"fund_rev_yoy": "fund_rev_yoy_q", "fund_rev_3m_yoy": "fund_rev_yoy_q"}
)


def _spans(*spans: str | tuple[str, str]) -> frozenset[str]:
    """Every month of the spans, both ends included; a single month may be given alone."""
    pairs = [(span, span) if isinstance(span, str) else span for span in spans]
    return frozenset(
        str(month) for first, last in pairs for month in pd.period_range(first, last, freq="M")
    )


MASKED_MONTHS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "fund_rev_yoy": _spans(("2010-01", "2015-02")),
        "fund_rev_3m_yoy": _spans(("2010-01", "2015-04"), "2016-03", "2017-01"),
        "fund_ep": _spans(
            ("2010-01", "2014-03"), ("2015-09", "2016-02"), ("2016-04", "2016-05"), "2017-01"
        ),
        "fund_eps_yoy": _spans(
            ("2010-04", "2015-03"),
            ("2015-08", "2015-09"),
            ("2016-04", "2017-06"),
            "2017-08",
            ("2021-08", "2021-09"),
            ("2022-02", "2022-03"),
            ("2022-09", "2023-01"),
        ),
        "fund_roe_ttm": _spans(
            ("2014-05", "2014-06"),
            "2014-09",
            ("2015-09", "2016-02"),
            "2016-05",
            "2017-04",
            "2017-12",
            ("2019-06", "2020-11"),
            ("2023-03", "2024-12"),
        ),
        "fund_gross_margin": _spans(("2010-01", "2013-05")),
        "fund_op_margin": _spans(("2010-01", "2013-05")),
        "fund_debt_ratio": _spans(
            "2013-05",
            "2017-04",
            ("2018-08", "2018-10"),
            ("2019-06", "2019-11"),
            ("2021-01", "2021-02"),
            "2021-06",
            ("2021-09", "2021-10"),
            "2022-10",
            ("2023-03", "2023-05"),
            ("2023-12", "2024-11"),
        ),
        "fund_gm_yoy_chg": _spans(("2010-01", "2014-05"), "2016-05", "2016-11"),
        "fund_opm_yoy_chg": _spans(("2010-01", "2014-05"), "2016-05", "2016-11"),
        "fund_cfo_to_ni": _spans(("2015-09", "2016-02"), "2016-05", "2022-10"),
        "fund_days_since_fs": _spans(("2010-01", "2013-05")),
        "fund_rev_yoy_q": _spans(("2010-01", "2014-05"), "2016-05", "2017-01", "2017-04"),
        "ind_ep_neutral": _spans(
            ("2010-01", "2014-03"), ("2015-09", "2016-02"), ("2016-04", "2016-05"), "2017-01"
        ),
        "fund_days_to_ex_date": _spans("2022-11"),
    }
)
GROUPS = {True: "delisted", False: "current"}
OBSERVED = "_observable"
PRESENT = "present_"
FILLED = "filled_"
ROWS = "rows_"
OBSERVABLE: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "fund_ep": ("eps_ttm",),
        "ind_ep_neutral": ("eps_ttm",),
        **{name: (name,) for name in STATEMENT_INPUTS if name != "eps_ttm"},
        "fund_rev_yoy": ("fund_rev_yoy", "fund_rev_yoy_q"),
        "fund_rev_3m_yoy": ("fund_rev_3m_yoy", "fund_rev_yoy_q"),
        **{name: (name,) for name in SOURCED_CHIPS},
    }
)
LISTING_WINDOWS: Mapping[str, int] = MappingProxyType(
    {"tech_mom_250_20": MOMENTUM_DAYS, "ind_mom_neutral": MOMENTUM_DAYS}
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


def _recorded(panel: pd.DataFrame, signal: str, column: str) -> pd.Series:
    """Whether the signal's inputs are on record, for a fallback also on one report basis."""
    recorded = panel[sourced_name(signal)].eq(1.0)
    if signal == column or signal not in BASIS_SIGNALS:
        return recorded
    return recorded & panel[crossed_name(signal)].ne(1.0)


def observable_gaps(
    panel: pd.DataFrame, exempt: Mapping[str, frozenset[str]] = NO_MARGINS
) -> pd.DataFrame:
    """Per row, whether each OBSERVABLE or LISTING_WINDOWS gap has a cause known on its day.

    Momentum gaps are known while the listing's age is below its LISTING_WINDOWS window.
    """
    industry = canonical_industry(panel["industry"])
    observed = {}
    for name, signals in OBSERVABLE.items():
        recorded = pd.concat([_recorded(panel, signal, name) for signal in signals], axis=1)
        observed[observable_name(name)] = recorded.any(axis=1) | industry.isin(
            exempt.get(name, frozenset())
        )
    young = {observable_name(name): panel["age"].lt(days) for name, days in LISTING_WINDOWS.items()}
    crossed = {crossed_name(name): panel[crossed_name(name)].eq(1.0) for name in BASIS_SIGNALS}
    return panel[["stock_id", "date"]].assign(**observed, **young, **crossed)


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


def fallback_filled(rows: pd.DataFrame, name: str) -> pd.Series:
    """Rows apply_fallbacks filled: a value while the feature's own source months are off record."""
    return rows[name].notna() & rows[sourced_name(name)].ne(1.0)


def fill_flags(current: pd.Series, delisted: pd.Series) -> pd.Series:
    """Whether fill shares still tell a later delisting: over MAX_FILL_RATIO or delisted only."""
    over = (delisted - MAX_FILL_RATIO * current).round(SHARE_DIGITS) > 0
    return (current.gt(0) & over) | (current.eq(0) & delisted.gt(0))


def monthly_coverage(
    rows: pd.DataFrame,
    column: str,
    delisted: pd.Series,
    fallbacks: Mapping[str, str] = FALLBACKS,
) -> pd.DataFrame:
    """Share of universe rows with a value or an observable gap, per month and later fate.

    The present_ shares count values only; a fallback column adds filled_ shares.
    """
    inside = rows[rows["in_universe"].astype(bool)]
    group = inside["stock_id"].map(delisted).eq(True).map(GROUPS)
    present = inside[column].notna()
    observed = observable_name(column)
    known = present | inside[observed].eq(True) if observed in inside else present
    months = _months(inside["date"])

    def shares(flag: pd.Series) -> pd.DataFrame:
        table = flag.astype("float64").groupby([months, group]).mean().unstack()
        return table.reindex(columns=list(GROUPS.values()))

    parts = [shares(known), shares(present).add_prefix(PRESENT)]
    if column in fallbacks:
        parts.append(shares(fallback_filled(inside, column)).add_prefix(FILLED))
    sizes = group.groupby([months, group]).size().unstack(fill_value=0)
    parts.append(sizes.reindex(columns=list(GROUPS.values()), fill_value=0).add_prefix(ROWS))
    table = pd.concat(parts, axis=1)
    return table.rename_axis(index=None, columns=None)


def _apart(more: pd.Series, less: pd.Series) -> pd.Series:
    """Whether one share is over ACCEPT_RATIO times the other and more than ACCEPT_GAP above it."""
    gap = (more - less).round(SHARE_DIGITS) > ACCEPT_GAP
    return gap & ((more - ACCEPT_RATIO * less).round(SHARE_DIGITS) > 0)


def affected_months(coverage: pd.DataFrame) -> list[str]:
    """Months where, on their own, the delisted group has gaps, values or fills far more often.

    A month where only the delisted group takes the fallback is affected too.
    """
    current, delisted = coverage["current"], coverage["delisted"]
    present = coverage[f"{PRESENT}current"], coverage[f"{PRESENT}delisted"]
    affected = _apart(1 - delisted, 1 - current) | _apart(present[1], present[0])
    if f"{FILLED}delisted" in coverage:
        filled = coverage[f"{FILLED}current"], coverage[f"{FILLED}delisted"]
        affected = affected | _apart(filled[1], filled[0]) | (filled[0].eq(0) & filled[1].gt(0))
    return sorted(str(month) for month in coverage.index[affected])


def unmasked_fills(coverage: pd.DataFrame, masked: Iterable[str]) -> pd.Series:
    """Fill shares of each group over the months outside masked, weighted by universe rows."""
    kept = coverage.loc[~coverage.index.isin(list(masked))]
    shares = {}
    for name in GROUPS.values():
        rows = kept[f"{ROWS}{name}"]
        filled = (kept[f"{FILLED}{name}"].fillna(0) * rows).sum()
        shares[name] = float(filled / rows.sum()) if rows.sum() else math.nan
    return pd.Series(shares)


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
