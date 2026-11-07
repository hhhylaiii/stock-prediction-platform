"""Fundamental features of SPEC §6, each visible from the day every input was announced."""

from collections.abc import Mapping, Sequence
from datetime import date, timedelta

import numpy as np
import pandas as pd
from django.db import models

from apps.market.calendar import TradingDays
from apps.market.models import DailyValuation, FinancialStatement, FinType, MonthlyRevenue
from apps.research.features.common import carried, feature_frame, ratio
from apps.research.features.registry import UNUSED, Group, feature_names
from apps.research.panel import require_sorted

TTM_QUARTERS = 4
TTM_LAGS = tuple(range(1, TTM_QUARTERS))
REVENUE_INPUTS: Mapping[str, tuple[int, ...]] = {
    "fund_rev_yoy": (12,),
    "fund_rev_3m_yoy": (1, 2, 12, 13, 14),
    "fund_rev_mom": (1,),
}
STATEMENT_INPUTS: Mapping[str, tuple[int, ...]] = {
    "eps_ttm": TTM_LAGS,
    "fund_eps_yoy": tuple(range(1, 2 * TTM_QUARTERS)),
    "fund_roe_ttm": TTM_LAGS,
    "fund_gross_margin": (),
    "fund_op_margin": (),
    "fund_debt_ratio": (),
    "fund_gm_yoy_chg": (TTM_QUARTERS,),
    "fund_opm_yoy_chg": (TTM_QUARTERS,),
    "fund_cfo_to_ni": TTM_LAGS,
    "fund_rev_yoy_q": (TTM_QUARTERS,),
}
STATEMENT_FIELDS: Mapping[str, tuple[str, ...]] = {
    "eps_ttm": ("eps",),
    "fund_eps_yoy": ("eps",),
    "fund_roe_ttm": ("roe",),
    "fund_gross_margin": ("gross_margin",),
    "fund_op_margin": ("op_margin",),
    "fund_debt_ratio": ("debt_ratio",),
    "fund_gm_yoy_chg": ("gross_margin",),
    "fund_opm_yoy_chg": ("op_margin",),
    "fund_cfo_to_ni": ("operating_cf", "net_income"),
    "fund_rev_yoy_q": ("revenue",),
}
BASIS_COLUMNS = frozenset({"revenue", "gross_margin", "op_margin", "operating_cf"})
CONSOLIDATED_MONTH = 2013 * 12
SIGNAL_COLUMNS = [
    "stock_id",
    "period",
    "number",
    "signal",
    "value",
    "announce_date",
    "first_known",
    "known_from",
]
REVENUE_STALE = timedelta(days=62)
STATEMENT_STALE = timedelta(days=200)
MONTH_SPAN = timedelta(days=31)
SOURCED = "_sourced"
QUARTER_SPAN = timedelta(days=92)
REVENUE_LOOKBACK = REVENUE_STALE + MONTH_SPAN * (max(map(max, REVENUE_INPUTS.values())) + 1)
STATEMENT_LOOKBACK = STATEMENT_STALE + QUARTER_SPAN * (
    max(max(lags, default=0) for lags in STATEMENT_INPUTS.values()) + 1
)
STATEMENT_COLUMNS = [
    "stock_id",
    "period",
    "fin_type",
    "eps",
    "gross_margin",
    "op_margin",
    "net_income",
    "roe",
    "debt_ratio",
    "operating_cf",
    "revenue",
    "announce_date",
]
NUMERIC = STATEMENT_COLUMNS[3:-1]
PASSED = [*REVENUE_INPUTS, *list(STATEMENT_INPUTS)[1:], "fund_days_since_fs"]


class _Lags:
    """Earlier periods of the same stock; consolidation-scope values break where the basis changed.

    Parent-attributable figures (eps, net income, roe) match on both bases, so they carry over.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.lookup = frame.set_index(["stock_id", "number"])
        self.basis = frame.get("fin_type")

    def __call__(self, column: str, lag: int) -> pd.Series:
        keys = pd.MultiIndex.from_arrays([self.frame["stock_id"], self.frame["number"] - lag])
        values = pd.Series(self.lookup[column].reindex(keys).to_numpy(), index=self.frame.index)
        if self.basis is None or column not in BASIS_COLUMNS:
            return values
        earlier = self.lookup["fin_type"].reindex(keys).to_numpy()
        return values.where(earlier == self.basis.to_numpy())

    def signal(self, name: str, value: pd.Series, lags: Sequence[int]) -> pd.DataFrame:
        """Rows of one signal with the first and last announce date of the periods it reads."""
        dates = [pd.to_datetime(self.frame["announce_date"])]
        dates += [pd.to_datetime(self("announce_date", lag)) for lag in lags]
        stacked = pd.concat(dates, axis=1)
        return self.frame[["stock_id", "period", "number"]].assign(
            signal=name,
            value=value.astype("float64"),
            announce_date=dates[0],
            first_known=stacked.min(axis=1),
            known_from=stacked.max(axis=1),
        )


def _numbered(frame: pd.DataFrame, separator: str, per_year: int) -> pd.DataFrame:
    year = frame["period"].str[:4].astype(int)
    part = frame["period"].str.split(separator).str[-1].str.lstrip("Q").astype(int)
    numbered = frame.assign(number=year * per_year + part - 1)
    return numbered.sort_values(["stock_id", "number"], ignore_index=True)


def _signals(
    lag: _Lags, values: Mapping[str, pd.Series], inputs: Mapping[str, Sequence[int]]
) -> pd.DataFrame:
    parts = [lag.signal(name, value, inputs[name]) for name, value in values.items()]
    return pd.concat(parts, ignore_index=True)[SIGNAL_COLUMNS]


def _growth(now: pd.Series, base: pd.Series) -> pd.Series:
    """now / base − 1, missing when the base is not positive."""
    return now / base.where(base > 0) - 1


def sourced_name(name: str) -> str:
    """Panel column that is 1 when every period and field a signal reads is on record, else 0."""
    return f"{name}{SOURCED}"


def _monthly(revenue: pd.DataFrame) -> _Lags:
    return _Lags(_numbered(revenue.assign(revenue=revenue["revenue"].astype("float64")), "-", 12))


def revenue_sourcing(revenue: pd.DataFrame) -> pd.DataFrame:
    """Per announced month and revenue signal, 1 when all the months it reads are on record."""
    lag = _monthly(revenue)
    values = {
        sourced_name(name): pd.concat(
            [lag.frame["revenue"], *(lag("revenue", k) for k in lags)], axis=1
        )
        .notna()
        .all(axis=1)
        .astype("float64")
        for name, lags in REVENUE_INPUTS.items()
    }
    inputs = {sourced_name(name): lags for name, lags in REVENUE_INPUTS.items()}
    return _signals(lag, values, inputs)


def revenue_signals(revenue: pd.DataFrame) -> pd.DataFrame:
    """Single-month yoy, three-month yoy and mom of each announced month, one row per signal."""
    lag = _monthly(revenue)
    frame = lag.frame
    now = frame["revenue"]
    three = now + lag("revenue", 1) + lag("revenue", 2)
    base = lag("revenue", 12) + lag("revenue", 13) + lag("revenue", 14)
    values = {
        "fund_rev_yoy": _growth(now, lag("revenue", 12)),
        "fund_rev_3m_yoy": _growth(three, base),
        "fund_rev_mom": _growth(now, lag("revenue", 1)),
    }
    kept = {
        name: value.mask(_across_consolidation(frame["number"], REVENUE_INPUTS[name]))
        for name, value in values.items()
    }
    return _signals(lag, kept, REVENUE_INPUTS)


def _across_consolidation(numbers: pd.Series, lags: Sequence[int]) -> pd.Series:
    """Whether a monthly signal reads months on both sides of 2013-01, when revenue consolidated."""
    return (numbers - max(lags) < CONSOLIDATED_MONTH) & (numbers >= CONSOLIDATED_MONTH)


def _preferred(statements: pd.DataFrame) -> pd.DataFrame:
    """First announced statement per stock and quarter (consolidated on a tie), numbered."""
    rank = statements["fin_type"].ne(FinType.CONSOLIDATED).astype(int)
    keys = ["stock_id", "period", "announce_date", "rank"]
    ordered = statements.assign(rank=rank).sort_values(keys)
    chosen = ordered.drop_duplicates(["stock_id", "period"]).drop(columns="rank")
    return _numbered(chosen.astype(dict.fromkeys(NUMERIC, "float64")), "-Q", 4)


def _ttm(lag: _Lags, frame: pd.DataFrame, column: str, offset: int = 0) -> pd.Series:
    """Sum of four consecutive quarters ending offset quarters back; missing if any is absent."""
    parts = [frame[column] if offset == 0 else lag(column, offset)]
    parts += [lag(column, offset + k) for k in range(1, TTM_QUARTERS)]
    return pd.concat(parts, axis=1).sum(axis=1, min_count=TTM_QUARTERS)


def statement_sourcing(statements: pd.DataFrame) -> pd.DataFrame:
    """Per chosen statement and signal, 1 when every field it reads is on record in each quarter.

    Quarters on the other basis count as on record: their gap is the basis rule, not the source.
    """
    frame = _preferred(statements)
    lag = _Lags(frame.drop(columns="fin_type"))
    values = {
        sourced_name(name): pd.concat(
            [
                frame[list(fields)],
                *(lag(field, k) for field in fields for k in STATEMENT_INPUTS[name]),
            ],
            axis=1,
        )
        .notna()
        .all(axis=1)
        .astype("float64")
        for name, fields in STATEMENT_FIELDS.items()
    }
    inputs = {sourced_name(name): STATEMENT_INPUTS[name] for name in STATEMENT_FIELDS}
    return _signals(lag, values, inputs)


def statement_signals(statements: pd.DataFrame) -> pd.DataFrame:
    """TTM EPS and its growth, TTM ROE and cash conversion, margins and their yoy changes."""
    frame = _preferred(statements)
    lag = _Lags(frame)
    eps = _ttm(lag, frame, "eps")
    income = _ttm(lag, frame, "net_income")
    values = {
        "eps_ttm": eps,
        "fund_eps_yoy": _growth(eps, _ttm(lag, frame, "eps", TTM_QUARTERS)),
        "fund_roe_ttm": _ttm(lag, frame, "roe"),
        "fund_gross_margin": frame["gross_margin"],
        "fund_op_margin": frame["op_margin"],
        "fund_debt_ratio": frame["debt_ratio"],
        "fund_gm_yoy_chg": frame["gross_margin"] - lag("gross_margin", TTM_QUARTERS),
        "fund_opm_yoy_chg": frame["op_margin"] - lag("op_margin", TTM_QUARTERS),
        "fund_cfo_to_ni": ratio(_ttm(lag, frame, "operating_cf"), income.where(income > 0)),
        "fund_rev_yoy_q": _growth(frame["revenue"], lag("revenue", TTM_QUARTERS)),
    }
    return _signals(lag, values, STATEMENT_INPUTS)


def fundamental_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Fundamental features of every row; needs the attached signals, valuation and close."""
    require_sorted(panel)
    close = carried(panel["close"], panel)
    columns = {
        **{name: panel[name] for name in PASSED},
        "fund_ep": ratio(panel["eps_ttm"], close),
        "fund_bp": pd.Series(1.0 / panel["pb"].where(panel["pb"] > 0), index=panel.index),
        "fund_dividend_yield": panel["dividend_yield"],
    }
    names = [*feature_names(Group.FUNDAMENTAL), *feature_names(Group.FUNDAMENTAL, batch=UNUSED)]
    return feature_frame(panel, {name: columns[name].astype(np.float64) for name in names}, names)


def _rows(
    model: type[models.Model],
    columns: list[str],
    stock_ids: Sequence[str] | None,
    **lookups: object,
) -> pd.DataFrame:
    rows = model._default_manager.filter(**lookups)
    if stock_ids is not None:
        rows = rows.filter(stock_id__in=list(stock_ids))
    return pd.DataFrame(list(rows.values_list(*columns)), columns=columns)


def segment_starts(panel: pd.DataFrame, calendar: TradingDays) -> pd.Series:
    """First trading day of the listing for rows of a reused code (segment 2 on); NaT otherwise."""
    later = panel["segment"] > 1
    days, ages = panel.loc[later, "date"].tolist(), panel.loc[later, "age"].astype(int).tolist()
    firsts = [pd.Timestamp(calendar.shift(day, -age)) for day, age in zip(days, ages, strict=True)]
    starts = pd.Series(pd.NaT, index=panel.index, dtype="datetime64[ns]")
    starts[later] = firsts
    return starts


def _events(rows: pd.DataFrame) -> pd.DataFrame:
    """Each period shows up on its announce date, missing until its last input is announced."""
    waiting = rows["known_from"] > rows["announce_date"]
    if not waiting.any():
        return rows
    pending = rows[waiting].assign(value=np.nan, known_from=rows.loc[waiting, "announce_date"])
    return pd.concat([rows, pending], ignore_index=True)


def _newest(events: pd.DataFrame) -> pd.DataFrame:
    """Events by visibility, without those older than a period of the stock already visible."""
    ordered = events.sort_values(["on", "number"], kind="stable")
    seen = ordered.groupby("stock_id")["number"].cummax()
    before = seen.groupby(ordered["stock_id"]).shift(1)
    return ordered[~(ordered["number"] < before)]


def _latest(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    stale: timedelta,
    starts: pd.Series,
    column: str = "value",
) -> pd.Series:
    """Column of the newest period visible on each row's date, stale ones and other listings out."""
    if events.empty or panel.empty:
        return pd.Series(np.nan, index=panel.index)
    left = pd.DataFrame(
        {
            "row": np.arange(len(panel)),
            "stock_id": panel["stock_id"].to_numpy(),
            "on": pd.to_datetime(panel["date"]),
        }
    ).sort_values("on")
    timed = _events(events)
    right = _newest(timed.assign(on=pd.to_datetime(timed["known_from"])))
    columns = ["stock_id", "on", "first_known", column]
    window = pd.Timedelta(stale)
    merged = pd.merge_asof(left, right[columns], on="on", by="stock_id", tolerance=window)
    merged = merged.sort_values("row")
    merged.index = panel.index
    return merged[column].mask(merged["first_known"] < starts)


def _days_since(panel: pd.DataFrame, announced: pd.Series, calendar: TradingDays) -> pd.Series:
    """Trading days after the announcement up to each row's date."""
    days = np.array([day.toordinal() for day in calendar.between(calendar.start, calendar.end)])
    now = np.searchsorted(days, [day.toordinal() for day in panel["date"]])
    known = announced.notna()
    then = np.full(len(panel), -1)
    ordinals = [stamp.toordinal() for stamp in announced[known]]
    then[known.to_numpy()] = np.searchsorted(days, ordinals, side="right") - 1
    return pd.Series(np.where(known, now - then, np.nan), index=panel.index)


def _attached(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    names: Sequence[str],
    stale: timedelta,
    starts: pd.Series,
) -> dict[str, np.ndarray]:
    return {
        name: _latest(panel, signals[signals["signal"] == name], stale, starts).to_numpy(
            dtype=np.float64
        )
        for name in names
    }


def _filings(statements: pd.DataFrame) -> pd.DataFrame:
    """One event per chosen statement, visible from its own announce date."""
    frame = _preferred(statements)
    return _Lags(frame).signal("filed", pd.Series(np.nan, index=frame.index), ())


def attach_fundamentals(
    panel: pd.DataFrame,
    calendar: TradingDays,
    start: date,
    end: date,
    stock_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """New panel with revenue and statement signals known on each day, plus that day's valuation."""
    revenue = _rows(
        MonthlyRevenue,
        ["stock_id", "period", "revenue", "announce_date"],
        stock_ids,
        announce_date__range=(start - REVENUE_LOOKBACK, end),
    )
    statements = _rows(
        FinancialStatement,
        STATEMENT_COLUMNS,
        stock_ids,
        announce_date__range=(start - STATEMENT_LOOKBACK, end),
    )
    valuation = _rows(
        DailyValuation,
        ["stock_id", "date", "pb", "dividend_yield"],
        stock_ids,
        date__range=(start, end),
    )
    starts = segment_starts(panel, calendar)
    empty = pd.DataFrame(columns=SIGNAL_COLUMNS)
    monthly = revenue_signals(revenue) if not revenue.empty else empty
    sourcing = revenue_sourcing(revenue) if not revenue.empty else empty
    quarterly = statement_signals(statements) if not statements.empty else empty
    recorded = statement_sourcing(statements) if not statements.empty else empty
    filings = _filings(statements) if not statements.empty else empty
    seen = _latest(panel, filings, STATEMENT_LOOKBACK, starts, "known_from")
    keys = pd.MultiIndex.from_arrays([panel["stock_id"], panel["date"]])
    daily = valuation.set_index(["stock_id", "date"]).astype("float64").reindex(keys)
    return panel.assign(
        **_attached(panel, monthly, list(REVENUE_INPUTS), REVENUE_STALE, starts),
        **_attached(
            panel, sourcing, list(map(sourced_name, REVENUE_INPUTS)), REVENUE_STALE, starts
        ),
        **_attached(panel, quarterly, list(STATEMENT_INPUTS), STATEMENT_STALE, starts),
        **_attached(
            panel, recorded, list(map(sourced_name, STATEMENT_INPUTS)), STATEMENT_STALE, starts
        ),
        fund_days_since_fs=_days_since(panel, pd.to_datetime(seen), calendar),
        pb=daily["pb"].to_numpy(),
        dividend_yield=daily["dividend_yield"].to_numpy(),
    )
