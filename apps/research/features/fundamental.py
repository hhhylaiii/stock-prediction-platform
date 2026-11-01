"""Fundamental features of SPEC §6, each visible from the day every input was announced."""

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
import pandas as pd
from django.db import models

from apps.market.calendar import TradingDays
from apps.market.models import DailyValuation, FinancialStatement, FinType, MonthlyRevenue
from apps.research.features.common import feature_frame, ratio
from apps.research.features.registry import Group, feature_names
from apps.research.panel import require_sorted

TTM_QUARTERS = 4
REVENUE_LAGS = (1, 2, 12, 13, 14)
STATEMENT_LAGS = tuple(range(1, 2 * TTM_QUARTERS))
REVENUE_SIGNALS = ["fund_rev_yoy", "fund_rev_3m_yoy", "fund_rev_mom"]
STATEMENT_SIGNALS = [
    "eps_ttm",
    "fund_eps_yoy",
    "fund_roe_ttm",
    "fund_gross_margin",
    "fund_op_margin",
    "fund_debt_ratio",
    "fund_gm_yoy_chg",
    "fund_opm_yoy_chg",
    "fund_cfo_to_ni",
]
REVENUE_LOOKBACK = timedelta(days=500)
STATEMENT_LOOKBACK = timedelta(days=900)
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
    "announce_date",
]
PASSED = [*REVENUE_SIGNALS, *STATEMENT_SIGNALS[1:], "fund_days_since_fs"]


class _Lags:
    """Values of earlier periods of the same stock, by period number."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.lookup = frame.set_index(["stock_id", "number"])

    def __call__(self, column: str, lag: int) -> pd.Series:
        keys = pd.MultiIndex.from_arrays([self.frame["stock_id"], self.frame["number"] - lag])
        return pd.Series(self.lookup[column].reindex(keys).to_numpy(), index=self.frame.index)

    def known_from(self, lags: Sequence[int]) -> pd.Series:
        """Latest announce date among the period and its lags that exist."""
        dates = [pd.to_datetime(self.frame["announce_date"])]
        dates += [pd.to_datetime(self("announce_date", lag)) for lag in lags]
        latest: pd.Series = pd.concat(dates, axis=1).max(axis=1).dt.date
        return latest


def _numbered(frame: pd.DataFrame, separator: str, per_year: int) -> pd.DataFrame:
    year = frame["period"].str[:4].astype(int)
    part = frame["period"].str.split(separator).str[-1].str.lstrip("Q").astype(int)
    numbered = frame.assign(number=year * per_year + part - 1)
    return numbered.sort_values(["stock_id", "number"], ignore_index=True)


def _growth(now: pd.Series, base: pd.Series) -> pd.Series:
    """now / base − 1, missing when the base is not positive."""
    return now / base.where(base > 0) - 1


def revenue_signals(revenue: pd.DataFrame) -> pd.DataFrame:
    """Single-month yoy, three-month yoy and mom of each announced month."""
    frame = _numbered(revenue.assign(revenue=revenue["revenue"].astype("float64")), "-", 12)
    lag = _Lags(frame)
    now = frame["revenue"]
    three = now + lag("revenue", 1) + lag("revenue", 2)
    base = lag("revenue", 12) + lag("revenue", 13) + lag("revenue", 14)
    return frame.assign(
        fund_rev_yoy=_growth(now, lag("revenue", 12)),
        fund_rev_3m_yoy=_growth(three, base),
        fund_rev_mom=_growth(now, lag("revenue", 1)),
        known_from=lag.known_from(REVENUE_LAGS),
    )[["stock_id", "period", "announce_date", "known_from", *REVENUE_SIGNALS]]


def _preferred(statements: pd.DataFrame) -> pd.DataFrame:
    """One statement per stock and quarter, consolidated over individual."""
    rank = statements["fin_type"].ne(FinType.CONSOLIDATED).astype(int)
    ordered = statements.assign(rank=rank).sort_values(["stock_id", "period", "rank"])
    return ordered.drop_duplicates(["stock_id", "period"]).drop(columns="rank")


def _ttm(lag: _Lags, frame: pd.DataFrame, column: str, offset: int = 0) -> pd.Series:
    """Sum of four consecutive quarters ending offset quarters back; missing if any is absent."""
    parts = [frame[column] if offset == 0 else lag(column, offset)]
    parts += [lag(column, offset + k) for k in range(1, TTM_QUARTERS)]
    return pd.concat(parts, axis=1).sum(axis=1, min_count=TTM_QUARTERS)


def statement_signals(statements: pd.DataFrame) -> pd.DataFrame:
    """TTM EPS and its growth, TTM ROE and cash conversion, margins and their yoy changes."""
    numeric = [
        "eps",
        "gross_margin",
        "op_margin",
        "net_income",
        "roe",
        "debt_ratio",
        "operating_cf",
    ]
    chosen = _preferred(statements)
    frame = _numbered(chosen.astype(dict.fromkeys(numeric, "float64")), "-Q", 4)
    lag = _Lags(frame)
    eps = _ttm(lag, frame, "eps")
    income = _ttm(lag, frame, "net_income")
    return frame.assign(
        eps_ttm=eps,
        fund_eps_yoy=_growth(eps, _ttm(lag, frame, "eps", TTM_QUARTERS)),
        fund_roe_ttm=_ttm(lag, frame, "roe"),
        fund_gross_margin=frame["gross_margin"],
        fund_op_margin=frame["op_margin"],
        fund_debt_ratio=frame["debt_ratio"],
        fund_gm_yoy_chg=frame["gross_margin"] - lag("gross_margin", TTM_QUARTERS),
        fund_opm_yoy_chg=frame["op_margin"] - lag("op_margin", TTM_QUARTERS),
        fund_cfo_to_ni=ratio(_ttm(lag, frame, "operating_cf"), income.where(income > 0)),
        known_from=lag.known_from(STATEMENT_LAGS),
    )[["stock_id", "period", "announce_date", "known_from", *STATEMENT_SIGNALS]]


def fundamental_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Fundamental features of every row; needs the attached signals, valuation and close."""
    require_sorted(panel)
    close = panel["close"].groupby([panel["stock_id"], panel["segment"]], sort=False).ffill()
    columns = {
        **{name: panel[name] for name in PASSED},
        "fund_ep": ratio(panel["eps_ttm"], close),
        "fund_bp": pd.Series(1.0 / panel["pb"].where(panel["pb"] > 0), index=panel.index),
        "fund_dividend_yield": panel["dividend_yield"],
    }
    names = feature_names(Group.FUNDAMENTAL)
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


def _segment_starts(panel: pd.DataFrame, calendar: TradingDays) -> pd.Series:
    """First trading day of each row's listing for reused codes (segment 2 on); NaT otherwise."""
    later = panel["segment"] > 1
    days, ages = panel.loc[later, "date"].tolist(), panel.loc[later, "age"].astype(int).tolist()
    firsts = [pd.Timestamp(calendar.shift(day, -age)) for day, age in zip(days, ages, strict=True)]
    starts = pd.Series(pd.NaT, index=panel.index, dtype="datetime64[ns]")
    starts[later] = firsts
    return starts


def _latest(
    panel: pd.DataFrame, events: pd.DataFrame, on: str, columns: list[str], starts: pd.Series
) -> pd.DataFrame:
    """Latest event per row with event[on] on or before the row's date and inside its listing."""
    if events.empty or panel.empty:
        return pd.DataFrame(np.nan, index=panel.index, columns=columns)
    left = pd.DataFrame(
        {
            "row": np.arange(len(panel)),
            "stock_id": panel["stock_id"].to_numpy(),
            "on": pd.to_datetime(panel["date"]),
        }
    ).sort_values("on")
    right = events.assign(on=pd.to_datetime(events[on]), at=pd.to_datetime(events["announce_date"]))
    ordered = right.sort_values(["on", "period"])[["stock_id", "on", "at", *columns]]
    merged = pd.merge_asof(left, ordered, on="on", by="stock_id").sort_values("row")
    merged.index = panel.index
    stale = merged["at"] < starts
    return merged[columns].mask(stale)


def _days_since(panel: pd.DataFrame, announced: pd.Series, calendar: TradingDays) -> pd.Series:
    """Trading days after the announcement up to each row's date."""
    days = np.array([day.toordinal() for day in calendar.between(calendar.start, calendar.end)])
    now = np.searchsorted(days, [day.toordinal() for day in panel["date"]])
    known = announced.notna()
    then = np.full(len(panel), -1)
    ordinals = [stamp.toordinal() for stamp in announced[known]]
    then[known.to_numpy()] = np.searchsorted(days, ordinals, side="right") - 1
    return pd.Series(np.where(known, now - then, np.nan), index=panel.index)


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
    starts = _segment_starts(panel, calendar)
    signals = statement_signals(statements) if not statements.empty else statements
    with_revenue = _latest(
        panel,
        revenue_signals(revenue) if not revenue.empty else revenue,
        "known_from",
        REVENUE_SIGNALS,
        starts,
    )
    with_statements = _latest(panel, signals, "known_from", STATEMENT_SIGNALS, starts)
    announced = _latest(
        panel, signals.assign(seen=signals.get("announce_date")), "announce_date", ["seen"], starts
    )
    keys = pd.MultiIndex.from_arrays([panel["stock_id"], panel["date"]])
    daily = valuation.set_index(["stock_id", "date"]).astype("float64").reindex(keys)
    return panel.assign(
        **{name: with_revenue[name].to_numpy(dtype=np.float64) for name in REVENUE_SIGNALS},
        **{name: with_statements[name].to_numpy(dtype=np.float64) for name in STATEMENT_SIGNALS},
        fund_days_since_fs=_days_since(panel, pd.to_datetime(announced["seen"]), calendar),
        pb=daily["pb"].to_numpy(),
        dividend_yield=daily["dividend_yield"].to_numpy(),
    )
