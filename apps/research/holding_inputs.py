"""Inputs of the holding books: cached research scores or stored predictions, prices and flags."""

from bisect import bisect_right
from collections.abc import Sequence
from datetime import date

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow.parquet as pq

from apps.market.calendar import TradingDays
from apps.market.models import IndexDaily, IndexId, StockMaster
from apps.research.holding import Market, Mask, Rule
from apps.research.holdout import SPEC_HOLDOUT_START
from apps.research.lgbm_runs import MODEL
from apps.research.models import LabelDaily, ModelObjective, Prediction
from apps.research.panel import load_price_panel
from apps.research.prediction_cache import MissingFoldsError, fold_path, holds_holdout
from apps.research.pricelimit import MATCH_TOLERANCE, apply_price_limits

HORIZON = 5
SCORE = f"{MODEL}_h{HORIZON}"
KEYS = ["date", "stock_id"]
SCORE_COLUMNS = [*KEYS, "in_universe", "score"]
HOLDOUT_RULES = frozenset({(0.70, 30), (0.90, 30), (0.90, 100)})
INDEX = IndexId.TAIEX_TR
SHADOW_START = date(2026, 11, 16)


class HoldoutRuleError(ValueError):
    """A rule the one holdout pass never ran, asked for on holdout folds."""


def check_holdout_rule(run_key: str, folds: Sequence[int], rule: Rule, end: date) -> None:
    """Holdout folds only alone and for H70/N30, H90/N30, H90/N100; research folds price before."""
    held = [index for index in folds if holds_holdout(run_key, MODEL, index)]
    if held and len(held) < len(folds):
        raise HoldoutRuleError(f"folds {held} of {run_key} are holdout folds; run them apart")
    if held and (rule.exit_below, rule.slots) not in HOLDOUT_RULES:
        raise HoldoutRuleError(
            f"folds {held} of {run_key} are holdout folds; only H70/N30, H90/N30 and H90/N100 "
            "may be rerun there"
        )
    if not held and end >= SPEC_HOLDOUT_START:
        raise HoldoutRuleError(
            f"research folds price before the holdout period from {SPEC_HOLDOUT_START}, "
            f"not through {end}"
        )


def research_scores(
    run_key: str, folds: Sequence[int], until: date, end: date, rule: Rule
) -> pd.DataFrame:
    """Cached fold scores through until, after the holdout checks for a book valued to end."""
    check_holdout_rule(run_key, folds, rule, end)
    return _cached_scores(run_key, folds, until)


def _cached_scores(run_key: str, folds: Sequence[int], until: date) -> pd.DataFrame:
    """LightGBM 5-day scores and universe flags of cached folds, signal days through until."""
    absent = [index for index in folds if not fold_path(run_key, MODEL, index).exists()]
    if absent:
        raise MissingFoldsError(run_key, MODEL, absent)
    columns = [*KEYS, "in_universe", SCORE]
    parts = [
        pq.read_table(fold_path(run_key, MODEL, index), columns=columns).to_pandas()
        for index in folds
    ]
    scores = pd.concat(parts, ignore_index=True).rename(columns={SCORE: "score"})
    scores = scores[scores["date"] <= until]
    if scores.duplicated(KEYS).any():
        raise ValueError("a stock and day is scored by more than one fold")
    chosen: pd.DataFrame = scores[SCORE_COLUMNS].reset_index(drop=True)
    return chosen


def stored_scores(start: date, end: date) -> pd.DataFrame:
    """Rank 5-day prediction rows of the signal days."""
    rows = Prediction.objects.filter(
        run__objective=ModelObjective.RANK, horizon=HORIZON, date__range=(start, end)
    ).values_list(*SCORE_COLUMNS)
    frame = pd.DataFrame(list(rows), columns=SCORE_COLUMNS)
    return frame.astype({"score": "float64", "in_universe": "bool"})


def labels_settled(day: date) -> bool:
    """Whether the day's 5-day labels exist and none still waits for the next day's prices."""
    rows = LabelDaily.objects.filter(date=day, horizon=HORIZON)
    return rows.exists() and not rows.filter(untradable_t1__isnull=True).exists()


def _wide(frame: pd.DataFrame, column: str, days: list[date], stocks: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(np.nan, index=days, columns=stocks)
    table = frame.pivot(index="date", columns="stock_id", values=column)
    return table.reindex(index=days, columns=stocks)


def _flags(frame: pd.DataFrame, column: str, days: list[date], stocks: list[str]) -> Mask:
    marked = frame.assign(flag=frame[column].astype("float64"))
    return _wide(marked, "flag", days, stocks).eq(1.0).to_numpy()


def _blocked(days: list[date], stocks: list[str]) -> Mask:
    rows = LabelDaily.objects.filter(
        horizon=HORIZON, untradable_t1=True, date__range=(days[0], days[-1]), stock_id__in=stocks
    ).values_list(*KEYS)
    frame = pd.DataFrame(list(rows), columns=KEYS).assign(untradable=True)
    return _flags(frame, "untradable", days, stocks)


def _delisted(days: list[date], stocks: list[str]) -> npt.NDArray[np.int64]:
    """Index of the first day after each stock's delist_date; len(days) without one."""
    rows = StockMaster.objects.filter(stock_id__in=stocks, delist_date__isnull=False)
    dates = {stock: day for stock, day in rows.values_list("stock_id", "delist_date") if day}
    return np.array(
        [bisect_right(days, dates[stock]) if stock in dates else len(days) for stock in stocks],
        dtype=np.int64,
    )


def _prices(calendar: TradingDays, days: list[date], stocks: list[str]) -> pd.DataFrame:
    panel = apply_price_limits(load_price_panel(calendar, days[0], days[-1], stocks))
    opened_down = panel["open"] <= panel["limit_down"] + MATCH_TOLERANCE
    return panel.assign(limit_down_open=opened_down)


def load_market(scores: pd.DataFrame, end: date, calendar: TradingDays) -> Market:
    """Grids of the scored stocks from the first signal day through end."""
    if scores.empty:
        raise ValueError("there are no scores to hold")
    days = list(calendar.between(scores["date"].min(), end))
    stocks = sorted(set(scores["stock_id"]))
    prices = _prices(calendar, days, stocks)
    return Market(
        days=days,
        stocks=stocks,
        opens=_wide(prices, "adj_open", days, stocks).to_numpy(dtype=np.float64),
        closes=_wide(prices, "adj_close", days, stocks).to_numpy(dtype=np.float64),
        score=_wide(scores, "score", days, stocks).to_numpy(dtype=np.float64),
        in_universe=_flags(scores, "in_universe", days, stocks),
        blocked=_blocked(days, stocks),
        delisted=_delisted(days, stocks),
        limit_down_open=_flags(prices, "limit_down_open", days, stocks),
    )


def index_levels(start: date, end: date) -> pd.Series:
    """TAIEX_TR closes of the days, by date."""
    rows = IndexDaily.objects.filter(index_id=INDEX, date__range=(start, end)).values_list(
        "date", "close"
    )
    return pd.Series({day: float(close) for day, close in rows}, dtype="float64").sort_index()
