"""Daily cross-sectional rank IC with fold, year, neutralized and bucketed views (SPEC §7)."""

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields

import numpy as np
import pandas as pd

from apps.research.metrics.stats import equal_count_buckets, is_flat, newey_west_t
from apps.research.walkforward import Fold

DATE = "date"
LABEL = "fwd_rank_pct"
MIN_STOCKS = 30
NEUTRAL_EXPOSURES = ("size_log_mcap", "risk_beta_250d", "tech_vol_20d")
EXPLAINED_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ICSummary:
    mean: float
    std: float
    icir: float
    nw_t: float
    positive_day_share: float
    n_days: int


SUMMARY_COLUMNS = [field.name for field in fields(ICSummary)]
FOLD_COLUMNS = ["fold", "test_start", "eval_end", "tuning", "holdout", *SUMMARY_COLUMNS]


def _empty_ic() -> pd.Series:
    return pd.Series([], dtype=np.float64, name="ic").rename_axis(DATE)


def daily_rank_ic(
    frame: pd.DataFrame, score: str, label: str = LABEL, min_stocks: int = MIN_STOCKS
) -> pd.Series:
    """Per-date Spearman correlation of score and label over rows holding both values."""
    rows = frame.loc[frame[score].notna() & frame[label].notna(), [DATE, score, label]]
    rows = rows.reset_index(drop=True)
    if rows.empty:
        return _empty_ic()
    keys = rows[DATE]
    ranks = rows.groupby(DATE)[[score, label]].rank()
    x = ranks[score] - ranks[score].groupby(keys).transform("mean")
    y = ranks[label] - ranks[label].groupby(keys).transform("mean")
    covariance = (x * y).groupby(keys).sum()
    scale = ((x * x).groupby(keys).sum() * (y * y).groupby(keys).sum()).pow(0.5)
    counts = keys.groupby(keys).size()
    ic = (covariance / scale.where(scale > 0)).where(counts >= min_stocks).dropna()
    return ic.rename("ic").rename_axis(DATE)


def summarize_ic(ic: pd.Series, lag: int) -> ICSummary:
    """Mean, std, ICIR, Newey-West t with lag = horizon, share of positive days."""
    values = ic.dropna().to_numpy(dtype=np.float64)
    n = int(values.size)
    mean = float(values.mean()) if n else math.nan
    std = float(values.std(ddof=1)) if n > 1 else math.nan
    icir = mean / std if n > 1 and not is_flat(std, mean) else math.nan
    positive = float((values > 0).mean()) if n else math.nan
    return ICSummary(mean, std, icir, newey_west_t(values, lag), positive, n)


def ic_by_fold(ic: pd.Series, folds: Iterable[Fold], lag: int) -> pd.DataFrame:
    """One summary row per fold over test_start..eval_end, keeping the tuning and holdout flags."""
    days = pd.Index(ic.index)
    rows = [
        {
            "fold": fold.index,
            "test_start": fold.test_start,
            "eval_end": fold.eval_end,
            "tuning": fold.tuning,
            "holdout": fold.holdout,
            **asdict(summarize_ic(ic[(days >= fold.test_start) & (days <= fold.eval_end)], lag)),
        }
        for fold in folds
    ]
    return pd.DataFrame(rows, columns=FOLD_COLUMNS).set_index("fold")


def positive_fold_share(by_fold: pd.DataFrame) -> float:
    """Share of research folds with IC days whose mean IC is above zero."""
    counted = ~by_fold["tuning"].astype(bool) & ~by_fold["holdout"].astype(bool)
    means = by_fold.loc[counted & (by_fold["n_days"] > 0), "mean"]
    return float((means > 0).mean()) if len(means) else math.nan


def research_ic(ic: pd.Series, folds: Iterable[Fold]) -> pd.Series:
    """IC days that count toward research results: research folds, test_start to eval_end."""
    days = pd.Index(ic.index)
    keep = np.zeros(len(days), dtype=bool)
    for fold in folds:
        if fold.research:
            keep |= (days >= fold.test_start) & (days <= fold.eval_end)
    return ic[keep]


def ic_by_year(ic: pd.Series, lag: int) -> pd.DataFrame:
    """One summary row per calendar year of the IC dates."""
    years = pd.Index([day.year for day in ic.index])
    rows = [
        {"year": int(year), **asdict(summarize_ic(ic[years == year], lag))}
        for year in sorted(years.unique())
    ]
    return pd.DataFrame(rows, columns=["year", *SUMMARY_COLUMNS]).set_index("year")


def _residuals(
    group: pd.DataFrame, score: str, industry: str, exposures: Sequence[str]
) -> pd.Series:
    dummies = pd.get_dummies(group[industry]).to_numpy(dtype=np.float64)
    design = np.column_stack([dummies, group[list(exposures)].to_numpy(dtype=np.float64)])
    target = group[score].to_numpy(dtype=np.float64)
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ coefficients
    if residual.std() <= EXPLAINED_TOLERANCE * target.std():
        residual = np.full_like(residual, np.nan)
    return pd.Series(residual, index=group.index)


def neutralized_rank_ic(
    frame: pd.DataFrame,
    score: str,
    label: str = LABEL,
    exposures: Sequence[str] = NEUTRAL_EXPOSURES,
    industry: str = "industry",
    min_stocks: int = MIN_STOCKS,
) -> pd.Series:
    """Rank IC of score residuals from a daily regression on industry dummies and exposures."""
    columns = [score, label, *exposures]
    named = frame[industry].notna() & (frame[industry].astype(str).str.strip() != "")
    rows = frame.loc[named & frame[columns].notna().all(axis=1), [DATE, industry, *columns]]
    rows = rows.reset_index(drop=True)
    parts = [
        _residuals(group, score, industry, exposures)
        for _, group in rows.groupby(DATE, sort=True)
        if len(group) >= min_stocks
    ]
    if not parts:
        return _empty_ic()
    residual = pd.concat(parts)
    scored = rows.loc[residual.index, [DATE, label]].assign(residual=residual)
    return daily_rank_ic(scored, "residual", label, min_stocks)


def rank_ic_by_bucket(
    frame: pd.DataFrame,
    score: str,
    label: str = LABEL,
    by: str = "liq_dollar_vol_med_20d",
    n_buckets: int = 3,
    min_stocks: int = MIN_STOCKS,
) -> pd.DataFrame:
    """Daily rank IC in equal-count buckets of by over rows with score and label; 1 is lowest."""
    needed = list(dict.fromkeys([DATE, score, label, by]))
    rows = frame.loc[frame[[score, label, by]].notna().all(axis=1), needed].reset_index(drop=True)
    bucket = equal_count_buckets(rows[by], rows[DATE], n_buckets)
    return pd.DataFrame(
        {
            number: daily_rank_ic(rows[bucket == number], score, label, min_stocks)
            for number in range(1, n_buckets + 1)
        }
    )
