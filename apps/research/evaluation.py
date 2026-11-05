"""Phase 4 evaluation: rank stability, deciles, interval coverage, leaks and consistency (§7)."""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from types import MappingProxyType

import numpy as np
import pandas as pd

from apps.market.calendar import TradingDays
from apps.research.baselines import INTERVAL_HORIZON, INTERVAL_Z, TRADING_YEAR
from apps.research.labels import UP_THRESHOLD
from apps.research.metrics.deciles import (
    LIQUIDITY,
    top_liquidity,
    tranche_returns,
    tranche_summary,
)
from apps.research.metrics.forecast import (
    coverage_by_bucket,
    daily_coverage,
    daily_width,
    research_rows,
    shared_rows,
)
from apps.research.metrics.ic import (
    DATE,
    SUMMARY_COLUMNS,
    ICSummary,
    daily_rank_ic,
    ic_by_fold,
    ic_by_year,
    neutralized_rank_ic,
    positive_fold_share,
    rank_ic_by_bucket,
    research_ic,
    summarize_ic,
)
from apps.research.walkforward import Fold

LEAK_LIMITS: Mapping[int, float] = MappingProxyType({5: 0.10, 20: 0.15})
TOP_FEATURES = 10
LIQUIDITY_BUCKETS = 3
VOLATILITY = "tech_vol_20d"
VOLATILITY_BUCKETS = 5
WIDTH_RATIO_BOUNDS = (0.5, 2.0)
EVEN_ODDS = 0.5
ALERT_SHARE = 0.10
CONSISTENCY_COLUMNS = ["mean_share", "max_share", "days_over", "n_days"]


def tradable_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Universe rows that trade at the next open: the rows rank metrics and deciles count."""
    return frame[frame["in_universe"].astype(bool) & frame["untradable_t1"].eq(False)]


def leaks(horizon: int, mean: float) -> bool:
    """Whether a research-period mean IC is high enough to be chased as a leak first."""
    return bool(mean > LEAK_LIMITS[horizon])


@dataclass(frozen=True)
class RankReport:
    horizon: int
    folds: tuple[Fold, ...]
    by_fold: pd.DataFrame
    by_year: pd.DataFrame
    total: ICSummary
    positive_share: float
    beat_share: float
    compared_folds: int
    by_liquidity: pd.DataFrame
    neutralized: ICSummary
    monthly: pd.Series
    monthly_days: pd.Series

    @property
    def leak(self) -> bool:
        return leaks(self.horizon, self.total.mean)


def _months(ic: pd.Series) -> pd.Index:
    months: pd.Index = pd.Index([day.replace(day=1) for day in ic.index], dtype=object)
    return months.rename("month")


def _beats(
    by_fold: pd.DataFrame,
    shared: pd.DataFrame,
    score: str,
    baseline: str,
    folds: Sequence[Fold],
    horizon: int,
) -> tuple[pd.DataFrame, float, int]:
    """Fold table with model and baseline IC on rows both score; share and count of folds compared.

    A fold without an IC on either side has no verdict and stays out of the share.
    """
    paired = ic_by_fold(daily_rank_ic(shared, score), folds, horizon)["mean"]
    compared = ic_by_fold(daily_rank_ic(shared, baseline), folds, horizon)["mean"]
    beats = (paired > compared).astype("boolean").mask(paired.isna() | compared.isna())
    table = by_fold.assign(paired_mean=paired, baseline_mean=compared, beats=beats)
    research = [fold.index for fold in folds if fold.research]
    verdicts = table.loc[research, "beats"].dropna()
    share = float(verdicts.astype(np.float64).mean()) if len(verdicts) else math.nan
    return table, share, len(verdicts)


def _by_liquidity(
    rows: pd.DataFrame, score: str, horizon: int, folds: Sequence[Fold]
) -> pd.DataFrame:
    table = rank_ic_by_bucket(rows, score, n_buckets=LIQUIDITY_BUCKETS)
    summaries = {
        bucket: asdict(summarize_ic(research_ic(table[bucket], folds), horizon))
        for bucket in range(1, LIQUIDITY_BUCKETS + 1)
    }
    return pd.DataFrame.from_dict(summaries, orient="index")[SUMMARY_COLUMNS].rename_axis("bucket")


def rank_report(
    frame: pd.DataFrame,
    score: str,
    horizon: int,
    folds: Sequence[Fold],
    baseline: str | None = None,
) -> RankReport:
    """Rank IC by fold, year and liquidity tercile, and neutralized, over tradable universe rows.

    A baseline is compared fold by fold on rows both score; the model's own metrics keep its rows.
    """
    chosen = tuple(folds)
    rows = tradable_rows(frame)
    ic = daily_rank_ic(rows, score)
    counted = research_ic(ic, chosen)
    by_fold = ic_by_fold(ic, chosen, horizon)
    beat, compared = math.nan, 0
    if baseline is not None:
        shared = shared_rows(rows, [score, baseline])
        by_fold, beat, compared = _beats(by_fold, shared, score, baseline, chosen, horizon)
    return RankReport(
        horizon=horizon,
        folds=chosen,
        by_fold=by_fold,
        by_year=ic_by_year(counted, horizon),
        total=summarize_ic(counted, horizon),
        positive_share=positive_fold_share(by_fold),
        beat_share=beat,
        compared_folds=compared,
        by_liquidity=_by_liquidity(rows, score, horizon, chosen),
        neutralized=summarize_ic(research_ic(neutralized_rank_ic(rows, score), chosen), horizon),
        monthly=counted.groupby(_months(counted)).mean(),
        monthly_days=counted.groupby(_months(counted)).size(),
    )


@dataclass(frozen=True)
class ImportanceDrift:
    by_fold: pd.DataFrame
    frequency: pd.Series


def importance_drift(
    importance: Mapping[int, Mapping[str, float]], top: int = TOP_FEATURES
) -> ImportanceDrift:
    """Each fold's top features by gain (ties by name), overlap with the fold before, counts."""
    tops = {
        fold: tuple(sorted(gains, key=lambda name: (-gains[name], name))[:top])
        for fold, gains in sorted(importance.items())
    }
    rows: list[dict[str, object]] = []
    previous: tuple[str, ...] | None = None
    for fold, names in tops.items():
        kept = math.nan if previous is None else len(set(names) & set(previous))
        rows.append({"fold": fold, "top": names, "kept": kept})
        previous = names
    counts = Counter(name for names in tops.values() for name in names)
    ordered = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    by_fold = pd.DataFrame(rows, columns=["fold", "top", "kept"]).set_index("fold")
    return ImportanceDrift(by_fold, pd.Series(ordered, dtype=np.int64, name="folds"))


def _last_hold(calendar: TradingDays, last_signal: date, horizon: int) -> date:
    """Close a tranche bought the day after last_signal exits on, or the calendar end."""
    following = calendar.between(last_signal, calendar.end)
    return following[min(horizon, len(following) - 1)]


def decile_report(
    frame: pd.DataFrame,
    score: str,
    horizon: int,
    folds: Sequence[Fold],
    prices: pd.DataFrame,
    calendar: TradingDays,
) -> pd.DataFrame:
    """Decile summaries net of costs on research-fold signals, for all rows and the liquid third.

    Prices after the last research tranche's exit are never read, so holdout prices stay unseen.
    """
    rows = research_rows(tradable_rows(frame), folds)
    rows = rows[rows[score].notna()]
    signal = rows[[DATE, "stock_id", LIQUIDITY]].assign(score=rows[score])
    held = prices.iloc[0:0]
    if len(signal):
        held = prices[prices[DATE] <= _last_hold(calendar, signal[DATE].max(), horizon)]
    samples = {"all": signal, "liquid": top_liquidity(signal)}
    tables = {
        name: tranche_summary(tranche_returns(part, held, calendar, horizon), horizon)
        for name, part in samples.items()
    }
    return pd.concat(tables, names=["sample"])


def coverage_report(
    frame: pd.DataFrame, label: str, lower: str, upper: str, folds: Sequence[Fold]
) -> pd.DataFrame:
    """Mean daily coverage and width per daily tech_vol_20d quintile of research rows, then all."""
    rows = research_rows(tradable_rows(frame), folds)
    daily = coverage_by_bucket(rows, label, lower, upper, VOLATILITY, VOLATILITY_BUCKETS)
    table = daily.groupby("bucket")[["coverage", "width"]].mean()
    scored = shared_rows(rows, [label, lower, upper])
    overall = {
        "coverage": daily_coverage(scored, label, lower, upper).mean(),
        "width": daily_width(scored, lower, upper).mean(),
    }
    return pd.concat([table, pd.DataFrame(overall, index=["all"])])


def band_width(vol: pd.Series) -> pd.Series:
    """Return width of the 80% volatility band exp(±1.28 × vol × √(20/252))."""
    half = INTERVAL_Z * vol * math.sqrt(INTERVAL_HORIZON / TRADING_YEAR)
    width: pd.Series = np.exp(half) - np.exp(-half)
    return width


def _shares(rows: pd.DataFrame, flagged: pd.Series) -> dict[str, float]:
    daily = flagged.astype(np.float64).groupby(rows[DATE]).mean()
    return {
        "mean_share": float(daily.mean()) if len(daily) else math.nan,
        "max_share": float(daily.max()) if len(daily) else math.nan,
        "days_over": int((daily > ALERT_SHARE).sum()),
        "n_days": len(daily),
    }


def consistency_report(
    frame: pd.DataFrame,
    q10: str,
    q50: str,
    q90: str,
    vol: str,
    prob: str,
    folds: Sequence[Fold],
) -> pd.DataFrame:
    """Daily shares of research universe rows whose bands disagree or whose odds contradict q50."""
    universe = research_rows(frame[frame["in_universe"].astype(bool)], folds)
    banded = shared_rows(universe, [q10, q90, vol])
    ratio = band_width(banded[vol]) / (banded[q90] - banded[q10])
    low, high = WIDTH_RATIO_BOUNDS
    voted = shared_rows(universe, [q50, prob])
    against = ((voted[prob] > EVEN_ODDS) & (voted[q50] < UP_THRESHOLD)) | (
        (voted[prob] < EVEN_ODDS) & (voted[q50] > UP_THRESHOLD)
    )
    table = {
        "width_ratio": _shares(banded, ~ratio.between(low, high)),
        "contradiction": _shares(voted, against),
    }
    return pd.DataFrame.from_dict(table, orient="index")[CONSISTENCY_COLUMNS]
