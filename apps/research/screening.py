"""Single-factor rank IC on the tuning folds' training windows, pruning, dedup (SPEC §6 步驟 7)."""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from apps.research.features.registry import feature
from apps.research.features.transforms import STOCK_LEVEL
from apps.research.metrics.ic import MIN_STOCKS, daily_rank_ic, summarize_ic
from apps.research.training import training_rows
from apps.research.walkforward import Fold

IC_FLOOR = 0.005
CORRELATION_CEILING = 0.95
RANK_LABEL = "fwd_rank_pct"


@dataclass(frozen=True)
class Screening:
    """Per-feature ICs and verdicts, the reason each dropped feature left, and the correlations."""

    table: pd.DataFrame
    dropped: dict[str, str]
    correlation: pd.DataFrame

    @property
    def kept(self) -> list[str]:
        return [name for name in self.table.index if name not in self.dropped]


def screening_window(folds: Iterable[Fold]) -> tuple[date, date]:
    """First and last training day of the tuning folds, whose rolling windows overlap."""
    tuning = [fold for fold in folds if fold.tuning]
    if not tuning:
        raise ValueError("there are no tuning folds to screen features on")
    return min(fold.train_start for fold in tuning), max(fold.train_end for fold in tuning)


def single_factor_ic(frame: pd.DataFrame, features: Sequence[str], horizon: int) -> pd.DataFrame:
    """Mean daily rank IC of each feature, its Newey-West t with lag = horizon and day count."""
    rows = {}
    for name in features:
        summary = summarize_ic(daily_rank_ic(frame, name, RANK_LABEL), lag=horizon)
        rows[name] = {"mean": summary.mean, "nw_t": summary.nw_t, "n_days": summary.n_days}
    return pd.DataFrame.from_dict(rows, orient="index")


def mean_daily_correlation(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Average over days of the cross-sectional Spearman matrix, each pair on its shared stocks."""
    names = list(features)
    total = np.zeros((len(names), len(names)))
    count = np.zeros((len(names), len(names)))
    for _, day in frame.groupby("date", sort=True):
        matrix = day[names].corr(method="spearman", min_periods=MIN_STOCKS).to_numpy()
        known = ~np.isnan(matrix)
        total += np.where(known, matrix, 0.0)
        count += known
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
    return pd.DataFrame(mean, index=names, columns=names)


def _low_ic(names: Sequence[str], strength: pd.Series, horizons: str) -> dict[str, str]:
    reasons = {}
    for name in names:
        if feature(name).conditional:
            continue
        if math.isnan(strength[name]):
            reasons[name] = f"no rank IC: fewer than {MIN_STOCKS} stocks with values every day"
        elif strength[name] < IC_FLOOR:
            reasons[name] = f"|rank IC| below {IC_FLOOR} at horizons {horizons}"
    return reasons


def select(ic: pd.DataFrame, correlation: pd.DataFrame) -> dict[str, str]:
    """Drop reasons: signal features under the IC floor, then the weaker of each correlated pair."""
    order = {name: position for position, name in enumerate(STOCK_LEVEL)}
    names = sorted(ic.index, key=order.__getitem__)
    strength = ic.abs().max(axis=1)
    reasons = _low_ic(names, strength, " and ".join(str(horizon) for horizon in ic.columns))
    ranked = np.nan_to_num(strength.to_numpy(dtype=np.float64), nan=-np.inf)
    power = dict(zip(strength.index, ranked, strict=True))
    pairs = [
        (first, second, float(correlation.loc[first, second]))
        for position, first in enumerate(names)
        for second in names[position + 1 :]
        if abs(correlation.loc[first, second]) > CORRELATION_CEILING
    ]
    for first, second, rho in sorted(pairs, key=lambda pair: -abs(pair[2])):
        if first in reasons or second in reasons:
            continue
        weaker, stronger = (first, second) if power[first] < power[second] else (second, first)
        reasons[weaker] = f"duplicate of {stronger} (mean daily Spearman {rho:+.3f})"
    return reasons


def screen(features: pd.DataFrame, labels: Mapping[int, pd.DataFrame]) -> Screening:
    """Screen the stock-level columns of raw features against each horizon's labels."""
    columns: dict[str, pd.Series] = {}
    means = {}
    for horizon, frame in labels.items():
        summary = single_factor_ic(training_rows(features, frame, RANK_LABEL), STOCK_LEVEL, horizon)
        columns[f"ic_h{horizon}"] = summary["mean"]
        columns[f"nw_t_h{horizon}"] = summary["nw_t"]
        means[horizon] = summary["mean"]
    universe = features[features["in_universe"].astype(bool)]
    correlation = mean_daily_correlation(universe, STOCK_LEVEL)
    dropped = select(pd.DataFrame(means), correlation)
    table = pd.DataFrame(columns).assign(
        conditional=[feature(name).conditional for name in STOCK_LEVEL],
        reason=[dropped.get(name, "") for name in STOCK_LEVEL],
    )
    return Screening(table, dropped, correlation)
