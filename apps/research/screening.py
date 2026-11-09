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
MIN_COVERAGE = 0.5
RANK_LABEL = "fwd_rank_pct"


@dataclass(frozen=True)
class Screening:
    """Per-feature ICs and verdicts, the reason each dropped feature left, and the correlations."""

    table: pd.DataFrame
    dropped: dict[str, str]
    correlation: pd.DataFrame
    days: int

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
    """Signal features under the floor at every horizon; one lacking an IC anywhere is kept."""
    return {
        name: f"|rank IC| below {IC_FLOOR} at horizons {horizons}"
        for name in names
        if not feature(name).conditional and strength[name] < IC_FLOOR
    }


def _twin(name: str, kept: Sequence[str], correlation: pd.DataFrame) -> str | None:
    row = np.abs(correlation.loc[[name], list(kept)].to_numpy(dtype=np.float64)[0])
    closeness = dict(zip(kept, row, strict=True))
    tied = [other for other in kept if closeness[other] > CORRELATION_CEILING]
    return max(tied, key=closeness.__getitem__, default=None)


def select(ic: pd.DataFrame, correlation: pd.DataFrame) -> dict[str, str]:
    """Drop reasons: signal features under the IC floor, then features tied to a stronger kept one.

    Features are kept in order of max |IC| over horizons, ties in the order ic lists them.
    """
    order = {name: position for position, name in enumerate(ic.index)}
    strength = ic.abs().max(axis=1)
    judged = strength.where(ic.notna().all(axis=1))
    reasons = _low_ic(list(ic.index), judged, " and ".join(str(h) for h in ic.columns))
    ranked = strength.fillna(-math.inf)
    kept: list[str] = []
    for name in sorted(set(ic.index) - set(reasons), key=lambda n: (-ranked[n], order[n])):
        twin = _twin(name, kept, correlation)
        if twin is None:
            kept.append(name)
        else:
            rho = float(correlation.loc[name, twin])
            reasons[name] = f"duplicate of {twin} (mean daily Spearman {rho:+.3f})"
    return reasons


def _require_labels(features: pd.DataFrame, labels: pd.DataFrame, horizon: int) -> None:
    days = pd.Index(features["date"].unique())
    missing = days.difference(pd.Index(labels["date"].unique()))
    if len(missing):
        raise ValueError(
            f"labels at horizon {horizon} are missing on {len(missing)} of {len(days)} days "
            f"from {min(missing)}; run build_labels over the screening window"
        )


def screen(
    features: pd.DataFrame,
    labels: Mapping[int, pd.DataFrame],
    names: Sequence[str] = STOCK_LEVEL,
) -> Screening:
    """Screen the stock-level columns of raw features against each horizon's labels.

    An IC over fewer than MIN_COVERAGE of the window's days is left out of select.
    """
    names = list(names)
    days = features["date"].nunique()
    columns: dict[str, pd.Series] = {}
    means = {}
    for horizon, frame in labels.items():
        _require_labels(features, frame, horizon)
        summary = single_factor_ic(training_rows(features, frame, RANK_LABEL), names, horizon)
        columns[f"ic_h{horizon}"] = summary["mean"]
        columns[f"nw_t_h{horizon}"] = summary["nw_t"]
        columns[f"days_h{horizon}"] = summary["n_days"]
        means[horizon] = summary["mean"].where(summary["n_days"] >= MIN_COVERAGE * days)
    universe = features[features["in_universe"].astype(bool)]
    correlation = mean_daily_correlation(universe, names)
    screened = pd.DataFrame(means)
    dropped = select(screened, correlation)
    table = pd.DataFrame(columns).assign(
        conditional=[feature(name).conditional for name in names],
        covered=screened.notna().all(axis=1),
        reason=[dropped.get(name, "") for name in names],
    )
    return Screening(table, dropped, correlation, days)
