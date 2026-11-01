"""Daily interval, return, volatility and probability metrics for walk-forward folds (SPEC §7)."""

import math
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType

import numpy as np
import numpy.typing as npt
import pandas as pd

from apps.research.metrics.ic import DATE, MIN_STOCKS
from apps.research.walkforward import Fold

QUANTILE_COLUMNS: Mapping[float, str] = MappingProxyType({0.1: "q10", 0.5: "q50", 0.9: "q90"})
FOLD_COLUMNS = ["fold", "test_start", "eval_end", "tuning", "holdout", "mean", "n_days"]
RELIABILITY_COLUMNS = ["lower", "upper", "mean_prob", "frac_positive", "n_rows"]

Values = npt.NDArray[np.float64]


def _rows(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    needed = list(dict.fromkeys([DATE, *columns]))
    keep = frame[list(columns)].notna().all(axis=1)
    return frame.loc[keep, needed].reset_index(drop=True)


def _by_day(rows: pd.DataFrame, values: npt.ArrayLike, name: str) -> pd.Series:
    series = pd.Series(np.asarray(values, dtype=np.float64), index=rows[DATE].to_numpy())
    return series.groupby(level=0, sort=True).mean().rename(name).rename_axis(DATE)


def _require_binary(values: pd.Series) -> None:
    if not values.isin([0, 1]).all():
        raise ValueError("labels must be 0 or 1")


def _require_probability(values: pd.Series) -> None:
    if ((values < 0) | (values > 1)).any():
        raise ValueError("probabilities must be between 0 and 1")


def pinball_loss(actual: npt.ArrayLike, predicted: npt.ArrayLike, alpha: float) -> Values:
    """Quantile loss per row: alpha above the prediction, 1 - alpha below it."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    miss = np.asarray(actual, dtype=np.float64) - np.asarray(predicted, dtype=np.float64)
    return np.maximum(alpha * miss, (alpha - 1) * miss)


def daily_pinball(
    frame: pd.DataFrame, label: str, quantiles: Mapping[float, str] = QUANTILE_COLUMNS
) -> pd.Series:
    """Per-day mean of the pinball loss summed over the quantile columns."""
    rows = _rows(frame, [label, *quantiles.values()])
    total = sum(
        (pinball_loss(rows[label], rows[column], alpha) for alpha, column in quantiles.items()),
        np.zeros(len(rows)),
    )
    return _by_day(rows, total, "pinball")


def daily_coverage(frame: pd.DataFrame, label: str, lower: str, upper: str) -> pd.Series:
    """Per-day share of labels inside [lower, upper], bounds included."""
    rows = _rows(frame, [label, lower, upper])
    inside = (rows[label] >= rows[lower]) & (rows[label] <= rows[upper])
    return _by_day(rows, inside, "coverage")


def daily_width(frame: pd.DataFrame, lower: str, upper: str) -> pd.Series:
    """Per-day mean interval width."""
    rows = _rows(frame, [lower, upper])
    return _by_day(rows, rows[upper] - rows[lower], "width")


def coverage_by_bucket(
    frame: pd.DataFrame, label: str, lower: str, upper: str, by: str, n_buckets: int = 5
) -> pd.DataFrame:
    """Mean daily coverage and width in equal-count daily buckets of by; 1 is lowest."""
    if n_buckets < 1:
        raise ValueError("n_buckets must be at least 1")
    rows = _rows(frame, [label, lower, upper, by])
    grouped = rows.groupby(DATE)[by]
    position = grouped.rank(method="first").astype(np.int64) - 1
    bucket = position * n_buckets // grouped.transform("size") + 1
    table = {
        number: {
            "coverage": daily_coverage(rows[bucket == number], label, lower, upper).mean(),
            "width": daily_width(rows[bucket == number], lower, upper).mean(),
        }
        for number in range(1, n_buckets + 1)
    }
    return pd.DataFrame.from_dict(table, orient="index").rename_axis("bucket")


def daily_absolute_error(frame: pd.DataFrame, label: str, predicted: str) -> pd.Series:
    """Per-day mean absolute error."""
    rows = _rows(frame, [label, predicted])
    return _by_day(rows, (rows[label] - rows[predicted]).abs(), "mae")


def daily_squared_error(frame: pd.DataFrame, label: str, predicted: str) -> pd.Series:
    """Per-day mean squared error; RMSE is the root of its mean over the evaluated days."""
    rows = _rows(frame, [label, predicted])
    return _by_day(rows, (rows[label] - rows[predicted]).pow(2), "mse")


def daily_direction_hits(frame: pd.DataFrame, label: str, predicted: str) -> pd.Series:
    """Per-day share of rows whose prediction has the label's sign; zero is its own sign."""
    rows = _rows(frame, [label, predicted])
    hits = np.sign(rows[label].to_numpy(np.float64)) == np.sign(
        rows[predicted].to_numpy(np.float64)
    )
    return _by_day(rows, hits, "direction")


def daily_qlike(frame: pd.DataFrame, realized: str, predicted: str) -> pd.Series:
    """Per-day mean QLIKE on variances, zero when exact; non-positive volatilities skipped."""
    rows = _rows(frame, [realized, predicted])
    rows = rows[(rows[realized] > 0) & (rows[predicted] > 0)]
    ratio = (rows[realized] / rows[predicted]).pow(2)
    return _by_day(rows, ratio - np.log(ratio) - 1, "qlike")


def daily_brier(frame: pd.DataFrame, label: str, prob: str) -> pd.Series:
    """Per-day mean squared gap between the probability and the 0 or 1 outcome."""
    rows = _rows(frame, [label, prob])
    _require_binary(rows[label])
    _require_probability(rows[prob])
    return _by_day(rows, (rows[prob] - rows[label]).pow(2), "brier")


def daily_auc(
    frame: pd.DataFrame, label: str, prob: str, min_stocks: int = MIN_STOCKS
) -> pd.Series:
    """Per-day ROC AUC with ties as half; days without both outcomes or enough rows skipped."""
    rows = _rows(frame, [label, prob])
    _require_binary(rows[label])
    keys = rows[DATE]
    positive = rows[label].astype(bool)
    ranks = rows.groupby(DATE)[prob].rank()
    counts = keys.groupby(keys).size()
    hits = positive.groupby(keys).sum()
    misses = counts - hits
    rank_sum = ranks.where(positive, 0.0).groupby(keys).sum()
    auc = (rank_sum - hits * (hits + 1) / 2) / (hits * misses)
    usable = (counts >= min_stocks) & (hits > 0) & (misses > 0)
    return auc[usable].astype(np.float64).rename("auc").rename_axis(DATE)


def reliability_table(frame: pd.DataFrame, label: str, prob: str, n_bins: int = 10) -> pd.DataFrame:
    """Pooled calibration table over equal-width probability bins; bin 1 starts at 0."""
    rows = _rows(frame, [label, prob])
    _require_binary(rows[label])
    _require_probability(rows[prob])
    probs = rows[prob].to_numpy(np.float64)
    bins = np.minimum(np.floor(probs * n_bins).astype(np.int64), n_bins - 1) + 1
    outcome = rows[label].to_numpy(np.float64)
    table = {
        number: {
            "lower": (number - 1) / n_bins,
            "upper": number / n_bins,
            "mean_prob": probs[bins == number].mean() if (bins == number).any() else math.nan,
            "frac_positive": outcome[bins == number].mean() if (bins == number).any() else math.nan,
            "n_rows": int((bins == number).sum()),
        }
        for number in range(1, n_bins + 1)
    }
    return pd.DataFrame.from_dict(table, orient="index", columns=RELIABILITY_COLUMNS).rename_axis(
        "bin"
    )


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Row-weighted mean gap between predicted and observed frequency over the bins."""
    total = int(table["n_rows"].sum())
    if total == 0:
        return math.nan
    filled = table[table["n_rows"] > 0]
    gaps = (filled["frac_positive"] - filled["mean_prob"]).abs()
    return float((gaps * filled["n_rows"]).sum() / total)


def mean_by_fold(daily: pd.Series, folds: Iterable[Fold]) -> pd.DataFrame:
    """Mean of a daily metric per fold over test_start..eval_end, with the fold flags."""
    days = pd.Index(daily.index)
    rows = []
    for fold in folds:
        values = daily[(days >= fold.test_start) & (days <= fold.eval_end)].dropna()
        rows.append(
            {
                "fold": fold.index,
                "test_start": fold.test_start,
                "eval_end": fold.eval_end,
                "tuning": fold.tuning,
                "holdout": fold.holdout,
                "mean": float(values.mean()) if len(values) else math.nan,
                "n_days": len(values),
            }
        )
    return pd.DataFrame(rows, columns=FOLD_COLUMNS).set_index("fold")
