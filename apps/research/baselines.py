"""Ridge, logistic and rule baselines for the three target families (SPEC §7 模型表)."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from apps.research.metrics.forecast import daily_brier
from apps.research.metrics.ic import daily_rank_ic
from apps.research.walkforward import Fold

PENALTY_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
CV_SPLITS = 4
INTERVAL_Z = 1.28
TRADING_YEAR = 252
INTERVAL_HORIZON = 20
TRAILING_DAYS = 60
QUANTILES = {"q10": 0.1, "q50": 0.5, "q90": 0.9}
LOGISTIC_ITERATIONS = 1_000
Values = npt.NDArray[np.float64]
Scorer = Callable[[pd.DataFrame, Values, str], float]


def _matrix(frame: pd.DataFrame, features: Sequence[str]) -> Values:
    values = frame[list(features)].to_numpy(dtype=np.float64)
    if np.isnan(values).any():
        raise ValueError("features have missing values; fill them before fitting")
    return values


def _predict(model: Any, values: Values) -> Values:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(values)[:, 1], dtype=np.float64)
    return np.asarray(model.predict(values), dtype=np.float64)


@dataclass(frozen=True)
class LinearFit:
    model: Any
    features: tuple[str, ...]
    penalty: float
    scores: dict[float, float]

    def predict(self, frame: pd.DataFrame) -> Values:
        """Ridge score, or logistic probability of the positive class, for each row."""
        return _predict(self.model, _matrix(frame, self.features))

    def coefficients(self) -> pd.DataFrame:
        """Weight and sign of each feature, largest absolute weight first."""
        weights = np.ravel(self.model.coef_).astype(np.float64)
        table = pd.DataFrame(
            {
                "feature": list(self.features),
                "weight": weights,
                "sign": np.sign(weights).astype(np.int64),
            }
        )
        order = table["weight"].abs().sort_values(ascending=False, kind="stable").index
        return table.loc[order].reset_index(drop=True)


def cv_splits(
    days: Sequence[date], gap: int, n_splits: int = CV_SPLITS
) -> list[tuple[tuple[date, ...], tuple[date, ...]]]:
    """Expanding time-ordered splits; each fit part ends gap days before its validation block."""
    blocks = np.array_split(np.array(list(days), dtype=object), n_splits + 1)
    splits = []
    for k in range(1, n_splits + 1):
        earlier = [day for block in blocks[:k] for day in block]
        fit = tuple(earlier[: max(len(earlier) - gap, 0)])
        if fit and len(blocks[k]):
            splits.append((fit, tuple(blocks[k])))
    return splits


def _rank_ic(valid: pd.DataFrame, predicted: Values, label: str) -> float:
    return float(daily_rank_ic(valid.assign(_score=predicted), "_score", label).mean())


def _negative_brier(valid: pd.DataFrame, predicted: Values, label: str) -> float:
    return -float(daily_brier(valid.assign(_prob=predicted), label, "_prob").mean())


def _fit(
    train: pd.DataFrame,
    features: Sequence[str],
    label: str,
    horizon: int,
    build: Callable[[float], Any],
    score: Scorer,
) -> LinearFit:
    """Penalty with the best mean validation score over the window's time splits, refit on all."""
    values, target = _matrix(train, features), train[label].to_numpy(dtype=np.float64)
    days = pd.Index(sorted(set(train["date"])))
    position = days.get_indexer(pd.Index(train["date"]))
    results: dict[float, list[float]] = {penalty: [] for penalty in PENALTY_GRID}
    for fit_days, valid_days in cv_splits(days, horizon):
        fit_rows = position < len(fit_days)
        start = days.get_loc(valid_days[0])
        valid_rows = (position >= start) & (position < start + len(valid_days))
        fitted, checked = values[fit_rows], values[valid_rows]
        valid = pd.DataFrame(
            {"date": train["date"].to_numpy()[valid_rows], label: target[valid_rows]}
        )
        for penalty in PENALTY_GRID:
            model = build(penalty).fit(fitted, target[fit_rows])
            results[penalty].append(score(valid, _predict(model, checked), label))
    scores = {key: float(np.mean(found)) if found else math.nan for key, found in results.items()}
    if all(math.isnan(value) for value in scores.values()):
        raise ValueError("the training window is too short to cross-validate the penalty")
    best = max(PENALTY_GRID, key=lambda penalty: np.nan_to_num(scores[penalty], nan=-np.inf))
    model = build(best).fit(values, target)
    return LinearFit(model, tuple(features), best, scores)


def fit_ridge(train: pd.DataFrame, features: Sequence[str], label: str, horizon: int) -> LinearFit:
    """Ridge with alpha chosen by mean daily validation rank IC (SPEC §7 排名 baseline)."""
    return _fit(train, features, label, horizon, lambda alpha: Ridge(alpha=alpha), _rank_ic)


def fit_logistic(
    train: pd.DataFrame, features: Sequence[str], label: str, horizon: int
) -> LinearFit:
    """Logistic regression with C chosen by mean daily validation Brier (SPEC §7 機率 baseline)."""
    return _fit(
        train,
        features,
        label,
        horizon,
        lambda c: LogisticRegression(C=c, max_iter=LOGISTIC_ITERATIONS),
        _negative_brier,
    )


def _raw(frame: pd.DataFrame, column: str, lowest: float) -> pd.Series:
    values = frame[column].astype(np.float64)
    if (values < lowest).any():
        raise ValueError(f"{column} looks standardized; rule baselines need raw values")
    return values


def zero_return(frame: pd.DataFrame) -> pd.Series:
    """The always-zero return forecast."""
    return pd.Series(0.0, index=frame.index)


def trailing_mean_return(frame: pd.DataFrame, horizon: int) -> pd.Series:
    """The past 60 days' mean return compounded over the horizon."""
    return (1 + _raw(frame, "tech_ret_60d", -1.0)) ** (horizon / TRAILING_DAYS) - 1


def vol_baseline(frame: pd.DataFrame) -> pd.Series:
    """Volatility forecast equal to the past 20 days' annualized volatility."""
    return _raw(frame, "tech_vol_20d", 0.0)


def vol_interval(frame: pd.DataFrame, horizon: int = INTERVAL_HORIZON) -> pd.DataFrame:
    """q50 of 0 and q10, q90 at ∓1.28 × tech_vol_20d × √(horizon / 252)."""
    half = INTERVAL_Z * _raw(frame, "tech_vol_20d", 0.0) * math.sqrt(horizon / TRADING_YEAR)
    return pd.DataFrame({"q10": -half, "q50": 0.0, "q90": half}, index=frame.index)


def historical_quantiles(train: pd.DataFrame, label: str) -> dict[str, float]:
    """Unconditional q10, q50 and q90 of the window's labels, a sanity floor for intervals."""
    values = train[label].dropna().astype(np.float64)
    return {name: float(values.quantile(level)) for name, level in QUANTILES.items()}


def shuffle_within_dates(frame: pd.DataFrame, column: str, seed: int) -> pd.DataFrame:
    """New frame with column permuted across stocks inside each date (SPEC §8 標籤打亂測試)."""
    rng = np.random.default_rng(seed)
    values = frame[column].to_numpy(copy=True)
    for positions in frame.groupby("date").indices.values():
        values[positions] = values[rng.permutation(positions)]
    return frame.assign(**{column: values})


def shuffled_oos_ic(
    frame: pd.DataFrame,
    fold: Fold,
    features: Sequence[str],
    label: str,
    horizon: int,
    seed: int,
) -> float:
    """Mean daily OOS rank IC of a Ridge refit on labels shuffled within dates; near 0 if clean.

    A linear fit flags labels leaking into features; it cannot memorize rows across a bad split.
    """
    shuffled = shuffle_within_dates(frame, label, seed)
    train = shuffled[shuffled["date"].between(fold.train_start, fold.train_end)]
    test = shuffled[shuffled["date"].between(fold.test_start, fold.eval_end)]
    fit = fit_ridge(train, features, label, horizon)
    return float(daily_rank_ic(test.assign(_score=fit.predict(test)), "_score", label).mean())
