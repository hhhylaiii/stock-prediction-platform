"""LightGBM return, quantile, volatility and probability models on raw features (SPEC §7)."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from apps.research.baselines import QUANTILES
from apps.research.features.registry import FEATURES as REGISTRY
from apps.research.features.transforms import STOCK_LEVEL, MarketBins, market_bins, winsorize
from apps.research.gbdt import MAX_ROUNDS, Boosted, bag_seeds, boost, refit, split_validation
from apps.research.metrics.forecast import expected_calibration_error, reliability_table
from apps.research.runs import Screened
from apps.research.training import training_rows
from apps.research.walkforward import Fold

RETURN_LABEL = "fwd_ret_open"
VOL_LABEL = "fwd_vol_20d"
UP_LABEL = "fwd_up_5pct"
LOG_VOL = "log_fwd_vol_20d"
UP = "up"
HUBER_MAD_SCALE = 1.5
VOL_FLOOR = 0.01
ECE_LIMIT = 0.05
ECE_BINS = 10
MARKET_LEVEL = [spec.name for spec in REGISTRY if spec.batch == 1 and spec.market_level]


@dataclass(frozen=True)
class TargetVariant:
    """A return or interval design choice under comparison; the defaults are the SPEC recipe."""

    market: bool = True
    relative_up: bool = False


TARGET_RECIPE = TargetVariant()


def raw_features(screened: Screened) -> list[str]:
    """Screened stock-level features, then the market features the screening left out."""
    misplaced = [name for name in screened.kept if name not in STOCK_LEVEL]
    misplaced += [name for name in screened.unscreened if name not in MARKET_LEVEL]
    if misplaced:
        raise ValueError(f"features in the wrong group for the raw-value models: {misplaced}")
    return [*screened.kept, *screened.unscreened]


def raw_inputs(
    raw: pd.DataFrame, fold: Fold, features: Sequence[str]
) -> tuple[pd.DataFrame, MarketBins]:
    """Stock features winsorized but not scored, market features cut into training quintiles."""
    stock = [name for name in features if name in STOCK_LEVEL]
    market = [name for name in features if name in MARKET_LEVEL]
    unknown = [name for name in features if name not in stock and name not in market]
    if unknown:
        raise ValueError(f"the raw-value models take stock or market features, not {unknown}")
    train = raw[raw["date"].between(fold.train_start, fold.train_end)]
    bins = market_bins(train, market)
    return bins.apply(winsorize(raw, stock)), bins


def _rows(inputs: pd.DataFrame, labels: pd.DataFrame, fold: Fold, label: str) -> pd.DataFrame:
    window = inputs[inputs["date"].between(fold.train_start, fold.train_end)]
    return training_rows(window, labels, label)


def huber_delta(values: pd.Series) -> float:
    """1.5 times the median absolute deviation of the training labels."""
    spread = float((values - values.median()).abs().median())
    if not spread > 0:
        raise ValueError("the training labels have no spread to scale the Huber delta")
    return HUBER_MAD_SCALE * spread


@dataclass(frozen=True)
class ReturnFit:
    boosted: Boosted
    delta: float


def fit_return(
    inputs: pd.DataFrame,
    labels: pd.DataFrame,
    fold: Fold,
    features: Sequence[str],
    seed: int,
    max_rounds: int = MAX_ROUNDS,
    bags: int = 1,
) -> ReturnFit:
    """Huber regression of the forward open-to-close return, delta from the window's MAD."""
    rows = _rows(inputs, labels, fold, RETURN_LABEL)
    delta = huber_delta(rows[RETURN_LABEL])
    params = {"objective": "huber", "alpha": delta, "metric": "huber"}
    boosted = boost(
        rows, fold, features, RETURN_LABEL, seed, params, max_rounds=max_rounds, bags=bags
    )
    return ReturnFit(boosted, delta)


def fit_quantiles(
    inputs: pd.DataFrame,
    labels: pd.DataFrame,
    fold: Fold,
    features: Sequence[str],
    seed: int,
    max_rounds: int = MAX_ROUNDS,
    bags: int = 1,
) -> dict[str, Boosted]:
    """One quantile booster per alpha of the forward return, each stopping on its own loss."""
    rows = _rows(inputs, labels, fold, RETURN_LABEL)
    return {
        name: boost(
            rows,
            fold,
            features,
            RETURN_LABEL,
            seed,
            {"objective": "quantile", "alpha": alpha, "metric": "quantile"},
            max_rounds=max_rounds,
            bags=bags,
        )
        for name, alpha in QUANTILES.items()
    }


def sorted_quantiles(predictions: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Each row's quantiles sorted so q10 ≤ q50 ≤ q90."""
    stacked = np.sort(np.column_stack([predictions[name] for name in QUANTILES]), axis=1)
    return {name: stacked[:, column] for column, name in enumerate(QUANTILES)}


def fit_volatility(
    inputs: pd.DataFrame,
    labels: pd.DataFrame,
    fold: Fold,
    features: Sequence[str],
    seed: int,
    max_rounds: int = MAX_ROUNDS,
    bags: int = 1,
) -> Boosted:
    """L2 regression of log forward volatility, floored so flat rows keep a finite label."""
    rows = _rows(inputs, labels, fold, VOL_LABEL)
    logged = rows.assign(**{LOG_VOL: np.log(rows[VOL_LABEL].clip(lower=VOL_FLOOR))})
    params = {"objective": "regression", "metric": "l2"}
    return boost(logged, fold, features, LOG_VOL, seed, params, max_rounds=max_rounds, bags=bags)


def volatility(boosted: Boosted, frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    """Annualized forward volatility, the exponential of the log prediction."""
    return np.exp(np.asarray(boosted.booster.predict(frame[list(features)])))


@dataclass(frozen=True)
class ProbabilityFit:
    """The binary booster, its bag's validation probabilities and ECE, isotonic if ECE was high."""

    boosted: Boosted
    valid_probs: np.ndarray
    ece: float
    calibrator: IsotonicRegression | None

    def predict(self, frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
        probs = np.asarray(self.boosted.booster.predict(frame[list(features)]))
        return probs if self.calibrator is None else np.asarray(self.calibrator.predict(probs))


def _bag_validation(
    boosted: Boosted,
    rows: pd.DataFrame,
    fold: Fold,
    features: Sequence[str],
    seed: int,
    bags: int,
    params: Mapping[str, str],
) -> np.ndarray:
    """Validation probabilities averaged over the bag, each seed fit before validation."""
    fit, valid = split_validation(rows, fold)
    others = [
        np.asarray(
            refit(fit, features, UP, member, boosted.rounds, params).predict(valid[list(features)])
        )
        for member in bag_seeds(seed, bags)[1:]
    ]
    return np.asarray(np.mean([boosted.valid_scores, *others], axis=0))


def fit_probability(
    inputs: pd.DataFrame,
    labels: pd.DataFrame,
    fold: Fold,
    features: Sequence[str],
    seed: int,
    max_rounds: int = MAX_ROUNDS,
    bags: int = 1,
) -> ProbabilityFit:
    """Binary booster of a 20-day gain above 5%, calibrated on its bag's validation if needed."""
    rows = _rows(inputs, labels, fold, UP_LABEL)
    rows = rows.assign(**{UP: rows[UP_LABEL].astype(np.float64)})
    params = {"objective": "binary", "metric": "binary_logloss"}
    boosted = boost(rows, fold, features, UP, seed, params, max_rounds=max_rounds, bags=bags)
    _, valid = split_validation(rows, fold)
    probs = _bag_validation(boosted, rows, fold, features, seed, bags, params)
    ece = expected_calibration_error(
        reliability_table(valid.assign(prob=probs), UP, "prob", ECE_BINS)
    )
    calibrator = None
    if ece > ECE_LIMIT:
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(probs, valid[UP].to_numpy())
    return ProbabilityFit(boosted, probs, ece, calibrator)
