import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from apps.research.baselines import (
    PENALTY_GRID,
    _fit,
    cv_splits,
    fit_logistic,
    fit_ridge,
    historical_quantiles,
    shuffle_within_dates,
    shuffled_oos_ic,
    trailing_mean_return,
    vol_baseline,
    vol_interval,
    zero_return,
)
from apps.research.metrics.forecast import daily_brier
from apps.research.metrics.ic import daily_rank_ic
from apps.research.walkforward import WalkForwardConfig, generate_folds
from tests.research.conftest import weekday_calendar

CALENDAR = weekday_calendar(date(2010, 1, 1), date(2012, 12, 31))
FEATURES = ["x", "z", "x_missing"]


def panel(days: tuple[date, ...], stocks: int, seed: int, signal: float = 1.0) -> pd.DataFrame:
    """Daily cross-sections where x drives the label, z is noise and x_missing a sparse flag."""
    rng = np.random.default_rng(seed)
    size = len(days) * stocks
    x, z, noise = rng.normal(size=size), rng.normal(size=size), rng.normal(size=size)
    frame = pd.DataFrame(
        {
            "date": np.repeat(np.array(days, dtype=object), stocks),
            "stock_id": np.tile([f"{1000 + k}" for k in range(stocks)], len(days)),
            "x": x,
            "z": z,
            "x_missing": (rng.random(size) < 0.05).astype(np.int8),
            "fwd_ret_open": signal * x + noise,
        }
    )
    return frame.assign(
        fwd_rank_pct=frame.groupby("date")["fwd_ret_open"].rank(pct=True),
        fwd_up_5pct=(frame["fwd_ret_open"] > 1.0).astype(np.float64),
    )


def test_cv_splits_grow_forward_and_keep_a_gap_before_each_validation_block() -> None:
    days = CALENDAR.between(date(2011, 1, 3), date(2011, 5, 20))

    splits = cv_splits(days, gap=5, n_splits=4)

    assert len(splits) == 4
    blocks = np.array_split(np.array(days, dtype=object), 5)
    for k, (fit, valid) in enumerate(splits, start=1):
        assert list(valid) == list(blocks[k])
        assert fit[0] == days[0]
        assert days.index(valid[0]) - days.index(fit[-1]) == 6
        assert set(fit) | set(valid) <= set(days)


def test_ridge_picks_alpha_by_validation_rank_ic_inside_the_window() -> None:
    days = CALENDAR.between(date(2011, 1, 3), date(2011, 6, 30))
    train = panel(days, 40, seed=1, signal=0.3)

    fit = fit_ridge(train, FEATURES, "fwd_rank_pct", horizon=5)

    expected = {}
    for alpha in PENALTY_GRID:
        scores = []
        for fit_days, valid_days in cv_splits(days, gap=5):
            part = train[train["date"].isin(fit_days)]
            valid = train[train["date"].isin(valid_days)]
            model = Ridge(alpha=alpha).fit(part[FEATURES], part["fwd_rank_pct"])
            scored = valid.assign(score=model.predict(valid[FEATURES]))
            scores.append(daily_rank_ic(scored, "score", "fwd_rank_pct").mean())
        expected[alpha] = float(np.mean(scores))
    assert (PENALTY_GRID[0], PENALTY_GRID[-1]) == (0.1, 100.0)
    assert fit.scores == pytest.approx(expected)
    assert fit.penalty == max(expected, key=lambda alpha: expected[alpha])
    final = Ridge(alpha=fit.penalty).fit(train[FEATURES], train["fwd_rank_pct"])
    assert fit.predict(train) == pytest.approx(final.predict(train[FEATURES]))


def test_the_coefficient_table_lists_weights_and_signs_by_size() -> None:
    days = CALENDAR.between(date(2011, 1, 3), date(2011, 6, 30))
    fit = fit_ridge(panel(days, 40, seed=2), ["z", "x_missing", "x"], "fwd_rank_pct", horizon=5)

    table = fit.coefficients()

    assert list(table.columns) == ["feature", "weight", "sign"]
    assert table["feature"].iloc[0] == "x"
    assert table["sign"].iloc[0] == 1
    assert table["weight"].abs().is_monotonic_decreasing
    assert sorted(table["feature"]) == sorted(FEATURES)


def test_logistic_picks_c_by_validation_brier_and_returns_probabilities() -> None:
    days = CALENDAR.between(date(2011, 1, 3), date(2011, 6, 30))
    train = panel(days, 40, seed=3)

    fit = fit_logistic(train, FEATURES, "fwd_up_5pct", horizon=20)

    expected = {}
    for c in PENALTY_GRID:
        scores = []
        for fit_days, valid_days in cv_splits(sorted(set(train["date"])), gap=20):
            part = train[train["date"].isin(fit_days)]
            valid = train[train["date"].isin(valid_days)]
            model = LogisticRegression(C=c, max_iter=1_000).fit(part[FEATURES], part["fwd_up_5pct"])
            scored = valid.assign(prob=model.predict_proba(valid[FEATURES])[:, 1])
            scores.append(-daily_brier(scored, "fwd_up_5pct", "prob").mean())
        expected[c] = float(np.mean(scores))
    assert fit.scores == pytest.approx(expected)
    assert fit.penalty == max(expected, key=lambda c: expected[c])
    probs = fit.predict(train)
    assert ((probs > 0) & (probs < 1)).all()
    assert np.corrcoef(probs, train["x"])[0, 1] > 0.5
    assert fit.coefficients()["feature"].iloc[0] == "x"
    assert all(score <= 0 for score in fit.scores.values())


def test_linear_fits_refuse_missing_values() -> None:
    days = CALENDAR.between(date(2011, 1, 3), date(2011, 6, 30))
    train = panel(days, 40, seed=4)
    train.loc[5, "x"] = math.nan

    with pytest.raises(ValueError, match="fill them before fitting"):
        fit_ridge(train, FEATURES, "fwd_rank_pct", horizon=5)
    with pytest.raises(ValueError, match="fill them before fitting"):
        fit_logistic(train, FEATURES, "fwd_up_5pct", horizon=20)


class Constant:
    """Model whose prediction is its penalty, so a scorer can rank penalties directly."""

    def __init__(self, penalty: float) -> None:
        self.penalty = penalty

    def fit(self, values: np.ndarray, target: np.ndarray) -> "Constant":
        return self

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.full(len(values), self.penalty)


@pytest.mark.parametrize(
    ("table", "chosen"),
    [
        ({0.1: 0.1, 0.3: 0.2, 1.0: 0.5, 3.0: 0.9, 10.0: 0.4, 30.0: 0.3, 100.0: 0.0}, 3.0),
        ({0.1: 0.1, 0.3: 0.7, 1.0: 0.7, 3.0: 0.2, 10.0: 0.4, 30.0: 0.3, 100.0: 0.0}, 0.3),
    ],
)
def test_the_best_cv_score_wins_and_ties_take_the_smaller_penalty(
    table: dict[float, float], chosen: float
) -> None:
    days = CALENDAR.between(date(2011, 1, 3), date(2011, 6, 30))

    fit = _fit(
        panel(days, 5, seed=8),
        FEATURES,
        "fwd_rank_pct",
        5,
        Constant,
        lambda valid, predicted, label: table[float(predicted[0])],
    )

    assert fit.penalty == chosen
    assert fit.scores == pytest.approx(table)


def test_a_window_too_short_to_cross_validate_is_refused() -> None:
    days = CALENDAR.between(date(2011, 1, 3), date(2011, 1, 14))

    with pytest.raises(ValueError, match="too short"):
        fit_ridge(panel(days, 40, seed=9), FEATURES, "fwd_rank_pct", horizon=20)


def test_return_volatility_and_interval_baselines() -> None:
    frame = pd.DataFrame({"tech_ret_60d": [0.2, -0.1, math.nan], "tech_vol_20d": [0.3, 0.5, 0.4]})

    assert zero_return(frame).tolist() == [0.0, 0.0, 0.0]
    trailing = trailing_mean_return(frame, 5)
    assert trailing.iloc[:2].tolist() == pytest.approx([1.2 ** (5 / 60) - 1, 0.9 ** (5 / 60) - 1])
    assert math.isnan(trailing.iloc[2])
    assert vol_baseline(frame).tolist() == [0.3, 0.5, 0.4]
    band = vol_interval(frame)
    half = 1.28 * 0.3 * math.sqrt(20 / 252)
    assert list(band.columns) == ["q10", "q50", "q90"]
    assert band.iloc[0].tolist() == pytest.approx([-half, 0.0, half])


def test_rule_baselines_refuse_standardized_inputs() -> None:
    scored = pd.DataFrame({"tech_ret_60d": [0.5, -1.4], "tech_vol_20d": [0.8, -0.6]})

    for baseline in (vol_baseline, vol_interval):
        with pytest.raises(ValueError, match="raw"):
            baseline(scored)
    with pytest.raises(ValueError, match="raw"):
        trailing_mean_return(scored, 5)


def test_historical_quantiles_come_from_the_training_labels() -> None:
    train = pd.DataFrame({"fwd_ret": [float(k) for k in range(101)] + [math.nan]})

    assert historical_quantiles(train, "fwd_ret") == pytest.approx(
        {"q10": 10.0, "q50": 50.0, "q90": 90.0}
    )


def test_shuffling_moves_label_columns_together_within_each_date_only() -> None:
    days = CALENDAR.between(date(2011, 1, 3), date(2011, 1, 14))
    frame = panel(days, 30, seed=5)
    labels = ["fwd_rank_pct", "fwd_ret_open"]

    shuffled = shuffle_within_dates(frame, labels, seed=11)

    assert shuffled[["date", "stock_id", "x"]].equals(frame[["date", "stock_id", "x"]])
    for day in days:
        before = frame.loc[frame["date"] == day, labels]
        after = shuffled.loc[shuffled["date"] == day, labels]
        assert sorted(map(tuple, before.to_numpy())) == sorted(map(tuple, after.to_numpy()))
    assert not shuffled["fwd_rank_pct"].equals(frame["fwd_rank_pct"])
    again = shuffle_within_dates(frame, ["fwd_rank_pct"], seed=11)
    assert again["fwd_rank_pct"].equals(shuffled["fwd_rank_pct"])
    assert again["fwd_ret_open"].equals(frame["fwd_ret_open"])
    assert frame[labels].equals(panel(days, 30, seed=5)[labels])


@pytest.mark.spec8
def test_label_shuffle_drops_oos_rank_ic_to_zero() -> None:
    config = WalkForwardConfig(
        train_months=12,
        test_months=3,
        step_months=3,
        valid_months=6,
        data_start=date(2010, 1, 1),
        first_test=date(2011, 1, 1),
        tuning_end=date(2010, 12, 31),
        holdout_start=date(2012, 7, 1),
    )
    fold = generate_folds(CALENDAR, date(2011, 6, 30), config)[0]
    frame = panel(CALENDAR.between(fold.train_start, fold.eval_end), 60, seed=6, signal=0.5)
    train = frame[frame["date"].between(fold.train_start, fold.train_end)]
    test = frame[frame["date"].between(fold.test_start, fold.eval_end)]
    fit = fit_ridge(train, FEATURES, "fwd_rank_pct", horizon=20)
    unshuffled = daily_rank_ic(test.assign(score=fit.predict(test)), "score", "fwd_rank_pct")

    shuffled = shuffle_within_dates(frame, ["fwd_rank_pct"], seed=7)
    refit = fit_ridge(
        shuffled[shuffled["date"].between(fold.train_start, fold.train_end)],
        FEATURES,
        "fwd_rank_pct",
        horizon=20,
    )
    scored = test.assign(
        score=refit.predict(test), fwd_rank_pct=shuffled.loc[test.index, "fwd_rank_pct"]
    )

    clean = shuffled_oos_ic(frame, fold, FEATURES, "fwd_rank_pct", horizon=20, seed=7)
    leaks = [
        shuffled_oos_ic(frame, fold, [*FEATURES, leak], "fwd_rank_pct", horizon=20, seed=7)
        for leak in ("fwd_rank_pct", "fwd_ret_open", "fwd_up_5pct")
    ]

    assert unshuffled.mean() > 0.3
    assert abs(clean) < 0.05
    assert clean == pytest.approx(daily_rank_ic(scored, "score", "fwd_rank_pct").mean())
    assert min(leaks) > 0.3
