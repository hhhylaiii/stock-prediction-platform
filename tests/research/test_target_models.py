from datetime import date
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from apps.research import gbdt, target_models
from apps.research.features.transforms import winsorize
from apps.research.gbdt import split_validation
from apps.research.metrics.forecast import expected_calibration_error, reliability_table
from apps.research.runs import Screened
from apps.research.target_models import (
    HUBER_MAD_SCALE,
    QUANTILES,
    RETURN_LABEL,
    UP,
    UP_LABEL,
    VOL_FLOOR,
    VOL_LABEL,
    fit_probability,
    fit_quantiles,
    fit_return,
    fit_volatility,
    huber_delta,
    raw_features,
    raw_inputs,
    sorted_quantiles,
)
from apps.research.training import training_rows
from apps.research.walkforward import Fold, WalkForwardConfig, generate_folds
from tests.research.conftest import weekday_calendar

CALENDAR = weekday_calendar(date(2010, 1, 1), date(2014, 12, 31))
CONFIG = WalkForwardConfig(train_months=12, valid_months=3, first_test=date(2012, 1, 1))
STOCK = ["tech_ret_20d", "tech_vol_20d", "fund_ep"]
MARKET = ["mkt_taiex_ret_5d"]
FEATURES = [*STOCK, *MARKET]
STOCKS = 60


def fold() -> Fold:
    return generate_folds(CALENDAR, date(2012, 6, 29), CONFIG)[0]


def panel(start: date, end: date, seed: int = 0) -> pd.DataFrame:
    """Raw features and labels, with a few rows out of universe, untradable or unlabelled."""
    days = CALENDAR.between(start, end)
    rng = np.random.default_rng(seed)
    size = len(days) * STOCKS
    signal = rng.normal(size=size)
    vol = np.tile(np.linspace(0.15, 0.6, STOCKS), len(days))
    ret = 0.02 * signal + vol * 0.08 * rng.normal(size=size)
    unlabelled = rng.random(size) < 0.03
    return pd.DataFrame(
        {
            "date": np.repeat(np.array(days, dtype=object), STOCKS),
            "stock_id": np.tile([f"{1000 + k}" for k in range(STOCKS)], len(days)),
            "in_universe": rng.random(size) > 0.05,
            "tech_ret_20d": signal,
            "tech_vol_20d": vol,
            "fund_ep": np.where(rng.random(size) < 0.1, np.nan, rng.normal(0.05, 0.02, size)),
            "mkt_taiex_ret_5d": np.repeat(rng.normal(0, 0.02, len(days)), STOCKS),
            RETURN_LABEL: np.where(unlabelled, np.nan, ret),
            VOL_LABEL: np.where(unlabelled, np.nan, vol * np.exp(0.1 * rng.normal(size=size))),
            UP_LABEL: pd.Series(ret > 0.05, dtype=object).where(~unlabelled, None),
            "untradable_t1": rng.random(size) < 0.05,
        }
    )


def split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = frame.drop(columns=[RETURN_LABEL, VOL_LABEL, UP_LABEL, "untradable_t1"])
    labels = frame[["stock_id", "date", RETURN_LABEL, VOL_LABEL, UP_LABEL, "untradable_t1"]]
    return features, labels


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    window = fold()
    raw, labels = split(panel(window.train_start, window.test_end))
    inputs, _ = raw_inputs(raw, window, FEATURES)
    test = inputs[inputs["date"].between(window.test_start, window.test_end)]
    return inputs, labels, training_rows(test, labels, RETURN_LABEL)


def trainable(inputs: pd.DataFrame, labels: pd.DataFrame, label: str) -> pd.DataFrame:
    """The training window's universe rows tradable at t+1 with a known label."""
    window = fold()
    inside = inputs[inputs["date"].between(window.train_start, window.train_end)]
    return training_rows(inside, labels, label)


@pytest.fixture
def train_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[dict[str, Any], np.ndarray]]:
    """Params and labels of every lgb.train call."""
    calls: list[tuple[dict[str, Any], np.ndarray]] = []
    train = lgb.train

    def spy(params: dict[str, Any], train_set: lgb.Dataset, **kwargs: Any) -> lgb.Booster:
        calls.append((dict(params), np.asarray(train_set.get_label())))
        return train(params, train_set, **kwargs)

    monkeypatch.setattr(gbdt.lgb, "train", spy)
    return calls


def test_raw_features_put_the_screened_stock_features_before_the_market_ones() -> None:
    screened = Screened(["fund_ep", "tech_ret_20d"], ["mkt_taiex_ret_5d", "mkt_sox_ret_5d"])

    assert raw_features(screened) == [
        "fund_ep",
        "tech_ret_20d",
        "mkt_taiex_ret_5d",
        "mkt_sox_ret_5d",
    ]


@pytest.mark.parametrize(
    ("kept", "unscreened", "bad"),
    [
        (["tech_ret_20d", "mkt_sox_ret_5d"], [], "mkt_sox_ret_5d"),
        (["tech_ret_20d"], ["fund_ep"], "fund_ep"),
        (["tech_ret_20d"], ["made_up"], "made_up"),
    ],
)
def test_raw_features_refuse_features_in_the_wrong_group(
    kept: list[str], unscreened: list[str], bad: str
) -> None:
    with pytest.raises(ValueError, match=bad):
        raw_features(Screened(kept, unscreened))


def test_raw_inputs_bin_market_features_on_the_training_window_only() -> None:
    window = fold()
    raw, _ = split(panel(window.train_start, window.test_end))
    shocked = raw.assign(
        mkt_taiex_ret_5d=raw["mkt_taiex_ret_5d"].where(raw["date"] <= window.train_end, 9.0)
    )
    shocked.loc[shocked["date"] == window.train_start, "mkt_taiex_ret_5d"] = np.nan

    inputs, bins = raw_inputs(shocked, window, FEATURES)
    reference, _ = raw_inputs(raw, window, FEATURES)

    train = shocked[shocked["date"].between(window.train_start, window.train_end)]
    daily = train.drop_duplicates("date")["mkt_taiex_ret_5d"]
    assert bins.edges["mkt_taiex_ret_5d"] == pytest.approx(
        tuple(daily.quantile([0.2, 0.4, 0.6, 0.8]))
    )
    assert set(inputs["mkt_taiex_ret_5d"].dropna().unique()) <= {0.0, 1.0, 2.0, 3.0, 4.0}
    assert (inputs.loc[inputs["date"] > window.train_end, "mkt_taiex_ret_5d"] == 4.0).all()
    assert inputs.loc[inputs["date"] == window.train_start, "mkt_taiex_ret_5d"].isna().all()
    assert list(bins.edges) == MARKET
    assert reference["mkt_taiex_ret_5d"].notna().all()


def test_raw_inputs_winsorize_stock_features_without_scoring_them() -> None:
    window = fold()
    raw, _ = split(panel(window.train_start, window.test_end))

    inputs, _ = raw_inputs(raw, window, FEATURES)

    pd.testing.assert_frame_equal(inputs[STOCK], winsorize(raw, STOCK)[STOCK])
    assert not inputs["tech_ret_20d"].equals(raw["tech_ret_20d"])
    assert inputs["tech_ret_20d"].std() == pytest.approx(raw["tech_ret_20d"].std(), rel=0.05)
    assert inputs["fund_ep"].isna().equals(raw["fund_ep"].isna())


def test_raw_inputs_refuse_features_outside_both_groups() -> None:
    window = fold()
    raw, _ = split(panel(window.train_start, window.test_end))

    with pytest.raises(ValueError, match="in_universe"):
        raw_inputs(raw, window, [*FEATURES, "in_universe"])


def test_huber_delta_is_one_and_a_half_median_absolute_deviations() -> None:
    assert huber_delta(pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])) == pytest.approx(1.5)
    assert HUBER_MAD_SCALE == 1.5


def test_huber_delta_refuses_labels_without_spread() -> None:
    with pytest.raises(ValueError, match="spread"):
        huber_delta(pd.Series([0.01] * 5))


def test_fit_return_uses_huber_with_the_window_delta(
    data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    train_calls: list[tuple[dict[str, Any], np.ndarray]],
) -> None:
    inputs, labels, test = data
    window = fold()

    fitted = fit_return(inputs, labels, window, FEATURES, seed=1, max_rounds=200)

    train = trainable(inputs, labels, RETURN_LABEL)
    assert len(train) < len(labels[labels["date"].between(window.train_start, window.train_end)])
    assert fitted.delta == pytest.approx(huber_delta(train[RETURN_LABEL]))
    params, label = train_calls[0]
    assert (params["objective"], params["alpha"], params["metric"]) == (
        "huber",
        pytest.approx(fitted.delta),
        "huber",
    )
    fit, _ = split_validation(train, window)
    np.testing.assert_allclose(label, fit[RETURN_LABEL])
    predicted = fitted.boosted.booster.predict(test[FEATURES])
    assert np.corrcoef(predicted, test[RETURN_LABEL])[0, 1] > 0.1


def test_fit_quantiles_trains_three_alphas_that_cover_about_eighty_percent(
    data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    train_calls: list[tuple[dict[str, Any], np.ndarray]],
) -> None:
    inputs, labels, test = data

    fitted = fit_quantiles(inputs, labels, fold(), FEATURES, seed=1, max_rounds=300)

    assert list(fitted) == list(QUANTILES)
    probes = [params for params, _ in train_calls if params["objective"] == "quantile"]
    assert sorted({params["alpha"] for params in probes}) == [0.1, 0.5, 0.9]
    assert {params["metric"] for params in probes} == {"quantile"}
    raw = {
        name: np.asarray(boosted.booster.predict(test[FEATURES]))
        for name, boosted in fitted.items()
    }
    ordered = sorted_quantiles(raw)
    inside = (test[RETURN_LABEL] >= ordered["q10"]) & (test[RETURN_LABEL] <= ordered["q90"])
    assert 0.65 < inside.mean() < 0.95


def test_sorted_quantiles_uncross_each_row() -> None:
    ordered = sorted_quantiles(
        {
            "q10": np.array([-0.05, 0.03, 0.0]),
            "q50": np.array([0.0, -0.01, 0.0]),
            "q90": np.array([0.04, 0.02, -0.02]),
        }
    )

    np.testing.assert_array_equal(ordered["q10"], [-0.05, -0.01, -0.02])
    np.testing.assert_array_equal(ordered["q50"], [0.0, 0.02, 0.0])
    np.testing.assert_array_equal(ordered["q90"], [0.04, 0.03, 0.0])


def test_fit_volatility_regresses_log_volatility_floored_for_flat_rows(
    data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    train_calls: list[tuple[dict[str, Any], np.ndarray]],
) -> None:
    inputs, labels, test = data
    window = fold()
    flat = labels.assign(
        **{
            VOL_LABEL: labels[VOL_LABEL]
            .where(labels["stock_id"] != "1000", 0.0)
            .where(labels["stock_id"] != "1001", 1e-9)
        }
    )

    fitted = fit_volatility(inputs, flat, window, FEATURES, seed=1, max_rounds=200)

    params, label = train_calls[0]
    assert (params["objective"], params["metric"]) == ("regression", "l2")
    fit, _ = split_validation(trainable(inputs, flat, VOL_LABEL), window)
    np.testing.assert_allclose(label, np.log(fit[VOL_LABEL].clip(lower=VOL_FLOOR)))
    assert label.min() == pytest.approx(np.log(VOL_FLOOR))
    assert VOL_FLOOR == 0.01
    predicted = target_models.volatility(fitted, test, FEATURES)
    assert (predicted > 0).all()
    assert np.corrcoef(predicted, test[VOL_LABEL])[0, 1] > 0.8


def test_probability_calibration_settings() -> None:
    assert (target_models.ECE_LIMIT, target_models.ECE_BINS) == (0.05, 10)


@pytest.mark.parametrize(("limit", "calibrated"), [(1.0, False), (-1.0, True)])
def test_fit_probability_calibrates_on_the_validation_tail_only_when_ece_is_high(
    data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
    limit: float,
    calibrated: bool,
) -> None:
    inputs, labels, test = data
    window = fold()
    monkeypatch.setattr(target_models, "ECE_LIMIT", limit)
    seen: list[tuple[np.ndarray, np.ndarray]] = []
    fit = target_models.IsotonicRegression.fit

    def spy(self: Any, x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append((np.asarray(x), np.asarray(y)))
        return fit(self, x, y, *args, **kwargs)

    monkeypatch.setattr(target_models.IsotonicRegression, "fit", spy)

    fitted = fit_probability(inputs, labels, window, FEATURES, seed=1, max_rounds=200)

    _, valid = split_validation(trainable(inputs, labels, UP_LABEL), window)
    np.testing.assert_array_equal(fitted.valid_probs, fitted.boosted.valid_scores)
    scored = valid.assign(prob=fitted.valid_probs)
    scored[UP] = scored[UP_LABEL].astype(float)
    assert fitted.ece == pytest.approx(
        expected_calibration_error(reliability_table(scored, UP, "prob", 10))
    )
    assert (fitted.calibrator is not None) is calibrated
    assert len(seen) == (1 if calibrated else 0)
    if seen:
        np.testing.assert_array_equal(seen[0][0], fitted.boosted.valid_scores)
        np.testing.assert_array_equal(seen[0][1], scored[UP].to_numpy())
    probs = fitted.predict(test, FEATURES)
    raw = np.asarray(fitted.boosted.booster.predict(test[FEATURES]))
    if fitted.calibrator is None:
        np.testing.assert_array_equal(probs, raw)
    else:
        np.testing.assert_array_equal(probs, fitted.calibrator.predict(raw))
        assert not np.allclose(probs, raw)
    assert ((probs >= 0) & (probs <= 1)).all()
    order = np.argsort(raw)
    assert (np.diff(probs[order]) >= -1e-12).all()


def test_fit_probability_calibrates_the_bag_average_of_seeds_fit_before_validation(
    data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, labels, _ = data
    window = fold()
    monkeypatch.setattr(target_models, "ECE_LIMIT", -1.0)

    fitted = fit_probability(inputs, labels, window, FEATURES, seed=1, max_rounds=200, bags=3)

    rows = trainable(inputs, labels, UP_LABEL)
    rows = rows.assign(**{UP: rows[UP_LABEL].astype(float)})
    before, valid = split_validation(rows, window)
    params = {"objective": "binary", "metric": "binary_logloss"}
    members = [
        gbdt.refit(before, FEATURES, UP, member, fitted.boosted.rounds, params)
        for member in gbdt.bag_seeds(1, 3)
    ]
    expected = np.mean([member.predict(valid[FEATURES]) for member in members], axis=0)
    np.testing.assert_allclose(fitted.valid_probs, expected, rtol=1e-9)
    assert not np.allclose(fitted.valid_probs, fitted.boosted.valid_scores)
    scored = valid.assign(prob=expected)
    assert fitted.ece == pytest.approx(
        expected_calibration_error(reliability_table(scored, UP, "prob", 10))
    )
    assert fitted.calibrator is not None
    alone = target_models.IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    alone.fit(expected, valid[UP].to_numpy())
    grid = np.linspace(0, 1, 101)
    np.testing.assert_allclose(fitted.calibrator.predict(grid), alone.predict(grid))
