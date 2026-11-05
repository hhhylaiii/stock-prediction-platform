from datetime import date
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from apps.research import gbdt
from apps.research.gbdt import (
    BASE_PARAMS,
    EARLY_STOPPING_ROUNDS,
    FEVAL_EVERY,
    DailyRankIC,
    boost,
    seeded,
    split_validation,
)
from apps.research.walkforward import Fold, WalkForwardConfig, generate_folds
from tests.research.conftest import weekday_calendar

CALENDAR = weekday_calendar(date(2010, 1, 1), date(2014, 12, 31))
CONFIG = WalkForwardConfig(train_months=12, valid_months=3, first_test=date(2012, 1, 1))
FEATURES = ["signal", "noise"]
LABEL = "fwd_rank_pct"


def fold() -> Fold:
    return generate_folds(CALENDAR, date(2012, 6, 29), CONFIG)[0]


def panel(
    start: date, end: date, stocks: int = 60, seed: int = 0, signal: float = 1.0
) -> pd.DataFrame:
    """Daily cross-sections where signal drives the label rank and noise does not."""
    days = CALENDAR.between(start, end)
    rng = np.random.default_rng(seed)
    size = len(days) * stocks
    frame = pd.DataFrame(
        {
            "date": np.repeat(np.array(days, dtype=object), stocks),
            "stock_id": np.tile([f"{1000 + k}" for k in range(stocks)], len(days)),
            "signal": rng.normal(size=size),
            "noise": rng.normal(size=size),
        }
    )
    target = signal * frame["signal"] + rng.normal(size=size)
    return frame.assign(**{LABEL: target.groupby(frame["date"]).rank(pct=True)})


def training_window() -> pd.DataFrame:
    window = fold()
    return panel(window.train_start, window.train_end)


def test_seeded_derives_three_distinct_lightgbm_seeds() -> None:
    seeds = seeded(7)

    assert set(seeds) == {"seed", "bagging_seed", "feature_fraction_seed"}
    assert len(set(seeds.values())) == 3
    assert seeded(7) == seeds
    assert seeded(8) != seeds


def test_base_params_follow_the_spec_starting_point() -> None:
    assert {key: BASE_PARAMS[key] for key in ("num_leaves", "learning_rate")} == {
        "num_leaves": 31,
        "learning_rate": 0.02,
    }
    assert BASE_PARAMS["feature_fraction"] == 0.7
    assert BASE_PARAMS["bagging_fraction"] == 0.8
    assert BASE_PARAMS["bagging_freq"] >= 1
    assert BASE_PARAMS["min_data_in_leaf"] == 200
    assert BASE_PARAMS["lambda_l2"] == 1
    assert EARLY_STOPPING_ROUNDS == 100
    assert FEVAL_EVERY == 10


def test_daily_rank_ic_matches_scipy_spearman_averaged_over_days() -> None:
    frame = panel(date(2011, 3, 1), date(2011, 3, 14), stocks=45, seed=3)
    scores = frame["signal"].to_numpy() + np.random.default_rng(4).normal(size=len(frame))
    expected = np.mean(
        [spearmanr(scores[day.index], day[LABEL]).statistic for _, day in frame.groupby("date")]
    )

    assert DailyRankIC(frame["date"], frame[LABEL]).value(scores) == pytest.approx(expected)


def test_feval_recomputes_only_every_ten_rounds() -> None:
    frame = panel(date(2011, 3, 1), date(2011, 3, 14), stocks=45, seed=3)
    feval = DailyRankIC(frame["date"], frame[LABEL])
    rng = np.random.default_rng(5)

    results = [feval(rng.normal(size=len(frame)), None) for _ in range(25)]

    values = [value for _, value, _ in results]
    changes = [k for k in range(1, 25) if values[k] != values[k - 1]]
    assert changes == [10, 20]
    assert {(name, higher) for name, _, higher in results} == {("daily_rank_ic", True)}


def test_feval_scores_a_day_without_spread_as_no_signal() -> None:
    frame = panel(date(2011, 3, 1), date(2011, 3, 4), stocks=45)

    assert DailyRankIC(frame["date"], frame[LABEL]).value(np.zeros(len(frame))) == 0.0


def test_feval_refuses_a_validation_set_without_a_full_day() -> None:
    frame = panel(date(2011, 3, 1), date(2011, 3, 4), stocks=10)

    with pytest.raises(ValueError, match="30 stocks"):
        DailyRankIC(frame["date"], frame[LABEL])


def test_split_validation_keeps_the_last_months_after_an_embargo() -> None:
    window = fold()

    fit, valid = split_validation(training_window(), window)

    assert fit["date"].max() == window.fit_end
    assert (valid["date"].min(), valid["date"].max()) == (window.valid_start, window.train_end)
    gap = CALENDAR.between(window.fit_end, window.valid_start)
    assert len(gap) - 2 >= CONFIG.embargo
    assert window.valid_start <= window.train_end < window.test_start


def test_split_validation_refuses_rows_past_the_training_window() -> None:
    window = fold()
    rows = panel(window.train_start, window.test_start)

    with pytest.raises(ValueError, match="after train_end"):
        split_validation(rows, window)


def test_boost_stops_on_validation_ic_then_refits_the_whole_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, window = training_window(), fold()
    calls: list[tuple[int, int, bool]] = []
    train = lgb.train

    def spy(params: dict[str, Any], train_set: lgb.Dataset, **kwargs: Any) -> lgb.Booster:
        booster = train(params, train_set, **kwargs)
        calls.append((train_set.num_data(), kwargs["num_boost_round"], "valid_sets" in kwargs))
        return booster

    monkeypatch.setattr(gbdt.lgb, "train", spy)

    boosted = boost(rows, window, FEATURES, LABEL, seed=1, feval=DailyRankIC, max_rounds=400)

    fit, _ = split_validation(rows, window)
    assert calls[0] == (len(fit), 400, True)
    assert calls[1] == (len(rows), boosted.rounds, False)
    assert boosted.booster.num_trees() == boosted.rounds
    assert 1 <= boosted.rounds < 400
    assert boosted.valid_score > 0.3
    importance = dict(zip(FEATURES, boosted.booster.feature_importance("gain"), strict=True))
    assert importance["signal"] > importance["noise"]


def test_boost_without_signal_stops_soon_after_its_best_round() -> None:
    window = fold()
    rows = panel(window.train_start, window.train_end, signal=0.0)

    boosted = boost(rows, window, FEATURES, LABEL, seed=1, feval=DailyRankIC, max_rounds=2000)

    assert boosted.rounds < 2000 - EARLY_STOPPING_ROUNDS
    assert abs(boosted.valid_score) < 0.1


def test_boost_is_reproducible_with_a_seed() -> None:
    rows, window = training_window(), fold()
    test = panel(window.test_start, window.test_end, seed=9)

    def scores(seed: int) -> np.ndarray:
        boosted = boost(rows, window, FEATURES, LABEL, seed=seed, feval=DailyRankIC, max_rounds=120)
        return np.asarray(boosted.booster.predict(test[FEATURES]))

    first = scores(1)
    np.testing.assert_array_equal(scores(1), first)
    assert not np.array_equal(scores(2), first)


def test_boost_with_a_builtin_metric_stops_on_its_loss() -> None:
    rows, window = training_window(), fold()

    boosted = boost(
        rows,
        window,
        FEATURES,
        LABEL,
        seed=1,
        params={"objective": "regression", "metric": "l2"},
        max_rounds=300,
    )

    assert 1 <= boosted.rounds <= 300
    assert 0 < boosted.valid_score < 0.1
