from datetime import date
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from apps.research import gbdt
from apps.research.gbdt import (
    BAG_STRIDE,
    BASE_PARAMS,
    EARLY_STOPPING_ROUNDS,
    FEVAL_EVERY,
    Bagged,
    DailyRankIC,
    bag_seeds,
    boost,
    refit,
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


@pytest.fixture
def probe_rounds(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Rounds each early-stopped lgb.train call actually ran."""
    ran: list[int] = []
    train = lgb.train

    def spy(params: dict[str, Any], train_set: lgb.Dataset, **kwargs: Any) -> lgb.Booster:
        if "valid_sets" not in kwargs:
            return train(params, train_set, **kwargs)
        seen: list[int] = []

        def record(env: Any) -> None:
            seen.append(env.iteration)

        record.order = 0  # type: ignore[attr-defined]
        booster = train(
            params, train_set, **{**kwargs, "callbacks": [record, *kwargs["callbacks"]]}
        )
        ran.append(len(seen))
        return booster

    monkeypatch.setattr(gbdt.lgb, "train", spy)
    return ran


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


def test_feval_leaves_a_day_without_spread_out_of_the_mean() -> None:
    frame = panel(date(2011, 3, 1), date(2011, 3, 14), stocks=45, seed=3)
    scores = frame["signal"].to_numpy() + np.random.default_rng(4).normal(size=len(frame))
    flat = frame["date"] == date(2011, 3, 8)
    scores[flat.to_numpy()] = 1.0
    expected = np.mean(
        [
            spearmanr(scores[day.index], day[LABEL]).statistic
            for _, day in frame[~flat].groupby("date")
        ]
    )

    assert DailyRankIC(frame["date"], frame[LABEL]).value(scores) == pytest.approx(expected)


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
    assert isinstance(boosted.booster, lgb.Booster)
    assert boosted.booster.num_trees() == boosted.rounds
    assert 1 <= boosted.rounds < 400
    assert (boosted.rounds - 1) % FEVAL_EVERY == 0
    assert boosted.valid_score > 0.3
    importance = dict(zip(FEATURES, boosted.booster.feature_importance("gain"), strict=True))
    assert importance["signal"] > importance["noise"]


def test_boost_keeps_the_probe_scores_on_the_validation_rows() -> None:
    rows, window = training_window(), fold()

    boosted = boost(rows, window, FEATURES, LABEL, seed=1, feval=DailyRankIC, max_rounds=200)

    _, valid = split_validation(rows, window)
    assert boosted.valid_scores.shape == (len(valid),)
    rescored = DailyRankIC(valid["date"], valid[LABEL]).value(boosted.valid_scores)
    assert rescored == pytest.approx(boosted.valid_score)


def test_boost_without_signal_stops_a_patience_after_its_best_round(
    probe_rounds: list[int],
) -> None:
    window = fold()
    rows = panel(window.train_start, window.train_end, signal=0.0)

    boosted = boost(rows, window, FEATURES, LABEL, seed=1, feval=DailyRankIC, max_rounds=2000)

    assert probe_rounds == [boosted.rounds + EARLY_STOPPING_ROUNDS]
    assert probe_rounds[0] < 2000
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


def test_boost_with_a_builtin_metric_stops_on_that_loss(probe_rounds: list[int]) -> None:
    window = fold()
    rows = panel(window.train_start, window.train_end, signal=0.0)

    boosted = boost(
        rows,
        window,
        FEATURES,
        LABEL,
        seed=1,
        params={"objective": "regression", "metric": "l1"},
        max_rounds=1000,
    )

    assert probe_rounds == [boosted.rounds + EARLY_STOPPING_ROUNDS]
    assert probe_rounds[0] < 1000
    assert 0.2 < boosted.valid_score < 0.3


def test_bag_seeds_never_share_a_lightgbm_seed() -> None:
    seeds = bag_seeds(7, 5)

    assert seeds == [7, 10, 13, 16, 19]
    assert BAG_STRIDE == 3
    derived = [value for seed in seeds for value in seeded(seed).values()]
    assert len(set(derived)) == len(derived)


def test_bag_seeds_need_at_least_one_bag() -> None:
    with pytest.raises(ValueError, match="at least one"):
        bag_seeds(0, 0)


def test_refit_trains_every_row_for_exactly_the_given_rounds() -> None:
    rows = training_window()

    booster = refit(rows, FEATURES, LABEL, seed=2, rounds=37)

    assert booster.num_trees() == 37
    again = refit(rows, FEATURES, LABEL, seed=2, rounds=37)
    np.testing.assert_array_equal(booster.predict(rows[FEATURES]), again.predict(rows[FEATURES]))


@pytest.fixture
def train_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Params and round counts of every lgb.train call, probes marked by valid_sets."""
    calls: list[dict[str, Any]] = []
    train = lgb.train

    def spy(params: dict[str, Any], train_set: lgb.Dataset, **kwargs: Any) -> lgb.Booster:
        calls.append(
            {**params, "_probe": "valid_sets" in kwargs, "_rounds": kwargs.get("num_boost_round")}
        )
        return train(params, train_set, **kwargs)

    monkeypatch.setattr(gbdt.lgb, "train", spy)
    return calls


def test_boost_with_bags_probes_once_and_refits_each_seed_for_the_same_rounds(
    train_calls: list[dict[str, Any]],
) -> None:
    rows, window = training_window(), fold()

    boosted = boost(
        rows, window, FEATURES, LABEL, seed=4, feval=DailyRankIC, max_rounds=200, bags=3
    )

    probes = [call for call in train_calls if call["_probe"]]
    refits = [call for call in train_calls if not call["_probe"]]
    assert [call["seed"] for call in probes] == [4]
    assert [call["seed"] for call in refits] == bag_seeds(4, 3)
    assert {call["_rounds"] for call in refits} == {boosted.rounds}
    assert isinstance(boosted.booster, Bagged)
    members = boosted.booster.boosters
    assert len(members) == 3
    test = panel(window.test_start, window.test_end, seed=9)
    expected = np.mean([member.predict(test[FEATURES]) for member in members], axis=0)
    np.testing.assert_allclose(boosted.booster.predict(test[FEATURES]), expected)
    gains = np.mean([member.feature_importance("gain") for member in members], axis=0)
    np.testing.assert_allclose(boosted.booster.feature_importance("gain"), gains)
    assert len({tuple(member.predict(test[FEATURES][:5])) for member in members}) == 3


def test_boost_with_one_bag_returns_the_booster_lightgbm_trains_directly() -> None:
    rows, window = training_window(), fold()

    one = boost(rows, window, FEATURES, LABEL, seed=4, feval=DailyRankIC, max_rounds=200)

    assert isinstance(one.booster, lgb.Booster)
    direct = lgb.train(
        {**BASE_PARAMS, **seeded(4)},
        lgb.Dataset(rows[FEATURES], label=rows[LABEL]),
        num_boost_round=one.rounds,
    )
    np.testing.assert_array_equal(
        one.booster.predict(rows[FEATURES]), direct.predict(rows[FEATURES])
    )


def day_sizes(part: pd.DataFrame) -> np.ndarray:
    return part.groupby("date", sort=False).size().to_numpy()


@pytest.fixture
def datasets(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Rows, weights and groups of every training and validation set lgb.train receives."""
    seen: list[dict[str, Any]] = []
    train = lgb.train

    def describe(data: lgb.Dataset) -> dict[str, Any]:
        return {"rows": data.data, "weight": data.weight, "group": data.group}

    def spy(params: dict[str, Any], train_set: lgb.Dataset, **kwargs: Any) -> lgb.Booster:
        valid = kwargs.get("valid_sets")
        seen.append({"fit": describe(train_set), "valid": valid and describe(valid[0])})
        return train(params, train_set, **kwargs)

    monkeypatch.setattr(gbdt.lgb, "train", spy)
    return seen


def test_sampled_keeps_every_nth_day_counted_back_from_the_last() -> None:
    frame = panel(date(2011, 3, 1), date(2011, 3, 14), stocks=3)
    days = sorted(frame["date"].unique())

    kept = gbdt.sampled(frame, 5)

    assert sorted(kept["date"].unique()) == [days[-6], days[-1]]
    assert len(kept) == 2 * 3
    assert gbdt.sampled(frame, 1) is frame
    with pytest.raises(ValueError, match="at least 1"):
        gbdt.sampled(frame, 0)


def test_boost_samples_fit_and_refit_rows_but_stops_on_every_validation_day(
    datasets: list[dict[str, Any]],
) -> None:
    window, rows = fold(), training_window()

    boost(rows, window, FEATURES, LABEL, seed=1, max_rounds=30, layout=gbdt.Layout(every=5))

    probe, final = datasets
    fit, valid = split_validation(rows, window)
    kept = gbdt.sampled(rows, 5)
    assert 0 < len(kept) < len(rows)
    assert list(probe["fit"]["rows"].index) == list(kept.index.intersection(fit.index))
    assert list(probe["valid"]["rows"].index) == list(valid.index)
    assert list(final["fit"]["rows"].index) == list(kept.index)
    assert final["valid"] is None


def test_boost_weights_fit_and_refit_rows_only(datasets: list[dict[str, Any]]) -> None:
    window = fold()
    rows = training_window()
    rows = rows.assign(weight=np.linspace(0.1, 1.0, len(rows)))

    boost(rows, window, FEATURES, LABEL, seed=1, max_rounds=30, layout=gbdt.Layout(weight="weight"))

    probe, final = datasets
    fit, _ = split_validation(rows, window)
    np.testing.assert_array_equal(probe["fit"]["weight"], fit["weight"].to_numpy())
    assert probe["valid"]["weight"] is None
    np.testing.assert_array_equal(final["fit"]["weight"], rows["weight"].to_numpy())


def test_boost_groups_each_set_by_date_for_ranking_objectives(
    datasets: list[dict[str, Any]],
) -> None:
    window, rows = fold(), training_window()
    rows = rows.assign(level=np.minimum((rows[LABEL] * 10).astype(int), 9))

    boost(
        rows,
        window,
        FEATURES,
        "level",
        seed=1,
        params={"objective": "lambdarank"},
        max_rounds=30,
        layout=gbdt.Layout(group=True),
    )

    probe, final = datasets
    fit, valid = split_validation(rows, window)
    np.testing.assert_array_equal(probe["fit"]["group"], day_sizes(fit))
    np.testing.assert_array_equal(probe["valid"]["group"], day_sizes(valid))
    np.testing.assert_array_equal(final["fit"]["group"], day_sizes(rows))
