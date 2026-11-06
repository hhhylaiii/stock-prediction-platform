import math
from datetime import date
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from apps.etl.upsert import upsert
from apps.research import gbdt, rank_model
from apps.research.features.transforms import standardize
from apps.research.metrics.ic import daily_rank_ic
from apps.research.models import ModelRun, Prediction
from apps.research.rank_model import (
    LAMBDARANK,
    LEVELS,
    RANK_LABEL,
    Variant,
    fit_rank,
    prediction_frame,
    rank_features,
    rank_inputs,
)
from apps.research.runs import SCREENING_MODEL, RunError, Screened, screened_features
from apps.research.training import training_rows
from apps.research.walkforward import Fold, WalkForwardConfig, generate_folds, time_decay_weights
from tests.research.conftest import weekday_calendar

CALENDAR = weekday_calendar(date(2010, 1, 1), date(2014, 12, 31))
CONFIG = WalkForwardConfig(train_months=12, valid_months=3, first_test=date(2012, 1, 1))
FEATURES = ["tech_ret_20d", "fund_ep"]
STOCKS = 70


def fold() -> Fold:
    return generate_folds(CALENDAR, date(2012, 6, 29), CONFIG)[0]


def raw_panel(start: date, end: date, seed: int = 0) -> pd.DataFrame:
    """Raw features where tech_ret_20d drives the forward return; the last stocks sit outside."""
    days = CALENDAR.between(start, end)
    rng = np.random.default_rng(seed)
    size = len(days) * STOCKS
    frame = pd.DataFrame(
        {
            "date": np.repeat(np.array(days, dtype=object), STOCKS),
            "stock_id": np.tile([f"{1000 + k}" for k in range(STOCKS)], len(days)),
            "in_universe": np.tile(np.arange(STOCKS) < STOCKS - 5, len(days)),
            "tech_ret_20d": rng.normal(0.01, 0.1, size=size),
            "fund_ep": rng.normal(0.05, 0.02, size=size),
            "mkt_taiex_ret_5d": np.repeat(rng.normal(size=len(days)), STOCKS),
        }
    )
    return frame.assign(fund_ep=frame["fund_ep"].mask(rng.random(size) < 0.1))


def labels_for(raw: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = raw["tech_ret_20d"] * 10 + rng.normal(size=len(raw))
    return pd.DataFrame(
        {
            "stock_id": raw["stock_id"],
            "date": raw["date"],
            "fwd_ret_open": ret,
            RANK_LABEL: ret.groupby(raw["date"]).rank(pct=True),
            "untradable_t1": rng.random(len(raw)) < 0.03,
        }
    )


def screening(**changes: Any) -> ModelRun:
    fields: dict[str, Any] = {
        "run_id": "screen-1.screening",
        "run_key": "screen-1",
        "model_type": SCREENING_MODEL,
        "objective": "rank",
        "version": "1",
        "train_start": date(2010, 1, 4),
        "train_end": date(2015, 11, 30),
        "params": {"unscreened": ["mkt_taiex_ret_5d", "mkt_sox_ret_5d"]},
        "feature_list": ["tech_ret_20d", "fund_ep", "size_log_mcap"],
        "lib_versions": {"python": "3.12"},
        "feature_hash": "0" * 64,
    }
    return ModelRun.objects.create(**{**fields, **changes})


@pytest.mark.django_db
def test_screened_features_read_the_recorded_screening() -> None:
    screening()

    assert screened_features() == Screened(
        kept=["tech_ret_20d", "fund_ep", "size_log_mcap"],
        unscreened=["mkt_taiex_ret_5d", "mkt_sox_ret_5d"],
    )


@pytest.mark.django_db
def test_screened_features_need_exactly_one_screening() -> None:
    with pytest.raises(RunError, match="no feature screening"):
        screened_features()

    screening()
    screening(run_id="screen-2.screening", run_key="screen-2")

    with pytest.raises(RunError, match="2 feature screenings"):
        screened_features()


def test_rank_features_keep_the_screened_stock_level_features_in_order() -> None:
    kept = ["fund_ep", "tech_ret_20d", "size_log_mcap"]

    assert rank_features(Screened(kept, ["mkt_taiex_ret_5d"])) == kept


@pytest.mark.parametrize("name", ["mkt_taiex_ret_5d", "made_up_feature"])
def test_rank_features_refuse_anything_but_stock_level_features(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        rank_features(Screened(["tech_ret_20d", name], []))


def test_fit_rank_refuses_features_that_are_not_stock_level() -> None:
    window = fold()
    raw = raw_panel(window.train_start, window.train_end)

    with pytest.raises(ValueError, match="mkt_taiex_ret_5d"):
        fit_rank(raw, labels_for(raw), window, [*FEATURES, "mkt_taiex_ret_5d"], seed=3)


def test_rank_inputs_zscore_the_features_and_keep_gaps_and_other_columns() -> None:
    raw = raw_panel(date(2011, 3, 1), date(2011, 3, 10))

    inputs = rank_inputs(raw, FEATURES)

    pd.testing.assert_frame_equal(inputs, standardize(raw, FEATURES))
    assert inputs["fund_ep"].isna().equals(raw["fund_ep"].isna())
    assert inputs["mkt_taiex_ret_5d"].equals(raw["mkt_taiex_ret_5d"])
    universe = inputs[inputs["in_universe"]]
    daily = universe.groupby("date")["tech_ret_20d"]
    assert daily.mean().abs().max() < 1e-9
    assert (daily.std() - 1).abs().max() < 1e-9


def test_fit_rank_trains_on_tradable_universe_rows_of_the_window_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = fold()
    raw = raw_panel(CALENDAR.shift(window.train_start, -30), window.test_end)
    labels = labels_for(raw)
    inputs = rank_inputs(raw, FEATURES)
    sizes: list[int] = []
    train = lgb.train

    def spy(params: dict[str, Any], train_set: lgb.Dataset, **kwargs: Any) -> lgb.Booster:
        booster = train(params, train_set, **kwargs)
        sizes.append(train_set.num_data())
        return booster

    monkeypatch.setattr(gbdt.lgb, "train", spy)

    boosted = fit_rank(inputs, labels, window, FEATURES, seed=3, max_rounds=200)

    joined = raw.merge(labels, on=["stock_id", "date"])
    usable = (
        joined["in_universe"]
        & ~joined["untradable_t1"]
        & joined["date"].between(window.train_start, window.train_end)
    )
    assert sizes[-1] == int(usable.sum())
    test = inputs[inputs["date"].between(window.test_start, window.test_end)]
    scored = test.assign(score=boosted.booster.predict(test[FEATURES])).merge(labels)
    assert daily_rank_ic(scored, "score").mean() > 0.3


def test_prediction_frame_ranks_the_universe_from_the_highest_score() -> None:
    rows = pd.DataFrame(
        {
            "date": [date(2026, 10, 30)] * 5 + [date(2026, 11, 2)] * 2,
            "stock_id": ["2330", "2317", "1101", "9999", "2454", "2330", "2317"],
            "in_universe": [True, True, True, False, True, True, True],
        }
    )

    frame = prediction_frame(rows, np.array([0.9, 0.2, 0.5, 0.95, 0.5, 0.1, 0.3]))

    first = frame[frame["date"] == date(2026, 10, 30)].set_index("stock_id")
    assert first.loc[["2330", "1101", "2454", "2317"], "rank"].tolist() == [1, 2, 3, 4]
    assert first.loc[["2330", "1101", "2454", "2317"], "pct"].tolist() == [1.0, 0.75, 0.5, 0.25]
    assert first.loc[["9999"], ["rank", "pct"]].isna().all(axis=None)
    assert first.loc["9999", "score"] == 0.95
    second = frame[frame["date"] == date(2026, 11, 2)].set_index("stock_id")
    assert second["rank"].to_dict() == {"2330": 2, "2317": 1}
    assert list(frame.columns) == ["date", "stock_id", "in_universe", "score", "rank", "pct"]


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_prediction_frame_refuses_scores_that_are_not_finite(bad: float) -> None:
    rows = pd.DataFrame(
        {"date": [date(2026, 10, 30)] * 2, "stock_id": ["2330", "2317"], "in_universe": True}
    )

    with pytest.raises(ValueError, match="finite"):
        prediction_frame(rows, np.array([0.5, bad]))


@pytest.mark.django_db
def test_prediction_frame_upserts_into_the_prediction_table() -> None:
    run = screening(run_id="prod-rank.lightgbm.h20", run_key="prod-rank", model_type="lightgbm")
    rows = pd.DataFrame(
        {
            "date": [date(2026, 10, 30)] * 3,
            "stock_id": ["2330", "2317", "9999"],
            "in_universe": [True, True, False],
        }
    )
    frame = prediction_frame(rows, np.array([0.7, 0.4, 0.9])).assign(run_id=run.run_id, horizon=20)
    keys = ["run_id", "stock_id", "date", "horizon"]

    upsert(Prediction, frame, keys, ["score", "rank", "pct", "in_universe"])

    stored = Prediction.objects.order_by("stock_id").values_list("stock_id", "rank", "pct")
    assert list(stored) == [("2317", 2, 0.5), ("2330", 1, 1.0), ("9999", None, None)]


@pytest.fixture
def boosts(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Arguments of every boost call fit_rank makes."""
    seen: list[dict[str, Any]] = []
    original = rank_model.boost

    def spy(
        rows: pd.DataFrame, fold: Fold, features: Any, label: str, *args: Any, **kwargs: Any
    ) -> Any:
        _seed, params, feval, _rounds, _bags, layout = args
        seen.append({"rows": rows, "label": label, "params": params, "feval": feval})
        seen[-1].update(
            layout=layout,
            min_rounds=kwargs.get("min_rounds"),
            fixed_rounds=kwargs.get("fixed_rounds"),
        )
        return original(rows, fold, features, label, *args, **kwargs)

    monkeypatch.setattr(rank_model, "boost", spy)
    return seen


def window_inputs() -> tuple[Fold, pd.DataFrame, pd.DataFrame]:
    window = fold()
    raw = raw_panel(CALENDAR.shift(window.train_start, -30), window.test_end)
    return window, rank_inputs(raw, FEATURES), labels_for(raw)


def test_variant_defaults_are_the_spec_recipe() -> None:
    assert Variant() == Variant(every=1, half_life=None, objective="regression", min_rounds=0)
    for bad, message in (
        ({"every": 0}, "every"),
        ({"min_rounds": -1}, "min_rounds"),
        ({"fixed_rounds": ((5, 0),)}, "fixed_rounds"),
        ({"fixed_rounds": ((5, 3), (5, 4))}, "fixed_rounds"),
        ({"fixed_rounds": ((10, 3),)}, "fixed_rounds"),
        ({"half_life": 0.0}, "half_life"),
        ({"objective": "xendcg"}, "objective"),
    ):
        with pytest.raises(ValueError, match=message):
            Variant(**bad)


def test_fit_rank_samples_every_fifth_day_through_the_layout(boosts: list[Any]) -> None:
    window, inputs, labels = window_inputs()

    fit_rank(inputs, labels, window, FEATURES, 3, 50, variant=Variant(every=5))

    (call,) = boosts
    assert call["layout"] == gbdt.Layout(every=5)
    assert call["label"] == RANK_LABEL
    assert call["params"] is None
    assert call["min_rounds"] == 0


def test_fit_rank_passes_the_variant_round_floor_to_boost(boosts: list[Any]) -> None:
    window, inputs, labels = window_inputs()

    boosted = fit_rank(inputs, labels, window, FEATURES, 3, 400, variant=Variant(min_rounds=120))

    (call,) = boosts
    assert call["min_rounds"] == 120
    assert boosted.rounds >= 120


def test_fit_rank_refits_the_fixed_rounds_it_is_given(boosts: list[Any]) -> None:
    window, inputs, labels = window_inputs()
    variant = Variant(fixed_rounds=((5, 11), (20, 13)))

    boosted = fit_rank(inputs, labels, window, FEATURES, 3, 400, variant=variant, fixed_rounds=11)

    (call,) = boosts
    assert call["fixed_rounds"] == 11
    assert boosted.rounds == 11
    assert variant.fixed_for(5) == 11
    assert variant.fixed_for(20) == 13
    assert Variant().fixed_for(5) == 0


def test_fit_rank_weights_rows_by_age_with_the_half_life(boosts: list[Any]) -> None:
    window, inputs, labels = window_inputs()

    fit_rank(inputs, labels, window, FEATURES, 3, 50, variant=Variant(half_life=24.0))

    (call,) = boosts
    rows, layout = call["rows"], call["layout"]
    expected = time_decay_weights(rows["date"], window.train_end, 24.0)
    np.testing.assert_allclose(rows[layout.weight].to_numpy(), expected)
    assert rows[layout.weight].max() <= 1.0


def test_fit_rank_lambdarank_grades_ten_levels_and_stops_on_the_rank_pct(
    boosts: list[Any],
) -> None:
    window, inputs, labels = window_inputs()

    boosted = fit_rank(
        inputs, labels, window, FEATURES, 3, 200, variant=Variant(objective="lambdarank")
    )

    (call,) = boosts
    rows = call["rows"]
    assert call["params"] == LAMBDARANK
    assert LAMBDARANK["lambdarank_truncation_level"] == 30
    assert LAMBDARANK["label_gain"] == [2.0**level - 1 for level in range(LEVELS)]
    assert call["layout"] == gbdt.Layout(group=True)
    grades = np.minimum(np.floor(rows[RANK_LABEL] * 10), 9).astype(int)
    np.testing.assert_array_equal(rows[call["label"]].to_numpy(), grades.to_numpy())
    assert set(rows[call["label"]]) <= set(range(10))
    _, valid = gbdt.split_validation(rows, window)
    metric = call["feval"](valid["date"], valid[call["label"]])
    np.testing.assert_array_equal(metric.frame["label"].to_numpy(), valid[RANK_LABEL].to_numpy())
    test = inputs[inputs["date"].between(window.test_start, window.test_end)]
    scored = test.assign(score=boosted.booster.predict(test[FEATURES])).merge(labels)
    assert daily_rank_ic(scored, "score").mean() > 0.3


def test_fit_rank_default_matches_the_plain_boost_bit_for_bit() -> None:
    window, inputs, labels = window_inputs()

    boosted = fit_rank(inputs, labels, window, FEATURES, 3, 120)

    rows = training_rows(
        inputs[inputs["date"].between(window.train_start, window.train_end)], labels, RANK_LABEL
    )
    plain = gbdt.boost(
        rows, window, FEATURES, RANK_LABEL, 3, feval=gbdt.DailyRankIC, max_rounds=120
    )
    test = inputs[inputs["date"].between(window.test_start, window.test_end)][FEATURES]
    assert boosted.rounds == plain.rounds
    np.testing.assert_array_equal(boosted.booster.predict(test), plain.booster.predict(test))
