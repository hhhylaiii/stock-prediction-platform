import math
from datetime import date
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from apps.etl.upsert import upsert
from apps.research import gbdt
from apps.research.features.transforms import standardize
from apps.research.metrics.ic import daily_rank_ic
from apps.research.models import ModelRun, Prediction
from apps.research.rank_model import (
    RANK_LABEL,
    Screened,
    fit_rank,
    prediction_frame,
    rank_features,
    rank_inputs,
    screened_features,
)
from apps.research.runs import SCREENING_MODEL, RunError
from apps.research.walkforward import Fold, WalkForwardConfig, generate_folds
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
