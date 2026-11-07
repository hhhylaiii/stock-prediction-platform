from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from apps.research import lgbm_runs
from apps.research.features.transforms import MarketBins
from apps.research.gbdt import Bagged
from apps.research.inference import (
    day_rows,
    model_inputs,
    run_contributions,
    run_values,
    top_contributions,
)
from apps.research.lgbm_runs import MODEL, run_id
from apps.research.models import REST_FEATURE, ModelObjective, ModelRun
from apps.research.prediction_cache import load_fold_predictions
from apps.research.rank_model import MODEL_TYPE
from apps.research.runs import load_model
from tests.research.run_panels import research, weekdays

LIVE = [fold for fold in research() if fold.research][-1]
DAY = LIVE.test_end


def live_run(scope: str, objective: str, horizon: int) -> ModelRun:
    return ModelRun.objects.get(run_id=run_id(f"{scope}-live", objective, horizon, LIVE))


def outputs(run: ModelRun, directory: Path, day: date = DAY) -> pd.DataFrame:
    rows = day_rows(day, directory, run.feature_list)
    values = run_values(run, load_model(run.run_key, run.run_id), rows)
    return rows[["stock_id", "in_universe"]].join(values)


def cached(scope: str, day: date = DAY, run_key: str | None = None) -> pd.DataFrame:
    frame = load_fold_predictions(run_key or f"{scope}-live", MODEL, LIVE)
    return frame[frame["date"] == day].drop(columns=["date", "in_universe"])


def live_days() -> list[date]:
    return [day for day in weekdays() if LIVE.test_start <= day <= LIVE.test_end]


@pytest.mark.django_db
def test_rank_inference_scores_the_day_as_its_fit_did(served_runs: Path) -> None:
    for horizon in (5, 20):
        produced = outputs(live_run("rank", "rank", horizon), served_runs)
        merged = produced.merge(cached("rank"), on="stock_id")

        assert len(merged) == len(produced) == 40
        assert not produced["in_universe"].all()
        np.testing.assert_allclose(merged["score"], merged[f"lightgbm_h{horizon}"], rtol=1e-6)
        assert list(produced.columns) == ["stock_id", "in_universe", "score"]


@pytest.mark.django_db
def test_a_single_seed_run_scores_and_explains_the_day_as_its_fit_did(served_runs: Path) -> None:
    lgbm_runs.run_folds("rank", [LIVE], "rank-solo", served_runs, bags=1)
    run = ModelRun.objects.get(run_id=run_id("rank-solo", "rank", 20, LIVE))
    model = load_model(run.run_key, run.run_id)
    rows = day_rows(DAY, served_runs, run.feature_list)

    produced = rows[["stock_id"]].join(run_values(run, model, rows))
    contrib = run_contributions(run, model, rows)

    assert isinstance(model, lgb.Booster)
    merged = produced.merge(cached("rank", run_key="rank-solo"), on="stock_id")
    np.testing.assert_allclose(merged["score"], merged["lightgbm_h20"], rtol=1e-6)
    gaps = produced.set_index("stock_id")["score"] - contrib.groupby("stock_id")["contrib"].sum()
    assert gaps.std() < 1e-9


PAIRS = [
    ("return", "regression", 5, "ret_pred", "lightgbm_ret_h5"),
    ("interval", "quantile", 20, "q10", "lightgbm_q10_h20"),
    ("interval", "quantile", 20, "q50", "lightgbm_q50_h20"),
    ("interval", "quantile", 20, "q90", "lightgbm_q90_h20"),
    ("interval", "volatility", 20, "vol_pred", "lightgbm_vol_h20"),
    ("interval", "classification", 20, "prob_up_5pct", "lightgbm_prob_h20"),
]


@pytest.mark.django_db
def test_return_and_interval_inference_match_their_fits_on_every_test_day(
    served_runs: Path,
) -> None:
    probability = live_run("interval", "classification", 20)
    assert load_model(probability.run_key, probability.run_id)["calibrator"] is not None
    for day in live_days():
        for scope, objective, horizon, column, stored in PAIRS:
            produced = outputs(live_run(scope, objective, horizon), served_runs, day)
            merged = produced.merge(cached(scope, day), on="stock_id")
            assert len(merged) == 40
            np.testing.assert_allclose(merged[column], merged[stored], rtol=1e-6)
    quantile = outputs(live_run("interval", "quantile", 20), served_runs)
    assert (quantile["ret_pred"] == quantile["q50"]).all()
    assert (quantile["q10"] <= quantile["q50"]).all()
    assert (quantile["q50"] <= quantile["q90"]).all()
    volatility = outputs(live_run("interval", "volatility", 20), served_runs)
    assert list(volatility.columns) == ["stock_id", "in_universe", "vol_pred"]


@pytest.mark.django_db
def test_an_uncalibrated_probability_model_serves_its_raw_probability(served_runs: Path) -> None:
    run = live_run("interval", "classification", 20)
    model = {**load_model(run.run_key, run.run_id), "calibrator": None}
    rows = day_rows(DAY, served_runs, run.feature_list)

    produced = rows[["stock_id"]].join(run_values(run, model, rows))

    merged = produced.merge(cached("interval"), on="stock_id")
    np.testing.assert_allclose(merged["prob_up_5pct"], merged["lightgbm_prob_raw_h20"], rtol=1e-6)
    assert not np.allclose(merged["prob_up_5pct"], merged["lightgbm_prob_h20"])


@pytest.mark.django_db
def test_market_features_enter_the_raw_models_as_their_training_bins(served_runs: Path) -> None:
    run = live_run("return", "regression", 5)
    model = load_model(run.run_key, run.run_id)
    seen = set()
    for day in live_days():
        rows = day_rows(day, served_runs, run.feature_list)
        inputs = model_inputs(run, model, rows)
        binned = MarketBins(model["market_bins"]).apply(rows)["mkt_taiex_ret_5d"]
        np.testing.assert_array_equal(inputs["mkt_taiex_ret_5d"], binned)
        seen |= set(inputs["mkt_taiex_ret_5d"])
    assert seen <= {0.0, 1.0, 2.0, 3.0, 4.0}
    assert len(seen) > 1


@pytest.mark.django_db
def test_rank_contributions_add_up_to_the_score_less_one_constant(served_runs: Path) -> None:
    run = live_run("rank", "rank", 20)
    model = load_model(run.run_key, run.run_id)
    rows = day_rows(DAY, served_runs, run.feature_list)

    contrib = run_contributions(run, model, rows)
    scores = rows[["stock_id"]].join(run_values(run, model, rows))

    assert len(contrib) == 3 * 40
    for stock, group in contrib.groupby("stock_id"):
        assert list(group["position"]) == [1, 2, 3]
        assert sorted(group["feature"]) == sorted(run.feature_list)
        assert group["contrib"].abs().is_monotonic_decreasing
        assert (group["n_features"] == 1).all()
        assert stock in set(scores["stock_id"])
    totals = contrib.groupby("stock_id")["contrib"].sum()
    gaps = scores.set_index("stock_id")["score"] - totals
    assert gaps.std() < 1e-9


def boosters(features: list[str]) -> Bagged:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(400, len(features))), columns=features)
    target = frame @ np.linspace(1.0, 0.1, len(features)) + rng.normal(scale=0.1, size=400)
    members = tuple(
        lgb.train(
            {"objective": "regression", "verbose": -1, "seed": seed, "num_leaves": 7},
            lgb.Dataset(frame, target),
            num_boost_round=30,
        )
        for seed in (1, 2)
    )
    return Bagged(members)


def test_tied_contributions_follow_the_feature_list() -> None:
    zeros = [f"z{k}" for k in range(14)]
    features = ["a", "b", "c", *zeros]
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(rng.normal(size=(400, 3)), columns=["a", "b", "c"]).assign(
        **dict.fromkeys(zeros, 0.0)
    )
    target = 3 * frame["a"] + 2 * frame["b"] + frame["c"]
    booster = lgb.train(
        {"objective": "regression", "verbose": -1, "num_leaves": 7},
        lgb.Dataset(frame[features], target),
        num_boost_round=30,
    )

    contrib = top_contributions((booster,), frame[features].head(3), pd.Series(["1", "2", "3"]))

    for _, group in contrib.groupby("stock_id"):
        assert list(group["feature"])[3:] == ["z0", "z1", REST_FEATURE]
        assert set(list(group["feature"])[:3]) == {"a", "b", "c"}
        assert group.iloc[-1]["n_features"] == 12
        assert group.iloc[-1]["contrib"] == 0.0


def test_contributions_keep_the_top_five_and_sum_the_rest() -> None:
    features = [f"f{k}" for k in range(8)]
    model = boosters(features)
    inputs = pd.DataFrame(np.random.default_rng(8).normal(size=(6, 8)), columns=features)
    stock_ids = pd.Series([f"{2300 + k}" for k in range(6)])

    contrib = top_contributions(model.boosters, inputs, stock_ids)

    full = np.mean([m.predict(inputs, pred_contrib=True) for m in model.boosters], axis=0)
    assert len(contrib) == 6 * 6
    for k, (stock, group) in enumerate(contrib.groupby("stock_id", sort=False)):
        assert stock == stock_ids[k]
        assert list(group["position"]) == [1, 2, 3, 4, 5, 6]
        top, rest = group.iloc[:5], group.iloc[5]
        assert rest["feature"] == REST_FEATURE
        assert rest["n_features"] == 3
        assert (top["n_features"] == 1).all()
        assert top["feature"].is_unique
        shown = [features.index(name) for name in top["feature"]]
        hidden = [j for j in range(8) if j not in shown]
        assert np.abs(full[k, shown]).min() >= np.abs(full[k, hidden]).max()
        np.testing.assert_allclose(top["contrib"], full[k, shown])
        assert rest["contrib"] == pytest.approx(full[k, hidden].sum())
        assert group["contrib"].sum() + full[k, -1] == pytest.approx(model.predict(inputs)[k])


@pytest.mark.django_db
def test_inference_refuses_a_model_fit_on_other_features(served_runs: Path) -> None:
    run = live_run("rank", "rank", 20)
    model = load_model(run.run_key, run.run_id)
    rows = day_rows(DAY, served_runs, run.feature_list)
    run.feature_list = list(reversed(run.feature_list))

    with pytest.raises(ValueError, match=r"fit on .* not its feature_list"):
        run_values(run, model, rows)


@pytest.mark.django_db
def test_inference_serves_lightgbm_runs_only(served_runs: Path) -> None:
    run = live_run("rank", "rank", 20)
    run.model_type = "ridge"

    with pytest.raises(ValueError, match="serves lightgbm runs, not ridge"):
        run_values(run, None, pd.DataFrame())


@pytest.mark.django_db
def test_a_day_without_feature_rows_is_refused(served_runs: Path) -> None:
    with pytest.raises(ValueError, match="no feature rows on 2016-12-31; run build_features"):
        day_rows(date(2016, 12, 31), served_runs, ["tech_ret_1d"])


def test_a_raw_value_run_listing_a_feature_no_model_takes_is_refused() -> None:
    features = ["fund_ep", "fund_rev_mom"]
    rng = np.random.default_rng(0)
    rows = pd.DataFrame({name: rng.normal(size=50) for name in features})
    booster = lgb.train(
        {"objective": "regression", "verbose": -1},
        lgb.Dataset(rows, rng.normal(size=50)),
        num_boost_round=2,
    )
    run = ModelRun(
        run_id="old-mom",
        model_type=MODEL_TYPE,
        objective=ModelObjective.REGRESSION,
        feature_list=features,
    )

    with pytest.raises(ValueError, match=r"old-mom reads .*\['fund_rev_mom'\]"):
        model_inputs(
            run, {"booster": booster, "market_bins": {}}, rows.assign(date=DAY, in_universe=True)
        )


def test_a_rank_run_listing_a_feature_no_model_takes_is_refused() -> None:
    features = ["fund_ep", "fund_rev_mom"]
    rng = np.random.default_rng(1)
    rows = pd.DataFrame({name: rng.normal(size=50) for name in features})
    booster = lgb.train(
        {"objective": "regression", "verbose": -1},
        lgb.Dataset(rows, rng.normal(size=50)),
        num_boost_round=2,
    )
    run = ModelRun(
        run_id="old-rank",
        model_type=MODEL_TYPE,
        objective=ModelObjective.RANK,
        feature_list=features,
    )

    with pytest.raises(ValueError, match=r"stock-level features only, not \['fund_rev_mom'\]"):
        model_inputs(run, booster, rows.assign(date=DAY, in_universe=True))
