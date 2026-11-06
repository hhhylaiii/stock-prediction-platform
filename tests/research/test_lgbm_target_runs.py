import math
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice, IndexDaily, StockMaster
from apps.research import lgbm_runs, target_models
from apps.research.baseline_runs import MAX_RUN_KEY
from apps.research.features.store import export_year
from apps.research.gbdt import BASE_PARAMS, Bagged, seeded
from apps.research.holdout import HoldoutError
from apps.research.lgbm_runs import MODEL, recipe, run_folds, run_id
from apps.research.management.commands import train
from apps.research.metrics.forecast import (
    expected_calibration_error,
    pinball_loss,
    reliability_table,
)
from apps.research.models import FeatureDaily, LabelDaily, ModelRun, ModelStatus
from apps.research.prediction_cache import (
    fold_notes,
    fold_path,
    load_fold_predictions,
    missing_folds,
)
from apps.research.relative_up import relative_up
from apps.research.runs import SCREENING_MODEL, label_hash, load_model
from apps.research.target_models import huber_delta, raw_inputs, sorted_quantiles
from apps.research.training import load_labels, training_rows
from tests.research.run_panels import (
    CONFIG,
    KEPT,
    STOCKS,
    UNSCREENED,
    folds,
    research,
    weekdays,
)

pytestmark = pytest.mark.django_db

FEATURES = [*KEPT, *UNSCREENED]
INTERVAL_COLUMNS = [
    "lightgbm_q10_h20",
    "lightgbm_q50_h20",
    "lightgbm_q90_h20",
    "lightgbm_vol_h20",
    "lightgbm_prob_h20",
    "lightgbm_prob_raw_h20",
]


@pytest.fixture
def panel(run_panel: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(lgbm_runs, "MAX_ROUNDS", 120)
    return run_panel


def raw_frame(panel: Path) -> pd.DataFrame:
    return pd.read_parquet(panel / "year=2016" / "part-0.parquet")


def labels(start: Any, end: Any, horizon: int) -> pd.DataFrame:
    rows = LabelDaily.objects.filter(date__range=(start, end), horizon=horizon)
    return pd.DataFrame(list(rows.values()))


def evaluated(panel: Path, cached: pd.DataFrame, fold: Any, horizon: int, label: str) -> Any:
    """Cached rows of test_start..eval_end that the fold's OOS metrics score."""
    known = labels(fold.test_start, fold.eval_end, horizon)
    inside = cached[cached["date"] <= fold.eval_end]
    return training_rows(inside, known, label)


def test_return_folds_cache_huber_predictions_on_five_day_labels(panel: Path) -> None:
    fold = research()[-1]

    assert run_folds("return", [fold], "return-wf1", panel, seed=2) == [fold.index]

    cached = load_fold_predictions("return-wf1", MODEL, fold)
    test_days = [day for day in weekdays() if fold.test_start <= day <= fold.test_end]
    assert cached.columns.tolist() == ["date", "stock_id", "in_universe", "lightgbm_ret_h5"]
    assert len(cached) == len(test_days) * len(STOCKS)
    run = ModelRun.objects.get(run_id=f"return-wf1.lightgbm.h5.f{fold.index:03d}")
    assert (run.model_type, run.objective, run.seed) == ("lightgbm", "regression", 2)
    assert run.feature_list == FEATURES
    assert run.params["horizon"] == 5
    assert run.params["label"] == "fwd_ret_open"
    assert run.params["lightgbm"] == {**BASE_PARAMS, **seeded(2)}
    assert 1 <= run.params["rounds"] <= 120
    assert list(run.params["market_bins"]) == UNSCREENED
    assert len(run.params["market_bins"]["mkt_taiex_ret_5d"]) == 4
    raw = raw_frame(panel)
    train = training_rows(
        raw.loc[raw["date"].between(fold.train_start, fold.train_end)],
        labels(fold.train_start, fold.train_end, 5),
        "fwd_ret_open",
    )
    assert run.params["delta"] == pytest.approx(huber_delta(train["fwd_ret_open"]))
    scored = evaluated(panel, cached, fold, 5, "fwd_ret_open")
    mae = (scored["lightgbm_ret_h5"] - scored["fwd_ret_open"]).abs().mean()
    assert run.params["oos_mae"] == pytest.approx(mae)
    assert run.params["oos_mae_zero"] == pytest.approx(scored["fwd_ret_open"].abs().mean())
    assert run.ic_mean is not None
    assert run.ic_mean > 0.3
    assert set(run.params["importance_gain"]) == set(FEATURES)


def test_saved_return_model_reproduces_the_cache(panel: Path) -> None:
    fold = research()[0]
    run_folds("return", [fold], "return-wf1", panel)

    saved = load_model("return-wf1", run_id("return-wf1", "regression", 5, fold))
    raw = raw_frame(panel)
    inputs, bins = raw_inputs(
        raw[raw["date"].between(fold.train_start, fold.test_end)], fold, FEATURES
    )
    test = inputs[inputs["date"].between(fold.test_start, fold.test_end)]
    assert saved["market_bins"] == bins.edges
    cached = load_fold_predictions("return-wf1", MODEL, fold)
    np.testing.assert_allclose(
        cached["lightgbm_ret_h5"].to_numpy(), saved["booster"].predict(test[FEATURES])
    )


def test_saved_interval_models_reproduce_the_cache(
    panel: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(target_models, "ECE_LIMIT", -1.0)
    fold = research()[0]
    run_folds("interval", [fold], "interval-wf1", panel)

    raw = raw_frame(panel)
    inputs, bins = raw_inputs(
        raw[raw["date"].between(fold.train_start, fold.test_end)], fold, FEATURES
    )
    test = inputs[inputs["date"].between(fold.test_start, fold.test_end)][FEATURES]
    cached = load_fold_predictions("interval-wf1", MODEL, fold)
    saved = {
        str(objective): load_model("interval-wf1", run_id("interval-wf1", objective, 20, fold))
        for objective in lgbm_runs.INTERVAL_TAGS
    }
    assert all(model["market_bins"] == bins.edges for model in saved.values())
    boosters = saved["quantile"]["boosters"]
    quantiles = sorted_quantiles({name: boosters[name].predict(test) for name in boosters})
    for name in ("q10", "q50", "q90"):
        np.testing.assert_allclose(cached[f"lightgbm_{name}_h20"], quantiles[name])
    vol = np.exp(saved["volatility"]["booster"].predict(test))
    np.testing.assert_allclose(cached["lightgbm_vol_h20"], vol)
    prob = saved["classification"]
    raw_prob = prob["booster"].predict(test)
    np.testing.assert_allclose(cached["lightgbm_prob_raw_h20"], raw_prob)
    np.testing.assert_allclose(cached["lightgbm_prob_h20"], prob["calibrator"].predict(raw_prob))
    for objective, booster in [
        ("quantile", boosters["q50"]),
        ("volatility", saved["volatility"]["booster"]),
        ("classification", prob["booster"]),
    ]:
        params = ModelRun.objects.get(run_id=run_id("interval-wf1", objective, 20, fold)).params
        gains = booster.feature_importance("gain")
        assert params["importance_gain"] == {
            name: float(gain) for name, gain in zip(FEATURES, gains, strict=True)
        }


def test_a_market_feature_missing_through_the_window_records_open_bins(panel: Path) -> None:
    fold = research()[0]
    FeatureDaily.objects.filter(date__range=(fold.train_start, fold.train_end)).update(
        mkt_taiex_ret_5d=None
    )
    export_year(2016, panel)

    assert run_folds("return", [fold], "return-wf1", panel) == [fold.index]

    run = ModelRun.objects.get(run_id=run_id("return-wf1", "regression", 5, fold))
    assert run.params["market_bins"] == {"mkt_taiex_ret_5d": [None, None, None, None]}


def test_interval_folds_cache_sorted_quantiles_volatility_and_probability(panel: Path) -> None:
    fold = research()[-1]

    assert run_folds("interval", [fold], "interval-wf1", panel, seed=1) == [fold.index]

    cached = load_fold_predictions("interval-wf1", MODEL, fold)
    assert cached.columns.tolist() == ["date", "stock_id", "in_universe", *INTERVAL_COLUMNS]
    assert cached[INTERVAL_COLUMNS].notna().all().all()
    assert (cached["lightgbm_q10_h20"] <= cached["lightgbm_q50_h20"]).all()
    assert (cached["lightgbm_q50_h20"] <= cached["lightgbm_q90_h20"]).all()
    assert (cached["lightgbm_vol_h20"] > 0).all()
    assert cached["lightgbm_prob_h20"].between(0, 1).all()
    runs = {
        run.objective: run
        for run in ModelRun.objects.filter(run_key="interval-wf1", model_type="lightgbm")
    }
    assert set(runs) == {"quantile", "volatility", "classification"}
    assert {run.params["horizon"] for run in runs.values()} == {20}
    assert {run.run_id for run in runs.values()} == {
        f"interval-wf1.lightgbm.{tag}.h20.f{fold.index:03d}" for tag in ("q", "vol", "prob")
    }
    quantile = runs["quantile"].params
    assert set(quantile["rounds"]) == {"q10", "q50", "q90"}
    scored = evaluated(panel, cached, fold, 20, "fwd_ret_open")
    actual = scored["fwd_ret_open"]
    inside = (actual >= scored["lightgbm_q10_h20"]) & (actual <= scored["lightgbm_q90_h20"])
    assert quantile["oos_coverage"] == pytest.approx(inside.mean())
    pinball = sum(
        pinball_loss(actual, scored[f"lightgbm_{name}_h20"], alpha).mean()
        for name, alpha in target_models.QUANTILES.items()
    )
    assert quantile["oos_pinball"] == pytest.approx(pinball)
    vol = evaluated(panel, cached, fold, 20, "fwd_vol_20d")
    mae = (vol["lightgbm_vol_h20"] - vol["fwd_vol_20d"]).abs().mean()
    assert runs["volatility"].params["oos_mae"] == pytest.approx(mae)
    assert runs["volatility"].params["vol_floor"] == target_models.VOL_FLOOR


def test_interval_folds_uncross_quantile_boosters_row_by_row(
    panel: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit = lgbm_runs.fit_quantiles

    def crossed(*args: Any, **kwargs: Any) -> Any:
        boosters = fit(*args, **kwargs)
        return {**boosters, "q10": boosters["q90"], "q90": boosters["q10"]}

    monkeypatch.setattr(lgbm_runs, "fit_quantiles", crossed)
    fold = research()[0]
    run_folds("interval", [fold], "interval-wf1", panel)

    cached = load_fold_predictions("interval-wf1", MODEL, fold)
    assert (cached["lightgbm_q10_h20"] <= cached["lightgbm_q50_h20"]).all()
    assert (cached["lightgbm_q50_h20"] <= cached["lightgbm_q90_h20"]).all()
    assert (cached["lightgbm_q10_h20"] < cached["lightgbm_q90_h20"]).any()


def test_interval_folds_record_ece_and_brier_before_and_after_calibration(
    panel: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(target_models, "ECE_LIMIT", -1.0)
    fold = research()[-1]
    run_folds("interval", [fold], "interval-wf1", panel)

    cached = load_fold_predictions("interval-wf1", MODEL, fold)
    params = ModelRun.objects.get(objective="classification", run_key="interval-wf1").params
    scored = evaluated(panel, cached, fold, 20, "fwd_up_5pct")
    scored = scored.assign(up=scored["fwd_up_5pct"].astype(float))
    for column, suffix in (("lightgbm_prob_h20", ""), ("lightgbm_prob_raw_h20", "_raw")):
        table = reliability_table(scored, "up", column, target_models.ECE_BINS)
        assert params[f"oos_ece{suffix}"] == pytest.approx(expected_calibration_error(table))
        brier = ((scored[column] - scored["up"]) ** 2).mean()
        assert params[f"oos_brier{suffix}"] == pytest.approx(brier)
    assert params["calibrated"] is True
    assert 0 <= params["valid_ece"] <= 1
    saved = load_model("interval-wf1", run_id("interval-wf1", "classification", 20, fold))
    assert saved["calibrator"] is not None
    assert not np.allclose(cached["lightgbm_prob_h20"], cached["lightgbm_prob_raw_h20"])


def test_each_target_fits_on_the_horizon_of_its_label(
    panel: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: dict[int, int] = {}
    fitted: list[tuple[str, int]] = []
    load = lgbm_runs.load_labels

    def tagged(start: Any, end: Any, horizon: int) -> pd.DataFrame:
        frame = load(start, end, horizon)
        loaded[id(frame)] = horizon
        return frame

    def spying(name: str) -> Any:
        real = getattr(lgbm_runs, name)

        def spy(inputs: pd.DataFrame, frame: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
            fitted.append((name, loaded[id(frame)]))
            return real(inputs, frame, *args, **kwargs)

        return spy

    monkeypatch.setattr(lgbm_runs, "load_labels", tagged)
    for name in ("fit_return", "fit_quantiles", "fit_volatility", "fit_probability"):
        monkeypatch.setattr(lgbm_runs, name, spying(name))
    fold = research()[0]

    run_folds("return", [fold], "return-wf1", panel)
    run_folds("interval", [fold], "interval-wf1", panel)

    assert fitted == [
        ("fit_return", 5),
        ("fit_quantiles", 20),
        ("fit_volatility", 20),
        ("fit_probability", 20),
    ]


def test_cached_target_folds_rerun_only_when_their_own_labels_change(panel: Path) -> None:
    fold = research()[0]
    run_folds("return", [fold], "return-wf1", panel)
    run_folds("interval", [fold], "interval-wf1", panel)

    assert fold_notes("return-wf1", MODEL, fold.index)["label_hash"] == label_hash(
        fold.train_start, fold.eval_end, (5,)
    )
    assert fold_notes("interval-wf1", MODEL, fold.index)["label_hash"] == label_hash(
        fold.train_start, fold.eval_end, (20,)
    )
    LabelDaily.objects.filter(date=fold.train_start, horizon=20).update(fwd_vol_20d=0.9)

    assert run_folds("return", [fold], "return-wf1", panel) == []
    assert run_folds("interval", [fold], "interval-wf1", panel) == [fold.index]


def test_recipes_differ_by_scope_and_follow_each_target_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scopes = ("rank", "return", "interval")
    assert len({recipe(scope, FEATURES, 0) for scope in scopes}) == 3
    for scope, name, value in [
        ("return", "HUBER_MAD_SCALE", 2.0),
        ("interval", "QUANTILES", {"q10": 0.05, "q50": 0.5, "q90": 0.95}),
        ("interval", "VOL_FLOOR", 0.02),
        ("interval", "ECE_LIMIT", 0.1),
        ("interval", "ECE_BINS", 20),
    ]:
        first = recipe(scope, FEATURES, 0)
        with monkeypatch.context() as patched:
            patched.setattr(target_models, name, value)
            assert recipe(scope, FEATURES, 0) != first, name


def test_the_longest_run_ids_fit_the_model_run_key() -> None:
    fold = research()[0]
    longest = "interval-" + "x" * (MAX_RUN_KEY - len("interval-"))
    limit = ModelRun._meta.get_field("run_id").max_length
    assert limit is not None

    ids = [run_id(longest, objective, 20, fold) for objective in lgbm_runs.INTERVAL_TAGS]
    assert max(len(value) for value in ids) <= limit


@pytest.mark.parametrize(
    ("scope", "fit", "horizon"),
    [("return", "fit_return", 5), ("interval", "fit_probability", 20)],
)
def test_labels_rewritten_during_a_target_fit_stop_the_fold(
    panel: Path, monkeypatch: pytest.MonkeyPatch, scope: str, fit: str, horizon: int
) -> None:
    fold = research()[0]
    real = getattr(lgbm_runs, fit)

    def relabelling(*args: Any, **kwargs: Any) -> Any:
        fitted = real(*args, **kwargs)
        LabelDaily.objects.filter(date=fold.train_start, horizon=horizon).update(fwd_ret_open=0.5)
        return fitted

    monkeypatch.setattr(lgbm_runs, fit, relabelling)

    with pytest.raises(RuntimeError, match="label_daily changed"):
        run_folds(scope, [fold], f"{scope}-wf1", panel)

    assert not ModelRun.objects.filter(run_key=f"{scope}-wf1").exists()
    assert missing_folds(f"{scope}-wf1", MODEL, [fold]) == [fold.index]


def test_an_interval_fold_with_a_promoted_model_keeps_its_cache(panel: Path) -> None:
    fold = research()[0]
    run_folds("interval", [fold], "interval-wf1", panel)
    promoted = run_id("interval-wf1", "volatility", 20, fold)
    ModelRun.objects.filter(run_id=promoted).update(status=ModelStatus.ACTIVE)

    with pytest.raises(lgbm_runs.RunError, match=promoted):
        run_folds("interval", [fold], "interval-wf1", panel, rerun=True)

    assert fold_path("interval-wf1", MODEL, fold.index).exists()


@pytest.mark.parametrize("scope", ["return", "interval"])
def test_target_holdout_folds_wait_for_the_frozen_design(panel: Path, scope: str) -> None:
    held = [fold for fold in folds() if fold.holdout]

    with pytest.raises(HoldoutError, match="holdout"):
        run_folds(scope, held, f"{scope}-wf1", panel)


def test_a_fold_without_test_labels_names_the_fold(panel: Path) -> None:
    fold = research()[0]
    LabelDaily.objects.filter(date__range=(fold.test_start, fold.eval_end), horizon=5).delete()

    with pytest.raises(RuntimeError, match=f"fold {fold.index} has no labelled fwd_ret_open"):
        run_folds("return", [fold], "return-wf1", panel)

    assert not ModelRun.objects.filter(run_key="return-wf1").exists()


def test_target_folds_need_the_screening(panel: Path) -> None:
    ModelRun.objects.filter(model_type=SCREENING_MODEL).delete()

    with pytest.raises(lgbm_runs.RunError, match="return scope fits on the screened features"):
        run_folds("return", research()[:1], "return-wf1", panel)


@pytest.fixture
def command(panel: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(train, "DEFAULT_CONFIG", CONFIG)
    StockMaster.objects.create(stock_id="1101", name="1101", market="TWSE", industry="水泥工業")
    DailyPrice.objects.create(
        stock_id="1101", date=weekdays()[-1], close=10, volume=1000, turnover=10_000
    )
    return panel


def test_command_trains_lightgbm_interval_folds(command: Path) -> None:
    out = StringIO()

    call_command(
        "train", model="lightgbm", scope="interval", run_key="interval-wf1", folds="0", stdout=out
    )

    assert "interval-wf1: ran folds [0]" in out.getvalue()
    assert ModelRun.objects.filter(run_key="interval-wf1", model_type="lightgbm").count() == 3


def test_a_full_lightgbm_return_run_checks_the_screening_before_the_shuffle_check(
    command: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(train, "shuffle_check", lambda *args: pytest.fail("shuffle ran"))
    ModelRun.objects.filter(model_type=SCREENING_MODEL).delete()

    with pytest.raises(CommandError, match="return scope fits on the screened features"):
        call_command("train", model="lightgbm", scope="return", run_key="return-wf1")


def test_bagged_target_folds_bag_every_booster(panel: Path) -> None:
    fold = research()[0]

    run_folds("return", [fold], "return-wf1", panel, seed=2, bags=2)
    run_folds("interval", [fold], "interval-wf1", panel, seed=2, bags=2)

    saved = {
        str(objective): load_model("interval-wf1", run_id("interval-wf1", objective, 20, fold))
        for objective in lgbm_runs.INTERVAL_TAGS
    }
    saved["regression"] = load_model("return-wf1", run_id("return-wf1", "regression", 5, fold))
    members = [
        *saved["quantile"]["boosters"].values(),
        saved["volatility"]["booster"],
        saved["classification"]["booster"],
        saved["regression"]["booster"],
    ]
    assert all(isinstance(member, Bagged) and len(member.boosters) == 2 for member in members)
    runs = ModelRun.objects.filter(model_type="lightgbm")
    assert runs.count() == 4
    assert all(run.params["bag_seeds"] == [2, 5] for run in runs)
    cached = load_fold_predictions("interval-wf1", MODEL, fold)
    assert cached[INTERVAL_COLUMNS].notna().all().all()
    assert (cached["lightgbm_q10_h20"] <= cached["lightgbm_q90_h20"]).all()


def test_target_recipes_keep_their_fingerprints_at_the_spec_design() -> None:
    assert recipe("return", FEATURES, 0) == "4-02656190ccdb"
    assert recipe("interval", FEATURES, 0) == "4-23c61a13464c"
    assert recipe("interval", FEATURES, 0, target=target_models.TARGET_RECIPE) == "4-23c61a13464c"
    assert recipe("interval", FEATURES, 0, target=target_models.TargetVariant(market=False)) != (
        "4-23c61a13464c"
    )


def test_target_scopes_drop_the_market_features_when_the_variant_asks(panel: Path) -> None:
    alone = target_models.TargetVariant(market=False)

    assert lgbm_runs.scope_features("return") == FEATURES
    assert lgbm_runs.scope_features("interval", alone) == KEPT
    assert lgbm_runs.scope_features("rank", alone) == lgbm_runs.scope_features("rank")


@pytest.mark.parametrize("scope", ["return", "interval"])
def test_a_target_fold_without_market_features_trains_on_stock_features_only(
    panel: Path, scope: str
) -> None:
    fold = research()[0]
    alone = target_models.TargetVariant(market=False)

    assert run_folds(scope, [fold], f"{scope}-wf1", panel, target=alone) == [fold.index]

    fitted = ModelRun.objects.filter(run_key=f"{scope}-wf1").exclude(model_type=SCREENING_MODEL)
    assert fitted.exists()
    for run in fitted:
        assert run.feature_list == KEPT
        assert set(run.params["importance_gain"]) == set(KEPT)
        assert run.params["market_bins"] == {}
    notes = fold_notes(f"{scope}-wf1", MODEL, fold.index)
    assert notes["recipe"] == recipe(scope, KEPT, 0, target=alone)
    assert run_folds(scope, [fold], f"{scope}-wf1", panel, target=alone) == []
    assert run_folds(scope, [fold], f"{scope}-wf1", panel) == [fold.index]


def test_rank_folds_refuse_a_target_variant(panel: Path) -> None:
    with pytest.raises(ValueError, match="target variants apply to the return and interval"):
        run_folds(
            "rank",
            research()[:1],
            "rank-wf1",
            panel,
            target=target_models.TargetVariant(market=False),
        )


def store_taiex() -> None:
    """TAIEX closes that swing, so the relative label differs from the absolute one."""
    IndexDaily.objects.bulk_create(
        IndexDaily(index_id="TAIEX", date=day, close=round(100 * (1 + 0.1 * math.sin(k / 6)), 4))
        for k, day in enumerate(weekdays())
    )


def test_an_interval_fold_on_the_relative_label_fits_its_probability_on_it(
    panel: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_taiex()
    seen: list[pd.DataFrame] = []
    original = lgbm_runs.fit_probability

    def spy(inputs: pd.DataFrame, given: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        seen.append(given)
        return original(inputs, given, *args, **kwargs)

    monkeypatch.setattr(lgbm_runs, "fit_probability", spy)
    fold = research()[0]
    relative = target_models.TargetVariant(relative_up=True)

    run_folds("interval", [fold], "interval-wf1", panel, target=relative)

    known = load_labels(fold.train_start, fold.eval_end, 20)
    expected = relative_up(known, TradingDays.from_db())
    assert seen[0]["fwd_up_5pct"].tolist() == expected.tolist()
    assert (expected != known["fwd_up_5pct"]).any()
    probability = ModelRun.objects.get(run_id=run_id("interval-wf1", "classification", 20, fold))
    assert probability.params["label"] == "fwd_up_5pct_vs_taiex"
    quantile = ModelRun.objects.get(run_id=run_id("interval-wf1", "quantile", 20, fold))
    assert quantile.params["label"] == "fwd_ret_open"


def test_relative_label_folds_rerun_when_the_taiex_closes_change(panel: Path) -> None:
    store_taiex()
    fold = research()[0]
    relative = target_models.TargetVariant(relative_up=True)
    run_folds("interval", [fold], "interval-rel", panel, target=relative)
    run_folds("interval", [fold], "interval-abs", panel)

    notes = fold_notes("interval-rel", MODEL, fold.index)
    assert notes["label_hash"] != fold_notes("interval-abs", MODEL, fold.index)["label_hash"]
    assert notes["label_hash"].startswith(label_hash(fold.train_start, fold.eval_end, (20,)))
    IndexDaily.objects.filter(index_id="TAIEX", date=fold.test_start).update(close=1)

    assert run_folds("interval", [fold], "interval-abs", panel) == []
    assert run_folds("interval", [fold], "interval-rel", panel, target=relative) == [fold.index]


def test_return_folds_refuse_the_relative_label(panel: Path) -> None:
    with pytest.raises(ValueError, match="relative_up applies to the interval scope"):
        run_folds(
            "return",
            research()[:1],
            "return-wf1",
            panel,
            target=target_models.TargetVariant(relative_up=True),
        )


def test_target_variants_record_only_the_fields_they_change() -> None:
    alone = target_models.TargetVariant(market=False)
    relative = target_models.TargetVariant(relative_up=True)

    assert lgbm_runs.target_note(target_models.TARGET_RECIPE) == {}
    assert lgbm_runs.target_note(alone) == {"market": False}
    assert lgbm_runs.target_note(relative) == {"relative_up": True}
    assert (
        len({recipe("interval", FEATURES, 0, target=target) for target in (alone, relative)}) == 2
    )
