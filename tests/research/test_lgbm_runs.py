from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.market.models import DailyPrice, StockMaster
from apps.research import baseline_runs, lgbm_runs
from apps.research.baseline_runs import BASELINE
from apps.research.features.store import export_year
from apps.research.gbdt import BASE_PARAMS, seeded
from apps.research.holdout import HoldoutError
from apps.research.lgbm_runs import MODEL, recipe, run_folds
from apps.research.management.commands import train
from apps.research.metrics.ic import daily_rank_ic
from apps.research.models import FeatureDaily, LabelDaily, ModelRun, ModelStatus
from apps.research.prediction_cache import (
    fold_notes,
    fold_path,
    load_fold_predictions,
    missing_folds,
)
from apps.research.rank_model import rank_inputs
from apps.research.runs import (
    SCREENING_MODEL,
    RunError,
    feature_hash,
    label_hash,
    lib_versions,
    load_model,
)
from tests.research.run_panels import CONFIG, KEPT, STOCKS, folds, research, weekdays

pytestmark = pytest.mark.django_db


@pytest.fixture
def panel(run_panel: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(lgbm_runs, "MAX_ROUNDS", 150)
    return run_panel


def test_rank_folds_cache_lightgbm_scores_and_record_both_horizons(panel: Path) -> None:
    fold = research()[-1]

    assert run_folds([fold], "rank-wf1", panel, seed=4) == [fold.index]

    cached = load_fold_predictions("rank-wf1", MODEL, fold)
    test_days = [day for day in weekdays() if fold.test_start <= day <= fold.test_end]
    assert cached.columns.tolist() == [
        "date",
        "stock_id",
        "in_universe",
        "lightgbm_h5",
        "lightgbm_h20",
    ]
    assert len(cached) == len(test_days) * len(STOCKS)
    assert cached[["lightgbm_h5", "lightgbm_h20"]].notna().all().all()
    for horizon in (5, 20):
        run = ModelRun.objects.get(run_id=f"rank-wf1.lightgbm.h{horizon}.f{fold.index:03d}")
        assert (run.model_type, run.objective, run.status) == ("lightgbm", "rank", "candidate")
        assert (run.train_start, run.train_end) == (fold.train_start, fold.train_end)
        assert run.feature_list == KEPT
        assert run.seed == 4
        assert run.lib_versions == lib_versions()
        assert run.feature_hash == feature_hash(fold.train_start, fold.test_end, panel)
        assert run.params["horizon"] == horizon
        assert run.params["label"] == "fwd_rank_pct"
        assert run.params["lightgbm"] == {**BASE_PARAMS, **seeded(4)}
        assert 1 <= run.params["rounds"] <= 150
        assert run.params["max_rounds"] == 150
        assert run.params["valid_ic"] > 0.3
        assert set(run.params["importance_gain"]) == set(KEPT)
        assert run.params["importance_gain"]["tech_ret_1d"] == max(
            run.params["importance_gain"].values()
        )
        labels = pd.DataFrame(
            list(
                LabelDaily.objects.filter(
                    date__range=(fold.test_start, fold.eval_end), horizon=horizon
                ).values("stock_id", "date", "fwd_rank_pct")
            )
        )
        scored = cached[cached["in_universe"]].merge(labels, on=["stock_id", "date"])
        daily = daily_rank_ic(scored, f"lightgbm_h{horizon}", "fwd_rank_pct")
        assert len(daily) == len([day for day in test_days if day <= fold.eval_end])
        assert run.ic_mean == pytest.approx(daily.mean())
        assert run.icir == pytest.approx(daily.mean() / daily.std())
        assert daily.mean() > 0.3


def test_cached_scores_come_from_the_saved_booster(panel: Path) -> None:
    fold = research()[0]
    run_folds([fold], "rank-wf1", panel)

    booster = load_model("rank-wf1", f"rank-wf1.lightgbm.h20.f{fold.index:03d}")
    cached = load_fold_predictions("rank-wf1", MODEL, fold)
    raw = pd.read_parquet(panel / "year=2016" / "part-0.parquet")
    inputs = rank_inputs(raw[raw["date"].between(fold.train_start, fold.test_end)], KEPT)
    test = inputs[inputs["date"].between(fold.test_start, fold.test_end)]
    expected = booster.predict(test[KEPT])
    np.testing.assert_allclose(cached["lightgbm_h20"].to_numpy(), expected)


def test_lightgbm_and_baselines_share_a_run_key_without_clobbering(panel: Path) -> None:
    fold = research()[0]

    baseline_runs.run_folds("rank", [fold], "rank-wf1", panel)
    run_folds([fold], "rank-wf1", panel)

    assert load_fold_predictions("rank-wf1", BASELINE, fold).columns[-1] == "ridge_h20"
    assert load_fold_predictions("rank-wf1", MODEL, fold).columns[-1] == "lightgbm_h20"
    assert baseline_runs.run_folds("rank", [fold], "rank-wf1", panel) == []
    kinds = set(ModelRun.objects.filter(run_key="rank-wf1").values_list("model_type", flat=True))
    assert kinds == {"ridge", "lightgbm"}


def test_cached_folds_note_their_inputs_and_rerun_only_when_one_changes(panel: Path) -> None:
    fold = research()[0]
    run_folds([fold], "rank-wf1", panel, seed=1)

    assert fold_notes("rank-wf1", MODEL, fold.index) == {
        "feature_hash": feature_hash(fold.train_start, fold.test_end, panel),
        "label_hash": label_hash(fold.train_start, fold.eval_end, (5, 20)),
        "recipe": recipe(KEPT, 1),
    }
    assert run_folds([fold], "rank-wf1", panel, seed=1) == []
    assert run_folds([fold], "rank-wf1", panel, seed=1, rerun=True) == [fold.index]
    assert run_folds([fold], "rank-wf1", panel, seed=2) == [fold.index]

    ModelRun.objects.filter(model_type=SCREENING_MODEL).update(feature_list=KEPT[:2])
    assert run_folds([fold], "rank-wf1", panel, seed=2) == [fold.index]

    LabelDaily.objects.filter(date=fold.train_start, horizon=5).update(fwd_rank_pct=0.5)
    assert run_folds([fold], "rank-wf1", panel, seed=2) == [fold.index]
    assert run_folds([fold], "rank-wf1", panel, seed=2) == []


def test_rebuilt_partitions_and_moved_boundaries_rerun_their_folds(panel: Path) -> None:
    fold = research()[0]
    run_folds([fold], "rank-wf1", panel)

    FeatureDaily.objects.filter(date=fold.test_start).update(tech_ret_60d=0.5)
    export_year(2016, panel)
    assert run_folds([fold], "rank-wf1", panel) == [fold.index]

    moved = replace(fold, tuning=not fold.tuning)
    assert run_folds([moved], "rank-wf1", panel) == [fold.index]
    assert run_folds([moved], "rank-wf1", panel) == []


def test_a_fold_with_a_promoted_model_keeps_its_cache(panel: Path) -> None:
    fold = research()[0]
    run_folds([fold], "rank-wf1", panel)
    promoted = f"rank-wf1.lightgbm.h5.f{fold.index:03d}"
    ModelRun.objects.filter(run_id=promoted).update(status=ModelStatus.ACTIVE)

    with pytest.raises(RunError, match=promoted):
        run_folds([fold], "rank-wf1", panel, rerun=True)

    assert fold_path("rank-wf1", MODEL, fold.index).exists()


def test_a_rerun_that_fails_leaves_its_fold_to_be_run_again(
    panel: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fold = research()[0]
    run_folds([fold], "rank-wf1", panel)

    def broken(*args: Any, **kwargs: Any) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(lgbm_runs, "save_fold_predictions", broken)
    with pytest.raises(OSError, match="disk full"):
        run_folds([fold], "rank-wf1", panel, rerun=True)

    assert missing_folds("rank-wf1", MODEL, [fold]) == [fold.index]


def test_labels_rewritten_during_the_fit_stop_the_fold(
    panel: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fold = research()[0]
    fitted = lgbm_runs.fit_rank

    def relabelling(*args: Any, **kwargs: Any) -> Any:
        boosted = fitted(*args, **kwargs)
        LabelDaily.objects.filter(date=fold.train_start).update(fwd_rank_pct=0.5)
        return boosted

    monkeypatch.setattr(lgbm_runs, "fit_rank", relabelling)

    with pytest.raises(RuntimeError, match="label_daily changed"):
        run_folds([fold], "rank-wf1", panel)

    assert not ModelRun.objects.filter(model_type="lightgbm").exists()
    assert missing_folds("rank-wf1", MODEL, [fold]) == [fold.index]


def test_recipe_changes_with_the_features_the_seed_and_the_round_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = recipe(KEPT, 0)

    assert recipe(KEPT, 0) == first
    assert recipe(KEPT[:2], 0) != first
    assert recipe(KEPT, 1) != first
    monkeypatch.setattr(lgbm_runs, "MAX_ROUNDS", 10)
    assert recipe(KEPT, 0) != first


def test_holdout_folds_wait_for_the_frozen_design(panel: Path) -> None:
    held = [fold for fold in folds() if fold.holdout]

    with pytest.raises(HoldoutError, match="holdout"):
        run_folds(held, "rank-wf1", panel)

    assert not ModelRun.objects.filter(model_type="lightgbm").exists()


def test_rank_folds_need_the_screening_and_a_rank_run_key(panel: Path) -> None:
    fold = research()[0]

    with pytest.raises(ValueError, match="start with rank-"):
        run_folds([fold], "interval-wf1", panel)

    ModelRun.objects.filter(model_type=SCREENING_MODEL).delete()
    with pytest.raises(RunError, match="no feature screening"):
        run_folds([fold], "rank-wf1", panel)
    assert missing_folds("rank-wf1", MODEL, [fold]) == [fold.index]


@pytest.fixture
def command(panel: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(train, "DEFAULT_CONFIG", CONFIG)
    StockMaster.objects.create(stock_id="1101", name="1101", market="TWSE", industry="水泥工業")
    DailyPrice.objects.create(
        stock_id="1101", date=weekdays()[-1], close=10, volume=1000, turnover=10_000
    )
    return panel


def test_command_trains_lightgbm_rank_folds(command: Path) -> None:
    out = StringIO()

    call_command(
        "train", model="lightgbm", scope="rank", run_key="rank-wf1", folds="0", seed=3, stdout=out
    )

    assert "rank-wf1: ran folds [0]" in out.getvalue()
    assert ModelRun.objects.get(run_id="rank-wf1.lightgbm.h20.f000").seed == 3
    assert missing_folds("rank-wf1", BASELINE, research()[:1]) == [0]


def test_command_keeps_lightgbm_to_the_rank_scope(command: Path) -> None:
    with pytest.raises(CommandError, match="rank scope"):
        call_command("train", model="lightgbm", scope="interval", run_key="interval-wf1")
