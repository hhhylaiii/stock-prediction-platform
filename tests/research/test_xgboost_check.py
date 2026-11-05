from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import xgboost as xgb
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.market.models import DailyPrice, StockMaster
from apps.research import xgboost_check
from apps.research.features.store import export_year, read_partitions
from apps.research.management.commands import xgb_check
from apps.research.models import FeatureDaily, LabelDaily, ModelRun
from apps.research.prediction_cache import MissingFoldsError, fold_path
from apps.research.rank_model import rank_inputs
from apps.research.training import load_labels, training_rows
from apps.research.xgboost_check import (
    DAILY_COLUMNS,
    GAP_LIMIT,
    XGB_PARAMS,
    fit_xgboost,
    relative_gap,
    xgboost_daily_ics,
    xgboost_report,
)
from tests.research.run_panels import CONFIG, KEPT, research, weekdays

pytestmark = pytest.mark.django_db


def chosen() -> list[Any]:
    return [fold for fold in research() if fold.research]


def test_relative_gap_scales_by_the_lightgbm_ic() -> None:
    assert relative_gap(0.05, 0.04) == pytest.approx(0.2)
    assert relative_gap(-0.05, -0.065) == pytest.approx(0.3)
    assert GAP_LIMIT == 0.2
    assert pd.isna(relative_gap(0.0, 0.01))


def test_xgboost_report_pools_days_and_flags_gaps_past_the_limit() -> None:
    day, later = date(2016, 10, 3), date(2016, 10, 4)
    rows = [
        [1, 5, day, 0.04, 0.036],
        [1, 5, later, 0.02, 0.018],
        [2, 5, day, 0.09, 0.081],
        [1, 20, day, 0.05, 0.03],
        [2, 20, day, 0.05, 0.04],
        [1, 1, day, 0.0, 0.02],
    ]
    daily = pd.DataFrame(rows, columns=DAILY_COLUMNS)

    text = xgboost_report("rank-wf1", daily)

    assert text.startswith("# XGBoost cross-check: rank-wf1")
    assert "| 5 | 0.0500 | 0.0450 | 10.0% | ok |" in text
    assert "| 20 | 0.0500 | 0.0350 | 30.0% | investigate the implementation |" in text
    assert "| 1 | 0.0000 | 0.0200 | n/a | investigate the implementation |" in text
    assert "| 1 | 20 | 0.0500 | 0.0300 |" in text
    assert "| 1 | 5 | 0.0300 | 0.0270 |" in text
    assert "| 2 | 5 | 0.0900 | 0.0810 |" in text


@pytest.fixture
def train_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Params, keyword options and the returned booster of every xgb.train call."""
    calls: list[dict[str, Any]] = []
    train = xgb.train

    def spy(params: dict[str, Any], dtrain: xgb.DMatrix, **kwargs: Any) -> xgb.Booster:
        booster = train(params, dtrain, **kwargs)
        calls.append({"params": dict(params), "rows": dtrain.num_row(), **kwargs, "out": booster})
        return booster

    monkeypatch.setattr(xgboost_check.xgb, "train", spy)
    return calls


def test_fit_xgboost_stops_on_validation_rank_ic_then_refits_every_row(
    rank_caches: Path, train_calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(xgboost_check, "MAX_ROUNDS", 400)
    fold = chosen()[0]
    inputs = rank_inputs(read_partitions(fold.train_start, fold.test_end, rank_caches), KEPT)
    labels = load_labels(fold.train_start, fold.eval_end, 20)
    inside = inputs[inputs["date"].between(fold.train_start, fold.train_end)]
    rows = training_rows(inside, labels, "fwd_rank_pct")

    booster = fit_xgboost(rows, fold, KEPT, seed=3)

    probe, final = train_calls
    assert probe["custom_metric"] is not None
    assert probe["maximize"] is True
    assert probe["early_stopping_rounds"] == 100
    assert probe["params"]["disable_default_eval_metric"] == 1
    assert probe["rows"] == int((rows["date"] <= fold.fit_end).sum())
    assert probe["evals"][0][0].num_row() == int((rows["date"] >= fold.valid_start).sum())
    assert final["rows"] == len(rows)
    assert "evals" not in final
    rounds = probe["out"].best_iteration + 1
    assert final["num_boost_round"] == rounds
    assert booster.num_boosted_rounds() == rounds < 400
    assert {key: final["params"][key] for key in XGB_PARAMS} == XGB_PARAMS
    assert (probe["params"]["seed"], final["params"]["seed"]) == (3, 3)
    assert XGB_PARAMS == {
        "tree_method": "hist",
        "max_depth": 6,
        "eta": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 200,
        "objective": "reg:squarederror",
    }


def test_xgboost_daily_ics_score_both_models_on_the_same_rows(
    rank_caches: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(xgboost_check, "MAX_ROUNDS", 120)
    folds = chosen()

    daily = xgboost_daily_ics("rank-wf1", folds, rank_caches, seed=0)

    assert daily.columns.tolist() == DAILY_COLUMNS
    pairs = set(zip(daily["fold"], daily["horizon"], strict=True))
    assert pairs == {(fold.index, horizon) for fold in folds for horizon in (5, 20)}
    means = daily.groupby(["fold", "horizon"])[["lightgbm", "xgboost"]].mean().reset_index()
    for index, horizon, lightgbm, other in means.itertuples(index=False):
        run = ModelRun.objects.get(run_id=f"rank-wf1.lightgbm.h{horizon}.f{index:03d}")
        assert lightgbm == pytest.approx(run.ic_mean)
        assert other > 0.3


def test_xgboost_daily_ics_refuse_stale_or_missing_lightgbm_folds(rank_caches: Path) -> None:
    folds = chosen()
    with pytest.raises(LookupError, match="no LightGBM rank run"):
        xgboost_daily_ics("rank-none", folds, rank_caches, seed=0)

    LabelDaily.objects.filter(date=folds[0].train_start, horizon=5).update(fwd_rank_pct=0.5)
    with pytest.raises(LookupError, match=rf"folds \[{folds[0].index}\] .* stale"):
        xgboost_daily_ics("rank-wf1", folds, rank_caches, seed=0)

    FeatureDaily.objects.filter(date=folds[-1].test_start).update(tech_ret_60d=0.5)
    export_year(2016, rank_caches)
    with pytest.raises(LookupError, match="stale"):
        xgboost_daily_ics("rank-wf1", folds[-1:], rank_caches, seed=0)


def test_xgboost_daily_ics_name_a_missing_lightgbm_cache(rank_caches: Path) -> None:
    folds = chosen()
    fold_path("rank-wf1", "lightgbm", folds[-1].index).unlink()

    with pytest.raises(MissingFoldsError, match=rf"rank-wf1/lightgbm .* \[{folds[-1].index}\]"):
        xgboost_daily_ics("rank-wf1", folds, rank_caches, seed=0)


def test_xgboost_daily_ics_refuse_labels_rebuilt_mid_run(
    rank_caches: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(xgboost_check, "MAX_ROUNDS", 120)
    folds = chosen()
    fit = xgboost_check.fit_xgboost

    def rebuilt(rows: pd.DataFrame, fold: Any, features: list[str], seed: int) -> xgb.Booster:
        LabelDaily.objects.filter(date=fold.train_start, horizon=20).update(fwd_rank_pct=0.5)
        return fit(rows, fold, features, seed)

    monkeypatch.setattr(xgboost_check, "fit_xgboost", rebuilt)

    with pytest.raises(LookupError, match=rf"folds \[{folds[0].index}\] .* stale"):
        xgboost_daily_ics("rank-wf1", folds, rank_caches, seed=0)


@pytest.fixture
def command(rank_caches: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(xgb_check, "DEFAULT_CONFIG", CONFIG)
    monkeypatch.setattr(xgboost_check, "MAX_ROUNDS", 120)
    StockMaster.objects.create(stock_id="1101", name="1101", market="TWSE", industry="水泥工業")
    DailyPrice.objects.create(
        stock_id="1101", date=weekdays()[-1], close=10, volume=1000, turnover=10_000
    )
    return rank_caches


def test_xgb_check_writes_the_report(command: Path, tmp_path: Path) -> None:
    report = tmp_path / "xgb.md"

    call_command("xgb_check", run_key="rank-wf1", folds="2", output=str(report), stdout=StringIO())

    expected = xgboost_report(
        "rank-wf1", xgboost_daily_ics("rank-wf1", chosen()[-1:], command, seed=0)
    )
    assert report.read_text() == expected


def test_xgb_check_refuses_folds_outside_the_research_period(command: Path) -> None:
    with pytest.raises(CommandError, match=r"folds \[0, 7\] are not research folds"):
        call_command("xgb_check", run_key="rank-wf1", folds="0-2,7", stdout=StringIO())
    with pytest.raises(CommandError, match="start with rank-"):
        call_command("xgb_check", run_key="return-wf1", stdout=StringIO())
