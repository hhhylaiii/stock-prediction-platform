from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.market.models import DailyPrice, StockMaster
from apps.research import baseline_runs, comparisons, lgbm_runs
from apps.research.comparisons import (
    GAP_LIMIT,
    XGB_PARAMS,
    adopt_ensemble,
    ensemble_ics,
    fit_xgboost,
    rank_average,
    relative_gap,
    xgboost_fold_ics,
)
from apps.research.features.store import read_partitions
from apps.research.management.commands import ensemble_check, xgb_check
from apps.research.metrics.ic import daily_rank_ic
from apps.research.models import LabelDaily, ModelRun
from apps.research.prediction_cache import load_fold_predictions
from apps.research.rank_model import rank_inputs
from apps.research.training import load_labels, training_rows
from tests.research.run_panels import CONFIG, KEPT, research, weekdays

pytestmark = pytest.mark.django_db


def test_rank_average_means_each_score_rank_within_the_day_universe() -> None:
    frame = pd.DataFrame(
        {
            "date": [date(2016, 1, 4)] * 4 + [date(2016, 1, 5)] * 2,
            "stock_id": ["a", "b", "c", "d", "a", "b"],
            "in_universe": [True, True, True, False, True, True],
            "ridge": [0.1, 0.3, 0.2, 9.0, 1.0, 2.0],
            "lightgbm": [3.0, 1.0, 2.0, 9.0, 5.0, 4.0],
        }
    )

    averaged = rank_average(frame, ["ridge", "lightgbm"])

    expected = [(1 / 3 + 1) / 2, (1 + 1 / 3) / 2, (2 / 3 + 2 / 3) / 2, np.nan, 0.75, 0.75]
    np.testing.assert_allclose(averaged.to_numpy(), expected)


def test_relative_gap_scales_by_the_lightgbm_ic_and_flags_past_the_limit() -> None:
    assert relative_gap(0.05, 0.04) == pytest.approx(0.2)
    assert relative_gap(-0.05, -0.065) == pytest.approx(0.3)
    assert GAP_LIMIT == 0.2
    assert np.isnan(relative_gap(0.0, 0.01))


def test_the_ensemble_is_adopted_only_when_it_beats_lightgbm_alone() -> None:
    assert adopt_ensemble({"lightgbm": 0.05, "ensemble": 0.051})
    assert not adopt_ensemble({"lightgbm": 0.05, "ensemble": 0.05})
    assert not adopt_ensemble({"lightgbm": 0.05, "ensemble": float("nan")})


@pytest.fixture
def cached(run_panel: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Ridge and LightGBM rank caches for the panel's research folds."""
    monkeypatch.setattr(lgbm_runs, "MAX_ROUNDS", 120)
    monkeypatch.setattr(comparisons, "MAX_ROUNDS", 120)
    folds = [fold for fold in research() if fold.research]
    baseline_runs.run_folds("rank", folds, "rank-wf1", run_panel)
    lgbm_runs.run_folds("rank", folds, "rank-wf1", run_panel)
    return run_panel


def research_folds() -> list[Any]:
    return [fold for fold in research() if fold.research]


def test_ensemble_ics_average_daily_oos_ic_over_research_folds(cached: Path) -> None:
    folds = research_folds()

    ics = ensemble_ics("rank-wf1", folds, 20)

    frames = []
    for fold in folds:
        ridge = load_fold_predictions("rank-wf1", "baseline", fold)
        lgbm = load_fold_predictions("rank-wf1", "lightgbm", fold)
        merged = ridge.merge(lgbm, on=["date", "stock_id", "in_universe"])
        merged = merged[merged["date"] <= fold.eval_end]
        known = pd.DataFrame(
            list(
                LabelDaily.objects.filter(
                    date__range=(fold.test_start, fold.eval_end), horizon=20
                ).values()
            )
        )
        frames.append(training_rows(merged, known, "fwd_rank_pct"))
    rows = pd.concat(frames, ignore_index=True)
    rows = rows.assign(ensemble=rank_average(rows, ["ridge_h20", "lightgbm_h20"]))
    assert set(ics) == {"ridge", "lightgbm", "ensemble"}
    for name, column in [("ridge", "ridge_h20"), ("lightgbm", "lightgbm_h20")]:
        assert ics[name] == pytest.approx(daily_rank_ic(rows, column).mean())
    assert ics["lightgbm"] > 0.3
    assert 0 < ics["ensemble"] <= 1


def test_fit_xgboost_stops_on_validation_rank_ic_then_refits_every_row(
    cached: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    train = xgb.train

    def spy(params: dict[str, Any], dtrain: xgb.DMatrix, **kwargs: Any) -> xgb.Booster:
        calls.append({**params, "_rows": dtrain.num_row(), "_probe": "evals" in kwargs})
        return train(params, dtrain, **kwargs)

    monkeypatch.setattr(comparisons.xgb, "train", spy)
    fold = research_folds()[0]
    inputs = rank_inputs(read_partitions(fold.train_start, fold.test_end, cached), KEPT)
    labels = load_labels(fold.train_start, fold.eval_end, 20)
    inside = inputs[inputs["date"].between(fold.train_start, fold.train_end)]
    rows = training_rows(inside, labels, "fwd_rank_pct")

    booster = fit_xgboost(rows, fold, KEPT, seed=3)

    probe, final = calls
    assert probe["_probe"]
    assert not final["_probe"]
    assert final["_rows"] == len(rows) > probe["_rows"]
    assert {key: final[key] for key in XGB_PARAMS} == XGB_PARAMS
    assert (probe["seed"], final["seed"]) == (3, 3)
    assert XGB_PARAMS == {
        "tree_method": "hist",
        "max_depth": 6,
        "eta": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "objective": "reg:squarederror",
    }
    assert 1 <= booster.num_boosted_rounds() <= 120


def test_xgboost_fold_ics_score_the_lightgbm_window_and_features(cached: Path) -> None:
    folds = research_folds()

    table = xgboost_fold_ics("rank-wf1", folds, cached, seed=0)

    assert table.columns.tolist() == ["fold", "horizon", "lightgbm_ic", "xgboost_ic"]
    assert sorted(zip(table["fold"], table["horizon"], strict=True)) == sorted(
        (fold.index, horizon) for fold in folds for horizon in (5, 20)
    )
    for fold, horizon, lightgbm, other in table.to_numpy().tolist():
        run = ModelRun.objects.get(run_id=f"rank-wf1.lightgbm.h{int(horizon)}.f{int(fold):03d}")
        assert lightgbm == pytest.approx(run.ic_mean)
        assert other > 0.3


def test_xgboost_fold_ics_need_the_lightgbm_runs(run_panel: Path) -> None:
    with pytest.raises(LookupError, match="no LightGBM rank run"):
        xgboost_fold_ics("rank-wf1", research_folds(), run_panel, seed=0)


@pytest.fixture
def command(cached: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(ensemble_check, "DEFAULT_CONFIG", CONFIG)
    monkeypatch.setattr(xgb_check, "DEFAULT_CONFIG", CONFIG)
    StockMaster.objects.create(stock_id="1101", name="1101", market="TWSE", industry="水泥工業")
    DailyPrice.objects.create(
        stock_id="1101", date=weekdays()[-1], close=10, volume=1000, turnover=10_000
    )
    return cached


def test_ensemble_check_reports_each_horizon_and_its_verdict(command: Path, tmp_path: Path) -> None:
    report = tmp_path / "ensemble.md"

    call_command("ensemble_check", run_key="rank-wf1", output=str(report), stdout=StringIO())

    text = report.read_text()
    assert text.startswith("# Rank-average ensemble: rank-wf1")
    for horizon in (5, 20):
        ics = ensemble_ics("rank-wf1", research_folds(), horizon)
        verdict = "adopt" if adopt_ensemble(ics) else "keep LightGBM alone"
        assert f"| {horizon} | {ics['ridge']:.4f} | {ics['lightgbm']:.4f} |" in text
        assert verdict in text


def test_xgb_check_reports_gaps_and_flags(command: Path, tmp_path: Path) -> None:
    report = tmp_path / "xgb.md"

    call_command("xgb_check", run_key="rank-wf1", output=str(report), stdout=StringIO())

    text = report.read_text()
    assert text.startswith("# XGBoost cross-check: rank-wf1")
    assert "| horizon | LightGBM IC | XGBoost IC | gap | check |" in text
    assert "## Per fold" in text
    assert text.count("\n| 5 |") >= 1


def test_checks_take_rank_run_keys_only(command: Path) -> None:
    for name in ("ensemble_check", "xgb_check"):
        with pytest.raises(CommandError, match="start with rank-"):
            call_command(name, run_key="interval-wf1", stdout=StringIO())


def test_ensemble_check_names_missing_caches(command: Path) -> None:
    with pytest.raises(CommandError, match="rank-none"):
        call_command("ensemble_check", run_key="rank-none", stdout=StringIO())
