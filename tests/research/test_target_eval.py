import math
import shutil
from collections.abc import Iterator
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from django.conf import settings as django_settings
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import transaction
from django.test import override_settings

from apps.market.models import DailyPrice, StockMaster, TradingCalendar
from apps.research import baseline_runs, lgbm_runs, target_eval
from apps.research.evaluation import consistency_report, coverage_report
from apps.research.features.store import export_year
from apps.research.management.commands import evaluate_targets
from apps.research.metrics.forecast import (
    daily_absolute_error,
    daily_auc,
    daily_brier,
    daily_coverage,
    daily_direction_hits,
    daily_pinball,
    daily_qlike,
    daily_squared_error,
    expected_calibration_error,
    mean_by_fold,
    reliability_table,
)
from apps.research.metrics.stats import newey_west_t, paired_nw_t
from apps.research.models import BacktestMetric, FeatureDaily, LabelDaily, ModelRun
from apps.research.prediction_cache import MissingFoldsError, fold_path
from apps.research.walkforward import Fold
from tests.research.run_panels import CONFIG, build_panel, folds, research, weekdays

pytestmark = pytest.mark.django_db

RETURN_KEY, INTERVAL_KEY = "return-wf1", "interval-wf1"
BUILT_TABLES = [DailyPrice, StockMaster, ModelRun, LabelDaily, FeatureDaily, TradingCalendar]


def build_runs(root: Path) -> None:
    """Return and interval caches of LightGBM and the baselines on the one-year panel."""
    with override_settings():
        with transaction.atomic():
            panel = build_panel(django_settings, root)
            DailyPrice.objects.create(
                stock_id="1101", date=weekdays()[-1], close=10, volume=1000, turnover=10_000
            )
        for scope, key in (("return", RETURN_KEY), ("interval", INTERVAL_KEY)):
            baseline_runs.run_folds(scope, research(), key, panel)
            lgbm_runs.run_folds(scope, research(), key, panel)


@pytest.fixture(scope="module")
def built(
    tmp_path_factory: pytest.TempPathFactory, django_db_setup: None, django_db_blocker: Any
) -> Iterator[Path]:
    """The runs built and committed once for the module, deleted after its last test."""
    root = tmp_path_factory.mktemp("target_eval")
    with django_db_blocker.unblock(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(lgbm_runs, "MAX_ROUNDS", 80)
        try:
            build_runs(root)
            yield root
        finally:
            for table in BUILT_TABLES:
                table._default_manager.all().delete()


@pytest.fixture
def runs(built: Path, settings: Any, tmp_path: Path) -> Path:
    """A copy of the built files for each test; database changes roll back with the test."""
    root = tmp_path / "built"
    shutil.copytree(built, root)
    settings.RESEARCH_CACHE_DIR = root / "predictions"
    settings.MODEL_DIR = root / "models"
    settings.FEATURE_PARQUET_DIR = root / "features"
    return root / "features"


def scored(frame: pd.DataFrame) -> pd.DataFrame:
    """Universe rows tradable at the next open, the rows every target metric counts."""
    return frame[frame["in_universe"] & frame["untradable_t1"].eq(False)]


def fold_means(daily: pd.Series, chosen: list[Fold]) -> pd.Series:
    return mean_by_fold(daily, chosen)["mean"]


def test_the_frame_joins_both_caches_and_the_labels_over_research_test_rows(runs: Path) -> None:
    chosen = research()

    frame = target_eval.target_frame(INTERVAL_KEY, chosen, runs)

    kept = [fold for fold in chosen if fold.research]
    assert kept
    assert len(kept) < len(chosen)
    assert set(frame["fold"]) == {fold.index for fold in kept}
    ends = frame["fold"].map({fold.index: fold.eval_end for fold in kept})
    assert (frame["date"] <= ends).all()
    keys = ["stock_id", "date"]
    labels = pd.DataFrame(
        list(LabelDaily.objects.filter(horizon=20).values(*keys, "fwd_ret_open", "fwd_vol_20d"))
    )
    joined = frame[keys].merge(labels, on=keys, how="left")
    np.testing.assert_allclose(frame["fwd_ret_open"], joined["fwd_ret_open"])
    vols = pd.DataFrame(list(FeatureDaily.objects.values(*keys, "tech_vol_20d")))
    expected = frame[keys].merge(vols, on=keys, how="left")["tech_vol_20d"]
    np.testing.assert_allclose(frame["tech_vol_20d"], expected.astype(float))
    for column in ("lightgbm_q10_h20", "volband_q90_h20", "logistic_h20", "lightgbm_prob_h20"):
        assert frame[column].notna().all(), column


def test_return_scores_mae_rmse_and_direction_against_both_baselines(runs: Path) -> None:
    chosen = research()
    kept = [fold for fold in chosen if fold.research]

    evaluated = target_eval.evaluate_return(
        target_eval.target_frame(RETURN_KEY, chosen, runs), chosen
    )

    rows = scored(target_eval.target_frame(RETURN_KEY, chosen, runs))
    model = daily_absolute_error(rows, "fwd_ret_open", "lightgbm_ret_h5")
    for name, column in (("zero", "zero_h5"), ("trailing", "trailing_h5")):
        shared = rows.dropna(subset=["lightgbm_ret_h5", column])
        first = daily_absolute_error(shared, "fwd_ret_open", "lightgbm_ret_h5")
        second = daily_absolute_error(shared, "fwd_ret_open", column)
        compared = evaluated.mae[name]
        assert compared.model == pytest.approx(first.mean())
        assert compared.baseline == pytest.approx(second.mean())
        assert compared.nw_t == pytest.approx(
            paired_nw_t(fold_means(first, kept), fold_means(second, kept), target_eval.FOLD_LAG)
        )
        assert list(compared.by_fold.index) == [fold.index for fold in kept]
        squared = daily_squared_error(shared, "fwd_ret_open", column)
        assert evaluated.rmse[name] == pytest.approx(math.sqrt(squared.mean()))
    assert evaluated.rmse["lightgbm"] == pytest.approx(
        math.sqrt(daily_squared_error(rows, "fwd_ret_open", "lightgbm_ret_h5").mean())
    )
    assert evaluated.mae_model == pytest.approx(model.mean())
    hits = daily_direction_hits(rows, "fwd_ret_open", "lightgbm_ret_h5")
    assert evaluated.direction == pytest.approx(hits.mean())
    assert evaluated.direction_t == pytest.approx(newey_west_t(hits - 0.5, 5))


def test_interval_scores_pinball_coverage_volatility_and_probability(runs: Path) -> None:
    chosen = research()
    kept = [fold for fold in chosen if fold.research]
    frame = target_eval.target_frame(INTERVAL_KEY, chosen, runs)

    evaluated = target_eval.evaluate_interval(frame, chosen)

    rows = scored(frame)
    model = {0.1: "lightgbm_q10_h20", 0.5: "lightgbm_q50_h20", 0.9: "lightgbm_q90_h20"}
    band = {0.1: "volband_q10_h20", 0.5: "volband_q50_h20", 0.9: "volband_q90_h20"}
    first = daily_pinball(rows, "fwd_ret_open", model)
    second = daily_pinball(rows, "fwd_ret_open", band)
    assert evaluated.pinball.model == pytest.approx(first.mean())
    assert evaluated.pinball.baseline == pytest.approx(second.mean())
    assert evaluated.pinball.nw_t == pytest.approx(
        paired_nw_t(fold_means(first, kept), fold_means(second, kept), target_eval.FOLD_LAG)
    )
    history = {0.1: "histq_q10_h20", 0.5: "histq_q50_h20", 0.9: "histq_q90_h20"}
    assert evaluated.history_pinball == pytest.approx(
        daily_pinball(rows, "fwd_ret_open", history).mean()
    )
    covered = daily_coverage(rows, "fwd_ret_open", "lightgbm_q10_h20", "lightgbm_q90_h20")
    assert evaluated.coverage == pytest.approx(covered.mean())
    pd.testing.assert_series_equal(
        evaluated.coverage_by_fold, fold_means(covered, kept), check_names=False
    )
    pd.testing.assert_frame_equal(
        evaluated.coverage_table,
        coverage_report(frame, "fwd_ret_open", "lightgbm_q10_h20", "lightgbm_q90_h20", chosen),
    )
    pd.testing.assert_frame_equal(
        evaluated.consistency,
        consistency_report(
            frame,
            "lightgbm_q10_h20",
            "lightgbm_q50_h20",
            "lightgbm_q90_h20",
            "lightgbm_vol_h20",
            "lightgbm_prob_h20",
            chosen,
        ),
    )
    vol = daily_qlike(rows, "fwd_vol_20d", "lightgbm_vol_h20")
    assert evaluated.qlike.model == pytest.approx(vol.mean())
    assert evaluated.qlike.baseline == pytest.approx(
        daily_qlike(rows, "fwd_vol_20d", "tech_vol_20d").mean()
    )
    assert evaluated.vol_mae.model == pytest.approx(
        daily_absolute_error(rows, "fwd_vol_20d", "lightgbm_vol_h20").mean()
    )
    up = rows.assign(fwd_up_5pct=rows["fwd_up_5pct"].astype(int))
    assert evaluated.auc == pytest.approx(daily_auc(up, "fwd_up_5pct", "lightgbm_prob_h20").mean())
    assert evaluated.logistic_auc == pytest.approx(
        daily_auc(up, "fwd_up_5pct", "logistic_h20").mean()
    )
    assert evaluated.brier.model == pytest.approx(
        daily_brier(up, "fwd_up_5pct", "lightgbm_prob_h20").mean()
    )
    for column, value in (
        ("lightgbm_prob_h20", evaluated.ece),
        ("lightgbm_prob_raw_h20", evaluated.ece_raw),
    ):
        table = reliability_table(up.dropna(subset=[column]), "fwd_up_5pct", column)
        assert value == pytest.approx(expected_calibration_error(table))


def comparison(model: float, baseline: float, nw_t: float) -> target_eval.Comparison:
    return target_eval.Comparison(model, baseline, nw_t, pd.DataFrame())


def test_lower_is_better_gates_need_a_newey_west_t_below_minus_two() -> None:
    assert target_eval.beats(comparison(0.01, 0.02, -2.5))
    assert not target_eval.beats(comparison(0.01, 0.02, -1.9))
    assert not target_eval.beats(comparison(0.03, 0.02, -2.5))
    assert not target_eval.beats(comparison(0.01, 0.02, math.nan))


@pytest.mark.parametrize(
    ("total", "by_fold", "passed", "outside"),
    [
        (0.80, [0.72, 0.88], True, []),
        (0.75, [0.70, 0.90], True, []),
        (0.749, [0.80, 0.80], False, []),
        (0.86, [0.80, 0.80], False, []),
        (0.80, [0.69, 0.80], False, [3]),
        (0.80, [0.80, 0.91], False, [4]),
        (0.80, [math.nan, 0.80], False, [3]),
    ],
)
def test_coverage_needs_the_total_and_every_fold_inside_their_bands(
    total: float, by_fold: list[float], passed: bool, outside: list[int]
) -> None:
    folds_coverage = pd.Series(by_fold, index=[3, 4])

    assert target_eval.coverage_outside(folds_coverage) == outside
    assert target_eval.coverage_passes(total, folds_coverage) is passed


@pytest.fixture
def command(runs: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(evaluate_targets, "DEFAULT_CONFIG", CONFIG)
    return runs


@pytest.mark.parametrize("run_key", [RETURN_KEY, INTERVAL_KEY])
def test_evaluate_targets_writes_the_report(command: Path, tmp_path: Path, run_key: str) -> None:
    output = tmp_path / "reports" / f"{run_key}.md"
    out = StringIO()

    call_command("evaluate_targets", run_key=run_key, output=str(output), stdout=out)

    evaluation = target_eval.evaluate_run(run_key, research(), command)
    assert output.read_text() == target_eval.report(run_key, evaluation)
    assert f"wrote {output}" in out.getvalue()
    assert not BacktestMetric.objects.exists()


def test_reports_name_each_gate_and_its_numbers(command: Path) -> None:
    returned = target_eval.report(
        RETURN_KEY, target_eval.evaluate_run(RETURN_KEY, research(), command)
    )
    interval = target_eval.report(
        INTERVAL_KEY, target_eval.evaluate_run(INTERVAL_KEY, research(), command)
    )

    assert "## 絕對報酬 5 日" in returned
    assert "MAE 低於零預測" in returned
    assert "MAE 低於過去 60 日平均" in returned
    assert "方向準確率" in returned
    for heading in ("## 區間 20 日", "## 波動率", "## 上漲機率", "## 輸出一致性"):
        assert heading in interval, heading
    assert "總覆蓋率" in interval
    assert "每摺覆蓋率" in interval
    assert "Brier 不高於 logistic" in interval
    assert "ECE" in interval


def test_evaluate_targets_refuses_other_scopes_missing_and_stale_caches(command: Path) -> None:
    with pytest.raises(CommandError, match="start with return- or interval-"):
        call_command("evaluate_targets", run_key="rank-wf1", stdout=StringIO())
    kept = [fold for fold in research() if fold.research]
    fold_path(INTERVAL_KEY, "baseline", kept[-1].index).unlink()
    with pytest.raises(CommandError, match=rf"interval-wf1/baseline .* \[{kept[-1].index}\]"):
        call_command("evaluate_targets", run_key=INTERVAL_KEY, stdout=StringIO())
    FeatureDaily.objects.filter(date=kept[0].test_start).update(tech_vol_20d=1.0)
    export_year(2016, command)
    with pytest.raises(CommandError, match=r"return-wf1 .* stale"):
        call_command("evaluate_targets", run_key=RETURN_KEY, stdout=StringIO())


def test_evaluate_run_names_missing_lightgbm_folds(runs: Path) -> None:
    kept = [fold for fold in research() if fold.research]
    fold_path(RETURN_KEY, "lightgbm", kept[0].index).unlink()

    with pytest.raises(MissingFoldsError, match=rf"return-wf1/lightgbm .* \[{kept[0].index}\]"):
        target_eval.evaluate_run(RETURN_KEY, research(), runs)


def test_evaluate_targets_never_reads_holdout_folds(command: Path, monkeypatch: Any) -> None:
    seen: list[list[Fold]] = []
    original = target_eval.evaluate_run

    def recorded(run_key: str, chosen: list[Fold], *args: Any) -> Any:
        seen.append(chosen)
        return original(run_key, chosen, *args)

    monkeypatch.setattr(evaluate_targets, "evaluate_run", recorded)

    call_command("evaluate_targets", run_key=RETURN_KEY, stdout=StringIO())

    assert [fold.index for fold in seen[0]] == [fold.index for fold in research()]
    assert not any(fold.holdout for fold in seen[0])
    assert any(fold.holdout for fold in folds())
