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
from django.db.models import F
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
from apps.research.prediction_cache import (
    MissingFoldsError,
    fold_notes,
    fold_path,
    load_fold_predictions,
    save_fold_predictions,
)
from apps.research.walkforward import Fold
from tests.research.run_panels import build_panel, folds, research, use_designs, weekdays

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


def shared(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    return frame.dropna(subset=list(columns))


def fold_means(daily: pd.Series, chosen: list[Fold]) -> pd.Series:
    return mean_by_fold(daily, chosen)["mean"]


def studied(chosen: list[Fold]) -> list[Fold]:
    return [fold for fold in chosen if fold.research]


def research_mean(daily: pd.Series, chosen: list[Fold]) -> float:
    """The mean of a daily metric over the research folds' days, built without target_eval."""
    days = pd.Index(daily.index)
    keep = np.zeros(len(days), dtype=bool)
    for fold in studied(chosen):
        keep |= (days >= fold.test_start) & (days <= fold.eval_end)
    return float(daily[keep].mean())


def check_paired(
    compared: target_eval.Comparison, first: pd.Series, second: pd.Series, chosen: list[Fold]
) -> None:
    """Research totals and NW t over research folds; every non-holdout fold in the table."""
    kept = studied(chosen)
    assert compared.model == pytest.approx(research_mean(first, chosen))
    assert compared.baseline == pytest.approx(research_mean(second, chosen))
    assert compared.nw_t == pytest.approx(
        paired_nw_t(fold_means(first, kept), fold_means(second, kept), target_eval.FOLD_LAG)
    )
    assert list(compared.by_fold.index) == [fold.index for fold in chosen]
    assert list(compared.by_fold["tuning"]) == [fold.tuning for fold in chosen]
    np.testing.assert_allclose(compared.by_fold["model"], fold_means(first, chosen))
    np.testing.assert_allclose(compared.by_fold["baseline"], fold_means(second, chosen))


def in_research(frame: pd.DataFrame, chosen: list[Fold]) -> pd.DataFrame:
    keep = pd.Series(False, index=frame.index)
    for fold in studied(chosen):
        keep |= (frame["date"] >= fold.test_start) & (frame["date"] <= fold.eval_end)
    return frame[keep]


def test_the_frame_joins_both_caches_and_the_labels_over_tuning_and_research_rows(
    runs: Path,
) -> None:
    chosen = research()

    frame = target_eval.target_frame(INTERVAL_KEY, chosen, runs)

    assert any(fold.tuning for fold in chosen)
    assert studied(chosen)
    assert set(frame["fold"]) == {fold.index for fold in chosen}
    ends = frame["fold"].map({fold.index: fold.eval_end for fold in chosen})
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
    frame = target_eval.target_frame(RETURN_KEY, chosen, runs)

    evaluated = target_eval.evaluate_return(frame, chosen)

    rows = scored(frame)
    for name, column in (("zero", "zero_h5"), ("trailing", "trailing_h5")):
        both = shared(rows, "fwd_ret_open", "lightgbm_ret_h5", column)
        check_paired(
            evaluated.mae[name],
            daily_absolute_error(both, "fwd_ret_open", "lightgbm_ret_h5"),
            daily_absolute_error(both, "fwd_ret_open", column),
            chosen,
        )
        pooled = in_research(both, chosen)
        assert evaluated.rmse[name] == pytest.approx(
            tuple(
                math.sqrt(daily_squared_error(pooled, "fwd_ret_open", value).mean())
                for value in ("lightgbm_ret_h5", column)
            )
        )
    hits = daily_direction_hits(in_research(rows, chosen), "fwd_ret_open", "lightgbm_ret_h5")
    assert evaluated.direction == pytest.approx(hits.mean())
    assert evaluated.direction_t == pytest.approx(newey_west_t(hits - 0.5, 5))


def test_interval_scores_pinball_coverage_volatility_and_probability(runs: Path) -> None:
    chosen = research()
    frame = target_eval.target_frame(INTERVAL_KEY, chosen, runs)

    evaluated = target_eval.evaluate_interval(frame, chosen)

    rows = scored(frame)
    model = {0.1: "lightgbm_q10_h20", 0.5: "lightgbm_q50_h20", 0.9: "lightgbm_q90_h20"}
    for compared, other in (
        (evaluated.pinball, "volband"),
        (evaluated.history, "histq"),
    ):
        baseline = {alpha: column.replace("lightgbm", other) for alpha, column in model.items()}
        both = shared(rows, "fwd_ret_open", *model.values(), *baseline.values())
        check_paired(
            compared,
            daily_pinball(both, "fwd_ret_open", model),
            daily_pinball(both, "fwd_ret_open", baseline),
            chosen,
        )
    pooled = in_research(rows, chosen)
    q10, q50, q90 = (pooled[model[alpha]] for alpha in (0.1, 0.5, 0.9))
    assert evaluated.crossed == int(((q10 > q50) | (q50 > q90)).sum())
    covered = daily_coverage(rows, "fwd_ret_open", "lightgbm_q10_h20", "lightgbm_q90_h20")
    assert evaluated.coverage == pytest.approx(research_mean(covered, chosen))
    pd.testing.assert_frame_equal(
        evaluated.coverage_by_fold, mean_by_fold(covered, chosen)[["mean", "tuning"]]
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
    vols = shared(rows, "fwd_vol_20d", "lightgbm_vol_h20", "tech_vol_20d")
    check_paired(
        evaluated.vol_mae,
        daily_absolute_error(vols, "fwd_vol_20d", "lightgbm_vol_h20"),
        daily_absolute_error(vols, "fwd_vol_20d", "tech_vol_20d"),
        chosen,
    )
    positive = vols[(vols[["fwd_vol_20d", "lightgbm_vol_h20", "tech_vol_20d"]] > 0).all(axis=1)]
    check_paired(
        evaluated.qlike,
        daily_qlike(positive, "fwd_vol_20d", "lightgbm_vol_h20"),
        daily_qlike(positive, "fwd_vol_20d", "tech_vol_20d"),
        chosen,
    )
    up = rows.assign(fwd_up_5pct=rows["fwd_up_5pct"].astype(int))
    both = shared(up, "lightgbm_prob_h20", "logistic_h20")
    check_paired(
        evaluated.brier,
        daily_brier(both, "fwd_up_5pct", "lightgbm_prob_h20"),
        daily_brier(both, "fwd_up_5pct", "logistic_h20"),
        chosen,
    )
    check_paired(
        evaluated.auc,
        daily_auc(both, "fwd_up_5pct", "lightgbm_prob_h20"),
        daily_auc(both, "fwd_up_5pct", "logistic_h20"),
        chosen,
    )
    research_up = in_research(up, chosen)
    pd.testing.assert_frame_equal(
        evaluated.reliability, reliability_table(research_up, "fwd_up_5pct", "lightgbm_prob_h20")
    )
    for column, value in (
        ("lightgbm_prob_h20", evaluated.ece),
        ("lightgbm_prob_raw_h20", evaluated.ece_raw),
    ):
        table = reliability_table(shared(research_up, column), "fwd_up_5pct", column)
        assert value == pytest.approx(expected_calibration_error(table))


def numbers(compared: target_eval.Comparison) -> tuple[float, float, float]:
    return compared.model, compared.baseline, compared.nw_t


def picked(frame: pd.DataFrame, chosen: list[Fold], count: int = 40) -> pd.Index:
    """Research rows the metrics count, spread over the period."""
    rows = in_research(scored(frame), chosen)
    return rows.index[:: max(len(rows) // count, 1)][:count]


def test_return_skips_untradable_rows_and_pairs_each_baseline_on_its_own_rows(runs: Path) -> None:
    chosen = research()
    frame = target_eval.target_frame(RETURN_KEY, chosen, runs)
    original = target_eval.evaluate_return(frame, chosen)
    rows = picked(frame, chosen)
    dropped = target_eval.evaluate_return(frame.drop(rows), chosen)

    marked = frame.index.isin(rows)
    untradable = frame.assign(
        untradable_t1=frame["untradable_t1"].mask(marked, True),
        fwd_ret_open=frame["fwd_ret_open"].mask(marked, 5.0),
    )
    gapped = frame.assign(zero_h5=frame["zero_h5"].mask(marked))

    skipped = target_eval.evaluate_return(untradable, chosen)
    paired = target_eval.evaluate_return(gapped, chosen)

    for name in ("zero", "trailing"):
        assert numbers(skipped.mae[name]) == pytest.approx(numbers(dropped.mae[name]))
        assert skipped.rmse[name] == pytest.approx(dropped.rmse[name])
    assert skipped.direction == pytest.approx(dropped.direction)
    assert numbers(paired.mae["zero"]) == pytest.approx(numbers(dropped.mae["zero"]))
    assert numbers(paired.mae["trailing"]) == pytest.approx(numbers(original.mae["trailing"]))
    assert paired.rmse["zero"] == pytest.approx(dropped.rmse["zero"])
    assert paired.rmse["trailing"] == pytest.approx(original.rmse["trailing"])
    assert paired.mae["zero"].model != pytest.approx(original.mae["zero"].model)


def test_interval_pairs_history_auc_and_qlike_on_their_shared_rows(runs: Path) -> None:
    chosen = research()
    frame = target_eval.target_frame(INTERVAL_KEY, chosen, runs)
    original = target_eval.evaluate_interval(frame, chosen)
    rows = frame.index.isin(picked(frame, chosen))
    dropped = target_eval.evaluate_interval(frame[~rows], chosen)

    gapped = target_eval.evaluate_interval(
        frame.assign(
            histq_q50_h20=frame["histq_q50_h20"].mask(rows),
            logistic_h20=frame["logistic_h20"].mask(rows),
            tech_vol_20d=frame["tech_vol_20d"].mask(rows, 0.0),
        ),
        chosen,
    )

    for compared, expected in (
        (gapped.history, dropped.history),
        (gapped.qlike, dropped.qlike),
        (gapped.brier, dropped.brier),
        (gapped.auc, dropped.auc),
        (gapped.pinball, original.pinball),
    ):
        assert numbers(compared) == pytest.approx(numbers(expected))
    assert gapped.auc.model != pytest.approx(original.auc.model)


def test_crossed_quantiles_count_tradable_research_rows_only(runs: Path) -> None:
    chosen = research()
    frame = target_eval.target_frame(INTERVAL_KEY, chosen, runs)
    counted, untradable = picked(frame, chosen, 6)[:5], picked(frame, chosen, 6)[5:]
    tuning = frame.index[frame["fold"].isin([fold.index for fold in chosen if fold.tuning])][:3]
    crossed = frame.index.isin(counted.union(untradable).union(tuning))
    broken = frame.assign(
        lightgbm_q10_h20=frame["lightgbm_q10_h20"].mask(crossed, frame["lightgbm_q50_h20"] + 0.01),
        untradable_t1=frame["untradable_t1"].mask(frame.index.isin(untradable), True),
    )

    assert target_eval.evaluate_interval(frame, chosen).crossed == 0
    assert target_eval.evaluate_interval(broken, chosen).crossed == len(counted) == 5


def test_return_reads_five_day_labels_and_refuses_them_once_changed(runs: Path) -> None:
    chosen = research()
    LabelDaily.objects.filter(horizon=5).update(fwd_ret_open=F("fwd_ret_open") + 0.1)

    frame = target_eval.target_frame(RETURN_KEY, chosen, runs)

    keys = ["stock_id", "date"]
    five = pd.DataFrame(list(LabelDaily.objects.filter(horizon=5).values(*keys, "fwd_ret_open")))
    expected = frame[keys].merge(five, on=keys, how="left")["fwd_ret_open"]
    np.testing.assert_allclose(frame["fwd_ret_open"], expected.astype(float))
    with pytest.raises(ValueError, match=r"return-wf1 .* stale"):
        target_eval.evaluate_run(RETURN_KEY, chosen, runs)


def comparison(model: float, baseline: float, nw_t: float) -> target_eval.Comparison:
    return target_eval.Comparison(model, baseline, nw_t, pd.DataFrame())


def test_lower_is_better_gates_need_a_newey_west_t_below_minus_two() -> None:
    assert target_eval.beats(comparison(0.01, 0.02, -2.5))
    assert not target_eval.beats(comparison(0.01, 0.02, -1.9))
    assert not target_eval.beats(comparison(0.01, 0.02, -2.0))
    assert not target_eval.beats(comparison(0.03, 0.02, -2.5))
    assert not target_eval.beats(comparison(0.01, 0.02, math.nan))


@pytest.mark.parametrize(
    ("model", "baseline", "passed"),
    [
        (0.01, 0.02, True),
        (0.02, 0.02, True),
        (0.03, 0.02, False),
        (math.nan, 0.02, False),
        (0.01, math.nan, False),
    ],
)
def test_bounded_gates_pass_when_the_model_is_not_above_its_baseline(
    model: float, baseline: float, passed: bool
) -> None:
    assert target_eval.no_worse(comparison(model, baseline, 3.0)) is passed


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


def table(name: str, model: float, baseline: float) -> str:
    """A fold table's header, its tuning fold 1 and research fold 5, as made() fills them."""
    return "\n".join(
        [
            f"| 摺 | 模型 {name} | baseline {name} |",
            "| --- | --- | --- |",
            "| 1（調參） | 0.90000 | 0.10000 |",
            f"| 5 | {model:.5f} | {baseline:.5f} |",
        ]
    )


def made(model: float, baseline: float, nw_t: float) -> target_eval.Comparison:
    """A comparison over a tuning fold 1 and research folds 5 and 6."""
    by_fold = pd.DataFrame(
        {"model": [0.9, model, model], "baseline": [0.1, baseline, baseline]},
        index=pd.Index([1, 5, 6], name="fold"),
    ).assign(tuning=[True, False, False])
    return target_eval.Comparison(model, baseline, nw_t, by_fold)


def test_the_return_report_states_each_gate_with_its_numbers() -> None:
    evaluated = target_eval.ReturnEvaluation(
        mae={"zero": made(0.01, 0.02, -2.5), "trailing": made(0.01, 0.011, -1.5)},
        rmse={"zero": (0.02, 0.03), "trailing": (0.02, 0.021)},
        direction=0.53,
        direction_t=2.4,
    )

    text = target_eval.report(RETURN_KEY, evaluated)

    assert (
        "MAE 低於零預測：模型 0.01000、baseline 0.02000，逐摺差 NW t -2.50（需 < −2）：通過" in text
    )
    assert "MAE 低於過去 60 日平均：模型 0.01000、baseline 0.01100，逐摺差 NW t -1.50" in text
    assert "（需 < −2）：未過" in text
    assert "RMSE 對零預測（同列）：模型 0.02000、baseline 0.03000" in text
    assert "方向準確率 0.5300，對 50% 的 NW t 2.40" in text
    assert table("MAE（對零預測）", 0.01, 0.02) in text
    assert table("MAE（對過去 60 日平均）", 0.01, 0.011) in text
    assert "調參摺列在每摺表並標註，不計入總體指標與關卡" in text


def frame_of(rows: dict[str, list[Any]], index: list[int]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=pd.Index(index, name="fold"))


def interval_evaluation(**changes: Any) -> target_eval.IntervalEvaluation:
    values: dict[str, Any] = {
        "pinball": made(0.10, 0.12, -3.0),
        "history": made(0.10, 0.11, 0.0),
        "crossed": 0,
        "coverage": 0.80,
        "coverage_by_fold": frame_of(
            {"mean": [0.95, 0.80, 0.78], "tuning": [True, False, False]}, [1, 5, 6]
        ),
        "coverage_table": pd.DataFrame(
            {"coverage": [0.8], "width": [0.2]}, index=pd.Index([1], name="bucket")
        ),
        "consistency": pd.DataFrame(
            {"mean_share": [0.01], "max_share": [0.05], "days_over": [0], "n_days": [30]},
            index=["width_ratio"],
        ),
        "vol_mae": made(0.010, 0.010, 0.5),
        "qlike": made(0.20, 0.19, -4.0),
        "auc": made(0.56, 0.55, 1.2),
        "brier": made(0.09, 0.10, -1.0),
        "reliability": pd.DataFrame(
            {
                "lower": [0.0],
                "upper": [0.1],
                "mean_prob": [0.05],
                "frac_positive": [0.06],
                "n_rows": [100],
            },
            index=[1],
        ),
        "ece": 0.03,
        "ece_raw": 0.08,
    }
    return target_eval.IntervalEvaluation(**{**values, **changes})


def test_the_interval_report_states_each_gate_with_its_numbers() -> None:
    text = target_eval.report(INTERVAL_KEY, interval_evaluation())

    assert "pinball 低於波動率區間：模型 0.10000、baseline 0.12000，逐摺差 NW t -3.00" in text
    assert (
        "pinball 低於歷史分位數（sanity 下限，同列）：模型 0.10000、歷史分位數 0.11000：通過"
        in text
    )
    assert "分位數交叉（q10 > q50 或 q50 > q90）的研究列數 0：通過" in text
    assert (
        "總覆蓋率 0.8000（需在 75% 到 85%），每摺覆蓋率需在 70% 到 90%，超出的研究摺：無：通過"
        in text
    )
    assert "| 1（調參） | 0.9500 |" in text
    assert "MAE 對 tech_vol_20d（參考）：模型 0.01000、baseline 0.01000，逐摺差 NW t 0.50" in text
    assert "MAE 不高於" not in text
    assert "日均 AUC 對 logistic（參考）：模型 0.56000、baseline 0.55000，逐摺差 NW t 1.20" in text
    assert (
        "QLIKE 不高於 tech_vol_20d：模型 0.20000、baseline 0.19000，逐摺差 NW t -4.00（參考）：未過"
        in text
    )
    assert (
        "Brier 不高於 logistic：模型 0.09000、baseline 0.10000，逐摺差 NW t -1.00（參考）：通過"
        in text
    )
    assert "ECE 低於 0.05：校準後 0.0300（校準前 0.0800）：通過" in text
    assert "| 0.0–0.1 | 0.0500 | 0.0600 | 100 |" in text
    assert "universe 內全部列（含 t+1 不可交易的列），研究摺。" in text
    for heading in ("## 區間 20 日", "## 波動率", "## 上漲機率", "## 輸出一致性"):
        assert heading in text, heading
    for name, model, baseline in (
        ("pinball（對波動率區間）", 0.10, 0.12),
        ("pinball（對歷史分位數）", 0.10, 0.11),
        ("MAE", 0.01, 0.01),
        ("QLIKE", 0.20, 0.19),
        ("AUC", 0.56, 0.55),
        ("Brier", 0.09, 0.10),
    ):
        assert table(name, model, baseline) in text, name


def test_the_interval_report_fails_coverage_crossing_and_ece_on_research_folds() -> None:
    text = target_eval.report(
        INTERVAL_KEY,
        interval_evaluation(
            crossed=3,
            ece=0.06,
            history=made(0.12, 0.11, 0.0),
            coverage_by_fold=frame_of(
                {"mean": [0.95, 0.69, math.nan], "tuning": [True, False, False]}, [1, 5, 6]
            ),
        ),
    )

    assert "超出的研究摺：5、6：未過" in text
    assert "研究列數 3：未過" in text
    assert "校準後 0.0600（校準前 0.0800）：未過" in text
    assert "歷史分位數 0.11000：未過" in text


@pytest.fixture
def command(runs: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    use_designs(monkeypatch)
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


def test_evaluate_targets_reports_an_output_it_cannot_write(command: Path, tmp_path: Path) -> None:
    with pytest.raises(CommandError, match="Is a directory"):
        call_command(
            "evaluate_targets", run_key=RETURN_KEY, output=str(tmp_path), stdout=StringIO()
        )


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


def test_evaluate_run_refuses_a_baseline_fit_on_other_features(runs: Path) -> None:
    fold = studied(research())[0]
    notes = fold_notes(RETURN_KEY, "baseline", fold.index)
    predictions = load_fold_predictions(RETURN_KEY, "baseline", fold)
    save_fold_predictions(
        RETURN_KEY, "baseline", fold, predictions, notes={**notes, "feature_hash": "other"}
    )

    assert target_eval.stale_folds("return", RETURN_KEY, research(), runs) == [fold.index]
    with pytest.raises(ValueError, match=rf"folds \[{fold.index}\] of return-wf1 .* stale"):
        target_eval.evaluate_run(RETURN_KEY, research(), runs)


def test_evaluate_run_names_missing_lightgbm_folds(runs: Path) -> None:
    kept = [fold for fold in research() if fold.research]
    fold_path(RETURN_KEY, "lightgbm", kept[0].index).unlink()

    with pytest.raises(MissingFoldsError, match=rf"return-wf1/lightgbm .* \[{kept[0].index}\]"):
        target_eval.evaluate_run(RETURN_KEY, research(), runs)


def test_evaluate_targets_never_reads_holdout_folds(command: Path, monkeypatch: Any) -> None:
    seen: list[list[Fold]] = []
    original = target_eval.evaluate_run

    def recorded(run_key: str, chosen: list[Fold], *args: Any, **kwargs: Any) -> Any:
        seen.append(chosen)
        return original(run_key, chosen, *args, **kwargs)

    monkeypatch.setattr(evaluate_targets, "evaluate_run", recorded)

    call_command("evaluate_targets", run_key=RETURN_KEY, stdout=StringIO())

    assert [fold.index for fold in seen[0]] == [fold.index for fold in research()]
    assert not any(fold.holdout for fold in seen[0])
    assert any(fold.holdout for fold in folds())
