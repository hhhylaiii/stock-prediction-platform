from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice, StockMaster, TradingCalendar
from apps.research import baseline_runs
from apps.research.baseline_runs import run_folds, shuffle_check
from apps.research.baselines import PENALTY_GRID, historical_quantiles
from apps.research.features.store import export_year
from apps.research.holdout import HoldoutError
from apps.research.management.commands import train
from apps.research.metrics.ic import daily_rank_ic
from apps.research.models import FeatureDaily, LabelDaily, ModelRun
from apps.research.prediction_cache import fold_record, load_fold_predictions
from apps.research.runs import feature_hash, lib_versions, load_model
from apps.research.training import LINEAR_FEATURES, linear_inputs
from apps.research.walkforward import Fold, WalkForwardConfig, generate_folds

pytestmark = pytest.mark.django_db
START, END = date(2016, 1, 1), date(2016, 12, 31)
STOCKS = [f"{1100 + k}" for k in range(40)]
CONFIG = WalkForwardConfig(
    train_months=6,
    test_months=1,
    step_months=1,
    valid_months=2,
    data_start=START,
    first_test=date(2016, 9, 1),
    tuning_end=date(2016, 9, 30),
    holdout_start=date(2016, 12, 1),
)


def weekdays() -> list[date]:
    days = (START + timedelta(days=n) for n in range((END - START).days + 1))
    return [day for day in days if day.weekday() < 5]


@pytest.fixture
def panel(settings: Any, tmp_path: Path) -> Path:
    """A year of 40 stocks whose tech_ret_1d drives the labels; 1139 is outside the universe."""
    settings.RESEARCH_CACHE_DIR = tmp_path / "predictions"
    settings.MODEL_DIR = tmp_path / "models"
    settings.FEATURE_PARQUET_DIR = tmp_path / "features"
    rng = np.random.default_rng(3)
    days = weekdays()
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=day.weekday() < 5)
        for day in (START + timedelta(days=n) for n in range((END - START).days + 1))
    )
    features, labels = [], []
    for day in days:
        signal = rng.normal(size=len(STOCKS))
        noise = rng.normal(size=len(STOCKS))
        ranks = pd.Series(signal + noise).rank(pct=True).to_numpy()
        for k, stock in enumerate(STOCKS):
            features.append(
                FeatureDaily(
                    stock_id=stock,
                    date=day,
                    in_universe=stock != "1139",
                    tech_ret_1d=float(signal[k]),
                    tech_ret_60d=0.01 * k,
                    tech_vol_20d=0.2 + 0.01 * k,
                )
            )
            ret = 0.01 * float(signal[k] + noise[k])
            for horizon in (5, 20):
                labels.append(
                    LabelDaily(
                        stock_id=stock,
                        date=day,
                        horizon=horizon,
                        fwd_ret=ret,
                        fwd_ret_open=ret,
                        fwd_rank_pct=float(ranks[k]),
                        fwd_vol_20d=0.3 if horizon == 20 else None,
                        fwd_up_5pct=bool(signal[k] + noise[k] > 0.5) if horizon == 20 else None,
                        untradable_t1=False,
                    )
                )
    FeatureDaily.objects.bulk_create(features)
    LabelDaily.objects.bulk_create(labels)
    export_year(2016, settings.FEATURE_PARQUET_DIR)
    return Path(settings.FEATURE_PARQUET_DIR)


def folds() -> list[Fold]:
    calendar = TradingDays(weekdays(), START, END)
    return generate_folds(calendar, END, CONFIG)


def research() -> list[Fold]:
    return [fold for fold in folds() if not fold.holdout]


def test_rank_folds_cache_ridge_scores_for_every_test_row(panel: Path) -> None:
    fold = research()[-1]
    assert fold.eval_end < fold.test_end

    assert run_folds("rank", [fold], "rank-wf1", panel) == [fold.index]

    cached = load_fold_predictions("rank-wf1", fold)
    test_days = [day for day in weekdays() if fold.test_start <= day <= fold.test_end]
    assert cached.columns.tolist() == ["date", "stock_id", "in_universe", "ridge_h5", "ridge_h20"]
    assert len(cached) == len(test_days) * len(STOCKS)
    assert not cached["in_universe"].all()
    assert cached[["ridge_h5", "ridge_h20"]].notna().all().all()
    run = ModelRun.objects.get(run_id=f"rank-wf1.ridge.h20.f{fold.index:03d}")
    assert run.run_key == "rank-wf1"
    assert run.objective == "rank"
    assert (run.train_start, run.train_end) == (fold.train_start, fold.train_end)
    assert run.feature_list == LINEAR_FEATURES
    assert run.feature_hash == feature_hash(fold.train_start, fold.test_end, panel)
    assert run.lib_versions == lib_versions()
    assert run.params["horizon"] == 20
    assert run.params["label"] == "fwd_rank_pct"
    assert run.params["fold"] == fold_record(fold)
    assert run.params["penalty"] in PENALTY_GRID
    assert set(run.params["cv_scores"]) == {str(penalty) for penalty in PENALTY_GRID}
    assert run.ic_mean is not None
    assert run.ic_mean > 0.2
    labels = pd.DataFrame(
        list(
            LabelDaily.objects.filter(
                date__range=(fold.test_start, fold.eval_end), horizon=20
            ).values("stock_id", "date", "fwd_rank_pct")
        )
    )
    scored = cached[cached["in_universe"]].merge(labels, on=["stock_id", "date"])
    daily = daily_rank_ic(scored, "ridge_h20", "fwd_rank_pct")
    assert len(daily) == len([day for day in test_days if day <= fold.eval_end])
    assert run.ic_mean == pytest.approx(daily.mean())
    assert run.icir == pytest.approx(daily.mean() / daily.std())
    assert ModelRun.objects.count() == 2


def test_cached_scores_come_from_the_saved_model(panel: Path) -> None:
    fold = research()[0]
    run_folds("rank", [fold], "rank-wf1", panel)

    model = load_model("rank-wf1", f"rank-wf1.ridge.h5.f{fold.index:03d}")
    cached = load_fold_predictions("rank-wf1", fold)
    frame = FeatureDaily.objects.filter(date__range=(fold.test_start, fold.test_end))
    assert frame.count() == len(cached)
    raw = pd.read_parquet(panel / "year=2016" / "part-0.parquet")
    inputs = linear_inputs(raw)
    test = inputs[inputs["date"].between(fold.test_start, fold.test_end)]
    expected = model.predict(test[LINEAR_FEATURES].to_numpy(dtype=np.float64))
    assert cached["ridge_h5"].to_numpy() == pytest.approx(expected)


def test_models_never_see_labels_after_the_training_window(panel: Path) -> None:
    fold = research()[0]
    run_folds("rank", [fold], "rank-wf1", panel)
    before = load_fold_predictions("rank-wf1", fold)
    LabelDaily.objects.filter(date__gt=fold.train_end).update(fwd_rank_pct=0.5)

    run_folds("rank", [fold], "rank-wf1", panel, rerun=True)

    after = load_fold_predictions("rank-wf1", fold)
    pd.testing.assert_frame_equal(before, after)


def test_return_folds_cache_the_rule_forecasts_without_model_runs(panel: Path) -> None:
    fold = research()[0]

    run_folds("return", [fold], "return-wf1", panel)

    cached = load_fold_predictions("return-wf1", fold)
    assert cached.columns.tolist() == ["date", "stock_id", "in_universe", "zero_h5", "trailing_h5"]
    assert (cached["zero_h5"] == 0).all()
    shares = cached.set_index("stock_id")["trailing_h5"].groupby(level=0).first()
    assert shares["1101"] == pytest.approx((1 + 0.01) ** (5 / 60) - 1)
    assert not ModelRun.objects.exists()


def test_interval_folds_cache_probability_quantile_and_volatility_baselines(panel: Path) -> None:
    fold = research()[0]

    run_folds("interval", [fold], "interval-wf1", panel)

    cached = load_fold_predictions("interval-wf1", fold)
    assert cached["logistic_h20"].between(0, 1).all()
    labels = pd.DataFrame(
        list(
            LabelDaily.objects.filter(date__range=(fold.train_start, fold.train_end), horizon=20)
            .exclude(stock_id="1139")
            .values("fwd_ret_open")
        )
    )
    expected = historical_quantiles(labels, "fwd_ret_open")
    for name, value in expected.items():
        assert cached[f"histq_{name}_h20"].to_numpy() == pytest.approx([value] * len(cached))
    first = cached[cached["stock_id"] == "1101"].iloc[0]
    half = 1.28 * 0.21 * np.sqrt(20 / 252)
    assert (first["volband_q10_h20"], first["volband_q50_h20"], first["volband_q90_h20"]) == (
        pytest.approx(-half),
        0.0,
        pytest.approx(half),
    )
    assert first["techvol_h20"] == pytest.approx(0.21)
    kinds = set(ModelRun.objects.values_list("model_type", "objective"))
    assert kinds == {("logistic", "classification"), ("histq", "quantile")}
    histq = ModelRun.objects.get(model_type="histq")
    assert histq.params["quantiles"] == pytest.approx(expected)


def test_cached_folds_are_skipped_unless_rerun(panel: Path) -> None:
    first, second = research()[:2]
    run_folds("return", [first], "return-wf1", panel)

    assert run_folds("return", [first, second], "return-wf1", panel) == [second.index]
    assert run_folds("return", [first, second], "return-wf1", panel) == []
    assert run_folds("return", [first], "return-wf1", panel, rerun=True) == [first.index]


def test_holdout_folds_wait_for_the_phase_4_models(panel: Path) -> None:
    held = [fold for fold in folds() if fold.holdout]

    with pytest.raises(HoldoutError, match="Phase 4"):
        run_folds("return", [research()[0], *held], "return-wf1", panel)

    assert not (Path(panel).parent / "predictions" / "return-wf1").exists()


@pytest.mark.parametrize(
    ("scope", "run_key", "message"),
    [
        ("rank", "interval-wf1", "start with rank-"),
        ("trend", "trend-wf1", "scope"),
        ("rank", "rank-" + "x" * 40, "at most"),
    ],
)
def test_run_keys_name_their_scope_and_leave_room_for_run_ids(
    tmp_path: Path, scope: str, run_key: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_folds(scope, research()[:1], run_key, tmp_path)


def test_shuffle_check_reports_an_oos_ic_near_zero(panel: Path) -> None:
    fold = next(fold for fold in folds() if fold.research)

    ics = [shuffle_check(fold, panel, seed) for seed in range(3)]

    assert max(abs(ic) for ic in ics) < 0.1
    assert len(set(ics)) == 3


@pytest.fixture
def command(panel: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(train, "DEFAULT_CONFIG", CONFIG)
    StockMaster.objects.create(stock_id="1101", name="1101", market="TWSE", industry="水泥工業")
    DailyPrice.objects.create(
        stock_id="1101", date=date(2016, 12, 27), close=10, volume=1000, turnover=10_000
    )
    return panel


def test_command_runs_the_chosen_research_folds(command: Path) -> None:
    out = StringIO()

    call_command("train", scope="return", run_key="return-wf1", folds="0-1", stdout=out)

    assert "return-wf1: ran folds [0, 1], 0 cached" in out.getvalue()
    assert load_fold_predictions("return-wf1", research()[1]).notna().all().all()


def test_command_runs_every_research_fold_by_default(command: Path) -> None:
    out = StringIO()

    call_command("train", scope="return", run_key="return-wf1", stdout=out)

    indices = [fold.index for fold in folds() if not fold.holdout]
    assert f"return-wf1: ran folds {indices}, 0 cached" in out.getvalue()
    assert len(indices) == len(folds()) - 1


def test_command_refuses_holdout_and_unknown_folds(command: Path) -> None:
    held = next(fold.index for fold in folds() if fold.holdout)

    with pytest.raises(CommandError, match=rf"holdout folds \[{held}\] wait"):
        call_command("train", scope="return", run_key="return-wf1", folds=str(held))
    with pytest.raises(CommandError, match="--scope and --run-key"):
        call_command("train", scope="return")


def test_command_shuffle_check_passes_and_fails_by_the_limit(command: Path, settings: Any) -> None:
    out = StringIO()
    call_command("train", shuffle_check=True, stdout=out)
    assert "shuffled OOS rank IC" in out.getvalue()

    settings.SHUFFLE_IC_LIMIT = 0.0
    with pytest.raises(CommandError, match="leak"):
        call_command("train", shuffle_check=True)


def test_a_partition_rewritten_during_the_read_stops_the_fold(
    panel: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fold = research()[0]
    reader = baseline_runs.read_partitions

    def rewritten(start: date, end: date, directory: Path) -> pd.DataFrame:
        frame = reader(start, end, directory)
        FeatureDaily.objects.filter(stock_id="1101").update(tech_ret_1d=9.0)
        export_year(2016, directory)
        return frame

    monkeypatch.setattr(baseline_runs, "read_partitions", rewritten)

    with pytest.raises(RuntimeError, match="changed while reading"):
        run_folds("return", [fold], "return-wf1", panel)

    assert not (panel.parent / "predictions" / "return-wf1").exists()


def test_run_folds_reads_the_feature_partitions(panel: Path) -> None:
    fold = research()[0]
    FeatureDaily.objects.all().delete()

    run_folds("return", [fold], "return-wf1", panel)

    assert len(load_fold_predictions("return-wf1", fold)) > 0
