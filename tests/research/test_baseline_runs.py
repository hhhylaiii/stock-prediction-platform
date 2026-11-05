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
from apps.research.baseline_runs import BASELINE, recipe, run_folds, shuffle_check
from apps.research.baselines import PENALTY_GRID, historical_quantiles
from apps.research.features.store import export_year
from apps.research.holdout import HoldoutError
from apps.research.management.commands import train
from apps.research.metrics.ic import daily_rank_ic
from apps.research.models import FeatureDaily, LabelDaily, ModelRun, ModelStatus
from apps.research.prediction_cache import (
    fold_notes,
    fold_path,
    fold_record,
    load_fold_predictions,
    missing_folds,
    save_fold_predictions,
)
from apps.research.runs import RunError, feature_hash, label_hash, lib_versions, load_model
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

    cached = load_fold_predictions("rank-wf1", BASELINE, fold)
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
    weights = load_model("rank-wf1", run.run_id).coef_
    assert run.params["coefficients"] == pytest.approx(
        dict(zip(LINEAR_FEATURES, weights, strict=True))
    )
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
    cached = load_fold_predictions("rank-wf1", BASELINE, fold)
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
    before = load_fold_predictions("rank-wf1", BASELINE, fold)
    LabelDaily.objects.filter(date__gt=fold.train_end).update(fwd_rank_pct=0.5)

    run_folds("rank", [fold], "rank-wf1", panel, rerun=True)

    after = load_fold_predictions("rank-wf1", BASELINE, fold)
    pd.testing.assert_frame_equal(before, after)


def test_return_folds_cache_the_rule_forecasts_without_model_runs(panel: Path) -> None:
    fold = research()[0]

    run_folds("return", [fold], "return-wf1", panel)

    cached = load_fold_predictions("return-wf1", BASELINE, fold)
    assert cached.columns.tolist() == ["date", "stock_id", "in_universe", "zero_h5", "trailing_h5"]
    assert (cached["zero_h5"] == 0).all()
    shares = cached.set_index("stock_id")["trailing_h5"].groupby(level=0).first()
    assert shares["1101"] == pytest.approx((1 + 0.01) ** (5 / 60) - 1)
    assert not ModelRun.objects.exists()


def test_interval_folds_cache_probability_quantile_and_volatility_baselines(panel: Path) -> None:
    fold = research()[0]

    run_folds("interval", [fold], "interval-wf1", panel)

    cached = load_fold_predictions("interval-wf1", BASELINE, fold)
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


def test_a_fold_without_evaluation_labels_still_caches_its_scores(panel: Path) -> None:
    fold = research()[-1]
    LabelDaily.objects.filter(date__gte=fold.test_start).delete()

    run_folds("rank", [fold], "rank-wf1", panel)

    assert load_fold_predictions("rank-wf1", BASELINE, fold)["ridge_h20"].notna().all()
    run = ModelRun.objects.get(run_id=f"rank-wf1.ridge.h20.f{fold.index:03d}")
    assert (run.ic_mean, run.icir) == (None, None)


def test_return_folds_read_only_their_test_period(
    panel: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fold = research()[0]
    reader = baseline_runs.read_partitions
    ranges = []

    def recorded(start: date, end: date, directory: Path) -> pd.DataFrame:
        ranges.append((start, end))
        return reader(start, end, directory)

    monkeypatch.setattr(baseline_runs, "read_partitions", recorded)

    run_folds("return", [fold], "return-wf1", panel)

    assert ranges == [(fold.test_start, fold.test_end)]


def test_rebuilt_partitions_rerun_their_cached_folds(panel: Path) -> None:
    first, second = research()[:2]
    run_folds("return", [first, second], "return-wf1", panel)
    FeatureDaily.objects.filter(date=second.test_start).update(tech_ret_60d=0.5)
    export_year(2016, panel)

    assert run_folds("return", [first, second], "return-wf1", panel) == [
        first.index,
        second.index,
    ]
    assert run_folds("return", [first, second], "return-wf1", panel) == []


def test_a_rerun_that_fails_leaves_its_fold_to_be_run_again(
    panel: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fold = research()[0]
    run_folds("return", [fold], "return-wf1", panel)

    def broken(*args: Any, **kwargs: Any) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(baseline_runs, "save_fold_predictions", broken)
    with pytest.raises(OSError, match="disk full"):
        run_folds("return", [fold], "return-wf1", panel, rerun=True)

    assert missing_folds("return-wf1", BASELINE, [fold]) == [fold.index]


@pytest.mark.parametrize(
    ("scope", "run_key", "first"),
    [("rank", "rank-wf1", "train_start"), ("return", "return-wf1", "test_start")],
)
def test_resumed_folds_hash_the_partitions_their_scope_reads(
    panel: Path, monkeypatch: pytest.MonkeyPatch, scope: str, run_key: str, first: str
) -> None:
    fold = research()[0]
    run_folds(scope, [fold], run_key, panel)
    ranges = []

    def recorded(start: date, end: date, directory: Path) -> str:
        ranges.append((start, end))
        return feature_hash(start, end, directory)

    monkeypatch.setattr(baseline_runs, "feature_hash", recorded)

    assert run_folds(scope, [fold], run_key, panel) == []
    assert ranges == [(getattr(fold, first), fold.test_end)]


def test_a_fold_with_a_promoted_model_keeps_its_cache(panel: Path) -> None:
    fold = research()[0]
    run_folds("rank", [fold], "rank-wf1", panel)
    promoted = f"rank-wf1.ridge.h5.f{fold.index:03d}"
    ModelRun.objects.filter(run_id=promoted).update(status=ModelStatus.ACTIVE)

    with pytest.raises(RunError, match=promoted):
        run_folds("rank", [fold], "rank-wf1", panel, rerun=True)

    assert fold_path("rank-wf1", BASELINE, fold.index).exists()


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
        ("rank", "rank-WF1", "lowercase"),
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
    assert load_fold_predictions("return-wf1", BASELINE, research()[1]).notna().all().all()


def test_command_runs_every_research_fold_by_default(command: Path, settings: Any) -> None:
    settings.SHUFFLE_IC_LIMIT = 0.2
    out = StringIO()

    call_command("train", scope="return", run_key="return-wf1", stdout=out)

    indices = [fold.index for fold in folds() if not fold.holdout]
    assert "shuffled OOS rank IC" in out.getvalue()
    assert f"return-wf1: ran folds {indices}, 0 cached" in out.getvalue()
    assert len(indices) == len(folds()) - 1


def test_a_full_run_stops_when_the_shuffle_check_fails(command: Path, settings: Any) -> None:
    settings.SHUFFLE_IC_LIMIT = 0.0

    with pytest.raises(CommandError, match="label column"):
        call_command("train", scope="return", run_key="return-wf1")
    call_command("train", scope="return", run_key="return-wf1", folds="0", stdout=StringIO())

    rest = [fold.index for fold in research()[1:]]
    assert missing_folds("return-wf1", BASELINE, research()) == rest


def test_command_refuses_holdout_and_unknown_folds(command: Path) -> None:
    held = next(fold.index for fold in folds() if fold.holdout)

    with pytest.raises(CommandError, match=rf"holdout folds \[{held}\] wait") as caught:
        call_command("train", scope="return", run_key="return-wf1", folds=str(held))
    assert "do not exist" not in str(caught.value)
    with pytest.raises(CommandError, match=r"folds \[99\] do not exist"):
        call_command("train", scope="return", run_key="return-wf1", folds="99")
    for written in ("3-1", "0-", "a"):
        with pytest.raises(CommandError, match="indices like"):
            call_command("train", scope="return", run_key="return-wf1", folds=written)
    with pytest.raises(CommandError, match="--scope and --run-key"):
        call_command("train", scope="return")
    for last in ("2030-01-01", "2016-08-01"):
        with pytest.raises(CommandError):
            call_command("train", scope="return", run_key="return-wf1", last_date=last)


def test_a_full_run_checks_the_run_key_before_the_shuffle_check(
    command: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(train, "shuffle_check", lambda *args: pytest.fail("shuffled first"))

    with pytest.raises(CommandError, match="start with rank-"):
        call_command("train", scope="rank", run_key="interval-x")


def test_command_stops_at_the_last_date(command: Path, settings: Any) -> None:
    settings.SHUFFLE_IC_LIMIT = 0.2
    out = StringIO()

    call_command("train", scope="return", run_key="return-wf1", last_date="2016-10-31", stdout=out)

    calendar = TradingDays(weekdays(), START, END)
    indices = [fold.index for fold in generate_folds(calendar, date(2016, 10, 31), CONFIG)]
    assert f"return-wf1: ran folds {indices}, 0 cached" in out.getvalue()
    assert len(indices) < len(research())


def test_command_rerun_refits_cached_folds(command: Path) -> None:
    call_command("train", scope="return", run_key="return-wf1", folds="0", stdout=StringIO())
    out = StringIO()

    call_command("train", scope="return", run_key="return-wf1", folds="0", rerun=True, stdout=out)

    assert "return-wf1: ran folds [0], 0 cached" in out.getvalue()


def test_command_passes_the_seed_to_the_shuffle_check(
    command: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeds = []

    def recorded(fold: Fold, directory: Path, seed: int) -> float:
        seeds.append(seed)
        return 0.0

    monkeypatch.setattr(train, "shuffle_check", recorded)

    call_command("train", shuffle_check=True, seed=7, stdout=StringIO())

    assert seeds == [7]


def test_command_shuffle_check_passes_and_fails_by_the_limit(command: Path, settings: Any) -> None:
    out = StringIO()
    call_command("train", shuffle_check=True, stdout=out)
    assert "shuffled OOS rank IC" in out.getvalue()

    settings.SHUFFLE_IC_LIMIT = 0.0
    with pytest.raises(CommandError, match="label column"):
        call_command("train", shuffle_check=True)


@pytest.mark.parametrize("ic", [0.02, -0.02])
def test_a_shuffled_ic_at_the_limit_fails(
    command: Path, monkeypatch: pytest.MonkeyPatch, settings: Any, ic: float
) -> None:
    settings.SHUFFLE_IC_LIMIT = 0.02
    monkeypatch.setattr(train, "shuffle_check", lambda *args: ic)

    with pytest.raises(CommandError, match="label column"):
        call_command("train", shuffle_check=True, stdout=StringIO())


def test_a_shuffle_check_without_an_ic_fails(
    command: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(train, "shuffle_check", lambda *args: float("nan"))

    with pytest.raises(CommandError, match="no OOS day"):
        call_command("train", shuffle_check=True, stdout=StringIO())


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

    assert len(load_fold_predictions("return-wf1", BASELINE, fold)) > 0


def scored(fold: Fold, score: float) -> pd.DataFrame:
    return pd.DataFrame({"date": [fold.test_start], "stock_id": ["1101"], "score": [score]})


def test_a_baseline_rerun_keeps_the_other_models_of_its_fold(panel: Path) -> None:
    fold = research()[0]
    run_folds("rank", [fold], "rank-wf1", panel)
    save_fold_predictions("rank-wf1", "lightgbm", fold, scored(fold, 0.7), notes={"seed": "7"})

    assert run_folds("rank", [fold], "rank-wf1", panel, rerun=True) == [fold.index]

    pd.testing.assert_frame_equal(
        load_fold_predictions("rank-wf1", "lightgbm", fold), scored(fold, 0.7)
    )
    assert fold_notes("rank-wf1", "lightgbm", fold.index) == {"seed": "7"}
    assert "ridge_h5" in load_fold_predictions("rank-wf1", BASELINE, fold)


def test_folds_another_model_cached_still_run_their_baselines(panel: Path) -> None:
    first, second = research()[:2]
    for fold in (first, second):
        save_fold_predictions("return-wf1", "lightgbm", fold, scored(fold, 0.7))

    assert run_folds("return", [first, second], "return-wf1", panel) == [first.index, second.index]
    assert run_folds("return", [first, second], "return-wf1", panel) == []


@pytest.mark.parametrize(
    ("scope", "run_key", "first", "last", "horizons"),
    [
        ("rank", "rank-wf1", "train_start", "eval_end", (5, 20)),
        ("interval", "interval-wf1", "train_start", "train_end", (20,)),
    ],
)
def test_cached_folds_note_their_features_labels_and_recipe(
    panel: Path, scope: str, run_key: str, first: str, last: str, horizons: tuple[int, ...]
) -> None:
    fold = research()[0]

    run_folds(scope, [fold], run_key, panel)

    assert fold_notes(run_key, BASELINE, fold.index) == {
        "feature_hash": feature_hash(getattr(fold, first), fold.test_end, panel),
        "label_hash": label_hash(getattr(fold, first), getattr(fold, last), horizons),
        "recipe": recipe(),
    }


def test_rule_only_folds_note_no_labels(panel: Path) -> None:
    fold = research()[0]

    run_folds("return", [fold], "return-wf1", panel)

    assert fold_notes("return-wf1", BASELINE, fold.index)["label_hash"] == ""


@pytest.mark.parametrize(
    ("scope", "run_key", "day", "horizon"),
    [
        ("rank", "rank-wf1", "train_start", 5),
        ("rank", "rank-wf1", "eval_end", 20),
        ("interval", "interval-wf1", "train_end", 20),
    ],
)
def test_relabelled_folds_rerun(
    panel: Path, scope: str, run_key: str, day: str, horizon: int
) -> None:
    fold = research()[0]
    run_folds(scope, [fold], run_key, panel)
    edge = getattr(fold, day)
    LabelDaily.objects.filter(date=edge, horizon=horizon, stock_id="1101").update(fwd_ret_open=0.9)

    assert run_folds(scope, [fold], run_key, panel) == [fold.index]
    assert run_folds(scope, [fold], run_key, panel) == []


def test_labels_a_scope_does_not_read_leave_its_cache(panel: Path) -> None:
    fold = research()[0]
    run_folds("interval", [fold], "interval-wf1", panel)
    run_folds("return", [fold], "return-wf1", panel)
    LabelDaily.objects.filter(date__gt=fold.train_end).update(fwd_up_5pct=True)
    LabelDaily.objects.filter(horizon=5).update(fwd_ret_open=0.9)

    assert run_folds("interval", [fold], "interval-wf1", panel) == []
    LabelDaily.objects.all().delete()
    assert run_folds("return", [fold], "return-wf1", panel) == []


def test_a_changed_recipe_reruns_cached_folds(panel: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fold = research()[0]
    run_folds("return", [fold], "return-wf1", panel)

    monkeypatch.setattr(baseline_runs, "RECIPE_VERSION", "2")
    assert run_folds("return", [fold], "return-wf1", panel) == [fold.index]
    assert run_folds("return", [fold], "return-wf1", panel) == []

    monkeypatch.setattr(baseline_runs, "LINEAR_FEATURES", LINEAR_FEATURES[:-1])
    assert run_folds("return", [fold], "return-wf1", panel) == [fold.index]


def test_labels_rewritten_during_the_fit_stop_the_fold(
    panel: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fold = research()[0]
    fitted = baseline_runs.SCOPES["rank"]

    def relabelled(inputs: Any, run_key: str) -> Any:
        result = fitted(inputs, run_key)
        LabelDaily.objects.filter(date=fold.train_start).update(fwd_rank_pct=0.5)
        return result

    monkeypatch.setitem(baseline_runs.SCOPES, "rank", relabelled)

    with pytest.raises(RuntimeError, match="label_daily changed"):
        run_folds("rank", [fold], "rank-wf1", panel)

    assert not ModelRun.objects.exists()
    assert missing_folds("rank-wf1", BASELINE, [fold]) == [fold.index]
