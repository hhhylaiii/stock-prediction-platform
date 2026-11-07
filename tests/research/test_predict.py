import shutil
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.etl.models import DataQualityLog
from apps.etl.quality import CHECKS
from apps.research import predict as predict_module
from apps.research.inference import day_rows, run_values
from apps.research.lgbm_runs import run_id
from apps.research.models import (
    REST_FEATURE,
    ModelRun,
    ModelStatus,
    Prediction,
    PredictionContrib,
)
from apps.research.predict import SLOTS, PredictionGateError, Slot, predict
from apps.research.runs import RunError, load_model, model_path
from tests.research.run_panels import research

LIVE = [fold for fold in research() if fold.research][-1]
DAY = LIVE.test_end
EARLIER = DAY - timedelta(days=1)
OUTSIDE = "1139"
LIVE_RUNS = {
    Slot("rank", 5): ("rank", 5),
    Slot("rank", 20): ("rank", 20),
    Slot("regression", 5): ("return", 5),
    Slot("quantile", 20): ("interval", 20),
    Slot("volatility", 20): ("interval", 20),
    Slot("classification", 20): ("interval", 20),
}


def live(slot: Slot) -> ModelRun:
    scope, horizon = LIVE_RUNS[slot]
    return ModelRun.objects.get(run_id=run_id(f"{scope}-live", slot.objective, horizon, LIVE))


def serve(*slots: Slot, status: str = ModelStatus.ACTIVE) -> list[ModelRun]:
    runs = [live(slot) for slot in slots]
    ModelRun.objects.filter(pk__in=[run.pk for run in runs]).update(status=status)
    return runs


def passed(day: date = DAY, severity: str = "info") -> None:
    for name in CHECKS:
        DataQualityLog.objects.create(date=day, check_name=name, severity=severity, detail={})


def clone(run: ModelRun, suffix: str, status: str) -> ModelRun:
    source = model_path(run.run_key, run.run_id)
    run.run_id = f"{run.run_id}{suffix}"
    run.status = status
    run.save(force_insert=True)
    shutil.copy(source, model_path(run.run_key, run.run_id))
    return run


def stored(run: ModelRun, day: date = DAY) -> dict[str, dict[str, Any]]:
    rows = Prediction.objects.filter(run=run, date=day).values()
    return {row["stock_id"]: dict(row) for row in rows}


@pytest.mark.django_db
def test_predict_writes_every_slot_for_every_stock_with_features(served_runs: Path) -> None:
    runs = serve(*SLOTS)
    passed()

    result = predict(DAY, served_runs)

    assert result.written == {run.run_id: 40 for run in runs}
    assert result.missing == []
    rank = live(Slot("rank", 20))
    rows = stored(rank)
    inside = {stock: row for stock, row in rows.items() if row["in_universe"]}
    assert len(inside) == 39
    assert sorted(row["rank"] for row in inside.values()) == list(range(1, 40))
    assert max(row["pct"] for row in inside.values()) == pytest.approx(1.0)
    best = min(inside.values(), key=lambda row: row["rank"])
    assert best["score"] == max(row["score"] for row in inside.values())
    assert rows[OUTSIDE]["rank"] is None
    assert rows[OUTSIDE]["pct"] is None
    assert rows[OUTSIDE]["score"] is not None
    assert rows[OUTSIDE]["in_universe"] is False
    assert all(row["ret_pred"] is None and row["q50"] is None for row in rows.values())
    frame = day_rows(DAY, served_runs, rank.feature_list)
    scores = run_values(rank, load_model(rank.run_key, rank.run_id), frame)["score"]
    np.testing.assert_allclose(
        [rows[stock]["score"] for stock in frame["stock_id"]], scores, rtol=1e-9
    )
    quantile = stored(live(Slot("quantile", 20)))
    assert all(row["ret_pred"] == row["q50"] for row in quantile.values())
    assert all(row["q10"] <= row["q50"] <= row["q90"] for row in quantile.values())
    assert all(row["score"] is None and row["vol_pred"] is None for row in quantile.values())
    regression = stored(live(Slot("regression", 5)))
    assert all(row["ret_pred"] is not None and row["q50"] is None for row in regression.values())
    assert all(row["vol_pred"] > 0 for row in stored(live(Slot("volatility", 20))).values())
    probability = stored(live(Slot("classification", 20))).values()
    assert all(0 <= row["prob_up_5pct"] <= 1 for row in probability)
    assert {row["horizon"] for row in rows.values()} == {20}
    for horizon in (5, 20):
        contrib = PredictionContrib.objects.filter(run=live(Slot("rank", horizon)), date=DAY)
        assert contrib.count() == 3 * 40
        assert set(contrib.values_list("horizon", flat=True)) == {horizon}
    assert not PredictionContrib.objects.exclude(run__objective="rank").exists()


@pytest.mark.django_db
@pytest.mark.parametrize("severity", [None, "error"])
def test_predict_waits_for_the_quality_gate(served_runs: Path, severity: str | None) -> None:
    serve(*SLOTS)
    if severity is not None:
        passed()
        DataQualityLog.objects.create(date=DAY, check_name="row_count", severity=severity)

    with pytest.raises(PredictionGateError, match=f"{DAY} has not passed the data quality gate"):
        predict(DAY, served_runs)

    assert not Prediction.objects.exists()


@pytest.mark.django_db
def test_predict_reruns_to_the_same_rows(served_runs: Path) -> None:
    serve(*SLOTS)
    passed()
    predict(DAY, served_runs)
    first = {run.run_id: stored(run) for run in ModelRun.objects.filter(status="active")}
    contrib = sorted(PredictionContrib.objects.values_list("stock_id", "feature", "contrib"))

    predict(DAY, served_runs)

    assert {run.run_id: stored(run) for run in ModelRun.objects.filter(status="active")} == first
    assert sorted(PredictionContrib.objects.values_list("stock_id", "feature", "contrib")) == (
        contrib
    )


def keep_rows(run: ModelRun, day: date, horizon: int, **values: float) -> None:
    Prediction.objects.create(
        run=run, stock_id="1100", date=day, horizon=horizon, in_universe=True, **values
    )


@pytest.mark.django_db
def test_predict_replaces_only_the_slot_rows_of_other_runs_on_the_day(served_runs: Path) -> None:
    ranked = serve(Slot("rank", 5), Slot("rank", 20))
    old = clone(live(Slot("rank", 20)), ".old", ModelStatus.RETIRED)
    interval = clone(live(Slot("quantile", 20)), ".old", ModelStatus.RETIRED)
    keep_rows(old, DAY, 20, score=0.5, rank=1, pct=1.0)
    keep_rows(old, EARLIER, 20, score=0.5, rank=1, pct=1.0)
    keep_rows(interval, DAY, 20, q10=-0.1, q50=0.0, q90=0.1, ret_pred=0.0)
    PredictionContrib.objects.create(
        run=old,
        stock_id="1100",
        date=DAY,
        horizon=20,
        feature=REST_FEATURE,
        position=1,
        contrib=0.1,
        n_features=3,
    )
    passed()

    result = predict(DAY, served_runs)

    assert result.written == {run.run_id: 40 for run in ranked}
    assert result.missing == [slot for slot in SLOTS if slot.objective != "rank"]
    assert not Prediction.objects.filter(run=old, date=DAY).exists()
    assert not PredictionContrib.objects.filter(run=old).exists()
    assert Prediction.objects.filter(run=old, date=EARLIER).exists()
    assert Prediction.objects.filter(run=interval, date=DAY).exists()


@pytest.mark.django_db
def test_predict_serves_the_active_run_then_the_degraded_one(served_runs: Path) -> None:
    passed()
    degraded = serve(Slot("rank", 20), status=ModelStatus.DEGRADED)[0]

    assert predict(DAY, served_runs).written == {degraded.run_id: 40}

    active = clone(live(Slot("rank", 20)), ".new", ModelStatus.ACTIVE)
    clone(live(Slot("rank", 20)), ".cand", ModelStatus.CANDIDATE)
    clone(live(Slot("rank", 20)), ".gone", ModelStatus.RETIRED)

    assert predict(DAY, served_runs).written == {active.run_id: 40}
    assert not Prediction.objects.filter(run=degraded, date=DAY).exists()


@pytest.mark.django_db
def test_predict_refuses_two_runs_of_one_status_in_a_slot(served_runs: Path) -> None:
    passed()
    serve(*SLOTS)
    twin = clone(live(Slot("rank", 20)), ".twin", ModelStatus.ACTIVE)

    with pytest.raises(RunError, match=rf"rank-live.*{twin.run_id}.*active.*rank 20"):
        predict(DAY, served_runs)

    assert not Prediction.objects.exists()


@pytest.mark.django_db
def test_predict_needs_a_serving_run(served_runs: Path) -> None:
    passed()

    with pytest.raises(RunError, match="no active or degraded run serves any slot"):
        predict(DAY, served_runs)


@pytest.mark.django_db
def test_a_failed_write_leaves_the_day_as_it_was(
    served_runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve(*SLOTS)
    old = clone(live(Slot("regression", 5)), ".old", ModelStatus.RETIRED)
    keep_rows(old, DAY, 5, ret_pred=0.01)
    passed()

    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("the database went away")

    monkeypatch.setattr(PredictionContrib.objects, "bulk_create", broken)
    with pytest.raises(RuntimeError, match="went away"):
        predict(DAY, served_runs)

    assert list(Prediction.objects.values_list("run_id", flat=True)) == [old.run_id]


@pytest.mark.django_db
def test_a_run_retired_while_the_day_is_scored_writes_nothing(
    served_runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rank = serve(*SLOTS)[1]
    passed()
    scored = predict_module.slot_rows

    def promoted(*args: Any) -> Any:
        ModelRun.objects.filter(pk=rank.pk).update(status=ModelStatus.RETIRED)
        return scored(*args)

    monkeypatch.setattr(predict_module, "slot_rows", promoted)
    with pytest.raises(RunError, match=f"{rank.run_id} changed status while predicting"):
        predict(DAY, served_runs)

    assert not Prediction.objects.exists()


@pytest.mark.django_db
def test_a_day_without_features_writes_nothing(served_runs: Path) -> None:
    serve(*SLOTS)
    passed(date(2016, 12, 31))

    with pytest.raises(ValueError, match="no feature rows on 2016-12-31"):
        predict(date(2016, 12, 31), served_runs)

    assert not Prediction.objects.exists()


@pytest.mark.django_db
def test_the_command_reports_each_run_and_the_missing_slots(
    served_runs: Path, settings: Any
) -> None:
    settings.FEATURE_PARQUET_DIR = served_runs
    ranked = serve(Slot("rank", 20))
    passed()
    out = StringIO()

    call_command("predict", date=DAY.isoformat(), stdout=out)

    text = out.getvalue()
    assert f"{ranked[0].run_id}: 40 rows" in text
    assert "no active or degraded run for regression 5 日" in text
    assert "no active or degraded run for rank 20 日" not in text


@pytest.mark.django_db
def test_the_command_reports_the_gate(served_runs: Path, settings: Any) -> None:
    settings.FEATURE_PARQUET_DIR = served_runs
    serve(*SLOTS)

    with pytest.raises(CommandError, match="data quality gate"):
        call_command("predict", date=DAY.isoformat())
