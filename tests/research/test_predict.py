import shutil
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.etl import tasks
from apps.etl.models import DataQualityLog, EtlRun
from apps.etl.quality import CHECKS
from apps.market.calendar import TradingDays
from apps.research import predict as predict_module
from apps.research.inference import day_rows, run_values, top_contributions
from apps.research.lgbm_runs import run_id
from apps.research.models import (
    REST_FEATURE,
    ModelMonitor,
    ModelRun,
    ModelStatus,
    Prediction,
    PredictionContrib,
)
from apps.research.predict import FIELDS, SLOTS, PredictionGateError, Slot, predict
from apps.research.runs import SCREENING_MODEL, RunError, load_model, model_path
from tests.research.run_panels import research
from tests.research.test_inference import boosters

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
OWNED = {
    "rank": {"score", "rank", "pct"},
    "regression": {"ret_pred"},
    "quantile": {"q10", "q50", "q90", "ret_pred"},
    "volatility": {"vol_pred"},
    "classification": {"prob_up_5pct"},
}
GATE_JOBS = {
    (job.source, job.dataset)
    for jobs in tasks.BATCHES.values()
    for job in jobs
    if job.dataset in tasks.GATE_DATASETS
}


def live(slot: Slot) -> ModelRun:
    scope, horizon = LIVE_RUNS[slot]
    return ModelRun.objects.get(run_id=run_id(f"{scope}-live", slot.objective, horizon, LIVE))


def serve(*slots: Slot, status: str = ModelStatus.ACTIVE) -> list[ModelRun]:
    runs = [live(slot) for slot in slots]
    ModelRun.objects.filter(pk__in=[run.pk for run in runs]).update(status=status)
    return runs


def landed(day: date = DAY) -> None:
    for source, dataset in GATE_JOBS:
        EtlRun.objects.create(
            source=source, dataset=dataset, start_date=day, end_date=day, status="success"
        )


def checked(day: date = DAY) -> None:
    for name in CHECKS:
        DataQualityLog.objects.create(date=day, check_name=name, severity="info", detail={})


def passed(day: date = DAY) -> None:
    landed(day)
    checked(day)


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


def keep_rows(run: ModelRun, day: date, horizon: int, **values: float) -> None:
    Prediction.objects.create(
        run=run, stock_id="1100", date=day, horizon=horizon, in_universe=True, **values
    )


@pytest.mark.django_db
def test_predict_writes_every_slot_for_every_stock_with_features(served_runs: Path) -> None:
    runs = serve(*SLOTS)
    passed()

    result = predict(DAY, served_runs)

    assert result.written == {run.run_id: 40 for run in runs}
    assert result.missing == []
    for slot in SLOTS:
        rows = Prediction.objects.filter(run=live(slot), date=DAY)
        assert rows.count() == 40
        assert set(rows.values_list("horizon", flat=True)) == {slot.horizon}
        for name in set(FIELDS) - OWNED[slot.objective]:
            assert not rows.filter(**{f"{name}__isnull": False}).exists()
        for name in OWNED[slot.objective] - {"rank", "pct"}:
            assert not rows.filter(**{f"{name}__isnull": True}).exists()
    rank = live(Slot("rank", 20))
    ranked = stored(rank)
    inside = {stock: row for stock, row in ranked.items() if row["in_universe"]}
    assert len(inside) == 39
    assert sorted(row["rank"] for row in inside.values()) == list(range(1, 40))
    assert max(row["pct"] for row in inside.values()) == pytest.approx(1.0)
    best = min(inside.values(), key=lambda row: row["rank"])
    assert best["score"] == max(row["score"] for row in inside.values())
    assert ranked[OUTSIDE]["rank"] is None
    assert ranked[OUTSIDE]["pct"] is None
    assert ranked[OUTSIDE]["score"] is not None
    assert ranked[OUTSIDE]["in_universe"] is False
    frame = day_rows(DAY, served_runs, rank.feature_list)
    scores = run_values(rank, load_model(rank.run_key, rank.run_id), frame)["score"]
    np.testing.assert_allclose(
        [ranked[stock]["score"] for stock in frame["stock_id"]], scores, rtol=1e-9
    )
    quantile = stored(live(Slot("quantile", 20))).values()
    assert all(row["ret_pred"] == row["q50"] for row in quantile)
    assert all(row["q10"] <= row["q50"] <= row["q90"] for row in quantile)
    assert all(row["vol_pred"] > 0 for row in stored(live(Slot("volatility", 20))).values())
    probability = stored(live(Slot("classification", 20))).values()
    assert all(0 <= row["prob_up_5pct"] <= 1 for row in probability)
    for horizon in (5, 20):
        contrib = PredictionContrib.objects.filter(run=live(Slot("rank", horizon)), date=DAY)
        assert contrib.count() == 3 * 40
        assert set(contrib.values_list("horizon", flat=True)) == {horizon}
    assert not PredictionContrib.objects.exclude(run__objective="rank").exists()


@pytest.mark.django_db
@pytest.mark.parametrize("gate", ["nothing", "no snapshots", "an error"])
def test_predict_waits_for_the_research_gate(served_runs: Path, gate: str) -> None:
    serve(*SLOTS)
    if gate == "no snapshots":
        checked()
    if gate == "an error":
        passed()
        DataQualityLog.objects.create(date=DAY, check_name="row_count", severity="error")

    with pytest.raises(PredictionGateError, match=f"{DAY} has not passed the research gate"):
        predict(DAY, served_runs)

    assert not Prediction.objects.exists()


@pytest.mark.django_db
def test_predict_reruns_to_the_same_rows(served_runs: Path) -> None:
    serve(*SLOTS)
    passed()
    predict(DAY, served_runs)
    first = {run.run_id: stored(run) for run in ModelRun.objects.filter(status="active")}
    contrib = sorted(PredictionContrib.objects.values_list("stock_id", "feature", "contrib"))
    rank = live(Slot("rank", 20))
    Prediction.objects.filter(run=rank, stock_id="1100", date=DAY).update(score=9.0)
    PredictionContrib.objects.filter(run=rank, stock_id="1100", date=DAY).update(contrib=9.0)

    predict(DAY, served_runs)

    assert {run.run_id: stored(run) for run in ModelRun.objects.filter(status="active")} == first
    assert sorted(PredictionContrib.objects.values_list("stock_id", "feature", "contrib")) == (
        contrib
    )


@pytest.mark.django_db
def test_rows_of_other_runs_on_the_day_are_replaced_only_on_purpose(served_runs: Path) -> None:
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

    with pytest.raises(RunError, match=rf"{DAY} already holds rows of {old.run_id}"):
        predict(DAY, served_runs)
    assert Prediction.objects.count() == 3

    result = predict(DAY, served_runs, replace=True)

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

    assert predict(DAY, served_runs, replace=True).written == {active.run_id: 40}
    assert not Prediction.objects.filter(run=degraded, date=DAY).exists()


@pytest.mark.django_db
def test_predict_leaves_the_feature_screening_out(served_runs: Path) -> None:
    passed()
    serve(Slot("rank", 20))
    ModelRun.objects.filter(model_type=SCREENING_MODEL).update(
        status=ModelStatus.ACTIVE, params={"horizon": 20}
    )

    assert predict(DAY, served_runs).written == {live(Slot("rank", 20)).run_id: 40}


@pytest.mark.django_db
@pytest.mark.parametrize("status", [ModelStatus.ACTIVE, ModelStatus.DEGRADED])
def test_predict_refuses_two_runs_of_one_status_in_a_slot(served_runs: Path, status: str) -> None:
    passed()
    serve(*SLOTS)
    serve(Slot("rank", 20), status=status)
    twin = clone(live(Slot("rank", 20)), ".twin", status)

    with pytest.raises(RunError, match=rf"rank-live.*{twin.run_id} are all {status} for rank 20"):
        predict(DAY, served_runs)

    assert not Prediction.objects.exists()


@pytest.mark.django_db
def test_predict_needs_a_serving_run(served_runs: Path) -> None:
    passed()

    with pytest.raises(RunError, match="no active or degraded run serves any slot"):
        predict(DAY, served_runs)


def refused_runs(error: pytest.ExceptionInfo[RunError]) -> set[str]:
    listed = str(error.value).split(" trained on labels")[0]
    return set(listed.split(", "))


@pytest.mark.django_db
def test_predict_refuses_days_whose_labels_overlap_a_runs_training(served_runs: Path) -> None:
    serve(*SLOTS)
    day = TradingDays.from_db().shift(LIVE.train_end, 10)
    passed(day)

    with pytest.raises(RunError, match="it would score in-sample") as error:
        predict(day, served_runs)

    assert refused_runs(error) == {live(slot).run_id for slot in SLOTS if slot.horizon == 20}
    assert not Prediction.objects.exists()


@pytest.mark.django_db
def test_any_one_run_trained_too_late_stops_the_day(served_runs: Path) -> None:
    serve(*SLOTS)
    late = live(Slot("volatility", 20))
    ModelRun.objects.filter(pk=late.pk).update(train_end=DAY)
    passed()

    with pytest.raises(RunError, match="it would score in-sample") as error:
        predict(DAY, served_runs)

    assert refused_runs(error) == {late.run_id}
    assert not Prediction.objects.exists()


@pytest.mark.django_db
def test_replace_leaves_a_day_the_monitor_already_scored(served_runs: Path) -> None:
    serve(Slot("rank", 20))
    old = clone(live(Slot("rank", 20)), ".old", ModelStatus.RETIRED)
    keep_rows(old, DAY, 20, score=0.5, rank=1, pct=1.0)
    ModelMonitor.objects.create(run=old, date=DAY, horizon=20, ic=0.1, n_stocks=39)
    passed()

    with pytest.raises(RunError, match=rf"model_monitor already scored {old.run_id} on {DAY}"):
        predict(DAY, served_runs, replace=True)

    assert list(Prediction.objects.values_list("run_id", flat=True)) == [old.run_id]


@pytest.mark.django_db
def test_a_failed_write_leaves_the_day_as_it_was(
    served_runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve(*SLOTS)
    old = clone(live(Slot("rank", 5)), ".old", ModelStatus.RETIRED)
    keep_rows(old, DAY, 5, score=0.5, rank=1, pct=1.0)
    passed()
    calls = []
    write = PredictionContrib.objects.bulk_create

    def broken(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("the database went away")
        return write(*args, **kwargs)

    monkeypatch.setattr(PredictionContrib.objects, "bulk_create", broken)
    with pytest.raises(RuntimeError, match="went away"):
        predict(DAY, served_runs, replace=True)

    assert len(calls) == 2
    assert list(Prediction.objects.values_list("run_id", flat=True)) == [old.run_id]
    assert not PredictionContrib.objects.exists()


@pytest.mark.django_db
def test_a_run_retired_while_the_day_is_scored_writes_nothing(
    served_runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rank = serve(*SLOTS)[1]
    passed()
    scored = predict_module.slot_rows

    def retired(*args: Any) -> Any:
        ModelRun.objects.filter(pk=rank.pk).update(status=ModelStatus.RETIRED)
        return scored(*args)

    monkeypatch.setattr(predict_module, "slot_rows", retired)
    with pytest.raises(RunError, match="serving run of rank 20 日 changed while predicting"):
        predict(DAY, served_runs)

    assert not Prediction.objects.exists()


@pytest.mark.django_db
def test_a_run_promoted_while_the_day_is_scored_writes_nothing(
    served_runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve(Slot("rank", 20), status=ModelStatus.DEGRADED)
    newer = clone(live(Slot("rank", 20)), ".new", ModelStatus.CANDIDATE)
    passed()
    scored = predict_module.slot_rows

    def promoted(*args: Any) -> Any:
        ModelRun.objects.filter(pk=newer.pk).update(status=ModelStatus.ACTIVE)
        return scored(*args)

    monkeypatch.setattr(predict_module, "slot_rows", promoted)
    with pytest.raises(RunError, match="serving run of rank 20 日 changed while predicting"):
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
def test_contributions_beyond_the_top_five_are_stored_as_the_rest(served_runs: Path) -> None:
    features = [f"f{k}" for k in range(8)]
    model = boosters(features)
    inputs = pd.DataFrame(np.random.default_rng(10).normal(size=(3, 8)), columns=features)
    rank = live(Slot("rank", 20))
    rows = top_contributions(model.boosters, inputs, pd.Series(["1100", "1101", "1102"]))

    PredictionContrib.objects.bulk_create(
        PredictionContrib(
            run=rank,
            date=DAY,
            horizon=20,
            stock_id=row["stock_id"],
            feature=row["feature"],
            position=int(row["position"]),
            contrib=float(row["contrib"]),
            n_features=int(row["n_features"]),
        )
        for row in rows.to_dict(orient="records")
    )

    rest = PredictionContrib.objects.filter(run=rank, feature=REST_FEATURE)
    assert rest.count() == 3
    assert set(rest.values_list("position", "n_features")) == {(6, 3)}
    assert PredictionContrib.objects.filter(run=rank).count() == 3 * 6


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
def test_the_command_replaces_other_runs_only_when_asked(served_runs: Path, settings: Any) -> None:
    settings.FEATURE_PARQUET_DIR = served_runs
    serve(Slot("rank", 20))
    old = clone(live(Slot("rank", 20)), ".old", ModelStatus.RETIRED)
    keep_rows(old, DAY, 20, score=0.5, rank=1, pct=1.0)
    passed()

    with pytest.raises(CommandError, match="already holds rows"):
        call_command("predict", date=DAY.isoformat())
    call_command("predict", date=DAY.isoformat(), replace=True, stdout=StringIO())

    assert not Prediction.objects.filter(run=old).exists()


@pytest.mark.django_db
def test_the_command_reports_its_errors(served_runs: Path, settings: Any) -> None:
    settings.FEATURE_PARQUET_DIR = served_runs
    serve(*SLOTS)

    with pytest.raises(CommandError, match="research gate"):
        call_command("predict", date=DAY.isoformat())
    with pytest.raises(CommandError, match="daily_price is empty"):
        call_command("predict")
    passed()
    clone(live(Slot("rank", 20)), ".twin", ModelStatus.ACTIVE)
    with pytest.raises(CommandError, match="are all active for rank 20"):
        call_command("predict", date=DAY.isoformat())


@pytest.mark.django_db
def test_predict_writes_only_the_slots_asked_for(
    served_runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = serve(*SLOTS)
    passed()
    asked = [Slot("rank", 5), Slot("quantile", 20)]
    elsewhere = live(Slot("volatility", 20))
    scored = predict_module.slot_rows

    def retired(*args: Any) -> Any:
        ModelRun.objects.filter(pk=elsewhere.pk).update(status=ModelStatus.RETIRED)
        return scored(*args)

    monkeypatch.setattr(predict_module, "slot_rows", retired)
    result = predict(DAY, served_runs, slots=asked)

    assert result.written == {live(slot).run_id: 40 for slot in asked}
    assert result.missing == []
    assert set(Prediction.objects.values_list("run_id", flat=True)) == {
        live(slot).run_id for slot in asked
    }
    assert len(runs) == len(SLOTS)


@pytest.mark.django_db
def test_predict_reports_a_missing_slot_among_those_asked_for(served_runs: Path) -> None:
    serve(Slot("rank", 5))
    passed()

    result = predict(DAY, served_runs, slots=[Slot("rank", 5), Slot("rank", 20)])

    assert result.missing == [Slot("rank", 20)]
    assert set(result.written) == {live(Slot("rank", 5)).run_id}
