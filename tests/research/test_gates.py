import shutil
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.etl.models import DataQualityLog
from apps.research import gates
from apps.research.gates import FIRST_LAUNCH, Gate, SlotGates, check_gates
from apps.research.lgbm_runs import MODEL
from apps.research.models import HoldoutRun, ModelRun, ModelStatus, Prediction
from apps.research.predict import Slot
from apps.research.prediction_cache import load_fold_predictions
from apps.research.production import build_run
from apps.research.runs import RunError, model_path
from tests.research.run_panels import claim_all, weekdays

LAST = weekdays()[-1]


def built(scope: str, research_key: str, directory: Path) -> list[ModelRun]:
    if not HoldoutRun.objects.exists():
        claim_all()
    if not ModelRun.objects.filter(run_key=research_key).exists():
        ModelRun.objects.create(
            run_id=f"{research_key}.lightgbm.h5.f000",
            run_key=research_key,
            model_type="lightgbm",
            objective="regression",
            version="4",
            train_start=date(2016, 1, 4),
            train_end=date(2016, 6, 1),
            params={},
            feature_list=[],
            lib_versions={},
            feature_hash="0" * 64,
        )
    build_run(scope, f"{scope}-prod", research_key, LAST, directory)
    return list(ModelRun.objects.filter(run_key=f"{scope}-prod").order_by("run_id"))


def logged(run: ModelRun, severity: str = "info", day: date | None = None) -> None:
    DataQualityLog.objects.create(
        date=day or run.train_start, check_name="row_count", severity=severity, detail={}
    )


def by_slot(results: list[SlotGates]) -> dict[Slot, dict[str, Gate]]:
    return {result.slot: {gate.name: gate for gate in result.gates} for result in results}


def serve_copy(run: ModelRun, scores: dict[str, float] | None = None) -> ModelRun:
    """An active copy of N whose as_of rows hold N's own cached scores unless given others."""
    source = model_path(run.run_key, run.run_id)
    original = run.run_id
    run.run_id, run.run_key, run.status = (
        f"rank-old.{original.split('.', 1)[1]}",
        "rank-old",
        "active",
    )
    run.params = {key: value for key, value in run.params.items() if key != "production"}
    run.save(force_insert=True)
    model_path(run.run_key, run.run_id).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, model_path(run.run_key, run.run_id))
    cached = load_fold_predictions(
        "rank-prod", MODEL, gates._fold(ModelRun.objects.get(pk=original))
    )
    column = f"{MODEL}_h{run.params['horizon']}"
    today = cached[cached["date"] == LAST]
    Prediction.objects.bulk_create(
        Prediction(
            run=run,
            stock_id=row["stock_id"],
            date=LAST,
            horizon=run.params["horizon"],
            score=(scores or {}).get(row["stock_id"], row[column]),
            in_universe=bool(row["in_universe"]),
        )
        for row in today.to_dict(orient="records")
    )
    return run


@pytest.mark.django_db
def test_a_first_launch_skips_the_comparisons_and_judges_the_shadow(researched: Path) -> None:
    runs = built("rank", "rank-wf1", researched)
    logged(runs[0])

    results = by_slot(check_gates("rank-prod", researched))

    assert set(results) == {Slot("rank", 5), Slot("rank", 20)}
    for slot, found in results.items():
        assert found["agreement"].passed is None
        assert found["agreement"].reason == FIRST_LAUNCH
        assert found["gain_overlap"].passed is None
        shadow = ModelRun.objects.get(
            run_key="rank-prod-shadow", objective="rank", params__horizon=slot.horizon
        )
        assert found["shadow_oos"].value == pytest.approx(shadow.ic_mean)
        assert found["shadow_oos"].sample >= gates.MIN_DAYS
        assert found["data_quality"].passed is True
        assert found["data_quality"].sample == 1
    stored = ModelRun.objects.get(run_id=runs[0].run_id).params["gates"]
    assert stored["serving"] is None
    assert [gate["name"] for gate in stored["gates"]] == [
        "agreement",
        "gain_overlap",
        "shadow_oos",
        "data_quality",
    ]
    assert set(stored["gates"][2]) == {"name", "passed", "value", "threshold", "sample", "reason"}
    assert ModelRun.objects.filter(run_key="rank-prod", status="candidate").count() == 2


@pytest.mark.django_db
def test_the_comparisons_judge_n_against_the_serving_run(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gates, "GAIN_TOP", 2)
    monkeypatch.setattr(gates, "GAIN_OVERLAP", 2)
    runs = built("rank", "rank-wf1", researched)
    logged(runs[0])
    twenty = next(run for run in runs if run.params["horizon"] == 20)
    same = serve_copy(ModelRun.objects.get(pk=twenty.pk))

    found = by_slot(check_gates("rank-prod", researched))[Slot("rank", 20)]

    assert found["agreement"].passed is True
    assert found["agreement"].value == pytest.approx(1.0)
    assert found["agreement"].sample == 39
    assert found["gain_overlap"].passed is True
    assert found["gain_overlap"].value == 2
    stored = ModelRun.objects.get(pk=twenty.pk).params["gates"]
    assert stored["serving"] == same.run_id

    scores = dict(Prediction.objects.filter(run=same).values_list("stock_id", "score"))
    Prediction.objects.filter(run=same).delete()
    ModelRun.objects.filter(pk=same.pk).delete()
    flipped = serve_copy(
        ModelRun.objects.get(pk=twenty.pk),
        {stock: -(value or 0.0) for stock, value in scores.items()},
    )
    ModelRun.objects.filter(pk=flipped.pk).update(
        params={**flipped.params, "importance_gain": {"x": 2.0, "y": 1.0}}
    )

    found = by_slot(check_gates("rank-prod", researched))[Slot("rank", 20)]

    assert found["agreement"].passed is False
    assert found["agreement"].value == pytest.approx(-1.0)
    assert found["gain_overlap"].passed is False
    assert found["gain_overlap"].value == 0


@pytest.mark.django_db
@pytest.mark.parametrize("case", ["no rows", "few stocks", "no gains"])
def test_the_comparisons_fail_without_data(researched: Path, case: str) -> None:
    runs = built("rank", "rank-wf1", researched)
    twenty = next(run for run in runs if run.params["horizon"] == 20)
    serving = serve_copy(ModelRun.objects.get(pk=twenty.pk))
    if case == "no rows":
        Prediction.objects.filter(run=serving).delete()
    if case == "few stocks":
        inside = Prediction.objects.filter(run=serving, in_universe=True)
        kept = sorted(inside.values_list("stock_id", flat=True))[:20]
        Prediction.objects.filter(run=serving).exclude(stock_id__in=kept).delete()
    if case == "no gains":
        ModelRun.objects.filter(pk=serving.pk).update(params={"horizon": 20})

    found = by_slot(check_gates("rank-prod", researched))[Slot("rank", 20)]

    if case == "no gains":
        assert found["gain_overlap"].passed is False
        assert "沒有 importance_gain" in found["gain_overlap"].reason
    else:
        assert found["agreement"].passed is False
        assert found["agreement"].value is None
        assert found["agreement"].sample == (0 if case == "no rows" else 20)


@pytest.mark.django_db
def test_the_shadow_gate_fails_without_enough_days_or_a_shadow(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built("rank", "rank-wf1", researched)
    monkeypatch.setattr(gates, "MIN_DAYS", 500)

    found = by_slot(check_gates("rank-prod", researched))[Slot("rank", 5)]["shadow_oos"]

    assert found.passed is False
    assert "少於 500" in found.reason
    assert found.sample > 0

    ModelRun.objects.filter(run_key="rank-prod-shadow").delete()
    found = by_slot(check_gates("rank-prod", researched))[Slot("rank", 5)]["shadow_oos"]

    assert found.passed is False
    assert found.sample == 0
    assert "沒有對應的 N′" in found.reason


@pytest.mark.django_db
def test_a_shadow_ranking_backwards_fails(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built("rank", "rank-wf1", researched)
    ranked = gates.daily_rank_ic
    monkeypatch.setattr(gates, "daily_rank_ic", lambda rows, column: -ranked(rows, column))

    found = by_slot(check_gates("rank-prod", researched))[Slot("rank", 20)]["shadow_oos"]

    assert found.value is not None
    assert found.value < 0
    assert found.passed is False
    assert found.reason == "未達門檻"


@pytest.mark.django_db
def test_the_quality_gate_lists_errors_and_needs_some_checks(researched: Path) -> None:
    runs = built("rank", "rank-wf1", researched)
    run = runs[0]

    empty = by_slot(check_gates("rank-prod", researched))[Slot("rank", 20)]["data_quality"]

    assert empty.passed is False
    assert empty.sample == 0
    assert "沒有任何品質檢查紀錄" in empty.reason

    logged(run)
    logged(run, "error", run.train_end)
    DataQualityLog.objects.create(
        date=run.train_end, check_name="etl_runs", severity="error", detail={}
    )
    logged(run, "error", LAST)

    found = by_slot(check_gates("rank-prod", researched))[Slot("rank", 20)]["data_quality"]

    assert found.passed is False
    assert found.value == 2
    assert found.reason == f"{run.train_end} etl_runs, {run.train_end} row_count"
    assert found.sample == 2


@pytest.mark.django_db
def test_the_target_shadow_gates_reuse_the_target_evaluation(researched: Path) -> None:
    for run in [*built("interval", "interval-wf1", researched)]:
        logged(run)
    built("return", "return-wf1", researched)

    results = by_slot(
        [*check_gates("interval-prod", researched), *check_gates("return-prod", researched)]
    )

    first = ModelRun.objects.filter(run_key="interval-prod-shadow").first()
    assert first is not None
    shadow = gates._fold(first)
    frame = gates.target_eval.target_frame("interval-prod-shadow", [shadow], researched)
    evaluated = gates.target_eval.evaluate_interval(frame, [shadow])
    quantile = results[Slot("quantile", 20)]["shadow_oos"]
    assert quantile.value == pytest.approx(evaluated.coverage)
    assert quantile.sample == evaluated.coverage_days > 0
    assert quantile.passed is (0.70 <= evaluated.coverage <= 0.90)
    volatility = results[Slot("volatility", 20)]["shadow_oos"]
    assert volatility.value == pytest.approx(evaluated.qlike.model)
    assert volatility.passed is (evaluated.qlike.model <= evaluated.qlike.baseline)
    probability = results[Slot("classification", 20)]["shadow_oos"]
    assert probability.value == pytest.approx(evaluated.brier.model)
    assert probability.sample == evaluated.brier.days > 0
    returned = results[Slot("regression", 5)]["shadow_oos"]
    folded = ModelRun.objects.filter(run_key="return-prod-shadow").first()
    assert folded is not None
    mae = gates.target_eval.evaluate_return(
        gates.target_eval.target_frame("return-prod-shadow", [gates._fold(folded)], researched),
        [gates._fold(folded)],
    ).mae["zero"]
    assert returned.value == pytest.approx(mae.model)
    assert returned.passed is (mae.model <= mae.baseline)
    assert returned.sample == mae.days > 0
    assert "零預測" in returned.threshold


@pytest.mark.django_db
def test_the_target_shadow_gates_fail_without_enough_days(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built("interval", "interval-wf1", researched)
    monkeypatch.setattr(gates, "MIN_DAYS", 500)

    results = by_slot(check_gates("interval-prod", researched))

    for objective in ("quantile", "volatility", "classification"):
        assert results[Slot(objective, 20)]["shadow_oos"].passed is False
        assert "少於 500" in results[Slot(objective, 20)]["shadow_oos"].reason


@pytest.mark.django_db
def test_gates_check_production_builds_only(researched: Path) -> None:
    built("rank", "rank-wf1", researched)

    for key in ("rank-prod-shadow", "rank-wf1"):
        with pytest.raises(RunError, match=f"{key} holds no production model N"):
            check_gates(key, researched)


@pytest.mark.django_db
def test_the_command_prints_each_gate(researched: Path, settings: Any) -> None:
    runs = built("rank", "rank-wf1", researched)
    logged(runs[0])
    settings.FEATURE_PARQUET_DIR = researched
    out = StringIO()

    call_command("check_gates", run_key="rank-prod", stdout=out)

    text = out.getvalue()
    assert "rank 20 日 rank-prod.lightgbm.h20.f000（首次上線）" in text
    assert f"agreement：{FIRST_LAUNCH}" in text
    assert "data_quality：通過" in text
    with pytest.raises(CommandError, match="holds no production model N"):
        call_command("check_gates", run_key="rank-prod-shadow")
    assert ModelRun.objects.filter(status=ModelStatus.ACTIVE).count() == 0
