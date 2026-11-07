import shutil
from dataclasses import replace
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.etl.models import DataQualityLog
from apps.research import gates
from apps.research.baseline_runs import BASELINE
from apps.research.gates import NO_SERVING, Gate, SlotGates, check_gates
from apps.research.lgbm_runs import MODEL
from apps.research.models import (
    HoldoutRun,
    LabelDaily,
    ModelObjective,
    ModelRun,
    ModelStatus,
    Prediction,
)
from apps.research.predict import Slot
from apps.research.prediction_cache import fold_path, load_fold_predictions
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


def serve_copy(run: ModelRun, values: dict[str, float] | None = None) -> ModelRun:
    """An active copy of N whose as_of rows hold N's own cached values unless given others."""
    source, original, prod = model_path(run.run_key, run.run_id), run.run_id, run.run_key
    column, name = gates._column(run), gates.FIELDS[ModelObjective(run.objective)]
    old = prod.replace("-prod", "-old")
    run.run_id, run.run_key, run.status = f"{old}.{original.split('.', 1)[1]}", old, "active"
    run.params = {key: value for key, value in run.params.items() if key != "production"}
    run.save(force_insert=True)
    model_path(run.run_key, run.run_id).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, model_path(run.run_key, run.run_id))
    cached = load_fold_predictions(prod, MODEL, gates._fold(ModelRun.objects.get(pk=original)))
    today = cached[(cached["date"] == LAST) & cached[column].notna()]
    Prediction.objects.bulk_create(
        Prediction(
            run=run,
            stock_id=row["stock_id"],
            date=LAST,
            horizon=run.params["horizon"],
            in_universe=bool(row["in_universe"]),
            **{name: float((values or {}).get(row["stock_id"], row[column]))},
        )
        for row in today.to_dict(orient="records")
    )
    return run


@pytest.mark.django_db
def test_a_first_launch_skips_the_comparisons_and_judges_the_shadow(researched: Path) -> None:
    LabelDaily.objects.filter(stock_id="1100").update(untradable_t1=True)
    runs = built("rank", "rank-wf1", researched)
    logged(runs[0])

    results = by_slot(check_gates("rank-prod", researched))

    assert set(results) == {Slot("rank", 5), Slot("rank", 20)}
    for slot, found in results.items():
        assert found["agreement"].passed is None
        assert found["agreement"].reason == NO_SERVING
        assert found["gain_overlap"].passed is None
        shadow = ModelRun.objects.get(
            run_key="rank-prod-shadow", objective="rank", params__horizon=slot.horizon
        )
        assert found["shadow_oos"].value == pytest.approx(shadow.ic_mean)
        assert found["shadow_oos"].sample >= gates.MIN_DAYS
        assert found["shadow_oos"].passed is True
        assert found["data_quality"].passed is True
        assert found["data_quality"].sample == 1
    stored = ModelRun.objects.get(run_id=runs[0].run_id).params["gates"]
    assert stored["serving"] is None
    assert stored["passed"] is True
    assert [gate["name"] for gate in stored["gates"]] == [
        "agreement",
        "gain_overlap",
        "shadow_oos",
        "data_quality",
    ]
    assert set(stored["gates"][2]) == {"name", "passed", "value", "threshold", "sample", "reason"}
    assert stored["gates"][3]["errors"] == []
    assert stored["gates"][3]["errors_total"] == 0
    assert set(gates.SAMPLE_UNITS) == {gate["name"] for gate in stored["gates"]}
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
    restored = ModelRun.objects.get(pk=twenty.pk).params["gates"]
    assert restored["serving"] == flipped.run_id
    assert restored["passed"] is False


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
def test_the_shadow_gate_fails_without_enough_days(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built("rank", "rank-wf1", researched)
    monkeypatch.setattr(gates, "MIN_DAYS", 500)

    found = by_slot(check_gates("rank-prod", researched))[Slot("rank", 5)]["shadow_oos"]

    assert found.passed is False
    assert "少於 500" in found.reason
    assert found.sample > 0


def other_shadow(run_key: str, **changed: str) -> None:
    for run in ModelRun.objects.filter(run_key=f"{run_key}-shadow"):
        note = {**run.params["production"], **changed}
        ModelRun.objects.filter(pk=run.pk).update(params={**run.params, "production": note})


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("scope", "case", "match"),
    [
        ("rank", "no cache", r"rank-prod-shadow has no cached fold 0"),
        ("rank", "no shadow", r"rank-prod-shadow holds no shadow N′ for rank"),
        ("interval", "other day", r"is the N′ of interval-prod as of 2016-11-30"),
        ("interval", "other build", r"is the N′ of interval-old as of"),
        ("interval", "no baselines", r"interval-prod-shadow has no baseline cache"),
    ],
)
def test_gates_refuse_a_build_whose_shadow_is_incomplete(
    researched: Path, scope: str, case: str, match: str
) -> None:
    runs = built(scope, f"{scope}-wf1", researched)
    logged(runs[0])
    key = f"{scope}-prod"
    if case == "no cache":
        fold_path(f"{key}-shadow", MODEL, 0).unlink()
    if case == "no shadow":
        ModelRun.objects.filter(run_key=f"{key}-shadow").delete()
    if case == "other day":
        other_shadow(key, as_of="2016-11-30")
    if case == "other build":
        other_shadow(key, shadow_of="interval-old")
    if case == "no baselines":
        fold_path(f"{key}-shadow", BASELINE, 0).unlink()

    with pytest.raises(RunError, match=match):
        check_gates(key, researched)

    assert not ModelRun.objects.filter(run_key=key, params__has_key="gates").exists()


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
    logged(run, "warning", run.train_end)
    logged(run, "error", run.train_start - timedelta(days=1))
    DataQualityLog.objects.create(
        date=run.train_end, check_name="etl_runs", severity="error", detail={}
    )
    logged(run, "error", LAST)
    logged(run, "error", LAST + timedelta(days=1))

    found = by_slot(check_gates("rank-prod", researched))[Slot("rank", 20)]["data_quality"]

    assert found.passed is False
    assert found.value == 3
    assert found.reason == f"2 天有 error，最近 {LAST} row_count"
    assert found.sample == 3
    assert found.extra == {
        "errors": [
            {"date": LAST.isoformat(), "check": "row_count"},
            {"date": run.train_end.isoformat(), "check": "etl_runs"},
            {"date": run.train_end.isoformat(), "check": "row_count"},
        ],
        "errors_total": 3,
    }


@pytest.mark.django_db
def test_the_quality_gate_names_every_check_of_its_latest_error_day_and_caps_the_list(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = built("rank", "rank-wf1", researched)
    run = runs[0]
    twenty = next(item for item in runs if item.params["horizon"] == 20)
    logged(run)
    logged(run, "error", run.train_end)
    for name in ("row_count", "missing_values", "etl_runs"):
        DataQualityLog.objects.create(date=LAST, check_name=name, severity="error", detail={})
    monkeypatch.setattr(gates, "ERROR_LIMIT", 2)

    found = by_slot(check_gates("rank-prod", researched))[Slot("rank", 20)]["data_quality"]

    assert found.reason == f"2 天有 error，最近 {LAST} etl_runs、missing_values、row_count"
    assert found.value == 4
    assert found.extra == {
        "errors": [
            {"date": LAST.isoformat(), "check": "etl_runs"},
            {"date": LAST.isoformat(), "check": "missing_values"},
        ],
        "errors_total": 4,
    }
    stored = ModelRun.objects.get(pk=twenty.pk).params["gates"]["gates"]
    quality = next(gate for gate in stored if gate["name"] == "data_quality")
    assert quality["errors"] == found.extra["errors"]
    assert quality["errors_total"] == 4
    assert all("errors" not in gate for gate in stored if gate["name"] != "data_quality")


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
    assert probability.passed is (evaluated.brier.model <= evaluated.brier.baseline)
    interval = load_fold_predictions("interval-prod", MODEL, gates._fold(built_n("interval-prod")))
    returns = load_fold_predictions("return-prod", MODEL, gates._fold(built_n("return-prod")))
    assert set(gates.COLUMNS.values()) <= {*interval.columns, *returns.columns}
    assert set(gates.FIELDS.values()) <= {item.name for item in Prediction._meta.get_fields()}
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
    built("return", "return-wf1", researched)
    monkeypatch.setattr(gates, "MIN_DAYS", 500)

    results = by_slot(
        [*check_gates("interval-prod", researched), *check_gates("return-prod", researched)]
    )

    for slot in (Slot("quantile", 20), Slot("volatility", 20), Slot("classification", 20)):
        assert results[slot]["shadow_oos"].passed is False
        assert "少於 500" in results[slot]["shadow_oos"].reason
    assert results[Slot("regression", 5)]["shadow_oos"].passed is False
    assert "少於 500" in results[Slot("regression", 5)]["shadow_oos"].reason


@pytest.mark.django_db
def test_gates_check_production_builds_only(researched: Path) -> None:
    built("rank", "rank-wf1", researched)

    for key in ("rank-prod-shadow", "rank-wf1"):
        with pytest.raises(RunError, match=f"{key} holds no production model N"):
            check_gates(key, researched)
    ModelRun.objects.filter(run_id="rank-prod.lightgbm.h5.f000").update(status="active")
    with pytest.raises(RunError, match=r"rank-prod\.lightgbm\.h5\.f000 left candidate"):
        check_gates("rank-prod", researched)
    assert not ModelRun.objects.filter(run_key="rank-prod", params__has_key="gates").exists()
    ModelRun.objects.filter(run_key="rank-prod").update(status="candidate")
    fold_path("rank-prod", MODEL, 0).unlink()
    with pytest.raises(RunError, match="rank-prod has no cached fold 0"):
        check_gates("rank-prod", researched)


@pytest.mark.django_db
def test_the_command_prints_each_gate(researched: Path, settings: Any) -> None:
    runs = built("rank", "rank-wf1", researched)
    logged(runs[0])
    settings.FEATURE_PARQUET_DIR = researched
    out = StringIO()

    call_command("check_gates", run_key="rank-prod", stdout=out)

    text = out.getvalue()
    assert "rank 20 日 rank-prod.lightgbm.h20.f000（沒有服務中的 run）" in text
    assert f"agreement：{NO_SERVING}" in text
    assert "data_quality：通過" in text
    assert text.strip().endswith("全部通過")
    with pytest.raises(CommandError, match="holds no production model N"):
        call_command("check_gates", run_key="rank-prod-shadow")
    assert ModelRun.objects.filter(status=ModelStatus.ACTIVE).count() == 0


def built_n(run_key: str) -> ModelRun:
    run = ModelRun.objects.filter(run_key=run_key).first()
    assert run is not None
    return run


def gate(passed: bool | None) -> Gate:
    return Gate("agreement", passed, None, "", 0, "")


def test_a_slot_passes_unless_some_gate_fails() -> None:
    slot = Slot("rank", 5)

    assert SlotGates(slot, "n", None, [gate(None), gate(True)]).passed is True
    assert SlotGates(slot, "n", None, [gate(None), gate(True), gate(False)]).passed is False
    assert SlotGates(slot, "n", None, [gate(None), gate(None)]).passed is True


def ranked(gains: dict[str, float]) -> ModelRun:
    return ModelRun(run_id="x", params={"importance_gain": gains})


def test_gain_overlap_counts_the_largest_gains_only() -> None:
    small = {f"b{k}": 1.0 for k in range(10)}
    mine = ranked({**{f"a{k}": 10.0 for k in range(10)}, **small})
    apart = ranked({**{f"c{k}": 10.0 for k in range(10)}, **small})
    six = ranked(
        {**{f"a{k}": 10.0 for k in range(6)}, **{f"c{k}": 10.0 for k in range(4)}, **small}
    )
    five = ranked(
        {**{f"a{k}": 10.0 for k in range(5)}, **{f"c{k}": 10.0 for k in range(5)}, **small}
    )

    assert gains_shared(mine, apart) == (0.0, False)
    assert gains_shared(mine, six) == (6.0, True)
    assert gains_shared(mine, five) == (5.0, False)


def test_gain_overlap_leaves_unused_features_out() -> None:
    unused = {f"z{k}": 0.0 for k in range(10)}
    mine = ranked({**{f"a{k}": 10.0 for k in range(3)}, **unused})
    theirs = ranked({**{f"b{k}": 10.0 for k in range(3)}, **unused})

    found = gates.gain_overlap(mine, theirs)

    assert (found.value, found.passed, found.sample) == (0.0, False, 3)


def gains_shared(mine: ModelRun, theirs: ModelRun) -> tuple[float | None, bool | None]:
    found = gates.gain_overlap(mine, theirs)
    return found.value, found.passed


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("scope", "slot", "field", "value", "passed"),
    [
        ("interval", Slot("quantile", 20), "coverage", 0.69, False),
        ("interval", Slot("quantile", 20), "coverage", 0.70, True),
        ("interval", Slot("quantile", 20), "coverage", 0.90, True),
        ("interval", Slot("quantile", 20), "coverage", 0.91, False),
        ("interval", Slot("volatility", 20), "qlike", 0.0, True),
        ("interval", Slot("volatility", 20), "qlike", 0.01, False),
        ("interval", Slot("classification", 20), "brier", 0.0, True),
        ("interval", Slot("classification", 20), "brier", 0.01, False),
        ("return", Slot("regression", 5), "mae", 0.0, True),
        ("return", Slot("regression", 5), "mae", 0.01, False),
    ],
)
def test_the_target_shadow_gates_hold_their_bounds(
    researched: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
    slot: Slot,
    field: str,
    value: float,
    passed: bool,
) -> None:
    built(scope, f"{scope}-wf1", researched)
    evaluate_interval = gates.target_eval.evaluate_interval
    evaluate_return = gates.target_eval.evaluate_return

    def interval(*args: Any) -> Any:
        found = evaluate_interval(*args)
        if field == "coverage":
            return replace(found, coverage=value)
        compared = getattr(found, field)
        return replace(found, **{field: replace(compared, model=compared.baseline + value)})

    def returned(*args: Any) -> Any:
        found = evaluate_return(*args)
        zero = found.mae["zero"]
        return replace(found, mae={**found.mae, "zero": replace(zero, model=zero.baseline + value)})

    monkeypatch.setattr(gates.target_eval, "evaluate_interval", interval)
    monkeypatch.setattr(gates.target_eval, "evaluate_return", returned)

    found = by_slot(check_gates(f"{scope}-prod", researched))[slot]["shadow_oos"]

    assert found.passed is passed


@pytest.mark.django_db
def test_gates_are_recorded_only_once_every_slot_is_checked(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built("rank", "rank-wf1", researched)
    checked = gates.data_quality
    calls: list[int] = []

    def broken(run: ModelRun) -> Gate:
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("the second slot broke")
        return checked(run)

    monkeypatch.setattr(gates, "data_quality", broken)
    with pytest.raises(RuntimeError, match="second slot"):
        check_gates("rank-prod", researched)

    assert not ModelRun.objects.filter(run_key="rank-prod", params__has_key="gates").exists()


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("scope", "slot"),
    [
        ("interval", Slot("quantile", 20)),
        ("interval", Slot("volatility", 20)),
        ("interval", Slot("classification", 20)),
        ("return", Slot("regression", 5)),
    ],
)
def test_each_objective_compares_its_own_prediction_field(
    researched: Path, monkeypatch: pytest.MonkeyPatch, scope: str, slot: Slot
) -> None:
    monkeypatch.setattr(gates, "GAIN_TOP", 2)
    monkeypatch.setattr(gates, "GAIN_OVERLAP", 2)
    built(scope, f"{scope}-wf1", researched)
    run = ModelRun.objects.get(
        run_key=f"{scope}-prod",
        model_type=MODEL,
        objective=slot.objective,
        params__horizon=slot.horizon,
    )
    serving = serve_copy(ModelRun.objects.get(pk=run.pk))

    found = by_slot(check_gates(f"{scope}-prod", researched))[slot]

    assert found["agreement"].passed is True
    assert found["agreement"].value == pytest.approx(1.0)
    assert found["agreement"].sample >= gates.MIN_STOCKS
    assert found["gain_overlap"].passed is True
    assert ModelRun.objects.get(pk=run.pk).params["gates"]["serving"] == serving.run_id


@pytest.mark.django_db
@pytest.mark.parametrize("change", ["status", "params"])
def test_gates_refuse_an_n_that_changed_while_they_were_checked(
    researched: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    runs = built("rank", "rank-wf1", researched)
    logged(runs[0])
    checked = gates.data_quality

    def moved(run: ModelRun) -> Gate:
        other = ModelRun.objects.filter(run_id="rank-prod.lightgbm.h5.f000")
        if change == "status":
            other.update(status="active")
        else:
            other.update(params={**other.get().params, "importance_gain": {"x": 1.0}})
        return checked(run)

    monkeypatch.setattr(gates, "data_quality", moved)
    with pytest.raises(RunError, match=r"h5\.f000 changed while its gates were checked"):
        check_gates("rank-prod", researched)

    assert not ModelRun.objects.filter(run_key="rank-prod", params__has_key="gates").exists()


def test_every_gate_names_the_unit_of_its_sample() -> None:
    assert set(gates.SAMPLE_UNITS) == {"agreement", "gain_overlap", "shadow_oos", "data_quality"}
    assert set(gates.SAMPLE_UNITS.values()) <= {"stocks", "features", "days"}
