import json
import math
from dataclasses import replace
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.market.calendar import TradingDays
from apps.research import (
    baseline_runs,
    holdout,
    holdout_runs,
    lgbm_runs,
    monitor_refs,
    production,
)
from apps.research.baseline_runs import BASELINE
from apps.research.holdout import HoldoutError, claim_holdout, production_pass
from apps.research.lgbm_runs import MODEL, run_fold, run_id
from apps.research.models import ModelRun, ModelStatus
from apps.research.monitor_refs import research_brier, research_ic_thresholds
from apps.research.prediction_cache import (
    MissingFoldsError,
    fold_notes,
    fold_path,
    load_fold_predictions,
)
from apps.research.production import (
    SHADOW_MONTHS,
    Design,
    build_run,
    production_fold,
    shadow_fold,
)
from apps.research.rank_model import SPEC_RECIPE, Variant
from apps.research.runs import RunError
from apps.research.walkforward import (
    WalkForwardConfig,
    WalkForwardError,
    add_months,
    generate_folds,
    oos_start,
)
from tests.research.run_panels import (
    CONFIG,
    KEPT,
    UNSCREENED,
    claim_all,
    panel_calendar,
    research,
    weekdays,
    write_holdout_markers,
)

AS_OF = date(2026, 10, 30)
SPEC = WalkForwardConfig()


def first_on_or_after(calendar: TradingDays, day: date) -> date:
    return calendar.between(day, calendar.end)[0]


def test_the_production_fold_trains_through_the_labels_known_on_its_day(
    long_calendar: TradingDays,
) -> None:
    fold = production_fold(long_calendar, AS_OF, SPEC)

    train_end = long_calendar.shift(AS_OF, -SPEC.embargo)
    valid_start = first_on_or_after(long_calendar, add_months(train_end, -6) + timedelta(days=1))
    assert fold.train_end == train_end
    assert fold.train_start == first_on_or_after(
        long_calendar, add_months(train_end, -36) + timedelta(days=1)
    )
    assert fold.valid_start == valid_start
    assert fold.fit_end == long_calendar.shift(valid_start, -(SPEC.embargo + 1))
    assert (fold.test_start, fold.test_end) == (AS_OF, AS_OF)
    assert fold.eval_end == train_end < fold.test_start
    assert not fold.tuning
    assert not fold.holdout
    assert holdout.is_holdout(fold)


def test_the_shadow_fold_tests_the_last_six_months_after_an_embargo(
    long_calendar: TradingDays,
) -> None:
    fold = shadow_fold(long_calendar, AS_OF, SPEC)

    test_start = first_on_or_after(
        long_calendar, add_months(AS_OF, -SHADOW_MONTHS) + timedelta(days=1)
    )
    assert (fold.test_start, fold.test_end) == (test_start, AS_OF)
    assert fold.train_end == long_calendar.shift(test_start, -(SPEC.embargo + 1))
    assert fold.eval_end == long_calendar.shift(AS_OF, -SPEC.embargo)
    assert fold.train_start == first_on_or_after(
        long_calendar, add_months(fold.train_end, -36) + timedelta(days=1)
    )
    assert not fold.tuning
    assert not fold.holdout


def test_the_production_model_is_out_of_sample_from_its_day(long_calendar: TradingDays) -> None:
    fold = production_fold(long_calendar, AS_OF, SPEC)

    assert oos_start(long_calendar, fold.train_end, 20) == AS_OF
    assert oos_start(long_calendar, fold.train_end, 5) == long_calendar.shift(fold.train_end, 5)
    assert oos_start(long_calendar, fold.train_end, 5) < AS_OF


def test_production_folds_follow_the_design_window(long_calendar: TradingDays) -> None:
    expanding = replace(SPEC, window="expanding")
    long = replace(SPEC, train_months=60)

    assert production_fold(long_calendar, AS_OF, expanding).train_start == date(2010, 1, 1)
    assert shadow_fold(long_calendar, AS_OF, expanding).train_start == date(2010, 1, 1)
    assert production_fold(long_calendar, AS_OF, long).train_start < (
        production_fold(long_calendar, AS_OF, SPEC).train_start
    )


def test_production_folds_refuse_a_window_before_the_data(long_calendar: TradingDays) -> None:
    with pytest.raises(WalkForwardError, match="before data_start"):
        production_fold(long_calendar, date(2012, 6, 29), SPEC)
    with pytest.raises(WalkForwardError, match="before data_start"):
        shadow_fold(long_calendar, date(2013, 3, 29), SPEC)


@pytest.fixture
def frozen(run_panel: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The one-year panel with its last month past the holdout start and every scope claimed."""
    monkeypatch.setattr(lgbm_runs, "MAX_ROUNDS", 80)
    monkeypatch.setattr(holdout, "SPEC_HOLDOUT_START", date(2016, 12, 1))
    for scope in ("rank", "return", "interval"):
        claim_holdout(scope, f"{scope}-final", weekdays()[-1])
    return run_panel


@pytest.mark.django_db
def test_a_production_fold_past_the_holdout_start_needs_the_pass(frozen: Path) -> None:
    fold = production_fold(panel_calendar(), weekdays()[-1], CONFIG)
    cached = fold_path("rank-prod", MODEL, fold.index)
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"kept")

    with pytest.raises(HoldoutError, match=f"fold {fold.index} is holdout"):
        run_fold("rank", fold, "rank-prod", frozen, KEPT, 0)
    assert cached.read_bytes() == b"kept"
    assert not ModelRun.objects.filter(run_key="rank-prod").exists()


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("run_key", "index", "message"),
    [
        ("interval-prod", 0, "start with rank-"),
        ("rank-final", 0, "holdout run"),
        ("rank-prod", 7, "production folds only"),
    ],
)
def test_run_fold_checks_a_production_write_before_touching_the_cache(
    frozen: Path, run_key: str, index: int, message: str
) -> None:
    fold = replace(production_fold(panel_calendar(), weekdays()[-1], CONFIG), index=index)
    cached = fold_path(run_key, MODEL, fold.index)
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"kept")

    with pytest.raises((ValueError, HoldoutError), match=message):
        run_fold("rank", fold, run_key, frozen, KEPT, 0, production=production_pass("rank"))

    assert cached.read_bytes() == b"kept"
    assert not ModelRun.objects.filter(run_key=run_key).exists()


@pytest.mark.django_db
def test_a_production_pass_fits_the_shadow_baselines_past_the_holdout_start(frozen: Path) -> None:
    fold = shadow_fold(panel_calendar(), weekdays()[-1], replace(CONFIG, train_months=5))
    features = baseline_runs.scope_features("interval")
    with pytest.raises(HoldoutError, match="holdout"):
        baseline_runs.run_fold("interval", fold, "interval-prod-shadow", frozen, features)

    baseline_runs.run_fold(
        "interval",
        fold,
        "interval-prod-shadow",
        frozen,
        features,
        production=production_pass("interval"),
        annotate=lambda run: {"research_run_key": "interval-wf1"},
    )

    runs = {run.model_type: run for run in ModelRun.objects.filter(run_key="interval-prod-shadow")}
    assert set(runs) == {"logistic", "histq"}
    assert {run.params["research_run_key"] for run in runs.values()} == {"interval-wf1"}
    cached = load_fold_predictions("interval-prod-shadow", BASELINE, fold)
    assert (cached["date"].min(), cached["date"].max()) == (fold.test_start, fold.test_end)
    assert cached["logistic_h20"].notna().all()


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("run_key", "index", "message"),
    [
        ("rank-prod-shadow", 0, "start with interval-"),
        ("interval-final", 0, "holdout run"),
        ("interval-prod-shadow", 7, "production folds only"),
    ],
)
def test_baseline_run_fold_checks_a_production_write_before_touching_the_cache(
    frozen: Path, run_key: str, index: int, message: str
) -> None:
    shadowed = shadow_fold(panel_calendar(), weekdays()[-1], replace(CONFIG, train_months=5))
    fold = replace(shadowed, index=index)
    cached = fold_path(run_key, BASELINE, fold.index)
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"kept")

    with pytest.raises((ValueError, HoldoutError), match=message):
        baseline_runs.run_fold(
            "interval",
            fold,
            run_key,
            frozen,
            baseline_runs.scope_features("interval"),
            production=production_pass("interval"),
        )

    assert cached.read_bytes() == b"kept"
    assert not ModelRun.objects.filter(run_key=run_key).exists()


@pytest.mark.django_db
def test_a_production_rank_fold_fits_without_evaluating_and_scores_its_day(frozen: Path) -> None:
    last = weekdays()[-1]
    fold = production_fold(panel_calendar(), last, CONFIG)

    run_fold("rank", fold, "rank-prod", frozen, KEPT, 0, bags=2, production=production_pass("rank"))

    for horizon in (5, 20):
        run = ModelRun.objects.get(run_id=run_id("rank-prod", "rank", horizon, fold))
        assert (run.train_start, run.train_end) == (fold.train_start, fold.train_end)
        assert run.ic_mean is None
        assert run.icir is None
        assert run.params["ic_nw_t"] is None
        assert run.params["bags"] == 2
        assert set(run.params["importance_gain"]) == set(KEPT)
    cached = load_fold_predictions("rank-prod", MODEL, fold)
    assert set(cached["date"]) == {last}
    assert cached["lightgbm_h20"].notna().all()
    assert fold_notes("rank-prod", MODEL, fold.index)["recipe"] == lgbm_runs.recipe(
        "rank", KEPT, 0, 2, SPEC_RECIPE
    )


@pytest.mark.django_db
def test_production_return_and_interval_folds_leave_their_oos_metrics_empty(
    frozen: Path,
) -> None:
    fold = production_fold(panel_calendar(), weekdays()[-1], CONFIG)
    features = [*KEPT, *UNSCREENED]

    run_fold(
        "return", fold, "return-prod", frozen, features, 0, production=production_pass("return")
    )
    granted = production_pass("interval")
    run_fold("interval", fold, "interval-prod", frozen, features, 0, production=granted)

    regression = ModelRun.objects.get(run_key="return-prod")
    assert regression.ic_mean is None
    assert regression.params["oos_mae"] is None
    assert regression.params["oos_mae_zero"] is None
    assert regression.params["delta"] > 0
    interval = {run.objective: run for run in ModelRun.objects.filter(run_key="interval-prod")}
    assert set(interval) == {"quantile", "volatility", "classification"}
    assert interval["quantile"].params["oos_coverage"] is None
    assert interval["quantile"].params["oos_pinball"] is None
    assert interval["volatility"].params["oos_mae"] is None
    probability = interval["classification"].params
    assert probability["oos_ece"] is None
    assert probability["oos_brier_raw"] is None
    assert probability["valid_ece"] is not None
    cached = load_fold_predictions("interval-prod", MODEL, fold)
    assert cached[["lightgbm_q50_h20", "lightgbm_prob_h20"]].notna().all().all()


@pytest.mark.django_db
def test_a_shadow_fold_is_evaluated_on_its_labelled_months(frozen: Path) -> None:
    fold = shadow_fold(panel_calendar(), weekdays()[-1], replace(CONFIG, train_months=5))

    run_fold("rank", fold, "rank-prod-shadow", frozen, KEPT, 0, production=production_pass("rank"))

    run = ModelRun.objects.get(run_id=run_id("rank-prod-shadow", "rank", 20, fold))
    assert run.ic_mean is not None
    cached = load_fold_predictions("rank-prod-shadow", MODEL, fold)
    assert cached["date"].min() == fold.test_start
    assert cached["date"].max() == fold.test_end


@pytest.mark.django_db
def test_build_run_waits_for_the_holdout_claim(researched: Path) -> None:
    with pytest.raises(HoldoutError, match="rank design is not frozen"):
        build_run("rank", "rank-prod", "rank-wf1", weekdays()[-1], researched)

    assert not ModelRun.objects.filter(run_key__startswith="rank-prod").exists()


def production_note(run: ModelRun) -> dict[str, Any]:
    note: dict[str, Any] = run.params["production"]
    return note


@pytest.mark.django_db
def test_build_run_fits_the_model_and_its_shadow_on_the_frozen_design(researched: Path) -> None:
    claim_all()
    last = weekdays()[-1]
    assert holdout.is_holdout(production_fold(panel_calendar(), last, CONFIG))

    built = build_run("rank", "rank-prod", "rank-wf1", last, researched)

    model = ModelRun.objects.filter(run_key="rank-prod").order_by("run_id")
    shadow = ModelRun.objects.filter(run_key="rank-prod-shadow").order_by("run_id")
    assert built.model == [run.run_id for run in model]
    assert built.shadow == [run.run_id for run in shadow]
    assert [run.params["horizon"] for run in model] == [20, 5]
    folds = [fold for fold in generate_folds(panel_calendar(), last, CONFIG) if fold.research]
    thresholds = research_ic_thresholds("rank-wf1", MODEL, folds)
    assert sorted(thresholds) == [5, 20]
    expected = {str(horizon): pytest.approx(value) for horizon, value in thresholds.items()}
    assert built.references == {"rolling_ic_threshold": expected}
    design = json.loads(json.dumps(production.design_note(production.DESIGNS["rank"])))
    shared = {"as_of": last.isoformat(), "frozen_by": "rank-final", "design": design}
    for run in model:
        assert run.status == "candidate"
        assert run.ic_mean is None
        assert run.feature_list == ["tech_ret_1d", "tech_vol_20d"]
        assert run.params["bags"] == 2
        assert run.params["variant"]["masked"] == ["tech_ret_60d"]
        assert run.params["research_run_key"] == "rank-wf1"
        assert production_note(run) == {
            **shared,
            "role": "model",
            "shadow_run_key": "rank-prod-shadow",
        }
        assert run.params["rolling_ic_threshold"] == expected
        assert run.train_end == production_fold(panel_calendar(), last, CONFIG).train_end
    for run in shadow:
        assert run.status == "candidate"
        assert run.ic_mean is not None
        assert run.params["research_run_key"] == "rank-wf1"
        assert production_note(run) == {**shared, "role": "shadow", "shadow_of": "rank-prod"}
        assert "rolling_ic_threshold" not in run.params
        assert run.train_end == shadow_fold(panel_calendar(), last, CONFIG).train_end
    fold = production_fold(panel_calendar(), last, CONFIG)
    assert set(load_fold_predictions("rank-prod", MODEL, fold)["date"]) == {last}
    assert not fold_path("rank-prod-shadow", BASELINE, 0).exists()


@pytest.mark.django_db
def test_build_run_command_gives_the_probability_model_the_research_brier(
    researched: Path,
) -> None:
    claim_all()
    last = weekdays()[-1]
    out = StringIO()

    call_command(
        "build_run",
        scope="interval",
        run_key="interval-prod",
        research_run_key="interval-wf1",
        as_of=last.isoformat(),
        stdout=out,
    )

    folds = [fold for fold in generate_folds(panel_calendar(), last, CONFIG) if fold.research]
    brier = research_brier("interval-wf1", folds)
    runs = {run.objective: run for run in ModelRun.objects.filter(run_key="interval-prod")}
    assert runs["classification"].params["brier_baseline"] == pytest.approx(brier)
    assert "brier_baseline" not in runs["quantile"].params
    assert {run.params["research_run_key"] for run in runs.values()} == {"interval-wf1"}
    shadow = ModelRun.objects.filter(run_key="interval-prod-shadow")
    assert not any("brier_baseline" in run.params for run in shadow)
    baselines = shadow.filter(model_type__in=["logistic", "histq"])
    assert baselines.count() == 2
    assert {production_note(run)["role"] for run in baselines} == {"shadow"}
    assert {run.params["research_run_key"] for run in baselines} == {"interval-wf1"}
    shadow_fold_ = shadow_fold(panel_calendar(), last, CONFIG)
    assert (
        load_fold_predictions("interval-prod-shadow", BASELINE, shadow_fold_)["logistic_h20"]
        .notna()
        .all()
    )
    assert not ModelRun.objects.filter(run_key="interval-prod", model_type="logistic").exists()
    assert not fold_path("interval-prod", BASELINE, 0).exists()
    assert f"Brier baseline from interval-wf1: {brier:.4f}" in out.getvalue()
    listed = list(shadow.order_by("run_id").values_list("run_id", flat=True))
    assert f"interval-prod-shadow: {', '.join(listed)}" in out.getvalue()
    assert "interval-prod-shadow.logistic.h20.f000" in listed
    assert "rolling IC" not in out.getvalue()


@pytest.mark.django_db
def test_build_run_fits_the_return_scope_without_references(researched: Path) -> None:
    claim_all()
    research_row("return-wf1")

    built = build_run("return", "return-prod", "return-wf1", weekdays()[-1], researched)

    assert built.references == {}
    assert built.model == ["return-prod.lightgbm.h5.f000"]
    assert built.shadow == ["return-prod-shadow.lightgbm.h5.f000"]
    shadow_fold_ = shadow_fold(panel_calendar(), weekdays()[-1], CONFIG)
    zero = load_fold_predictions("return-prod-shadow", BASELINE, shadow_fold_)["zero_h5"]
    assert (zero == 0).all()
    assert not fold_path("return-prod", BASELINE, 0).exists()
    model = ModelRun.objects.get(run_key="return-prod")
    assert production_note(model)["role"] == "model"
    assert "rolling_ic_threshold" not in model.params
    assert "brier_baseline" not in model.params


def model_fields(run_key: str) -> dict[str, Any]:
    return {
        "run_id": f"{run_key}.lightgbm.h20.f000",
        "run_key": run_key,
        "model_type": "lightgbm",
        "objective": "rank",
        "version": "4",
        "train_start": date(2016, 1, 4),
        "train_end": date(2016, 6, 1),
        "params": {},
        "feature_list": [],
        "lib_versions": {},
        "feature_hash": "0" * 64,
    }


def research_row(run_key: str, **params: Any) -> ModelRun:
    return ModelRun.objects.create(**{**model_fields(run_key), "params": params})


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("scope", "run_key", "research_key", "message"),
    [
        ("rank", "interval-prod", "rank-wf1", "start with rank-"),
        ("rank", "rank-" + "x" * 30, "rank-wf1", "at most"),
        ("rank", "rank-prod", "interval-wf1", "start with rank-"),
        ("rank", "rank-prod", "rank-none", "no model_run"),
        ("return", "return-prod", "return-none", "no model_run"),
        ("rank", "rank-wf1", "rank-wf1", "is the research run"),
        ("rank", "rank-wf1", "rank-wf1-shadow", "is the research run"),
        ("rank", "rank-final", "rank-wf1", "rank-final is a holdout run"),
        ("rank", "rank-used", "rank-wf1", "not a production build"),
        ("rank", "rank-y", "rank-wf1", "rank-y-shadow holds runs that are not a production"),
        ("rank", "rank-wf1-shadow", "rank-wf2", "names a shadow"),
        ("rank", "rank-prod", "rank-made", "is a production build"),
    ],
)
def test_build_run_checks_its_keys_before_fitting(
    researched: Path, scope: str, run_key: str, research_key: str, message: str
) -> None:
    claim_all()
    research_row("rank-used")
    research_row("rank-y-shadow")
    research_row("rank-made", production={"role": "model"})
    rows = {run.run_id: run.params for run in ModelRun.objects.all()}
    cached = sorted(path.name for path in fold_path("rank-wf1", MODEL, 0).parent.iterdir())

    with pytest.raises((ValueError, RunError, HoldoutError), match=message):
        build_run(scope, run_key, research_key, weekdays()[-1], researched)

    assert {run.run_id: run.params for run in ModelRun.objects.all()} == rows
    assert sorted(path.name for path in fold_path("rank-wf1", MODEL, 0).parent.iterdir()) == cached


@pytest.mark.django_db
def test_build_run_refuses_a_shadow_key_that_is_a_holdout_run(researched: Path) -> None:
    claim_holdout("rank", "rank-x-shadow", weekdays()[-1])

    with pytest.raises(HoldoutError, match="rank-x-shadow is a holdout run"):
        build_run("rank", "rank-x", "rank-wf1", weekdays()[-1], researched)

    assert not ModelRun.objects.filter(run_key__startswith="rank-x").exists()


@pytest.mark.django_db
def test_build_run_leaves_a_settled_build_and_its_shadow_alone(researched: Path) -> None:
    claim_all()
    research_row("rank-prod", production={"role": "model"})
    research_row("rank-prod-shadow", production={"role": "shadow"})
    ModelRun.objects.filter(run_key="rank-prod").update(status=ModelStatus.ACTIVE)
    cached = fold_path("rank-prod-shadow", MODEL, 0)
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"evidence")
    rows = {run.run_id: (run.status, run.params) for run in ModelRun.objects.all()}

    with pytest.raises(RunError, match=r"rank-prod\.lightgbm\.h20\.f000 is active"):
        build_run("rank", "rank-prod", "rank-wf1", weekdays()[-1], researched)

    assert {run.run_id: (run.status, run.params) for run in ModelRun.objects.all()} == rows
    assert cached.read_bytes() == b"evidence"


@pytest.mark.django_db
def test_build_run_takes_rank_thresholds_only_from_the_frozen_design(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(production.DESIGNS, "rank", Design(CONFIG))
    claim_all()

    with pytest.raises(RunError, match="rank-wf1 was not fit on the frozen rank design"):
        build_run("rank", "rank-prod", "rank-wf1", weekdays()[-1], researched)

    assert not ModelRun.objects.filter(run_key__startswith="rank-prod").exists()


@pytest.mark.django_db
def test_build_run_needs_the_research_caches_before_fitting(researched: Path) -> None:
    claim_all()
    fitted = ModelRun.objects.filter(run_key="rank-wf1", model_type=MODEL).first()
    assert fitted is not None
    research_row("rank-wf2", variant=fitted.params["variant"])
    ModelRun.objects.filter(run_key="rank-wf2").update(feature_list=fitted.feature_list)

    with pytest.raises(MissingFoldsError):
        build_run("rank", "rank-prod", "rank-wf2", weekdays()[-1], researched)

    assert not ModelRun.objects.filter(run_key__startswith="rank-prod").exists()


@pytest.mark.django_db
def test_build_run_refuses_missing_references_before_fitting(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claim_all()
    monkeypatch.setattr(monitor_refs, "rolling_ic_threshold", lambda ic: math.nan)
    monkeypatch.setattr(production, "research_brier", lambda run_key, folds: math.nan)

    with pytest.raises(RunError, match=r"no rolling IC threshold at \[5, 20\] days"):
        build_run("rank", "rank-prod", "rank-wf1", weekdays()[-1], researched)
    with pytest.raises(RunError, match="no logistic baseline Brier"):
        build_run("interval", "interval-prod", "interval-wf1", weekdays()[-1], researched)

    assert not ModelRun.objects.filter(run_key__endswith="-prod").exists()
    assert not ModelRun.objects.filter(run_key__endswith="-prod-shadow").exists()


@pytest.mark.django_db
def test_build_run_waits_for_a_holdout_that_ran_the_frozen_design(researched: Path) -> None:
    claim_holdout("rank", "rank-final", weekdays()[-1])
    design = production.DESIGNS["rank"]
    note = json.loads(json.dumps(production.design_note(design)))
    features = lgbm_runs.design_features("rank", design.variant, design.target)

    def refused(message: str) -> None:
        with pytest.raises(RunError, match=message):
            build_run("rank", "rank-prod", "rank-wf1", weekdays()[-1], researched)

    refused("holdout rank-final recorded no run")
    bare = research_row("rank-final")
    refused("1 runs under holdout rank-final recorded no design")
    bare.delete()
    other = research_row("rank-final", design={"config": {}, "variant": {}, "target": {}})
    refused("not the one holdout rank-final ran")
    other.delete()
    ModelRun.objects.create(
        **{**model_fields("rank-final"), "params": {production.DESIGN: note}, "feature_list": KEPT}
    )
    refused("holdout rank-final fit other features")
    ModelRun.objects.filter(run_key="rank-final").update(feature_list=features)
    refused(r"holdout rank-final has not written folds \[\d+\]")
    write_holdout_markers("rank", "rank-final", models=(BASELINE,))
    refused(r"holdout rank-final has not written folds \[\d+\]")
    write_holdout_markers("rank", "rank-final")

    build_run("rank", "rank-prod", "rank-wf1", weekdays()[-1], researched)

    assert ModelRun.objects.filter(run_key="rank-prod").exists()


@pytest.mark.django_db
def test_build_run_accepts_the_holdout_the_runner_really_wrote(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    last = weekdays()[-1]
    frozen = replace(production.DESIGNS["rank"], research_run_key="rank-wf1")
    monkeypatch.setitem(production.DESIGNS, "rank", frozen)
    monkeypatch.setattr(holdout_runs, "latest_price_date", lambda: last)

    outcome = holdout_runs.run_holdout("rank", "rank-final", last, researched)

    built = build_run("rank", "rank-prod", "rank-wf1", last, researched)
    assert outcome.ran
    assert built.model
    assert ModelRun.objects.filter(run_key="rank-prod").exists()


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("scope", "design", "message"),
    [
        ("rank", Design(CONFIG, Variant(masked=tuple(KEPT))), "masks every rank feature"),
        ("return", Design(CONFIG, Variant(masked=("tech_ret_60d",))), "rank scope only"),
    ],
)
def test_build_run_checks_the_design_before_fitting(
    researched: Path, monkeypatch: pytest.MonkeyPatch, scope: str, design: Design, message: str
) -> None:
    claim_all()
    research_row("return-wf1")
    monkeypatch.setitem(production.DESIGNS, scope, design)

    with pytest.raises(ValueError, match=message):
        build_run(scope, f"{scope}-prod", f"{scope}-wf1", weekdays()[-1], researched)

    assert not ModelRun.objects.filter(run_key__startswith=f"{scope}-prod").exists()


@pytest.mark.django_db
def test_a_failed_build_leaves_an_annotated_shadow_and_builds_again(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claim_all()
    last = weekdays()[-1]
    save = lgbm_runs.save_fold_predictions

    def broken(*args: Any, **kwargs: Any) -> Path:
        raise OSError("the cache disk is full")

    monkeypatch.setattr(lgbm_runs, "save_fold_predictions", broken)
    with pytest.raises(OSError, match="disk is full"):
        build_run("rank", "rank-prod", "rank-wf1", last, researched)

    shadow = list(ModelRun.objects.filter(run_key="rank-prod-shadow"))
    assert shadow
    assert all(production_note(run)["role"] == "shadow" for run in shadow)
    assert all(run.params["research_run_key"] == "rank-wf1" for run in shadow)
    assert not ModelRun.objects.filter(run_key="rank-prod").exists()

    monkeypatch.setattr(lgbm_runs, "save_fold_predictions", save)
    built = build_run("rank", "rank-prod", "rank-wf1", last, researched)

    assert len(built.model) == 2
    assert built.shadow == sorted(run.run_id for run in shadow)


@pytest.mark.django_db
def test_a_failed_rebuild_keeps_no_earlier_gates(
    researched: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claim_all()
    last = weekdays()[-1]
    build_run("rank", "rank-prod", "rank-wf1", last, researched)
    for run in ModelRun.objects.filter(run_key__in=["rank-prod", "rank-prod-shadow"]):
        ModelRun.objects.filter(pk=run.pk).update(params={**run.params, "gates": {"passed": True}})
    fit = lgbm_runs.FITS["rank"]

    def broken(job: Any) -> Any:
        if job.run_key == "rank-prod":
            raise MemoryError("the fit ran out of memory")
        return fit(job)

    monkeypatch.setitem(lgbm_runs.FITS, "rank", broken)
    with pytest.raises(MemoryError, match="out of memory"):
        build_run("rank", "rank-prod", "rank-wf1", last, researched)

    assert ModelRun.objects.filter(run_key="rank-prod").count() == 2
    assert not ModelRun.objects.filter(run_key="rank-prod", params__has_key="gates").exists()


@pytest.mark.django_db
def test_research_fits_refuse_a_production_run_key(researched: Path) -> None:
    research_row("rank-made", production={"role": "model"})
    research_row("interval-made", production={"role": "model"})
    chosen = [fold for fold in research() if fold.research]

    with pytest.raises(RunError, match="rank-made holds a production build"):
        lgbm_runs.run_folds("rank", chosen, "rank-made", researched)
    with pytest.raises(RunError, match="interval-made holds a production build"):
        baseline_runs.run_folds("interval", chosen, "interval-made", researched)

    assert not fold_path("rank-made", MODEL, chosen[0].index).exists()


@pytest.mark.django_db
def test_build_run_command_reports_both_runs(researched: Path) -> None:
    claim_all()
    out = StringIO()

    call_command(
        "build_run",
        scope="rank",
        run_key="rank-prod",
        research_run_key="rank-wf1",
        as_of=weekdays()[-1].isoformat(),
        stdout=out,
    )

    text = out.getvalue()
    limits = ModelRun.objects.get(run_id="rank-prod.lightgbm.h20.f000").params[
        "rolling_ic_threshold"
    ]
    assert "rank-prod: rank-prod.lightgbm.h20.f000, rank-prod.lightgbm.h5.f000" in text
    assert "rank-prod-shadow: rank-prod-shadow.lightgbm.h20.f000" in text
    assert (
        f"rolling IC thresholds from rank-wf1: 5 日 {limits['5']:.4f}, 20 日 {limits['20']:.4f}"
        in text
    )
    assert "Brier" not in text


@pytest.mark.django_db
def test_build_run_command_reports_a_design_that_is_not_frozen(researched: Path) -> None:
    with pytest.raises(CommandError, match="rank design is not frozen"):
        call_command(
            "build_run",
            scope="rank",
            run_key="rank-prod",
            research_run_key="rank-wf1",
            as_of=weekdays()[-1].isoformat(),
        )
