from dataclasses import replace
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.market.calendar import TradingDays
from apps.research import baseline_runs, holdout, lgbm_runs, monitor_refs, production
from apps.research.holdout import HoldoutError, claim_holdout, production_pass
from apps.research.lgbm_runs import MODEL, run_fold, run_id
from apps.research.models import ModelRun
from apps.research.monitor_refs import research_brier, research_ic_thresholds
from apps.research.prediction_cache import MissingFoldsError, fold_notes, load_fold_predictions
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
)
from tests.research.run_panels import CONFIG, KEPT, UNSCREENED, research, weekdays

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


def panel_calendar() -> TradingDays:
    days = weekdays()
    return TradingDays(days, days[0], days[-1])


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

    with pytest.raises(HoldoutError, match="holdout"):
        run_fold("rank", fold, "rank-prod", frozen, KEPT, 0)
    assert not ModelRun.objects.filter(run_key="rank-prod").exists()


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


@pytest.fixture
def researched(run_panel: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Research caches of a rank and an interval walk-forward on the panel, before any claim.

    The panel's research folds hold fewer than 60 IC days, so the rolling window is 5 here.
    """
    monkeypatch.setattr(lgbm_runs, "MAX_ROUNDS", 80)
    chosen = [fold for fold in research() if fold.research]
    lgbm_runs.run_folds("rank", chosen, "rank-wf1", run_panel)
    baseline_runs.run_folds("interval", chosen, "interval-wf1", run_panel)
    rolling = monitor_refs.rolling_ic_threshold
    monkeypatch.setattr(monitor_refs, "rolling_ic_threshold", lambda ic: rolling(ic, 5))
    monkeypatch.setattr(production, "SHADOW_MONTHS", 4)
    monkeypatch.setattr(production, "BAGS", 2)
    masked = Variant(masked=("tech_ret_60d",))
    designs = {scope: Design(config=CONFIG) for scope in ("rank", "return", "interval")}
    monkeypatch.setattr(production, "DESIGNS", {**designs, "rank": Design(CONFIG, masked)})
    return run_panel


def claim_all() -> None:
    for scope in ("rank", "return", "interval"):
        claim_holdout(scope, f"{scope}-final", weekdays()[-1])


@pytest.mark.django_db
def test_build_run_waits_for_the_holdout_claim(researched: Path) -> None:
    with pytest.raises(HoldoutError, match="rank design is not frozen"):
        build_run("rank", "rank-prod", "rank-wf1", weekdays()[-1], researched)

    assert not ModelRun.objects.filter(run_key__startswith="rank-prod").exists()


@pytest.mark.django_db
def test_build_run_fits_the_model_and_its_shadow_on_the_frozen_design(researched: Path) -> None:
    claim_all()
    last = weekdays()[-1]

    built = build_run("rank", "rank-prod", "rank-wf1", last, researched)

    model = ModelRun.objects.filter(run_key="rank-prod").order_by("run_id")
    shadow = ModelRun.objects.filter(run_key="rank-prod-shadow").order_by("run_id")
    assert built.model == [run.run_id for run in model]
    assert built.shadow == [run.run_id for run in shadow]
    assert [run.params["horizon"] for run in model] == [20, 5]
    folds = [fold for fold in generate_folds(panel_calendar(), last, CONFIG) if fold.research]
    thresholds = research_ic_thresholds("rank-wf1", MODEL, folds)
    assert len(thresholds) == 2
    assert all(value == value for value in thresholds.values())
    for run in model:
        assert run.status == "candidate"
        assert run.ic_mean is None
        assert run.feature_list == ["tech_ret_1d", "tech_vol_20d"]
        assert run.params["bags"] == 2
        assert run.params["variant"]["masked"] == ["tech_ret_60d"]
        assert run.params["research_run_key"] == "rank-wf1"
        assert run.params["shadow_run_key"] == "rank-prod-shadow"
        assert run.params["as_of"] == last.isoformat()
        assert run.params["design"]["config"]["train_months"] == CONFIG.train_months
        assert run.params["rolling_ic_threshold"] == {
            str(horizon): pytest.approx(value) for horizon, value in thresholds.items()
        }
        assert run.train_end == production_fold(panel_calendar(), last, CONFIG).train_end
    for run in shadow:
        assert run.ic_mean is not None
        assert run.params["shadow_of"] == "rank-prod"
        assert "rolling_ic_threshold" not in run.params
        assert run.train_end == shadow_fold(panel_calendar(), last, CONFIG).train_end
    assert set(
        load_fold_predictions("rank-prod", MODEL, production_fold(panel_calendar(), last, CONFIG))[
            "date"
        ]
    ) == {last}


@pytest.mark.django_db
def test_build_run_gives_the_probability_model_the_research_brier(researched: Path) -> None:
    claim_all()
    last = weekdays()[-1]

    build_run("interval", "interval-prod", "interval-wf1", last, researched)

    folds = [fold for fold in generate_folds(panel_calendar(), last, CONFIG) if fold.research]
    runs = {run.objective: run for run in ModelRun.objects.filter(run_key="interval-prod")}
    assert runs["classification"].params["brier_baseline"] == pytest.approx(
        research_brier("interval-wf1", folds)
    )
    assert "brier_baseline" not in runs["quantile"].params
    assert {run.params["research_run_key"] for run in runs.values()} == {"interval-wf1"}


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("scope", "run_key", "research_key", "message"),
    [
        ("rank", "interval-prod", "rank-wf1", "start with rank-"),
        ("rank", "rank-" + "x" * 30, "rank-wf1", "at most"),
        ("rank", "rank-prod", "interval-wf1", "start with rank-"),
        ("rank", "rank-prod", "rank-none", "no model_run"),
        ("return", "return-prod", "return-none", "no model_run"),
    ],
)
def test_build_run_checks_its_keys_before_fitting(
    researched: Path, scope: str, run_key: str, research_key: str, message: str
) -> None:
    claim_all()

    with pytest.raises((ValueError, RunError), match=message):
        build_run(scope, run_key, research_key, weekdays()[-1], researched)

    assert not ModelRun.objects.filter(run_key__in=[run_key, f"{run_key}-shadow"]).exists()


@pytest.mark.django_db
def test_build_run_needs_the_research_caches_before_fitting(researched: Path) -> None:
    claim_all()
    ModelRun.objects.create(
        run_id="rank-wf2.lightgbm.h20.f000",
        run_key="rank-wf2",
        model_type="lightgbm",
        objective="rank",
        version="4",
        train_start=date(2016, 1, 4),
        train_end=date(2016, 6, 1),
        params={},
        feature_list=[],
        lib_versions={},
        feature_hash="0" * 64,
    )

    with pytest.raises(MissingFoldsError):
        build_run("rank", "rank-prod", "rank-wf2", weekdays()[-1], researched)

    assert not ModelRun.objects.filter(run_key="rank-prod").exists()


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
    assert "rank-prod: rank-prod.lightgbm.h20.f000, rank-prod.lightgbm.h5.f000" in text
    assert "rank-prod-shadow: rank-prod-shadow.lightgbm.h20.f000" in text
    assert "rolling IC thresholds from rank-wf1:" in text


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
