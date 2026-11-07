from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from apps.market.calendar import TradingDays
from apps.research import holdout, lgbm_runs
from apps.research.holdout import HoldoutError, claim_holdout, production_pass
from apps.research.lgbm_runs import MODEL, run_fold, run_id
from apps.research.models import ModelRun
from apps.research.prediction_cache import fold_notes, load_fold_predictions
from apps.research.production import SHADOW_MONTHS, production_fold, shadow_fold
from apps.research.rank_model import SPEC_RECIPE
from apps.research.walkforward import WalkForwardConfig, WalkForwardError, add_months
from tests.research.run_panels import CONFIG, KEPT, UNSCREENED, weekdays

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
