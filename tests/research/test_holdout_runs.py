from datetime import date
from pathlib import Path
from typing import Any

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.market.models import IndexDaily
from apps.research import holdout, holdout_runs, lgbm_runs, prediction_cache, production
from apps.research.baseline_runs import BASELINE
from apps.research.holdout import HoldoutError, claim_holdout
from apps.research.holdout_runs import (
    recorded_design,
    rehearse,
    research_fold,
    run_holdout,
)
from apps.research.lgbm_runs import MODEL
from apps.research.models import HoldoutRun, ModelRun
from apps.research.prediction_cache import fold_cached, fold_path, load_fold_predictions
from apps.research.production import DESIGN, Design
from apps.research.runs import RunError
from apps.research.target_models import TargetVariant
from apps.research.walkforward import Fold
from tests.research.run_panels import CONFIG, folds, use_designs, weekdays

LAST = weekdays()[-1]
DAY = LAST.isoformat()
RELATIVE = TargetVariant(market=False, relative_up=True)


@pytest.fixture
def frozen(run_panel: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The panel with December as holdout and every scope's design frozen on a research run."""
    monkeypatch.setattr(lgbm_runs, "MAX_ROUNDS", 40)
    monkeypatch.setattr(holdout, "SPEC_HOLDOUT_START", CONFIG.holdout_start)
    monkeypatch.setattr(prediction_cache, "SPEC_HOLDOUT_START", CONFIG.holdout_start)
    use_designs(monkeypatch)
    plain = TargetVariant(market=False)
    frozen = {
        "rank": Design(CONFIG, research_run_key="rank-wf1"),
        "return": Design(CONFIG, target=plain, research_run_key="return-wf1"),
        "interval": Design(CONFIG, target=plain, research_run_key="interval-wf1"),
    }
    monkeypatch.setattr(production, "DESIGNS", frozen)
    return run_panel


def held() -> list[Fold]:
    return [fold for fold in folds() if fold.holdout]


def runs_of(run_key: str) -> list[ModelRun]:
    return list(ModelRun.objects.filter(run_key=run_key).order_by("run_id"))


@pytest.mark.django_db
def test_the_holdout_claims_then_fits_every_holdout_fold_once_on_the_frozen_design(
    frozen: Path,
) -> None:
    outcome = run_holdout("rank", "rank-final", LAST, frozen, bags=2)

    assert [fold.index for fold in held()] == outcome.ran
    assert outcome.kept == []
    assert (outcome.ticket.scope, outcome.ticket.attempt) == ("rank", 1)
    assert list(HoldoutRun.objects.values_list("run_key", flat=True)) == ["rank-final"]
    fold = held()[0]
    assert fold_cached("rank-final", fold.index)
    scores = load_fold_predictions("rank-final", MODEL, fold)
    ridge = load_fold_predictions("rank-final", BASELINE, fold)
    assert {"lightgbm_h5", "lightgbm_h20"} <= set(scores.columns)
    assert {"ridge_h5", "ridge_h20"} <= set(ridge.columns)
    note = recorded_design(production.DESIGNS["rank"])
    fitted = runs_of("rank-final")
    assert {run.model_type for run in fitted} == {"lightgbm", "ridge"}
    assert all(run.params[DESIGN] == note for run in fitted)
    assert {run.params["bags"] for run in fitted if run.model_type == "lightgbm"} == {2}


@pytest.mark.django_db
def test_the_holdout_resumes_its_own_claim_and_refits_only_unwritten_folds(
    frozen: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fold = held()[0]

    def broken(*args: Any, **kwargs: Any) -> list[Path]:
        raise OSError("disk full")

    monkeypatch.setattr(holdout_runs, "save_holdout_fold", broken)
    with pytest.raises(OSError, match="disk full"):
        run_holdout("return", "return-final", LAST, frozen, bags=1)
    assert not fold_cached("return-final", fold.index)
    monkeypatch.setattr(holdout_runs, "save_holdout_fold", prediction_cache.save_holdout_fold)

    again = run_holdout("return", "return-final", date(2017, 3, 1), frozen, bags=1)
    assert (again.ran, again.ticket.last_date) == ([fold.index], LAST)
    done = run_holdout("return", "return-final", LAST, frozen, bags=1)
    assert (done.ran, done.kept) == ([], [fold.index])
    assert HoldoutRun.objects.count() == 1


@pytest.mark.django_db
def test_a_second_holdout_under_another_key_is_refused_without_force_and_a_reason(
    frozen: Path,
) -> None:
    run_holdout("return", "return-final", LAST, frozen, bags=1)

    with pytest.raises(HoldoutError, match="rerun only with force and a reason"):
        run_holdout("return", "return-again", LAST, frozen, bags=1)
    with pytest.raises(HoldoutError, match="needs a reason"):
        run_holdout("return", "return-again", LAST, frozen, bags=1, force=True)

    forced = run_holdout(
        "return", "return-again", LAST, frozen, bags=1, force=True, reason="labels rebuilt"
    )
    assert forced.ticket.attempt == 2
    assert runs_of("return-again")


@pytest.mark.django_db
def test_a_design_that_is_not_frozen_runs_no_holdout(
    frozen: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(production.DESIGNS, "rank", Design(CONFIG))

    with pytest.raises(HoldoutError, match="rank design is not frozen"):
        run_holdout("rank", "rank-final", LAST, frozen)

    assert not HoldoutRun.objects.exists()


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("run_key", "message"),
    [
        ("rank-rehearsal-1", "names a rehearsal"),
        ("return-final", "must start with rank-"),
    ],
)
def test_the_holdout_refuses_rehearsal_and_foreign_keys_before_claiming(
    frozen: Path, run_key: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_holdout("rank", run_key, LAST, frozen)

    assert not HoldoutRun.objects.exists()


@pytest.mark.django_db
def test_the_holdout_refuses_a_key_holding_a_production_build(frozen: Path) -> None:
    ModelRun.objects.create(
        run_id="rank-prod.lightgbm.h5.f000",
        run_key="rank-prod",
        model_type="lightgbm",
        objective="rank",
        version="4",
        train_start=date(2016, 1, 4),
        train_end=date(2016, 6, 1),
        params={"production": {"role": "model"}},
        feature_list=[],
        lib_versions={},
        feature_hash="0" * 64,
    )

    with pytest.raises(RunError, match="holds a production build"):
        run_holdout("rank", "rank-prod", LAST, frozen)
    with pytest.raises(HoldoutError, match="holds a production build"):
        claim_holdout("rank", "rank-prod", LAST)

    assert not HoldoutRun.objects.exists()


@pytest.mark.django_db
def test_a_rehearsal_fits_a_research_fold_on_the_same_path_without_a_claim(
    frozen: Path,
) -> None:
    fold = research_fold("interval", None, LAST, holdout_runs.TradingDays.from_db())

    paths = rehearse("interval", "interval-rehearsal", fold, frozen, bags=1)

    assert paths == [
        fold_path("interval-rehearsal", BASELINE, fold.index),
        fold_path("interval-rehearsal", MODEL, fold.index),
    ]
    assert fold.index == max(f.index for f in folds() if f.research)
    assert not HoldoutRun.objects.exists()
    fitted = runs_of("interval-rehearsal")
    note = recorded_design(production.DESIGNS["interval"])
    assert {run.objective for run in fitted} >= {"quantile", "volatility", "classification"}
    assert all(run.params[DESIGN] == note for run in fitted)
    again = rehearse("interval", "interval-rehearsal", fold, frozen, bags=1)
    assert again == paths


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("run_key", "index", "message"),
    [
        ("interval-final", None, "starts with interval-rehearsal"),
        ("interval-rehearsal", "holdout", "not a research fold"),
        ("interval-rehearsal", "tuning", "not a research fold"),
    ],
)
def test_a_rehearsal_runs_only_on_a_rehearsal_key_and_a_research_fold(
    frozen: Path, run_key: str, index: str | None, message: str
) -> None:
    calendar = holdout_runs.TradingDays.from_db()
    if index is None:
        fold = research_fold("interval", None, LAST, calendar)
        with pytest.raises(ValueError, match=message):
            rehearse("interval", run_key, fold, frozen)
        return
    chosen = next(f for f in folds() if (f.holdout if index == "holdout" else f.tuning))
    with pytest.raises(ValueError, match=message):
        research_fold("interval", chosen.index, LAST, calendar)


@pytest.mark.django_db
def test_rehearsing_a_relative_design_uses_its_up_label(
    frozen: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    IndexDaily.objects.bulk_create(
        IndexDaily(index_id="TAIEX_TR", date=day, close=round(1000 * 0.9977**k, 4))
        for k, day in enumerate(weekdays())
    )
    monkeypatch.setitem(
        production.DESIGNS,
        "interval",
        Design(CONFIG, target=RELATIVE, research_run_key="interval-wf1"),
    )
    fold = research_fold("interval", None, LAST, holdout_runs.TradingDays.from_db())

    rehearse("interval", "interval-rehearsal", fold, frozen, bags=1)

    labels = {
        run.params["label"]
        for run in runs_of("interval-rehearsal")
        if run.objective == "classification"
    }
    assert labels == {"fwd_up_5pct_vs_taiex_tr"}


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"rehearse": True, "force": True}, "a rehearsal claims none"),
        ({"rehearse": True, "reason": "why"}, "a rehearsal claims none"),
        ({"fold": 3}, "add --rehearse"),
        ({"bags": 0}, "at least 1"),
    ],
)
def test_the_command_refuses_options_that_do_not_go_together(
    frozen: Path, options: dict[str, Any], message: str
) -> None:
    with pytest.raises(CommandError, match=message):
        call_command("holdout", scope="return", run_key="return-final", **options)

    assert not HoldoutRun.objects.exists()
