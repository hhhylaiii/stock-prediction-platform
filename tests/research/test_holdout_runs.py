import shutil
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.market.models import IndexDaily
from apps.research import (
    baseline_runs,
    holdout,
    holdout_runs,
    lgbm_runs,
    prediction_cache,
    production,
)
from apps.research.baseline_runs import BASELINE
from apps.research.holdout import HoldoutError, claim_holdout
from apps.research.holdout_runs import (
    recorded_design,
    rehearse,
    research_fold,
    run_holdout,
)
from apps.research.lgbm_runs import MODEL
from apps.research.management.commands import holdout as holdout_command
from apps.research.models import HoldoutRun, LabelDaily, ModelRun
from apps.research.prediction_cache import fold_cached, fold_path, load_fold_predictions
from apps.research.production import DESIGN, Design
from apps.research.runs import SCREENING_MODEL, RunError
from apps.research.target_models import TargetVariant
from apps.research.walkforward import Fold
from tests.research.run_panels import CONFIG, folds, use_designs, weekdays

LAST = weekdays()[-1]
PLAIN = TargetVariant(market=False)
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
    monkeypatch.setattr(production, "BAGS", 1)
    monkeypatch.setattr(holdout_runs, "latest_price_date", lambda: LAST)
    monkeypatch.setattr(holdout_command, "latest_price_date", lambda: LAST)
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
def test_the_command_runs_the_holdout_once_and_names_the_folds(frozen: Path) -> None:
    out = StringIO()

    call_command("holdout", scope="return", run_key="return-final", stdout=out)
    call_command("holdout", scope="return", run_key="return-final", stdout=out)

    index = held()[0].index
    lines = out.getvalue().splitlines()
    assert lines[0] == (
        f"return-final: holdout attempt 1 of return through {LAST}; "
        f"fitted folds [{index}], already written []"
    )
    assert lines[1].endswith(f"fitted folds [], already written [{index}]")
    with pytest.raises(CommandError, match="rerun only with force and a reason"):
        call_command("holdout", scope="return", run_key="return-other")


@pytest.mark.django_db
def test_the_command_rehearses_a_research_fold_without_a_claim(frozen: Path) -> None:
    out = StringIO()
    fold = next(f for f in folds() if f.research)

    call_command(
        "holdout",
        scope="return",
        run_key="return-rehearsal",
        rehearse=True,
        fold=fold.index,
        bags=1,
        last_date=DAY,
        stdout=out,
    )

    assert out.getvalue().strip() == f"return-rehearsal: rehearsed research fold {fold.index}"
    assert fold_path("return-rehearsal", MODEL, fold.index).exists()
    assert not HoldoutRun.objects.exists()


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"rehearse": True, "force": True}, "a rehearsal claims none"),
        ({"rehearse": True, "reason": "why"}, "a rehearsal claims none"),
        ({"fold": 3}, "--fold apply to a rehearsal"),
        ({"last_date": DAY}, "--last-date apply to a rehearsal"),
        ({"bags": 5, "seed": 1}, "--bags, --seed apply to a rehearsal"),
        ({"rehearse": True, "bags": 0}, "at least 1"),
    ],
)
def test_the_command_refuses_options_that_do_not_go_together(
    frozen: Path, options: dict[str, Any], message: str
) -> None:
    with pytest.raises(CommandError, match=message):
        call_command("holdout", scope="return", run_key="return-final", **options)

    assert not HoldoutRun.objects.exists()


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("last", "message"),
    [
        (date(2017, 1, 6), "past the latest price day"),
        (date(2016, 11, 15), "no holdout fold through 2016-11-15"),
    ],
)
def test_a_claim_the_holdout_could_not_finish_is_never_written(
    frozen: Path, last: date, message: str
) -> None:
    with pytest.raises(HoldoutError, match=message):
        run_holdout("return", "return-final", last, frozen)

    assert not HoldoutRun.objects.exists()


@pytest.mark.django_db
def test_a_claim_waits_for_every_holdout_fold_to_have_a_settled_label(frozen: Path) -> None:
    fold = held()[0]
    LabelDaily.objects.filter(date__gte=fold.test_start, horizon=5).update(fwd_ret_open=None)

    with pytest.raises(HoldoutError, match=f"no settled label yet for fold {fold.index} at 5 days"):
        run_holdout("return", "return-final", LAST, frozen)

    assert not HoldoutRun.objects.exists()


@pytest.mark.django_db
@pytest.mark.parametrize("left", ["runs", "caches"])
def test_a_claim_needs_a_key_holding_neither_runs_nor_caches(
    frozen: Path, settings: Any, left: str
) -> None:
    cached = Path(settings.RESEARCH_CACHE_DIR) / "interval-used"
    if left == "runs":
        fold = next(f for f in folds() if f.research)
        baseline_runs.run_folds("interval", [fold], "interval-used", frozen, target=PLAIN)
        shutil.rmtree(cached)
    else:
        cached.mkdir(parents=True)
    assert ModelRun.objects.filter(run_key="interval-used").exists() is (left == "runs")

    with pytest.raises(HoldoutError, match="already holds runs or caches"):
        run_holdout("interval", "interval-used", LAST, frozen)

    assert not HoldoutRun.objects.exists()


@pytest.mark.django_db
def test_a_claim_needs_a_design_the_runners_accept_and_a_screening(
    frozen: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = Design(CONFIG, target=PLAIN, research_run_key="rank-wf1")
    monkeypatch.setitem(production.DESIGNS, "rank", target)
    with pytest.raises(ValueError, match="target variants apply"):
        run_holdout("rank", "rank-final", LAST, frozen)
    ModelRun.objects.filter(model_type=SCREENING_MODEL).delete()
    with pytest.raises(RunError, match="no feature screening is recorded"):
        run_holdout("interval", "interval-final", LAST, frozen)

    assert not HoldoutRun.objects.exists()


def disk_full(*args: Any, **kwargs: Any) -> list[Path]:
    raise OSError("disk full")


@pytest.mark.django_db
def test_a_resume_on_another_design_or_other_features_is_refused(
    frozen: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(holdout_runs, "save_holdout_fold", disk_full)
    with pytest.raises(OSError, match="disk full"):
        run_holdout("return", "return-final", LAST, frozen)
    monkeypatch.setattr(holdout_runs, "save_holdout_fold", prediction_cache.save_holdout_fold)
    began = production.DESIGNS["return"]
    other = Design(CONFIG, target=TargetVariant(market=True), research_run_key="return-wf1")
    monkeypatch.setitem(production.DESIGNS, "return", other)

    with pytest.raises(
        HoldoutError,
        match="not the one holdout return-final ran; resume it on the design it began with",
    ):
        run_holdout("return", "return-final", LAST, frozen)
    monkeypatch.setitem(production.DESIGNS, "return", began)
    screening = ModelRun.objects.filter(model_type=SCREENING_MODEL)
    kept = screening.get().feature_list
    screening.update(feature_list=kept[1:])
    with pytest.raises(HoldoutError, match="fit other features"):
        run_holdout("return", "return-final", LAST, frozen)
    screening.update(feature_list=kept)

    assert run_holdout("return", "return-final", LAST, frozen).ran == [held()[0].index]
    assert len({str(run.params[DESIGN]) for run in runs_of("return-final")}) == 1


@pytest.mark.django_db
@pytest.mark.parametrize("linked", [(), (BASELINE,)])
def test_a_fold_taken_but_not_fully_written_stops_the_resume(
    frozen: Path, settings: Any, linked: tuple[str, ...]
) -> None:
    run_holdout("return", "return-final", LAST, frozen)
    index = held()[0].index
    for model in (BASELINE, MODEL):
        if model not in linked:
            fold_path("return-final", model, index).unlink()

    with pytest.raises(HoldoutError, match=rf"folds \[{index}\] of return-final were taken but"):
        run_holdout("return", "return-final", LAST, frozen)
    for model in linked:
        fold_path("return-final", model, index).unlink()
    (Path(settings.RESEARCH_CACHE_DIR) / "return-final" / f"fold-{index:03d}.holdout").unlink()

    again = run_holdout("return", "return-final", LAST, frozen)

    assert (again.ran, again.kept) == ([index], [])
    assert HoldoutRun.objects.count() == 1


@pytest.mark.django_db
def test_the_latest_claim_resumes_under_the_command_that_forced_it(
    frozen: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_holdout("return", "return-final", LAST, frozen)
    forced: dict[str, Any] = {"force": True, "reason": "labels rebuilt"}
    monkeypatch.setattr(holdout_runs, "save_holdout_fold", disk_full)
    with pytest.raises(OSError, match="disk full"):
        run_holdout("return", "return-again", LAST, frozen, **forced)
    monkeypatch.setattr(holdout_runs, "save_holdout_fold", prediction_cache.save_holdout_fold)

    again = run_holdout("return", "return-again", LAST, frozen, **forced)

    assert (again.ticket.attempt, again.ran) == (2, [held()[0].index])
    assert HoldoutRun.objects.count() == 2
    with pytest.raises(HoldoutError, match="already holds runs or caches"):
        run_holdout("return", "return-final", LAST, frozen, **forced)
    assert HoldoutRun.objects.count() == 2


@pytest.mark.django_db
def test_a_resume_with_other_bags_or_seed_is_refused(
    frozen: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> list[Path]:
        raise OSError("disk full")

    monkeypatch.setattr(holdout_runs, "save_holdout_fold", broken)
    with pytest.raises(OSError, match="disk full"):
        run_holdout("return", "return-final", LAST, frozen, bags=2)

    with pytest.raises(HoldoutError, match=r"fit with \(seed, bags\) \[\(0, 2\)\]"):
        run_holdout("return", "return-final", LAST, frozen, bags=1)
    with pytest.raises(HoldoutError, match="resume it with the same"):
        run_holdout("return", "return-final", LAST, frozen, seed=7, bags=2)


@pytest.mark.django_db
def test_a_resume_after_one_of_two_folds_fits_only_the_unwritten_one(
    frozen: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(holdout, "SPEC_HOLDOUT_START", date(2016, 11, 1))
    monkeypatch.setattr(prediction_cache, "SPEC_HOLDOUT_START", date(2016, 11, 1))
    november, december = (f.index for f in folds() if f.test_end >= date(2016, 11, 1))
    save = prediction_cache.save_holdout_fold

    def second_fails(run_key: str, fold: Fold, *args: Any, **kwargs: Any) -> list[Path]:
        if fold.index == december:
            raise OSError("disk full")
        return save(run_key, fold, *args, **kwargs)

    monkeypatch.setattr(holdout_runs, "save_holdout_fold", second_fails)
    with pytest.raises(OSError, match="disk full"):
        run_holdout("return", "return-final", LAST, frozen)
    monkeypatch.setattr(holdout_runs, "save_holdout_fold", save)

    again = run_holdout("return", "return-final", LAST, frozen)

    assert (again.ran, again.kept) == ([december], [november])


@pytest.mark.django_db
def test_a_rehearsal_refuses_a_fold_that_is_not_research(frozen: Path) -> None:
    tuning = next(f for f in folds() if f.tuning)

    with pytest.raises(ValueError, match="runs on a research fold"):
        rehearse("return", "return-rehearsal", tuning, frozen)


@pytest.mark.django_db
def test_research_fits_refuse_a_claimed_holdout_key(frozen: Path) -> None:
    run_holdout("return", "return-final", LAST, frozen)
    research = [f for f in folds() if f.research]

    with pytest.raises(RunError, match="holds a holdout claim"):
        baseline_runs.run_folds("return", research, "return-final", frozen, target=PLAIN)
    with pytest.raises(RunError, match="holds a holdout claim"):
        lgbm_runs.run_folds("return", research, "return-final", frozen, target=PLAIN)
