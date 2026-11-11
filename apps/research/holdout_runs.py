"""The one holdout run of a scope's frozen design, and its rehearsal on a research fold (§7)."""

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from django.conf import settings

from apps.market.calendar import TradingDays
from apps.research import baseline_runs, lgbm_runs, production
from apps.research.gbdt import bag_seeds
from apps.research.holdout import (
    HoldoutError,
    HoldoutTicket,
    claim_holdout,
    is_holdout,
    latest_ticket,
)
from apps.research.labels import latest_price_date
from apps.research.models import LabelDaily, ModelRun
from apps.research.prediction_cache import (
    fold_cached,
    fold_path,
    save_fold_predictions,
    save_holdout_fold,
)
from apps.research.production import DESIGN, Design, design_note
from apps.research.runs import FittedFold, record_run, refuse_production_key
from apps.research.walkforward import Fold, generate_folds

REHEARSAL = "rehearsal"
MODELS = (baseline_runs.BASELINE, lgbm_runs.MODEL)


@dataclass(frozen=True)
class HoldoutOutcome:
    ticket: HoldoutTicket
    ran: list[int]
    kept: list[int]


def frozen_design(scope: str) -> Design:
    """The scope's frozen design; one naming no research run is not frozen yet."""
    design = production.DESIGNS[scope]
    if design.research_run_key is None:
        raise HoldoutError(
            f"the {scope} design is not frozen: production.DESIGNS names no research run for it"
        )
    return design


def recorded_design(design: Design) -> dict[str, Any]:
    """The design as a run's params store it, after a JSON round trip."""
    loaded: dict[str, Any] = json.loads(json.dumps(design_note(design)))
    return loaded


def holdout_folds(scope: str, last: date, calendar: TradingDays) -> list[Fold]:
    """The frozen design's folds testing from the holdout start, on data through last."""
    folds = generate_folds(calendar, last, frozen_design(scope).config)
    return [fold for fold in folds if is_holdout(fold)]


def _rehearsal_key(scope: str) -> str:
    return f"{scope}-{REHEARSAL}"


def _check_key(scope: str, run_key: str, rehearsal: bool) -> None:
    baseline_runs.check_run_key(scope, run_key)
    refuse_production_key(run_key)
    marked = run_key.startswith(_rehearsal_key(scope))
    if rehearsal and not marked:
        raise ValueError(f"a rehearsal run_key starts with {_rehearsal_key(scope)}")
    if marked and not rehearsal:
        raise ValueError(f"{run_key} names a rehearsal; the holdout needs its own run_key")


def _fit(
    scope: str, fold: Fold, run_key: str, directory: Path, seed: int, bags: int
) -> dict[str, FittedFold]:
    """The fold's baselines and boosters on the frozen design, fitted but not written."""
    design = frozen_design(scope)
    features = lgbm_runs.design_features(scope, design.variant, design.target)
    return {
        baseline_runs.BASELINE: baseline_runs.fit_fold(
            scope, fold, run_key, directory, baseline_runs.scope_features(scope), design.target
        ),
        lgbm_runs.MODEL: lgbm_runs.fit_fold(
            scope, fold, run_key, directory, features, seed, bags, design.variant, design.target
        ),
    }


def _record(
    scope: str, fold: Fold, fitted: dict[str, FittedFold], ticket: HoldoutTicket | None
) -> None:
    note = recorded_design(frozen_design(scope))
    for part in fitted.values():
        for run, model in part.trained:
            run.params = {**run.params, DESIGN: note}
            record_run(run, model, fold=fold, ticket=ticket)


def _unlabelled(scope: str, folds: list[Fold]) -> list[str]:
    """Fold and horizon pairs without a settled label in test_start..eval_end."""
    return [
        f"fold {fold.index} at {horizon} days"
        for fold in folds
        for horizon in lgbm_runs.SCOPE_HORIZONS[scope]
        if not LabelDaily.objects.filter(
            date__range=(fold.test_start, fold.eval_end),
            horizon=horizon,
            fwd_ret_open__isnull=False,
        ).exists()
    ]


def _check_claim(scope: str, run_key: str, last: date, calendar: TradingDays) -> None:
    """Refuse a claim the holdout could not finish: no data, no folds, no labels, a used key."""
    price = latest_price_date()
    if price is None or last > price:
        raise HoldoutError(f"{last} is past the latest price day {price}")
    folds = holdout_folds(scope, last, calendar)
    if not folds:
        raise HoldoutError(f"the {scope} design has no holdout fold through {last}")
    missing = _unlabelled(scope, folds)
    if missing:
        raise HoldoutError(f"no settled label yet for {', '.join(missing)}; claim once it is built")
    cached = Path(settings.RESEARCH_CACHE_DIR) / run_key
    if ModelRun.objects.filter(run_key=run_key).exists() or cached.exists():
        raise HoldoutError(f"{run_key} already holds runs or caches; the holdout needs a new key")


def _check_resume(run_key: str, seed: int, bags: int) -> None:
    """Refuse resuming a claim with another seed or bag count than its written folds used."""
    runs = ModelRun.objects.filter(run_key=run_key, model_type=lgbm_runs.MODEL)
    used = {(run.seed, run.params.get("bags")) for run in runs}
    if used and used != {(seed, bags)}:
        raise HoldoutError(
            f"{run_key} was fit with (seed, bags) {sorted(used)}; resume it with the same"
        )


def run_holdout(
    scope: str,
    run_key: str,
    last: date,
    directory: Path,
    *,
    seed: int = 0,
    bags: int | None = None,
    force: bool = False,
    reason: str = "",
) -> HoldoutOutcome:
    """Claim the scope's holdout once it can finish, then fit and write every unwritten fold.

    A second call with the latest claim's run_key resumes it on that claim's last date.
    """
    _check_key(scope, run_key, rehearsal=False)
    design = frozen_design(scope)
    lgbm_runs.check_design(scope, design.variant, design.target)
    bagged = production.BAGS if bags is None else bags
    bag_seeds(seed, bagged)
    calendar = TradingDays.from_db()
    latest = latest_ticket(scope)
    if latest is not None and latest.run_key == run_key and not force:
        _check_resume(run_key, seed, bagged)
        ticket = latest
    else:
        _check_claim(scope, run_key, last, calendar)
        ticket = claim_holdout(scope, run_key, last, force=force, reason=reason)
    folds = holdout_folds(scope, ticket.last_date, calendar)
    kept = [fold.index for fold in folds if fold_cached(run_key, fold.index)]
    ran = []
    for fold in folds:
        if fold.index in kept:
            continue
        fitted = _fit(scope, fold, run_key, directory, seed, bagged)
        _record(scope, fold, fitted, ticket)
        predictions = {model: part.predictions for model, part in fitted.items()}
        notes = {model: part.notes for model, part in fitted.items()}
        save_holdout_fold(run_key, fold, predictions, ticket, notes)
        ran.append(fold.index)
    return HoldoutOutcome(ticket, ran, kept)


def research_fold(scope: str, index: int | None, last: date, calendar: TradingDays) -> Fold:
    """A research fold of the frozen design, the last one when no index is given."""
    research = [
        fold
        for fold in generate_folds(calendar, last, frozen_design(scope).config)
        if fold.research
    ]
    if not research:
        raise ValueError(f"the {scope} design has no research fold through {last}")
    if index is None:
        return research[-1]
    chosen = [fold for fold in research if fold.index == index]
    if not chosen:
        raise ValueError(f"fold {index} is not a research fold of the {scope} design")
    return chosen[0]


def rehearse(
    scope: str, run_key: str, fold: Fold, directory: Path, *, seed: int = 0, bags: int = 1
) -> list[Path]:
    """Fit and record one research fold as the holdout would, writing research caches only."""
    _check_key(scope, run_key, rehearsal=True)
    if not fold.research:
        raise ValueError(f"a rehearsal runs on a research fold, not fold {fold.index}")
    baseline_runs.require_candidates(run_key, fold)
    for model in MODELS:
        fold_path(run_key, model, fold.index).unlink(missing_ok=True)
    fitted = _fit(scope, fold, run_key, directory, seed, bags)
    _record(scope, fold, fitted, None)
    return [
        save_fold_predictions(run_key, model, fold, part.predictions, notes=part.notes)
        for model, part in fitted.items()
    ]
