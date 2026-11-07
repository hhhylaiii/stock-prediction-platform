"""Production runs (SPEC §7): the frozen design fit through the latest labels, and its shadow."""

import math
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from django.db import transaction

from apps.market.calendar import TradingDays
from apps.research.baseline_runs import check_run_key
from apps.research.holdout import production_pass
from apps.research.lgbm_runs import MODEL, run_fold, scope_features, target_note, variant_note
from apps.research.models import HoldoutScope, ModelObjective, ModelRun
from apps.research.monitor_refs import research_brier, research_ic_thresholds, store_references
from apps.research.rank_model import SPEC_RECIPE, Variant
from apps.research.runs import RunError
from apps.research.target_models import TARGET_RECIPE, TargetVariant
from apps.research.walkforward import (
    Fold,
    WalkForwardConfig,
    WalkForwardError,
    add_months,
    generate_folds,
    training_window,
)

SHADOW_MONTHS = 6
PRODUCTION_FOLD = 0


def _fold(
    calendar: TradingDays,
    config: WalkForwardConfig,
    train_end: date,
    test: tuple[date, date],
    eval_end: date,
) -> Fold:
    window = training_window(calendar, config, train_end, test[0])
    if window is None:
        raise WalkForwardError(
            f"the training window before {test[0]} starts before data_start {config.data_start}"
        )
    train_start, fit_end, valid_start = window
    return Fold(
        index=PRODUCTION_FOLD,
        train_start=train_start,
        fit_end=fit_end,
        valid_start=valid_start,
        train_end=train_end,
        test_start=test[0],
        test_end=test[1],
        eval_end=eval_end,
        tuning=False,
        holdout=False,
    )


def production_fold(calendar: TradingDays, as_of: date, config: WalkForwardConfig) -> Fold:
    """Fold of model N: training ends where as_of's labels end, the test is as_of alone."""
    train_end = calendar.shift(as_of, -config.embargo)
    return _fold(calendar, config, train_end, (as_of, as_of), train_end)


def shadow_fold(calendar: TradingDays, as_of: date, config: WalkForwardConfig) -> Fold:
    """Fold of shadow N′: it tests the SHADOW_MONTHS through as_of after the usual embargo."""
    test_start = calendar.between(add_months(as_of, -SHADOW_MONTHS) + timedelta(days=1), as_of)[0]
    train_end = calendar.shift(test_start, -(config.embargo + 1))
    eval_end = calendar.shift(as_of, -config.embargo)
    return _fold(calendar, config, train_end, (test_start, as_of), eval_end)


@dataclass(frozen=True)
class Design:
    """A scope's frozen design: walk-forward layout, rank variant and target variant."""

    config: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    variant: Variant = SPEC_RECIPE
    target: TargetVariant = TARGET_RECIPE


@dataclass(frozen=True)
class Built:
    model: list[str]
    shadow: list[str]
    thresholds: dict[int, float]
    brier: float


DESIGNS: dict[str, Design] = {scope: Design() for scope in HoldoutScope.values}
BAGS = 5
SHADOW = "shadow"


def shadow_key(run_key: str) -> str:
    """The run_key of a production run's shadow N′."""
    return f"{run_key}-{SHADOW}"


def design_note(design: Design) -> dict[str, Any]:
    """The design as production params record it."""
    config = {
        name: value.isoformat() if isinstance(value, date) else value
        for name, value in asdict(design.config).items()
    }
    return {
        "config": config,
        "variant": variant_note(design.variant),
        "target": target_note(design.target),
    }


def _references(
    scope: str, research_run_key: str, folds: list[Fold]
) -> tuple[dict[int, float], float]:
    if scope == HoldoutScope.RANK:
        return research_ic_thresholds(research_run_key, MODEL, folds), math.nan
    if scope == HoldoutScope.INTERVAL:
        return {}, research_brier(research_run_key, folds)
    return {}, math.nan


def _annotate(run_key: str, note: dict[str, Any]) -> list[str]:
    runs = list(ModelRun.objects.filter(run_key=run_key).order_by("run_id"))
    for run in runs:
        run.params = {**run.params, **note}
        run.save(update_fields=["params"])
    return [run.run_id for run in runs]


def _check_keys(scope: str, run_key: str, research_run_key: str) -> None:
    for key in (run_key, shadow_key(run_key), research_run_key):
        check_run_key(scope, key)
    if not ModelRun.objects.filter(run_key=research_run_key).exists():
        raise RunError(f"no model_run of the research run {research_run_key}")


def build_run(
    scope: str, run_key: str, research_run_key: str, as_of: date, directory: Path, seed: int = 0
) -> Built:
    """Fit N and its shadow N′ on scope's frozen design through as_of, with monitor references."""
    granted = production_pass(scope)
    _check_keys(scope, run_key, research_run_key)
    design = DESIGNS[scope]
    calendar = TradingDays.from_db()
    research = [fold for fold in generate_folds(calendar, as_of, design.config) if fold.research]
    thresholds, brier = _references(scope, research_run_key, research)
    masked = design.variant.masked
    features = [name for name in scope_features(scope, design.target) if name not in masked]
    folds = {
        run_key: production_fold(calendar, as_of, design.config),
        shadow_key(run_key): shadow_fold(calendar, as_of, design.config),
    }
    for key, fold in folds.items():
        run_fold(
            scope,
            fold,
            key,
            directory,
            features,
            seed,
            BAGS,
            design.variant,
            design.target,
            granted,
        )
    note = {"research_run_key": research_run_key, "design": design_note(design)}
    note["as_of"] = as_of.isoformat()
    with transaction.atomic():
        model = _annotate(run_key, {**note, "shadow_run_key": shadow_key(run_key)})
        shadow = _annotate(shadow_key(run_key), {**note, "shadow_of": run_key})
        for run in ModelRun.objects.filter(run_key=run_key):
            if run.objective == ModelObjective.RANK:
                store_references(run.run_id, thresholds, math.nan)
            elif run.objective == ModelObjective.CLASSIFICATION:
                store_references(run.run_id, {}, brier)
    return Built(model, shadow, thresholds, brier)
