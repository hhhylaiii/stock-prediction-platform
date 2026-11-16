"""Shadow candidates C1 and C2 (holding-shadow-prereg.md §7): their fits, files and freezes."""

import hashlib
import math
import os
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from time import monotonic
from typing import Any

import pandas as pd
from django.conf import settings
from django.db import transaction
from django.db.models import Max
from django.utils import timezone

from apps.market.calendar import TradingDays
from apps.research import lgbm_runs
from apps.research.baseline_runs import RANK_LABEL, fold_inputs, scope_features
from apps.research.baselines import fit_ridge
from apps.research.gbdt import BASE_PARAMS, bag_seeds, seeded
from apps.research.holdout import authorize_production, production_pass, require_key
from apps.research.lock import research_lock
from apps.research.models import (
    HoldoutScope,
    ModelObjective,
    ModelRun,
    Prediction,
    ShadowArtifact,
    ShadowCandidate,
)
from apps.research.predict import Slot, serving_run
from apps.research.prediction_cache import fold_from_record, fold_record
from apps.research.production import BAGS, DESIGNS
from apps.research.relative_up import market_digest, market_returns
from apps.research.resources import peak_rss_mb
from apps.research.runs import (
    CHUNK,
    MODEL_SUFFIX,
    PRODUCTION,
    RunError,
    label_hash,
    lib_versions,
    pickle_beside,
)
from apps.research.target_models import RETURN_LABEL, fit_return, raw_inputs
from apps.research.training import load_labels
from apps.research.walkforward import Fold

HORIZON = 5
SEED = 0
CANDIDATE_DIR = "shadow"
RELATIVE = f"{RETURN_LABEL} - TAIEX_TR close {HORIZON}d"


@dataclass(frozen=True)
class Fitted:
    model: Any
    recipe: dict[str, Any]


@dataclass(frozen=True)
class Serving:
    run: ModelRun
    fold: Fold
    as_of: date


@dataclass(frozen=True)
class Candidate:
    """A candidate follows its slot's serving run and is fit on that run's production fold."""

    slot: Slot
    scope: str
    fit: Callable[[Serving, Path], Fitted]


@dataclass(frozen=True)
class Measured:
    path: Path
    sha256: str
    seconds: float
    peak_mb: float
    recipe: dict[str, Any]


def relative_labels(labels: pd.DataFrame, calendar: TradingDays) -> pd.DataFrame:
    """The 5-day open return less TAIEX_TR's close-to-close move over the same five sessions."""
    market = labels["date"].map(market_returns(calendar, labels["date"], HORIZON))
    return labels.assign(**{RETURN_LABEL: labels[RETURN_LABEL] - market})


def _unchanged(before: str, after: str, fold: Fold) -> None:
    if before != after:
        raise RuntimeError(f"labels changed while fitting {fold.train_start}..{fold.train_end}")


def _ridge(serving: Serving, directory: Path) -> Fitted:
    """The rank Ridge of baseline_runs on the screened features and their gap flags."""
    fold = serving.fold
    features = scope_features(HoldoutScope.RANK)
    read = fold_inputs(fold, fold.train_start, fold.train_end, directory, features)
    digest = label_hash(fold.train_start, fold.train_end, (HORIZON,))
    labels = load_labels(fold.train_start, fold.train_end, HORIZON)
    fit = fit_ridge(read.train(labels, RANK_LABEL), features, RANK_LABEL, HORIZON)
    _unchanged(digest, label_hash(fold.train_start, fold.train_end, (HORIZON,)), fold)
    recipe = {
        "model": "ridge",
        "label": RANK_LABEL,
        "features": features,
        "penalty": fit.penalty,
        "cv": {
            str(key): value if math.isfinite(value) else None for key, value in fit.scores.items()
        },
        "feature_hash": read.feature_hash,
        "label_hash": digest,
    }
    return Fitted(fit, recipe)


def _relative_digest(fold: Fold, calendar: TradingDays) -> str:
    reach = calendar.shift(fold.train_end, HORIZON)
    labels = label_hash(fold.train_start, fold.train_end, (HORIZON,))
    return f"{labels}:{market_digest(fold.train_start, reach)}"


def _relative(serving: Serving, directory: Path) -> Fitted:
    """The frozen return design fit on the 5-day return relative to TAIEX_TR."""
    fold, scope = serving.fold, HoldoutScope.RETURN
    design = DESIGNS[scope]
    features = lgbm_runs.design_features(scope, design.variant, design.target)
    if features != list(serving.run.feature_list):
        raise RunError(f"{serving.run.run_id} reads other features than the frozen return design")
    calendar = TradingDays.from_db()
    read = fold_inputs(fold, fold.train_start, fold.train_end, directory, features)
    inputs, bins = raw_inputs(read.raw, fold, features)
    digest = _relative_digest(fold, calendar)
    labels = relative_labels(load_labels(fold.train_start, fold.train_end, HORIZON), calendar)
    rounds = lgbm_runs.MAX_ROUNDS
    fitted = fit_return(inputs, labels, fold, features, SEED, rounds, BAGS)
    _unchanged(digest, _relative_digest(fold, calendar), fold)
    recipe = {
        "model": "lightgbm",
        "label": RELATIVE,
        "features": features,
        "lightgbm": {**BASE_PARAMS, **seeded(SEED)},
        "bags": BAGS,
        "bag_seeds": bag_seeds(SEED, BAGS),
        "max_rounds": rounds,
        "rounds": fitted.boosted.rounds,
        "delta": fitted.delta,
        "feature_hash": read.feature_hash,
        "label_hash": digest,
    }
    return Fitted({"booster": fitted.boosted.booster, "market_bins": dict(bins.edges)}, recipe)


CANDIDATES: dict[str, Candidate] = {
    ShadowCandidate.C1: Candidate(Slot(ModelObjective.RANK, HORIZON), HoldoutScope.RANK, _ridge),
    ShadowCandidate.C2: Candidate(
        Slot(ModelObjective.REGRESSION, HORIZON), HoldoutScope.RETURN, _relative
    ),
}


def candidate(name: str) -> Candidate:
    if name not in CANDIDATES:
        raise ValueError(f"candidate must be one of {sorted(CANDIDATES)}, not {name!r}")
    return CANDIDATES[name]


def serving(chosen: Candidate) -> Serving:
    """The slot's serving production model N and its fold; only a production fold is fit on."""
    run = serving_run(chosen.slot)
    if run is None:
        raise RunError(f"no serving run for {chosen.slot}; a candidate follows one")
    note = run.params.get(PRODUCTION) or {}
    if note.get("role") != "model" or "fold" not in run.params:
        raise RunError(f"{run.run_id} is not a production model N with a recorded fold")
    fold = fold_from_record(run.params["fold"])
    authorize_production(run.run_key, production_pass(chosen.scope), fold)
    return Serving(run, fold, date.fromisoformat(note["as_of"]))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def artifact_path(name: str, run_key: str, generation: int) -> str:
    """The file's path under MODEL_DIR, the form shadow_artifact keeps."""
    folder = Path(CANDIDATE_DIR) / name.lower() / require_key(run_key)
    return str(folder / f"g{generation}{MODEL_SUFFIX}")


def first_signal(slot: Slot, as_of: date, calendar: TradingDays, today: date) -> date:
    """The first trading day from today on, after as_of, whose slot predictions are unwritten."""
    rows = Prediction.objects.filter(run__objective=slot.objective, horizon=slot.horizon)
    latest = rows.aggregate(last=Max("date"))["last"]
    unwritten = calendar.shift(as_of if latest is None else max(latest, as_of), 1)
    return max(
        unwritten, calendar.shift(calendar.latest_on_or_before(today - timedelta(days=1)), 1)
    )


def _recipe(fitted: Fitted, seconds: float, peak: float) -> dict[str, Any]:
    return {**fitted.recipe, "lib_versions": lib_versions(), "seconds": seconds, "peak_mb": peak}


def measure(name: str, directory: Path, output: Path) -> Measured:
    """Fit the candidate into output without the lock or any database write, for timing it."""
    chosen = candidate(name)
    found = serving(chosen)
    began = monotonic()
    fitted = chosen.fit(found, directory)
    seconds, peak = round(monotonic() - began, 1), round(peak_rss_mb(), 1)
    path = output / f"{name.lower()}{MODEL_SUFFIX}"
    temporary = Path(pickle_beside(path, fitted.model))
    os.replace(temporary, path)
    return Measured(path, file_sha256(path), seconds, peak, _recipe(fitted, seconds, peak))


def freeze(name: str, directory: Path, replace: bool = False) -> ShadowArtifact:
    """Fit, write and freeze the candidate's next generation under the research lock."""
    chosen = candidate(name)
    with research_lock(f"shadow_train {name}"):
        found = serving(chosen)
        kept = ShadowArtifact.objects.filter(candidate=name, serving_run=found.run)
        generation = (kept.aggregate(last=Max("generation"))["last"] or 0) + 1
        if generation > 1 and not replace:
            raise RunError(
                f"{name} is frozen on {found.run.run_id}; replace writes generation {generation}"
            )
        began = monotonic()
        fitted = chosen.fit(found, directory)
        seconds, peak = round(monotonic() - began, 1), round(peak_rss_mb(), 1)
        relative = artifact_path(name, found.run.run_key, generation)
        path = Path(settings.MODEL_DIR) / relative
        temporary = Path(pickle_beside(path, fitted.model))
        try:
            with transaction.atomic(durable=True):
                row = ShadowArtifact.objects.create(
                    candidate=name,
                    serving_run=found.run,
                    generation=generation,
                    as_of=found.as_of,
                    train_start=found.fold.train_start,
                    train_end=found.fold.train_end,
                    fold=fold_record(found.fold),
                    path=relative,
                    sha256=file_sha256(temporary),
                    recipe=_recipe(fitted, seconds, peak),
                    first_signal=first_signal(
                        chosen.slot, found.as_of, TradingDays.from_db(), timezone.localdate()
                    ),
                    frozen_at=timezone.now(),
                )
                os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return row


def load_artifact(row: ShadowArtifact) -> Any:
    """The frozen model, refused when its file is missing or no longer matches its hash."""
    path = Path(settings.MODEL_DIR) / row.path
    if not path.exists():
        raise RunError(f"{row} has no file at {row.path}")
    with path.open("rb") as handle:
        data = handle.read()
    if hashlib.sha256(data).hexdigest() != row.sha256:
        raise RunError(f"{row}'s file no longer matches its frozen sha256")
    return pickle.loads(data)  # noqa: S301
