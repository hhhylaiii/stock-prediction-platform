"""A rank run's cached folds scored against its Ridge baseline: report and metrics (SPEC §7)."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from django.db import transaction

from apps.market.calendar import TradingDays
from apps.market.models import StockMaster
from apps.research import lgbm_runs
from apps.research.baseline_runs import BASELINE, check_run_key
from apps.research.ensemble import cached_scores
from apps.research.eval_report import render_report, write_rank_metrics
from apps.research.evaluation import (
    ImportanceDrift,
    RankReport,
    decile_report,
    importance_drift,
    rank_report,
)
from apps.research.features.size_risk import canonical_industry
from apps.research.features.store import read_columns
from apps.research.labels import HORIZONS
from apps.research.metrics.deciles import LIQUIDITY
from apps.research.metrics.ic import NEUTRAL_EXPOSURES
from apps.research.models import HoldoutScope, ModelObjective, ModelRun
from apps.research.panel import load_price_panel
from apps.research.prediction_cache import MissingFoldsError, fold_notes, missing_folds
from apps.research.rank_model import RANK_LABEL
from apps.research.runs import feature_hash
from apps.research.training import KEYS, load_labels
from apps.research.walkforward import Fold

MODEL = lgbm_runs.MODEL
RIDGE = "ridge"
EXPOSURES = [LIQUIDITY, *NEUTRAL_EXPOSURES]


@dataclass(frozen=True)
class RankEvaluation:
    """One horizon's LightGBM and Ridge reports, LightGBM deciles and importance drift."""

    model: RankReport
    baseline: RankReport
    deciles: pd.DataFrame
    importance: ImportanceDrift


def _industries() -> pd.Series:
    rows = StockMaster.objects.values_list("stock_id", "industry")
    stocks, names = zip(*rows, strict=True) if rows else ((), ())
    return canonical_industry(pd.Series(list(names), index=list(stocks), dtype=object))


def evaluation_frame(
    run_key: str, folds: Sequence[Fold], horizon: int, directory: Path
) -> pd.DataFrame:
    """Both cached scores up to each eval_end with the label, exposures and industry."""
    scores = cached_scores(run_key, folds, horizon)
    ends = scores["fold"].map({fold.index: fold.eval_end for fold in folds})
    scores = scores[scores["date"] <= ends]
    start, end = min(fold.test_start for fold in folds), max(fold.eval_end for fold in folds)
    labels = load_labels(start, end, horizon)[[*KEYS, RANK_LABEL, "untradable_t1"]]
    exposures = read_columns(start, end, directory, EXPOSURES)
    frame = scores.merge(labels, on=KEYS, how="left", validate="one_to_one")
    frame = frame.merge(exposures, on=KEYS, how="left", validate="one_to_one")
    return frame.assign(industry=frame["stock_id"].map(_industries()))


def fold_gains(run_key: str, folds: Sequence[Fold], horizon: int) -> dict[int, dict[str, float]]:
    """Each fold's LightGBM gain by feature, read from its model_run params."""
    ids = {lgbm_runs.run_id(run_key, ModelObjective.RANK, horizon, fold): fold for fold in folds}
    rows = ModelRun.objects.filter(run_id__in=ids).values_list("run_id", "params")
    gains = {ids[run_id].index: params["importance_gain"] for run_id, params in rows}
    missing = sorted({fold.index for fold in folds} - gains.keys())
    if missing:
        raise LookupError(
            f"no LightGBM model_run of {run_key} at {horizon} days for folds {missing}"
        )
    return gains


def price_window(folds: Sequence[Fold], horizon: int, calendar: TradingDays) -> tuple[date, date]:
    """Days the deciles read: the first research signal through the last tranche's exit."""
    research = [fold for fold in folds if fold.research]
    if not research:
        raise ValueError("no research folds to evaluate")
    following = calendar.between(max(fold.eval_end for fold in research), calendar.end)
    return min(fold.test_start for fold in research), following[min(horizon, len(following) - 1)]


def evaluate(
    run_key: str,
    folds: Sequence[Fold],
    horizon: int,
    directory: Path,
    prices: pd.DataFrame,
    calendar: TradingDays,
) -> RankEvaluation:
    """One horizon of a rank run: LightGBM against Ridge fold by fold, Ridge alone, deciles."""
    frame = evaluation_frame(run_key, folds, horizon, directory)
    boosted, ridge = f"{MODEL}_h{horizon}", f"{RIDGE}_h{horizon}"
    return RankEvaluation(
        model=rank_report(frame, boosted, horizon, folds, baseline=ridge),
        baseline=rank_report(frame, ridge, horizon, folds),
        deciles=decile_report(frame, boosted, horizon, folds, prices, calendar),
        importance=importance_drift(fold_gains(run_key, folds, horizon)),
    )


def stale_folds(run_key: str, folds: Sequence[Fold], directory: Path) -> list[int]:
    """Folds whose LightGBM cache was fit on features or labels other than today's."""
    stale = []
    for fold in folds:
        notes = fold_notes(run_key, MODEL, fold.index)
        current = {
            "feature_hash": feature_hash(fold.train_start, fold.test_end, directory),
            "label_hash": lgbm_runs.label_digest(HoldoutScope.RANK, fold),
        }
        if {key: notes.get(key) for key in current} != current:
            stale.append(fold.index)
    return stale


def evaluate_run(
    run_key: str, folds: Sequence[Fold], directory: Path, calendar: TradingDays
) -> dict[int, RankEvaluation]:
    """Every horizon of a rank run; missing caches are named first, then stale ones refused."""
    check_run_key(HoldoutScope.RANK, run_key)
    start, end = price_window(folds, max(HORIZONS), calendar)
    for model in (BASELINE, MODEL):
        missing = missing_folds(run_key, model, folds)
        if missing:
            raise MissingFoldsError(run_key, model, missing)
    stale = stale_folds(run_key, folds, directory)
    if stale:
        raise ValueError(
            f"folds {stale} of {run_key} cache LightGBM fit on features or labels that are "
            "stale; rerun train for Ridge and LightGBM"
        )
    prices = load_price_panel(calendar, start, end)
    return {
        horizon: evaluate(run_key, folds, horizon, directory, prices, calendar)
        for horizon in HORIZONS
    }


def write_metrics(run_key: str, evaluations: Mapping[int, RankEvaluation]) -> int:
    """Replace the run's LightGBM and Ridge rank metrics at every horizon in one transaction."""
    with transaction.atomic():
        return sum(
            write_rank_metrics(evaluated.model, run_key, MODEL)
            + write_rank_metrics(evaluated.baseline, run_key, RIDGE)
            for evaluated in evaluations.values()
        )


def report(run_key: str, evaluations: Mapping[int, RankEvaluation]) -> str:
    """Markdown of each horizon's LightGBM section with its deciles and importance drift."""
    sections = [
        render_report(
            f"{run_key}（{horizon} 日）",
            [evaluated.model],
            {horizon: evaluated.deciles},
            evaluated.importance,
        )
        for horizon, evaluated in sorted(evaluations.items())
    ]
    return "\n".join(sections) + "\n"
