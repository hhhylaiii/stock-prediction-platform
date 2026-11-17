"""Each morning's shadow candidate scores (holding-shadow-prereg.md §7), kept out of prediction."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from django.conf import settings
from django.utils import timezone

from apps.market.calendar import TradingDays
from apps.research.candidates import HORIZON, load_artifact
from apps.research.ensemble import rank_average
from apps.research.features.store import columns_for, read_partitions
from apps.research.inference import day_rows, run_values
from apps.research.lgbm_runs import MODEL
from apps.research.models import (
    ModelObjective,
    ModelRun,
    Prediction,
    ShadowArtifact,
    ShadowCandidate,
    ShadowScore,
)
from apps.research.runs import RunError
from apps.research.training import linear_inputs

ROW_KEYS = ["stock_id", "in_universe", "score"]
RIDGE, BASE = "ridge", "base"


@dataclass
class Scored:
    scored: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    notes: list[dict[str, Any]] = field(default_factory=list)


def label(row: ShadowArtifact) -> str:
    return f"{row.serving_run_id} g{row.generation}"


def run_label(row: ShadowArtifact) -> str:
    """The run_id a candidate book records for its lists and fills (H0 §1.3)."""
    return f"{row.candidate}:{row.serving_run_id}:g{row.generation}"


def artifacts(name: str) -> list[ShadowArtifact]:
    return list(ShadowArtifact.objects.filter(candidate=name).order_by("frozen_at"))


def first_signal(name: str) -> date | None:
    """The first signal day of the candidate's first frozen artifact, where its sample starts."""
    found = artifacts(name)
    return found[0].first_signal if found else None


def _serving_rows(day: date) -> pd.DataFrame:
    """The serving rank 5-day run's rows of the day: the universe and C0's score."""
    rows = Prediction.objects.filter(
        run__objective=ModelObjective.RANK, horizon=HORIZON, date=day
    ).values_list(*ROW_KEYS)
    frame = pd.DataFrame(list(rows), columns=ROW_KEYS)
    return frame.astype({"in_universe": "bool", "score": "float64"}).rename(columns={"score": BASE})


def _within(frame: pd.DataFrame, column: str) -> pd.Series:
    """rank(pct=True) of a column among the day's universe rows; NaN elsewhere."""
    return frame[column].where(frame["in_universe"]).rank(pct=True)


def _ridge(model: Any, row: ShadowArtifact, day: date, base: pd.DataFrame) -> pd.DataFrame:
    """C1: the Ridge score and C0's score, each ranked in the universe, averaged."""
    features = list(row.recipe["features"])
    read = read_partitions(day, day, Path(settings.FEATURE_PARQUET_DIR), columns_for(features))
    inputs = linear_inputs(read)
    raw = pd.Series(model.predict(inputs), index=inputs["stock_id"].to_numpy())
    frame = base.assign(date=day, raw=base["stock_id"].map(raw))
    score = rank_average(frame.rename(columns={"raw": RIDGE}), [RIDGE, BASE])
    return frame.assign(score=score)


def _relative(model: Any, row: ShadowArtifact, day: date, base: pd.DataFrame) -> pd.DataFrame:
    """C2: the predicted 5-day return relative to TAIEX_TR, on the run's own inputs."""
    features = list(row.recipe["features"])
    rows = day_rows(day, Path(settings.FEATURE_PARQUET_DIR), features)
    shadow = ModelRun(
        run_id=run_label(row),
        model_type=MODEL,
        objective=ModelObjective.REGRESSION,
        feature_list=features,
    )
    predicted = run_values(shadow, model, rows)["ret_pred"].to_numpy()
    raw = pd.Series(predicted, index=rows["stock_id"].to_numpy())
    frame = base.assign(raw=base["stock_id"].map(raw))
    return frame.assign(score=frame["raw"])


SCORERS: dict[str, Callable[[Any, ShadowArtifact, date, pd.DataFrame], pd.DataFrame]] = {
    ShadowCandidate.C1: _ridge,
    ShadowCandidate.C2: _relative,
}


def _value(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _rows(name: str, row: ShadowArtifact, day: date, frame: pd.DataFrame) -> list[ShadowScore]:
    pct = _within(frame, "score")
    now = timezone.now()
    return [
        ShadowScore(
            candidate=name,
            date=day,
            stock_id=stock,
            serving_run_id=row.serving_run_id,
            generation=row.generation,
            score=_value(score),
            raw=_value(raw),
            pct=_value(share),
            in_universe=bool(inside),
            recorded_at=now,
        )
        for stock, inside, score, raw, share in zip(
            frame["stock_id"], frame["in_universe"], frame["score"], frame["raw"], pct, strict=True
        )
    ]


def _unscored(name: str, start: date, session: date, calendar: TradingDays) -> list[date]:
    done = set(ShadowScore.objects.filter(candidate=name).values_list("date", flat=True).distinct())
    served = Prediction.objects.filter(
        run__objective=ModelObjective.RANK, horizon=HORIZON, date__range=(start, session)
    )
    predicted = set(served.values_list("date", flat=True).distinct())
    return [day for day in calendar.between(start, session) if day in predicted - done]


def _chosen(found: list[ShadowArtifact], day: date) -> ShadowArtifact | None:
    usable = [row for row in found if row.first_signal <= day]
    return usable[-1] if usable else None


def score_candidate(name: str, session: date, calendar: TradingDays) -> Scored:
    """Score each unscored signal day from the candidate's first signal day through session.

    Days under a missing or changed file get no rows; the newest artifact's gets one note a day.
    """
    found = artifacts(name)
    result = Scored()
    if not found or found[0].first_signal > session:
        return result
    newest = (found[-1].serving_run_id, found[-1].generation)
    loaded: dict[tuple[str, int], Any] = {}
    invalid: dict[tuple[str, int], str] = {}
    for day in _unscored(name, found[0].first_signal, session, calendar):
        row = _chosen(found, day)
        if row is None:
            continue
        key = (row.serving_run_id, row.generation)
        if key not in loaded and key not in invalid:
            try:
                loaded[key] = load_artifact(row)
            except RunError as error:
                invalid[key] = str(error)
                if key == newest:
                    note = {
                        "book": name,
                        "signal_date": day.isoformat(),
                        "artifact_invalid": str(error),
                    }
                    result.notes.append(note)
        if key in invalid:
            continue
        frame = SCORERS[name](loaded[key], row, day, _serving_rows(day))
        ShadowScore.objects.bulk_create(_rows(name, row, day, frame))
        result.scored.append(day.isoformat())
        if label(row) not in result.artifacts:
            result.artifacts.append(label(row))
    return result
