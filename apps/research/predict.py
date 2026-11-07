"""Daily predictions (SPEC §7): each slot's serving run scores every stock with features."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from django.db import transaction

from apps.etl.tasks import ready_for_research
from apps.market.calendar import TradingDays
from apps.research.inference import day_rows, run_contributions, run_values
from apps.research.models import (
    ModelMonitor,
    ModelObjective,
    ModelRun,
    ModelStatus,
    Prediction,
    PredictionContrib,
)
from apps.research.rank_model import prediction_frame
from apps.research.runs import SCREENING_MODEL, RunError, load_model
from apps.research.walkforward import oos_start

SERVING = (ModelStatus.ACTIVE, ModelStatus.DEGRADED)
FIELDS = ("score", "rank", "pct", "ret_pred", "q10", "q50", "q90", "vol_pred", "prob_up_5pct")
BATCH = 2000


class PredictionGateError(RuntimeError):
    """The day's data quality gate has not passed, so nothing is predicted."""


@dataclass(frozen=True)
class Slot:
    objective: str
    horizon: int

    def __str__(self) -> str:
        return f"{self.objective} {self.horizon} 日"


SLOTS = (
    Slot(ModelObjective.RANK, 5),
    Slot(ModelObjective.RANK, 20),
    Slot(ModelObjective.REGRESSION, 5),
    Slot(ModelObjective.QUANTILE, 20),
    Slot(ModelObjective.VOLATILITY, 20),
    Slot(ModelObjective.CLASSIFICATION, 20),
)


@dataclass(frozen=True)
class Predicted:
    written: dict[str, int] = field(default_factory=dict)
    missing: list[Slot] = field(default_factory=list)


@dataclass(frozen=True)
class SlotRows:
    slot: Slot
    run: ModelRun
    predictions: list[Prediction]
    contributions: list[PredictionContrib]


def serving_run(slot: Slot) -> ModelRun | None:
    """The slot's active run, else its degraded one; two of either is refused."""
    runs = ModelRun.objects.filter(objective=slot.objective, params__horizon=slot.horizon).exclude(
        model_type=SCREENING_MODEL
    )
    for status in SERVING:
        found = list(runs.filter(status=status).order_by("run_id"))
        if len(found) > 1:
            listed = ", ".join(run.run_id for run in found)
            raise RunError(f"{listed} are all {status} for {slot}; retire all but one")
        if found:
            return found[0]
    return None


def _value(value: Any) -> Any:
    """A database value: missing becomes None, numpy scalars become Python ones."""
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(name): _value(value) for name, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def _predictions(slot: Slot, rows: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    if slot.objective == ModelObjective.RANK:
        return prediction_frame(rows, values["score"].to_numpy())
    return rows[["date", "stock_id", "in_universe"]].join(values)


def slot_rows(slot: Slot, run: ModelRun, rows: pd.DataFrame, day: date) -> SlotRows:
    """The run's prediction rows for every stock in rows, and its contributions if it ranks."""
    model = load_model(run.run_key, run.run_id)
    table = _predictions(slot, rows, run_values(run, model, rows))
    present = [name for name in FIELDS if name in table]
    predictions = [
        Prediction(
            run_id=run.run_id,
            stock_id=record["stock_id"],
            date=day,
            horizon=slot.horizon,
            in_universe=bool(record["in_universe"]),
            **{name: record[name] for name in present},
        )
        for record in _records(table)
    ]
    contributions = []
    if slot.objective == ModelObjective.RANK:
        contributions = [
            PredictionContrib(run_id=run.run_id, date=day, horizon=slot.horizon, **record)
            for record in _records(run_contributions(run, model, rows))
        ]
    return SlotRows(slot, run, predictions, contributions)


def _same_slot(day: date, slot: Slot) -> dict[str, Any]:
    return {"date": day, "horizon": slot.horizon, "run__objective": slot.objective}


def _write(day: date, written: SlotRows) -> None:
    same = _same_slot(day, written.slot)
    PredictionContrib.objects.filter(**same).delete()
    Prediction.objects.filter(**same).delete()
    Prediction.objects.bulk_create(written.predictions, batch_size=BATCH)
    if written.contributions:
        PredictionContrib.objects.bulk_create(written.contributions, batch_size=BATCH)


def _chosen(runs: dict[Slot, ModelRun | None]) -> dict[Slot, tuple[str, str] | None]:
    return {slot: None if run is None else (run.run_id, run.status) for slot, run in runs.items()}


def _recheck(runs: dict[Slot, ModelRun | None]) -> None:
    """Lock the serving runs and refuse if any slot would pick another run than it scored with."""
    list(ModelRun.objects.select_for_update(no_key=True).filter(status__in=SERVING))
    before, now = _chosen(runs), _chosen({slot: serving_run(slot) for slot in runs})
    moved = [str(slot) for slot in runs if before[slot] != now[slot]]
    if moved:
        raise RunError(f"the serving run of {', '.join(moved)} changed while predicting; run again")


def _check_day(day: date, serving: dict[Slot, ModelRun]) -> None:
    """Refuse a day whose label window overlaps the labels a run trained on."""
    calendar = TradingDays.from_db()
    trained = sorted(
        run.run_id
        for run in serving.values()
        if day < oos_start(calendar, run.train_end, int(run.params["horizon"]))
    )
    if trained:
        raise RunError(
            f"{', '.join(trained)} trained on labels reaching past {day}; it would score in-sample"
        )


def _others(day: date, serving: dict[Slot, ModelRun], table: Any) -> list[str]:
    return sorted(
        {
            run_id
            for slot, run in serving.items()
            for run_id in table.objects.filter(**_same_slot(day, slot))
            .exclude(run_id=run.run_id)
            .values_list("run_id", flat=True)
            .distinct()
        }
    )


def _check_rows(day: date, serving: dict[Slot, ModelRun], replace: bool) -> None:
    """Refuse other runs' rows on the day unless replace was asked for, and their scored days."""
    others = _others(day, serving, Prediction)
    if others and not replace:
        raise RunError(
            f"{day} already holds rows of {', '.join(others)}; replace them only on purpose"
        )
    scored = _others(day, serving, ModelMonitor)
    if scored:
        raise RunError(
            f"model_monitor already scored {', '.join(scored)} on {day}; its history stays"
        )


def predict(
    day: date, directory: Path, replace: bool = False, slots: Sequence[Slot] = SLOTS
) -> Predicted:
    """Write the slots' predictions for day in one transaction; replace swaps other runs' rows."""
    if not ready_for_research(day):
        raise PredictionGateError(
            f"{day} has not passed the research gate (every gate snapshot loaded, every quality "
            "check run, no error); predictions wait for it"
        )
    runs = {slot: serving_run(slot) for slot in slots}
    serving = {slot: run for slot, run in runs.items() if run is not None}
    if not serving:
        raise RunError("no active or degraded run serves any slot")
    _check_day(day, serving)
    _check_rows(day, serving, replace)
    features = [name for run in serving.values() for name in run.feature_list]
    rows = day_rows(day, directory, features)
    prepared = [slot_rows(slot, run, rows, day) for slot, run in serving.items()]
    with transaction.atomic():
        _recheck(runs)
        _check_rows(day, serving, replace)
        for written in prepared:
            _write(day, written)
    return Predicted(
        written={item.run.run_id: len(item.predictions) for item in prepared},
        missing=[slot for slot, run in runs.items() if run is None],
    )
