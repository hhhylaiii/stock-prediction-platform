import math
from datetime import date
from typing import Any

import pandas as pd
import pytest
from django.db import IntegrityError, connection, transaction
from django.db.models import ProtectedError

from apps.etl.upsert import upsert
from apps.research.models import ModelMonitor, ModelRun
from tests.market.schema import primary_key_columns

pytestmark = pytest.mark.django_db
DAY = date(2026, 9, 30)
RUN_ID = "prod-interval.lightgbm.h20"
SPEC_COLUMNS = {"run", "date", "horizon", "ic", "coverage", "brier", "direction_acc", "n_stocks"}


@pytest.fixture
def model_run() -> ModelRun:
    return ModelRun.objects.create(
        run_id=RUN_ID,
        run_key="prod-interval",
        model_type="lightgbm",
        objective="quantile",
        version="2026-10",
        train_start=date(2023, 9, 1),
        train_end=date(2026, 9, 30),
        lib_versions={"lightgbm": "4.5.0"},
        feature_hash="0" * 64,
    )


def monitor(**changes: Any) -> ModelMonitor:
    fields: dict[str, Any] = {
        "run_id": RUN_ID,
        "date": DAY,
        "horizon": 20,
        "ic": 0.04,
        "coverage": 0.79,
        "brier": 0.231,
        "direction_acc": 0.54,
        "mae": 0.058,
        "width": 0.131,
        "auc": 0.58,
        "n_stocks": 1842,
    }
    return ModelMonitor(**{**fields, **changes})


def test_monitor_keeps_every_spec_column_and_its_key() -> None:
    assert {field.name for field in ModelMonitor._meta.get_fields()} >= SPEC_COLUMNS
    assert primary_key_columns("model_monitor") == [["run_id", "date", "horizon"]]


def test_monitor_round_trip_keeps_only_the_run_metrics(model_run: ModelRun) -> None:
    monitor(ic=None, brier=None, direction_acc=None, mae=None, auc=None).save()

    stored = ModelMonitor.objects.get(pk=(RUN_ID, DAY, 20))
    assert (stored.coverage, stored.width, stored.n_stocks) == (0.79, 0.131, 1842)
    assert (stored.ic, stored.brier, stored.auc) == (None, None, None)
    assert list(model_run.monitors.values_list("date", flat=True)) == [DAY]


def test_monitor_upsert_is_idempotent(model_run: ModelRun) -> None:
    frame = pd.DataFrame(
        [{"run_id": RUN_ID, "date": DAY, "horizon": 20, "coverage": 0.8, "n_stocks": 1800}]
    )
    keys, values = ["run_id", "date", "horizon"], ["coverage", "n_stocks"]

    upsert(ModelMonitor, frame, keys, values)
    upsert(ModelMonitor, frame.assign(coverage=0.77), keys, values)
    upsert(ModelMonitor, frame.assign(coverage=0.77), keys, values)

    assert list(ModelMonitor.objects.values_list("coverage", "n_stocks")) == [(0.77, 1800)]


@pytest.mark.parametrize(
    ("changes", "constraint"),
    [
        ({"horizon": 1}, "model_monitor_horizon_known"),
        ({"ic": math.nan}, "model_monitor_finite"),
        ({"mae": math.inf}, "model_monitor_finite"),
        ({"ic": 1.01}, "model_monitor_ic_range"),
        ({"ic": -1.01}, "model_monitor_ic_range"),
        ({"coverage": 1.2}, "model_monitor_unit_interval"),
        ({"brier": -0.1}, "model_monitor_unit_interval"),
        ({"direction_acc": 1.5}, "model_monitor_unit_interval"),
        ({"auc": -0.2}, "model_monitor_unit_interval"),
        ({"mae": -0.01}, "model_monitor_spread_gte_0"),
        ({"width": -0.01}, "model_monitor_spread_gte_0"),
        ({"n_stocks": -1}, "n_stocks"),
    ],
)
def test_monitor_check_constraints(
    model_run: ModelRun, changes: dict[str, Any], constraint: str
) -> None:
    with pytest.raises(IntegrityError, match=constraint), transaction.atomic():
        monitor(**changes).save()


def test_monitor_needs_a_recorded_run() -> None:
    with connection.cursor() as cursor:
        cursor.execute("SET CONSTRAINTS ALL IMMEDIATE")

    with pytest.raises(IntegrityError, match="model_monitor_run_id"), transaction.atomic():
        monitor(run_id="never-trained").save()


def test_a_monitored_run_cannot_be_deleted(model_run: ModelRun) -> None:
    monitor().save()

    with pytest.raises(ProtectedError), transaction.atomic():
        model_run.delete()

    assert ModelMonitor.objects.count() == 1
