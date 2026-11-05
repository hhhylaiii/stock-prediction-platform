from datetime import date

import pytest
import time_machine
from django.test import Client
from pytest_django import DjangoAssertNumQueries

from apps.etl.models import DataQualityLog, EtlRun
from apps.etl.quality import CHECKS
from apps.research.models import ModelRun
from tests.api.conftest import (
    add_bars,
    add_model_run,
    add_predictions,
    add_stock,
    envelope_of,
    taipei,
)

pytestmark = pytest.mark.usefixtures("calendar_in_db")

OCT_29, OCT_30 = date(2026, 10, 29), date(2026, 10, 30)
FRIDAY_NIGHT = taipei(2026, 10, 30, 23, 30)


def add_run(dataset: str, day: date, finished: tuple[int, int] | None) -> EtlRun:
    return EtlRun.objects.create(
        source="twse",
        dataset=dataset,
        start_date=day,
        end_date=day,
        status=EtlRun.Status.SUCCESS if finished else EtlRun.Status.RUNNING,
        started_at=taipei(day.year, day.month, day.day, 15, 30),
        finished_at=None if finished is None else taipei(day.year, day.month, day.day, *finished),
    )


@pytest.fixture
def loaded(db: None) -> None:
    add_stock()
    add_bars("2330", [OCT_29, OCT_30], [990.0, 1000.0])
    for check in CHECKS:
        DataQualityLog.objects.create(date=OCT_30, check_name=check, severity="info", detail={})
    add_run("daily_price", OCT_30, (15, 31))
    add_run("margin", OCT_30, (23, 4))
    add_run("qfii", OCT_30, None)
    add_run("daily_price", OCT_29, (23, 50))
    for objective, horizon in (("rank", 20), ("rank", 5), ("quantile", 20)):
        run = add_model_run(f"{objective}-h{horizon}", objective)
        add_predictions(run, OCT_30, horizon, {"2330": {"score": 1.0, "rank": 1, "pct": 1.0}})


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_status_reports_prices_predictions_quality_and_last_update(
    client: Client, loaded: None
) -> None:
    response = client.get("/api/status")

    assert response.status_code == 200
    body = envelope_of(response)
    assert body["data"] == {
        "session": "2026-10-30",
        "price_date": "2026-10-30",
        "prediction_date": "2026-10-30",
        "predictions_ready": True,
        "quality": {"passed": True, "worst": "info", "missing_checks": []},
        "updated_at": "2026-10-30T23:04:00+08:00",
        "degraded": [],
    }


@time_machine.travel(taipei(2026, 11, 2, 16), tick=False)
def test_a_new_session_before_its_checks_is_not_ready(client: Client, loaded: None) -> None:
    data = envelope_of(client.get("/api/status"))["data"]

    assert data["session"] == "2026-11-02"
    assert (data["price_date"], data["prediction_date"]) == ("2026-10-30", "2026-10-30")
    assert data["predictions_ready"] is False
    assert data["quality"] == {"passed": False, "worst": None, "missing_checks": list(CHECKS)}
    assert data["updated_at"] is None


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_degraded_serving_models_are_listed(client: Client, loaded: None) -> None:
    ModelRun.objects.filter(pk="quantile-h20").update(status="degraded")

    data = envelope_of(client.get("/api/status"))["data"]

    assert data["degraded"] == [{"run_id": "quantile-h20", "objective": "quantile", "horizon": 20}]


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_screening_and_candidates_are_never_reported(client: Client, loaded: None) -> None:
    add_model_run("screening", "rank", status="degraded", model_type="screening")
    add_model_run("cand", "rank", status="candidate")

    data = envelope_of(client.get("/api/status"))["data"]

    assert data["degraded"] == []


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_an_empty_database_reports_nothing_loaded(client: Client, db: None) -> None:
    data = envelope_of(client.get("/api/status"))["data"]

    assert data == {
        "session": "2026-10-30",
        "price_date": None,
        "prediction_date": None,
        "predictions_ready": False,
        "quality": {"passed": False, "worst": None, "missing_checks": list(CHECKS)},
        "updated_at": None,
        "degraded": [],
    }


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_status_query_count(
    client: Client, loaded: None, django_assert_max_num_queries: DjangoAssertNumQueries
) -> None:
    with django_assert_max_num_queries(6):
        client.get("/api/status")
