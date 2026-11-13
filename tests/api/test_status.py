from datetime import date

import pytest
import time_machine
from django.test import Client
from pytest_django import DjangoAssertNumQueries

from apps.etl.models import DataQualityLog, EtlRun
from apps.etl.quality import CHECKS
from apps.research.models import ModelRun, Prediction
from tests.api.conftest import (
    add_bars,
    add_model_run,
    add_predictions,
    add_stock,
    envelope_of,
    taipei,
)

pytestmark = pytest.mark.usefixtures("calendar_in_db")

OCT_29, OCT_30, NOV_2 = date(2026, 10, 29), date(2026, 10, 30), date(2026, 11, 2)
FRIDAY_NIGHT = taipei(2026, 10, 30, 23, 30)
MONDAY_MORNING = taipei(2026, 11, 2, 8)


def add_run(
    dataset: str,
    day: date,
    finished: tuple[int, int] | None,
    status: str = EtlRun.Status.SUCCESS,
    end: date | None = None,
) -> EtlRun:
    return EtlRun.objects.create(
        source="twse",
        dataset=dataset,
        start_date=day,
        end_date=end or day,
        status=status if finished else EtlRun.Status.RUNNING,
        started_at=taipei(day.year, day.month, day.day, 15, 30),
        finished_at=None if finished is None else taipei(day.year, day.month, day.day, *finished),
    )


def rank_rows(
    run_id: str,
    day: date,
    horizon: int = 20,
    objective: str = "rank",
    status: str = "active",
    model_type: str = "lightgbm",
) -> None:
    run = add_model_run(run_id, objective, status=status, model_type=model_type)
    add_predictions(run, day, horizon, {"2330": {"score": 1.0, "rank": 1, "pct": 1.0}})


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
        "horizons": [
            {"horizon": 5, "prediction_date": "2026-10-30", "ready": True},
            {"horizon": 20, "prediction_date": "2026-10-30", "ready": True},
        ],
        "quality": {
            "date": "2026-10-30",
            "passed": True,
            "worst": "info",
            "missing_checks": [],
            "warnings": 0,
            "warning_rows": 0,
        },
        "updated_at": "2026-10-30T23:04:00+08:00",
        "degraded": [],
    }


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_warnings_are_counted_for_the_strip(client: Client, loaded: None) -> None:
    DataQualityLog.objects.filter(date=OCT_30, check_name__in=CHECKS[:2]).update(severity="warning")
    DataQualityLog.objects.create(
        date=OCT_30, check_name="output_width_ratio", severity="warning", detail={"share": 0.5}
    )

    quality = envelope_of(client.get("/api/status"))["data"]["quality"]

    assert (quality["passed"], quality["worst"], quality["warnings"]) == (True, "warning", 2)
    assert quality["warning_rows"] == 2


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_a_check_with_a_row_per_table_counts_once_and_keeps_its_rows(
    client: Client, loaded: None
) -> None:
    DataQualityLog.objects.filter(date=OCT_30, check_name="row_count").delete()
    for table in ("daily_price", "daily_chip", "daily_valuation"):
        DataQualityLog.objects.create(
            date=OCT_30, check_name="row_count", severity="warning", detail={"table": table}
        )

    quality = envelope_of(client.get("/api/status"))["data"]["quality"]

    assert (quality["worst"], quality["warnings"], quality["warning_rows"]) == ("warning", 1, 3)


@time_machine.travel(MONDAY_MORNING, tick=False)
def test_before_the_open_the_last_session_s_predictions_are_ready(
    client: Client, loaded: None
) -> None:
    data = envelope_of(client.get("/api/status"))["data"]

    assert data["session"] == "2026-11-02"
    assert (data["price_date"], data["prediction_date"]) == ("2026-10-30", "2026-10-30")
    assert data["predictions_ready"] is True
    assert data["quality"]["date"] == "2026-10-30"
    assert data["quality"]["passed"] is True
    assert data["updated_at"] == "2026-10-30T23:04:00+08:00"


@time_machine.travel(taipei(2026, 11, 2, 16), tick=False)
def test_new_prices_before_their_checks_are_not_ready(client: Client, loaded: None) -> None:
    add_bars("2330", [NOV_2], [1010.0])
    add_run("daily_price", NOV_2, (15, 40))

    data = envelope_of(client.get("/api/status"))["data"]

    assert (data["price_date"], data["prediction_date"]) == ("2026-11-02", "2026-10-30")
    assert data["predictions_ready"] is False
    assert data["quality"] == {
        "date": "2026-11-02",
        "passed": False,
        "worst": None,
        "missing_checks": list(CHECKS),
        "warnings": 0,
        "warning_rows": 0,
    }
    assert data["updated_at"] == "2026-11-02T15:40:00+08:00"


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_failed_and_multi_day_runs_do_not_move_the_update_time(
    client: Client, loaded: None
) -> None:
    add_run("qfii", OCT_30, (23, 20), status=EtlRun.Status.FAILED)
    add_run("finmind", OCT_30, (23, 25), end=NOV_2)

    data = envelope_of(client.get("/api/status"))["data"]

    assert data["updated_at"] == "2026-10-30T23:04:00+08:00"


@time_machine.travel(taipei(2026, 11, 2, 23), tick=False)
def test_each_horizon_s_serving_rank_dates_its_own_predictions(
    client: Client, loaded: None
) -> None:
    add_bars("2330", [NOV_2], [1010.0])
    rank_rows("rank-h5-new", NOV_2, horizon=5)
    rank_rows("quantile-new", NOV_2, objective="quantile")
    rank_rows("rank-old", NOV_2, status="retired")
    rank_rows("rank-cand", NOV_2, status="candidate")
    rank_rows("screening", NOV_2, model_type="screening", status="degraded")

    data = envelope_of(client.get("/api/status"))["data"]

    assert data["horizons"] == [
        {"horizon": 5, "prediction_date": "2026-11-02", "ready": True},
        {"horizon": 20, "prediction_date": "2026-10-30", "ready": False},
    ]
    assert (data["prediction_date"], data["predictions_ready"]) == ("2026-11-02", True)


@time_machine.travel(taipei(2026, 11, 2, 23), tick=False)
def test_the_twenty_day_ranking_dates_the_predictions_when_it_is_the_newer(
    client: Client, loaded: None
) -> None:
    add_bars("2330", [NOV_2], [1010.0])
    rank_rows("rank-h20-new", NOV_2)

    data = envelope_of(client.get("/api/status"))["data"]

    assert data["horizons"] == [
        {"horizon": 5, "prediction_date": "2026-10-30", "ready": False},
        {"horizon": 20, "prediction_date": "2026-11-02", "ready": True},
    ]
    assert (data["prediction_date"], data["predictions_ready"]) == ("2026-11-02", True)


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_a_horizon_without_a_serving_rank_has_no_date(client: Client, loaded: None) -> None:
    ModelRun.objects.filter(pk="rank-h20").update(status="candidate")

    data = envelope_of(client.get("/api/status"))["data"]

    assert data["horizons"] == [
        {"horizon": 5, "prediction_date": "2026-10-30", "ready": True},
        {"horizon": 20, "prediction_date": None, "ready": False},
    ]
    assert (data["prediction_date"], data["predictions_ready"]) == ("2026-10-30", True)


@time_machine.travel(taipei(2026, 11, 2, 16), tick=False)
def test_no_horizon_is_ready_before_the_day_s_predictions(client: Client, loaded: None) -> None:
    add_bars("2330", [NOV_2], [1010.0])

    data = envelope_of(client.get("/api/status"))["data"]

    assert [entry["ready"] for entry in data["horizons"]] == [False, False]
    assert (data["prediction_date"], data["predictions_ready"]) == ("2026-10-30", False)


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_degraded_serving_models_are_listed(client: Client, loaded: None) -> None:
    ModelRun.objects.filter(pk="quantile-h20").update(status="degraded")

    data = envelope_of(client.get("/api/status"))["data"]

    assert data["degraded"] == [{"run_id": "quantile-h20", "objective": "quantile", "horizon": 20}]


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_degraded_models_are_listed_without_a_twenty_day_ranking(
    client: Client, loaded: None
) -> None:
    ModelRun.objects.filter(pk="rank-h20").update(status="candidate")
    ModelRun.objects.filter(pk__in=["rank-h5", "quantile-h20"]).update(status="degraded")

    data = envelope_of(client.get("/api/status"))["data"]

    assert data["degraded"] == [
        {"run_id": "quantile-h20", "objective": "quantile", "horizon": 20},
        {"run_id": "rank-h5", "objective": "rank", "horizon": 5},
    ]


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_a_degraded_model_is_read_on_its_own_latest_day(client: Client, loaded: None) -> None:
    Prediction.objects.filter(run_id="quantile-h20").update(date=OCT_29)
    ModelRun.objects.filter(pk="quantile-h20").update(status="degraded")

    data = envelope_of(client.get("/api/status"))["data"]

    assert data["degraded"] == [{"run_id": "quantile-h20", "objective": "quantile", "horizon": 20}]


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_a_retired_run_s_later_rows_do_not_hide_the_degraded_one(
    client: Client, loaded: None
) -> None:
    ModelRun.objects.filter(pk="quantile-h20").update(status="degraded")
    Prediction.objects.filter(run_id="quantile-h20").update(date=OCT_29)
    rank_rows("quantile-old", OCT_30, objective="quantile", status="retired")

    data = envelope_of(client.get("/api/status"))["data"]

    assert data["degraded"] == [{"run_id": "quantile-h20", "objective": "quantile", "horizon": 20}]


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_screening_and_candidates_are_never_reported(client: Client, loaded: None) -> None:
    rank_rows(
        "screening", OCT_30, objective="volatility", status="degraded", model_type="screening"
    )
    rank_rows("cand", OCT_30, objective="classification", status="candidate")

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
        "horizons": [
            {"horizon": 5, "prediction_date": None, "ready": False},
            {"horizon": 20, "prediction_date": None, "ready": False},
        ],
        "quality": {
            "date": "2026-10-30",
            "passed": False,
            "worst": None,
            "missing_checks": list(CHECKS),
            "warnings": 0,
            "warning_rows": 0,
        },
        "updated_at": None,
        "degraded": [],
    }


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_status_query_count(
    client: Client, loaded: None, django_assert_max_num_queries: DjangoAssertNumQueries
) -> None:
    with django_assert_max_num_queries(6):
        client.get("/api/status")
