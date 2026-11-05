from datetime import date, datetime

import pytest
import time_machine
from django.test import Client
from pytest_django import DjangoAssertNumQueries

from apps.etl.models import EtlRun
from tests.api.conftest import envelope_of, taipei

pytestmark = pytest.mark.usefixtures("calendar_in_db")

OCT_29, OCT_30 = date(2026, 10, 29), date(2026, 10, 30)


def add_run(
    dataset: str,
    start: date,
    end: date,
    started_at: datetime,
    source: str = "twse",
    status: str = EtlRun.Status.SUCCESS,
) -> EtlRun:
    return EtlRun.objects.create(
        source=source,
        dataset=dataset,
        start_date=start,
        end_date=end,
        status=status,
        rows=1047,
        started_at=started_at,
        finished_at=None if status == EtlRun.Status.RUNNING else started_at,
    )


def test_default_scope_lists_that_day_s_increment_in_start_order(client: Client) -> None:
    tpex = add_run("daily_price", OCT_30, OCT_30, taipei(2026, 10, 30, 15, 33), source="tpex")
    twse = add_run("daily_price", OCT_30, OCT_30, taipei(2026, 10, 30, 15, 31))
    add_run("TaiwanStockPrice", date(2010, 1, 4), OCT_30, taipei(2026, 10, 1), source="finmind")
    add_run("trading_calendar_refresh", date(2026, 10, 1), OCT_29, taipei(2026, 10, 30, 15, 10))
    add_run("trading_calendar_refresh", OCT_30, date(2026, 11, 30), taipei(2026, 11, 2, 15, 10))
    add_run("daily_price", OCT_29, OCT_29, taipei(2026, 10, 29, 15, 31))

    response = client.get("/api/etl/runs", {"date": "2026-10-30"})

    assert response.status_code == 200
    body = envelope_of(response)
    assert [run["run_id"] for run in body["data"]] == [twse.pk, tpex.pk]
    assert body["data"][0] == {
        "run_id": twse.pk,
        "source": "twse",
        "dataset": "daily_price",
        "start_date": "2026-10-30",
        "end_date": "2026-10-30",
        "status": "success",
        "rows": 1047,
        "error": "",
        "started_at": "2026-10-30T15:31:00+08:00",
        "finished_at": "2026-10-30T15:31:00+08:00",
    }
    assert body["meta"] == {
        "date": "2026-10-30",
        "scope": "day",
        "total": 2,
        "page": 1,
        "page_size": 20,
        "pages": 1,
    }


def test_covering_scope_adds_multi_day_runs_over_the_date(client: Client) -> None:
    twse = add_run("daily_price", OCT_30, OCT_30, taipei(2026, 10, 30, 15, 31))
    backfill = add_run(
        "TaiwanStockPrice", date(2010, 1, 4), OCT_30, taipei(2026, 10, 1), source="finmind"
    )
    refresh = add_run(
        "trading_calendar_refresh", date(2026, 10, 1), OCT_30, taipei(2026, 11, 2, 15, 10)
    )
    add_run("trading_calendar_refresh", date(2026, 10, 1), OCT_29, taipei(2026, 10, 30, 15, 10))
    add_run("daily_price", date(2026, 11, 2), date(2026, 11, 2), taipei(2026, 11, 2, 15, 31))

    body = envelope_of(client.get("/api/etl/runs", {"date": "2026-10-30", "scope": "covering"}))

    assert [run["run_id"] for run in body["data"]] == [backfill.pk, twse.pk, refresh.pk]
    assert body["meta"]["scope"] == "covering"


@pytest.mark.parametrize("scope", ["all", "DAY"])
def test_unknown_scope_is_invalid(client: Client, scope: str) -> None:
    response = client.get("/api/etl/runs", {"date": "2026-10-30", "scope": scope})

    assert response.status_code == 400
    assert set(envelope_of(response)["error"]["fields"]) == {"scope"}


def test_failed_and_running_runs_are_listed_with_their_error(client: Client) -> None:
    failed = add_run("margin", OCT_30, OCT_30, taipei(2026, 10, 30, 21, 31), status="failed")
    EtlRun.objects.filter(pk=failed.pk).update(error="HTTPError: 503")
    add_run("qfii", OCT_30, OCT_30, taipei(2026, 10, 30, 18, 30), status="running")

    runs = envelope_of(client.get("/api/etl/runs", {"date": "2026-10-30"}))["data"]

    assert [(run["status"], run["error"]) for run in runs] == [
        ("running", ""),
        ("failed", "HTTPError: 503"),
    ]
    assert runs[0]["finished_at"] is None


@time_machine.travel(taipei(2026, 10, 31, 10), tick=False)
def test_date_defaults_to_the_latest_session_on_or_before_today(client: Client) -> None:
    run = add_run("daily_price", OCT_30, OCT_30, taipei(2026, 10, 30, 15, 31))

    body = envelope_of(client.get("/api/etl/runs"))

    assert body["meta"]["date"] == "2026-10-30"
    assert [entry["run_id"] for entry in body["data"]] == [run.pk]


@time_machine.travel(taipei(2026, 11, 2, 7), tick=False)
def test_default_date_is_today_in_taipei_not_utc(client: Client) -> None:
    body = envelope_of(client.get("/api/etl/runs"))

    assert body["meta"]["date"] == "2026-11-02"


@time_machine.travel(taipei(2026, 12, 1, 10), tick=False)
def test_today_past_the_calendar_reports_it_outdated(client: Client) -> None:
    response = client.get("/api/etl/runs")

    assert response.status_code == 503
    assert envelope_of(response)["error"]["code"] == "calendar_outdated"


def test_pages_hold_twenty_runs(client: Client) -> None:
    for minute in range(25):
        add_run(f"set{minute:02d}", OCT_30, OCT_30, taipei(2026, 10, 30, 16, minute))

    first = envelope_of(client.get("/api/etl/runs", {"date": "2026-10-30"}))
    second = envelope_of(client.get("/api/etl/runs", {"date": "2026-10-30", "page": "2"}))

    assert len(first["data"]) == 20
    assert first["meta"] == {
        "date": "2026-10-30",
        "scope": "day",
        "total": 25,
        "page": 1,
        "page_size": 20,
        "pages": 2,
    }
    assert [run["dataset"] for run in second["data"]] == [f"set{m}" for m in range(20, 25)]
    assert second["meta"]["page"] == 2


def test_page_size_is_capped_at_100(client: Client) -> None:
    add_run("daily_price", OCT_30, OCT_30, taipei(2026, 10, 30, 15, 31))

    small = envelope_of(client.get("/api/etl/runs", {"date": "2026-10-30", "page_size": "5"}))
    huge = envelope_of(client.get("/api/etl/runs", {"date": "2026-10-30", "page_size": "1000"}))

    assert small["meta"]["page_size"] == 5
    assert huge["meta"]["page_size"] == 100


@pytest.mark.parametrize("page", ["2", "abc", "last", "0"])
def test_page_out_of_range_is_not_found(client: Client, page: str) -> None:
    add_run("daily_price", OCT_30, OCT_30, taipei(2026, 10, 30, 15, 31))

    response = client.get("/api/etl/runs", {"date": "2026-10-30", "page": page})

    assert response.status_code == 404
    assert envelope_of(response)["error"]["code"] == "not_found"


def test_an_empty_day_is_one_empty_page(client: Client) -> None:
    body = envelope_of(client.get("/api/etl/runs", {"date": "2026-10-30"}))

    assert body["data"] == []
    assert body["meta"] == {
        "date": "2026-10-30",
        "scope": "day",
        "total": 0,
        "page": 1,
        "page_size": 20,
        "pages": 1,
    }


@pytest.mark.parametrize("value", ["2026-13-01", "yesterday", "2026/10/30"])
def test_malformed_date_is_invalid(client: Client, value: str) -> None:
    response = client.get("/api/etl/runs", {"date": value})

    assert response.status_code == 400
    assert set(envelope_of(response)["error"]["fields"]) == {"date"}


@time_machine.travel(taipei(2026, 10, 31, 10), tick=False)
def test_etl_runs_query_count(
    client: Client, django_assert_max_num_queries: DjangoAssertNumQueries
) -> None:
    for minute in range(5):
        add_run(f"set{minute}", OCT_30, OCT_30, taipei(2026, 10, 30, 16, minute))

    with django_assert_max_num_queries(2):
        given = client.get("/api/etl/runs", {"date": "2026-10-30"})
    with django_assert_max_num_queries(3):
        default = client.get("/api/etl/runs")

    assert [given.status_code, default.status_code] == [200, 200]
    assert len(envelope_of(default)["data"]) == 5
