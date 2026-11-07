from datetime import date

import pytest
import time_machine
from django.db import connection
from django.test import Client
from django.test.utils import CaptureQueriesContext
from pytest_django import DjangoAssertNumQueries

from apps.etl.models import DataQualityLog
from apps.etl.quality import CHECKS, should_run_prediction
from apps.market.models import TradingCalendar
from apps.research.evaluation import ALERT_SHARE
from apps.research.promotion import _note
from tests.api.conftest import envelope_of, sessions_until, taipei

pytestmark = pytest.mark.usefixtures("calendar_in_db")

OCT_27, OCT_28, OCT_29, OCT_30 = (date(2026, 10, d) for d in (27, 28, 29, 30))
FRIDAY_NIGHT = taipei(2026, 10, 30, 23, 30)
Severity = DataQualityLog.Severity


def log(day: date, check: str, severity: str = Severity.INFO, **detail: object) -> None:
    DataQualityLog.objects.create(date=day, check_name=check, severity=severity, detail=detail)


def log_all(day: date, severity: str = Severity.INFO) -> None:
    for check in CHECKS:
        log(day, check, severity)


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_each_session_reports_its_checks_newest_first(client: Client) -> None:
    log_all(OCT_30)
    log(OCT_30, "row_count", table="daily_chip", count=1796)
    log_all(OCT_29)
    log(OCT_29, "missing_values", Severity.ERROR, table="daily_chip")
    log(OCT_27, "row_count", Severity.WARNING)
    log(OCT_27, "big_move")
    log(OCT_27, "trading_calendar")

    response = client.get("/api/data-quality", {"days": 4})

    assert response.status_code == 200
    body = envelope_of(response)
    days = body["data"]
    assert [entry["date"] for entry in days] == [
        "2026-10-30",
        "2026-10-29",
        "2026-10-28",
        "2026-10-27",
    ]
    assert [entry["passed"] for entry in days] == [True, False, False, False]
    assert [entry["worst"] for entry in days] == ["info", "error", None, "warning"]
    assert [entry["missing_checks"] for entry in days] == [
        [],
        [],
        list(CHECKS),
        ["missing_values", "index_freshness"],
    ]
    assert days[2]["checks"] == []
    assert [entry["model_checks"] for entry in days] == [[], [], [], []]
    assert days[0]["checks"][-1] == {
        "check_name": "trading_calendar",
        "severity": "info",
        "detail": {},
    }
    assert {
        "check_name": "row_count",
        "severity": "info",
        "detail": {"table": "daily_chip", "count": 1796},
    } in days[0]["checks"]
    assert [check["check_name"] for check in days[0]["checks"]] == sorted([*CHECKS, "row_count"])
    assert body["meta"] == {
        "days": 4,
        "start_date": "2026-10-27",
        "end_date": "2026-10-30",
        "checks": list(CHECKS),
        "output_alert_share": ALERT_SHARE,
    }


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_model_output_checks_are_listed_apart(client: Client) -> None:
    log_all(OCT_30)
    log(OCT_30, "output_width_ratio", Severity.WARNING, share=0.2, flagged=9, n_stocks=45)
    log(OCT_30, "monitor_unscored", Severity.WARNING, runs=[])
    log(OCT_30, "model_alert", Severity.WARNING, reason="below_threshold")
    log(OCT_30, "outputs_note", Severity.WARNING)

    day = envelope_of(client.get("/api/data-quality", {"days": 1}))["data"][0]

    assert (day["passed"], day["worst"]) == (True, "warning")
    assert [check["check_name"] for check in day["model_checks"]] == [
        "model_alert",
        "monitor_unscored",
        "output_width_ratio",
    ]
    assert day["model_checks"][-1] == {
        "id": DataQualityLog.objects.get(check_name="output_width_ratio").pk,
        "check_name": "output_width_ratio",
        "severity": "warning",
        "detail": {"share": 0.2, "flagged": 9, "n_stocks": 45},
    }
    assert "outputs_note" in [check["check_name"] for check in day["checks"]]
    assert not {"model_alert", "monitor_unscored"} & {
        check["check_name"] for check in day["checks"]
    }


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_research_schedule_checks_sit_with_the_model_checks(client: Client) -> None:
    log_all(OCT_30)
    log(OCT_30, "research_morning", Severity.WARNING, step="predict")
    log(OCT_30, "research_evening", Severity.INFO)

    day = envelope_of(client.get("/api/data-quality", {"days": 1}))["data"][0]
    quality = envelope_of(client.get("/api/status"))["data"]["quality"]

    assert [check["check_name"] for check in day["model_checks"]] == [
        "research_evening",
        "research_morning",
    ]
    assert not [check for check in day["checks"] if check["check_name"].startswith("research_")]
    assert (day["worst"], quality["worst"], quality["warnings"]) == ("info", "info", 0)


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_model_checks_carry_their_row_id_in_write_order(client: Client) -> None:
    log_all(OCT_30)
    refused = [{"slot": "rank 20 日", "run_id": "rank-a1", "gates": []}]
    _note(OCT_30, "rank", "rank-a", refused, "", promoted=False)
    _note(OCT_30, "rank", "rank-a", [], "", promoted=True)

    day = envelope_of(client.get("/api/data-quality", {"days": 1}))["data"][0]

    written = list(
        DataQualityLog.objects.filter(check_name="model_promotion")
        .order_by("pk")
        .values_list("pk", flat=True)
    )
    assert [check["id"] for check in day["model_checks"]] == written
    assert [check["detail"]["promoted"] for check in day["model_checks"]] == [False, True]
    assert not [check for check in day["checks"] if "id" in check]


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_model_output_checks_alone_leave_the_day_clean(client: Client) -> None:
    log_all(OCT_30)
    log(OCT_30, "output_odds_vs_q50", Severity.WARNING, share=0.5, flagged=1, n_stocks=2)

    day = envelope_of(client.get("/api/data-quality", {"days": 1}))["data"][0]

    assert (day["passed"], day["worst"], len(day["model_checks"])) == (True, "info", 1)


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_a_model_check_error_fails_the_day_as_the_gate_does(client: Client) -> None:
    log_all(OCT_30)
    log(OCT_30, "model_alert", Severity.ERROR, reason="below_threshold")

    body = envelope_of(client.get("/api/data-quality", {"days": 1}))
    day = body["data"][0]

    assert (day["passed"], day["worst"]) == (should_run_prediction(OCT_30), "info")
    assert day["passed"] is False


def test_the_output_alert_share_follows_the_research_constant(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.api.ops.ALERT_SHARE", 0.2)

    with time_machine.travel(FRIDAY_NIGHT, tick=False):
        meta = envelope_of(client.get("/api/data-quality", {"days": 1}))["meta"]

    assert meta["output_alert_share"] == 0.2


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_passed_matches_the_prediction_gate(client: Client) -> None:
    log_all(OCT_30)
    log_all(OCT_29)
    log(OCT_29, "consistency", Severity.ERROR)
    log_all(OCT_28, Severity.WARNING)
    log(OCT_27, "row_count")

    days = envelope_of(client.get("/api/data-quality", {"days": 4}))["data"]

    assert {entry["date"]: entry["passed"] for entry in days} == {
        day.isoformat(): should_run_prediction(day) for day in (OCT_27, OCT_28, OCT_29, OCT_30)
    }


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_window_counts_sessions_and_skips_holidays(client: Client) -> None:
    body = envelope_of(client.get("/api/data-quality", {"days": 20}))

    dates = [entry["date"] for entry in body["data"]]
    assert dates == [day.isoformat() for day in reversed(sessions_until(OCT_30)[-20:])]
    assert "2026-10-09" not in dates


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_only_logs_inside_the_window_are_loaded(client: Client) -> None:
    log(OCT_28, "row_count", Severity.ERROR)
    log(date(2026, 10, 31), "row_count", Severity.ERROR)

    with CaptureQueriesContext(connection) as queries:
        days = envelope_of(client.get("/api/data-quality", {"days": 2}))["data"]

    assert [entry["checks"] for entry in days] == [[], []]
    loaded = [query["sql"] for query in queries if "data_quality_log" in query["sql"]]
    assert len(loaded) == 1
    assert "BETWEEN '2026-10-29'::date AND '2026-10-30'::date" in loaded[0]


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_default_window_is_30_sessions(client: Client) -> None:
    body = envelope_of(client.get("/api/data-quality"))

    assert len(body["data"]) == 30
    assert body["meta"]["days"] == 30


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_window_stops_at_the_calendar_start(client: Client) -> None:
    body = envelope_of(client.get("/api/data-quality", {"days": 120}))

    assert len(body["data"]) == len(sessions_until(OCT_30))
    assert body["meta"]["start_date"] == "2026-07-01"


@time_machine.travel(taipei(2026, 11, 1, 9), tick=False)
def test_weekend_anchors_on_the_last_session(client: Client) -> None:
    body = envelope_of(client.get("/api/data-quality", {"days": 1}))

    assert body["meta"]["end_date"] == "2026-10-30"


@time_machine.travel(taipei(2026, 12, 1, 9), tick=False)
def test_today_past_the_calendar_reports_it_outdated(client: Client) -> None:
    response = client.get("/api/data-quality", {"days": 1})

    assert response.status_code == 503
    assert envelope_of(response)["error"]["code"] == "calendar_outdated"


@time_machine.travel(taipei(2026, 11, 30, 23), tick=False)
def test_the_calendar_s_last_day_is_still_served(client: Client) -> None:
    body = envelope_of(client.get("/api/data-quality", {"days": 1}))

    assert body["meta"]["end_date"] == "2026-11-30"


@time_machine.travel(taipei(2026, 6, 1, 9), tick=False)
def test_today_before_the_calendar_reports_it_unavailable(client: Client) -> None:
    response = client.get("/api/data-quality")

    assert response.status_code == 503
    assert envelope_of(response)["error"]["code"] == "calendar_unavailable"


def test_empty_calendar_reports_it_unavailable(client: Client) -> None:
    TradingCalendar.objects.all().delete()

    response = client.get("/api/data-quality")

    assert response.status_code == 503
    assert envelope_of(response)["error"]["code"] == "calendar_unavailable"


@pytest.mark.parametrize("days", ["0", "121", "x"])
def test_days_outside_one_to_120_is_invalid(client: Client, days: str) -> None:
    response = client.get("/api/data-quality", {"days": days})

    assert response.status_code == 400
    assert set(envelope_of(response)["error"]["fields"]) == {"days"}


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_data_quality_query_count(
    client: Client, django_assert_max_num_queries: DjangoAssertNumQueries
) -> None:
    for day in sessions_until(OCT_30)[-30:]:
        log_all(day)

    with django_assert_max_num_queries(2):
        client.get("/api/data-quality")
