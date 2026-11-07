from datetime import date

import pytest
import time_machine
from django.test import Client
from pytest_django import DjangoAssertNumQueries

from apps.etl.corrections import CORRECTION_CHECKS
from apps.etl.models import DataQualityLog
from apps.etl.quality import CHECKS
from tests.api.conftest import envelope_of, taipei

pytestmark = [pytest.mark.django_db, pytest.mark.usefixtures("calendar_in_db")]

OCT_30 = date(2026, 10, 30)
FRIDAY_NIGHT = taipei(2026, 10, 30, 23, 30)
Severity = DataQualityLog.Severity


def note(day: date, check: str, stock: str, severity: str = Severity.INFO) -> None:
    DataQualityLog.objects.create(
        date=day, check_name=check, severity=severity, detail={"stock_id": stock}
    )


def test_each_correction_check_reads_as_one_summary_row(client: Client) -> None:
    note(date(2010, 1, 4), "fi_holding_zero_replaced", "1101")
    note(date(2010, 1, 5), "fi_holding_zero_replaced", "1101")
    note(date(2026, 10, 15), "fi_holding_zero_replaced", "2330")
    note(date(2026, 11, 7), "statement_filed_later", "2330", Severity.WARNING)
    note(date(2026, 11, 7), "statement_filed_later", "2317", Severity.WARNING)
    note(OCT_30, "row_count", "2330")

    body = envelope_of(client.get("/api/data-quality/corrections"))

    assert body["data"] == [
        {
            "check_name": "fi_holding_zero_replaced",
            "severity": "info",
            "rows": 3,
            "stocks": 2,
            "first_date": "2010-01-04",
            "last_date": "2026-10-15",
        },
        {
            "check_name": "statement_filed_later",
            "severity": "warning",
            "rows": 2,
            "stocks": 2,
            "first_date": "2026-11-07",
            "last_date": "2026-11-07",
        },
    ]
    assert body["meta"] == {"checks": sorted(CORRECTION_CHECKS), "cache_seconds": 60}


def test_no_correction_is_an_empty_list(client: Client) -> None:
    assert envelope_of(client.get("/api/data-quality/corrections"))["data"] == []


def test_the_summary_is_one_query_then_served_from_the_cache(
    client: Client, django_assert_num_queries: DjangoAssertNumQueries
) -> None:
    note(OCT_30, "statement_restated", "2330")

    with django_assert_num_queries(1):
        first = envelope_of(client.get("/api/data-quality/corrections"))["data"]
    note(OCT_30, "statement_restated", "2317")
    with django_assert_num_queries(0):
        second = envelope_of(client.get("/api/data-quality/corrections"))["data"]

    assert first == second


@time_machine.travel(FRIDAY_NIGHT, tick=False)
def test_corrections_leave_the_day_s_checks_and_the_strip(client: Client) -> None:
    for check in CHECKS:
        DataQualityLog.objects.create(date=OCT_30, check_name=check, severity="info", detail={})
    for stock in ("1101", "2330"):
        note(OCT_30, "statement_filed_later", stock, Severity.WARNING)

    day = envelope_of(client.get("/api/data-quality", {"days": 1}))["data"][0]
    quality = envelope_of(client.get("/api/status"))["data"]["quality"]

    assert not [check for check in day["checks"] if check["check_name"] in CORRECTION_CHECKS]
    assert (day["worst"], day["passed"]) == ("info", True)
    assert (quality["worst"], quality["warnings"], quality["warning_rows"]) == ("info", 0, 0)
