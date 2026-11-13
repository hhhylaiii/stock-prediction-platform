import json
import logging
from typing import Any

import pytest
import responses
from django.core.management import call_command
from django.test import override_settings

from apps.etl.alerts import announce, dispatch
from apps.etl.models import DataQualityLog
from apps.etl.quality import run_checks
from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice
from tests.etl.test_quality import CODES, DAY, _two_clean_days

WEBHOOK = "https://hooks.example.test/alerts"


@pytest.fixture
def calendar(october_2026: TradingDays) -> TradingDays:
    return october_2026


@responses.activate
@pytest.mark.django_db
@override_settings(ALERT_WEBHOOK_URL=WEBHOOK, FINMIND_API_TOKEN="secret-token")
def test_warnings_and_errors_go_to_the_webhook_without_secrets(
    calendar: TradingDays, caplog: pytest.LogCaptureFixture
) -> None:
    responses.post(WEBHOOK, status=200)
    _two_clean_days()
    DailyPrice.objects.filter(date=DAY, stock_id__in=CODES[94:]).delete()
    with caplog.at_level(logging.WARNING, logger="apps.etl.alerts"):
        dispatch(run_checks(DAY, calendar))
    (call,) = responses.calls
    raw = call.request.body
    text = raw.decode() if isinstance(raw, bytes) else str(raw)
    body = json.loads(text)
    assert {(alert["check"], alert["severity"]) for alert in body["alerts"]} == {
        ("row_count", "warning"),
        ("missing_values", "warning"),
    }
    assert body["alerts"][0]["date"] == "2026-10-28"
    assert "secret-token" not in text
    assert WEBHOOK not in text
    assert "row_count" in caplog.text


@responses.activate
@pytest.mark.django_db
@override_settings(ALERT_WEBHOOK_URL=WEBHOOK)
def test_a_failing_webhook_is_logged_and_does_not_raise(
    calendar: TradingDays, caplog: pytest.LogCaptureFixture
) -> None:
    responses.post(WEBHOOK, status=500)
    _two_clean_days()
    DailyPrice.objects.filter(date=DAY).delete()
    with caplog.at_level(logging.ERROR, logger="apps.etl.alerts"):
        dispatch(run_checks(DAY, calendar))
    assert "webhook" in caplog.text


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(("webhook", "broken"), [(WEBHOOK, False), ("", True)])
def test_nothing_is_posted_for_clean_days_or_without_a_webhook(
    calendar: TradingDays, webhook: str, broken: bool, caplog: pytest.LogCaptureFixture
) -> None:
    _two_clean_days()
    if broken:
        DailyPrice.objects.filter(date=DAY).delete()
    with override_settings(ALERT_WEBHOOK_URL=webhook):
        dispatch(run_checks(DAY, calendar))
    assert len(responses.calls) == 0
    assert "webhook" not in caplog.text


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
@override_settings(ALERT_WEBHOOK_URL=WEBHOOK)
def test_check_quality_command_sends_the_alerts() -> None:
    responses.post(WEBHOOK, status=200)
    _two_clean_days()
    DailyPrice.objects.filter(date=DAY).delete()
    call_command("check_quality", date="2026-10-28")
    assert len(responses.calls) == 1


@responses.activate
@override_settings(ALERT_WEBHOOK_URL=WEBHOOK)
def test_an_info_row_is_posted_only_when_announced(caplog: pytest.LogCaptureFixture) -> None:
    responses.post(WEBHOOK, status=200)
    row = DataQualityLog(
        date=DAY, check_name="calendar_reach", severity="info", detail={"level": 0}
    )
    dispatch([row])
    assert len(responses.calls) == 0
    with caplog.at_level(logging.INFO, logger="apps.etl.alerts"):
        announce([row])
    (call,) = responses.calls
    raw = call.request.body
    body = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    assert body["alerts"] == [
        {
            "date": "2026-10-28",
            "check": "calendar_reach",
            "severity": "info",
            "detail": {"level": 0},
        }
    ]
    assert "calendar_reach" in caplog.text
    assert "ERROR" not in caplog.text


def _log(check: str, severity: str, detail: dict[str, Any] | None = None) -> DataQualityLog:
    return DataQualityLog(date=DAY, check_name=check, severity=severity, detail=detail or {})


def _text(call: responses.Call) -> str:
    raw = call.request.body
    return str(json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))["text"])


@responses.activate
@override_settings(ALERT_WEBHOOK_URL=WEBHOOK)
def test_the_headline_names_each_check_and_severity_worst_first() -> None:
    responses.post(WEBHOOK, status=200)
    announce(
        [
            _log("row_count", "warning", {"table": "daily_price"}),
            _log("row_count", "warning", {"table": "daily_chip"}),
            _log("etl_runs", "error", {"missing": ["twse/foreign_holding"]}),
            _log("calendar_reach", "info", {"level": 0}),
        ]
    )
    assert _text(responses.calls[0]) == (
        "4 data quality alert(s) for 2026-10-28: "
        "etl_runs error; row_count warning ×2; calendar_reach info"
    )


@responses.activate
@override_settings(ALERT_WEBHOOK_URL=WEBHOOK)
def test_a_long_headline_lists_five_kinds_and_counts_the_rest() -> None:
    responses.post(WEBHOOK, status=200)
    announce([_log(f"check_{n}", "warning") for n in range(7)])
    assert _text(responses.calls[0]) == (
        "7 data quality alert(s) for 2026-10-28: check_0 warning; check_1 warning; "
        "check_2 warning; check_3 warning; check_4 warning; +2 more"
    )


@responses.activate
@override_settings(ALERT_WEBHOOK_URL=WEBHOOK)
def test_the_headline_never_carries_the_detail() -> None:
    responses.post(WEBHOOK, status=200)
    detail = {"reason": "TWSE request failed: https://www.twse.com.tw/rwd/zh/x?date=20270101"}
    announce([_log("calendar_reach", "warning", detail)])
    assert _text(responses.calls[0]) == (
        "1 data quality alert(s) for 2026-10-28: calendar_reach warning"
    )
    raw = responses.calls[0].request.body
    body = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    assert body["alerts"][0]["detail"] == detail
