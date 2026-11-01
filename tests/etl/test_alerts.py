import json
import logging

import pytest
import responses
from django.core.management import call_command
from django.test import override_settings

from apps.etl.alerts import dispatch
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
