from collections.abc import Callable
from datetime import date
from typing import Any

import pytest
import responses
from django.conf import settings
from django.core.management import call_command
from django.core.management.base import CommandError
from responses import matchers

from apps.etl.adapters import build_adapter
from apps.etl.models import EtlRun, RawTwse
from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice

URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_phase0_acceptance_command_writes_raw_and_standard(
    load_fixture: Callable[[str], Any],
) -> None:
    responses.get(URL, json=load_fixture("twse/daily_price/mi_index_20261028.json"))
    call_command("fetch", source="twse", dataset="daily_price", date="2026-10-28")
    assert RawTwse.objects.filter(dataset="daily_price").count() == 1
    assert DailyPrice.objects.filter(date=date(2026, 10, 28)).count() == 6
    run = EtlRun.objects.get()
    assert (run.source, run.dataset, run.status, run.rows) == ("twse", "daily_price", "success", 6)
    assert (run.start_date, run.end_date) == (date(2026, 10, 28), date(2026, 10, 28))


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_running_twice_does_not_duplicate_rows(load_fixture: Callable[[str], Any]) -> None:
    payload = load_fixture("twse/daily_price/mi_index_20261028.json")
    responses.get(URL, json=payload)
    responses.get(URL, json=payload)
    for _ in range(2):
        call_command(
            "fetch", "--source", "twse", "--dataset", "daily_price", "--date", "2026-10-28"
        )
    assert DailyPrice.objects.count() == 6
    assert EtlRun.objects.filter(status="success").count() == 2


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_date_range_fetches_each_trading_day(load_fixture: Callable[[str], Any]) -> None:
    payload = load_fixture("twse/daily_price/mi_index_20261028.json")
    for day in ("20261022", "20261027", "20261028"):
        params = {"date": day, "type": "ALLBUT0999", "response": "json"}
        responses.get(
            URL, json={**payload, "date": day}, match=[matchers.query_param_matcher(params)]
        )
    call_command(
        "fetch", source="twse", dataset="daily_price", start="2026-10-22", end="2026-10-28"
    )
    assert len(responses.calls) == 3
    assert DailyPrice.objects.count() == 18
    run = EtlRun.objects.get()
    assert (run.rows, run.checkpoint) == (18, {"last_date": "2026-10-28"})


@pytest.mark.usefixtures("october_2026_in_db")
def test_non_trading_day_is_a_command_error() -> None:
    with pytest.raises(CommandError, match="not a trading day"):
        call_command("fetch", source="twse", dataset="daily_price", date="2026-10-23")


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_unpublished_day_fails_the_run(load_fixture: Callable[[str], Any]) -> None:
    responses.get(URL, json=load_fixture("twse/daily_price/mi_index_20261025_no_data.json"))
    with pytest.raises(CommandError, match="SourceNotReadyError"):
        call_command("fetch", source="twse", dataset="daily_price", date="2026-10-28")
    assert EtlRun.objects.get().status == "failed"


@pytest.mark.usefixtures("october_2026_in_db")
@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"source": "nope", "dataset": "daily_price", "date": "2026-10-28"}, "unknown source"),
        ({"source": "twse", "dataset": "nope", "date": "2026-10-28"}, "no dataset"),
        ({"source": "twse", "dataset": "daily_price"}, "give --date"),
        ({"source": "twse", "dataset": "daily_price", "start": "2026-10-27"}, "give --date"),
        (
            {
                "source": "twse",
                "dataset": "daily_price",
                "date": "2026-10-28",
                "start": "2026-10-27",
            },
            "either --date",
        ),
        (
            {
                "source": "twse",
                "dataset": "daily_price",
                "start": "2026-10-28",
                "end": "2026-10-27",
            },
            "after --end",
        ),
        ({"source": "twse", "dataset": "daily_price", "date": "2026/10/28"}, "ISO date"),
    ],
)
def test_bad_arguments_are_command_errors(options: dict[str, str], message: str) -> None:
    with pytest.raises(CommandError, match=message):
        call_command("fetch", **options)


@pytest.mark.django_db
def test_empty_calendar_is_a_command_error() -> None:
    with pytest.raises(CommandError, match="trading_calendar is empty"):
        call_command("fetch", source="twse", dataset="daily_price", date="2026-10-28")


def test_october_fixture_matches_twse(october_2026: TradingDays) -> None:
    assert len(october_2026.between(date(2026, 10, 1), date(2026, 10, 31))) == 19
    assert not october_2026.is_trading_day(date(2026, 10, 9))
    assert not october_2026.is_trading_day(date(2026, 10, 26))


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_refused_request_is_a_command_error() -> None:
    responses.get(URL, status=403)
    with pytest.raises(CommandError, match="SourceRequestError"):
        call_command("fetch", source="twse", dataset="daily_price", date="2026-10-28")
    assert EtlRun.objects.get().status == "failed"


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_exhausted_retries_are_a_command_error() -> None:
    responses.get(URL, status=503)
    with pytest.raises(CommandError, match="RetryableError"):
        call_command("fetch", source="twse", dataset="daily_price", date="2026-10-28")
    assert EtlRun.objects.get().status == "failed"
    assert len(responses.calls) == settings.ETL_RATE_LIMITS["twse"]["max_retries"] + 1


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_future_date_is_refused_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("apps.etl.adapters.base.timezone.localdate", lambda: date(2026, 10, 27))
    with pytest.raises(CommandError, match="after today"):
        call_command("fetch", source="twse", dataset="daily_price", date="2026-10-28")
    assert len(responses.calls) == 0
    assert EtlRun.objects.count() == 0


def test_adapters_of_one_source_share_a_limiter(
    october_2026: TradingDays, monkeypatch: pytest.MonkeyPatch
) -> None:
    slept: list[float] = []
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", slept.append)
    first = build_adapter("twse", october_2026)
    second = build_adapter("twse", october_2026)
    assert first.limiter is second.limiter
    first.limiter.call(lambda: 1)
    second.limiter.call(lambda: 2)
    assert len(slept) == 1
    assert slept[0] > settings.ETL_RATE_LIMITS["twse"]["min_interval"] - 1.0
