from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import pytest
import responses
from django.core.management import call_command
from django.core.management.base import CommandError
from responses import matchers

from apps.etl.models import EtlRun
from apps.etl.trading_calendar import UNLABELLED_CLOSURE, WEEKEND_TRADING, corrections
from apps.market.models import TradingCalendar

FMTQIK_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK"
OCTOBER = "twse/reference/fmtqik_202610.json"

pytestmark = pytest.mark.usefixtures("october_2026_in_db", "quiet")


@pytest.fixture
def quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)


def _today(monkeypatch: pytest.MonkeyPatch, day: date) -> None:
    monkeypatch.setattr(
        "apps.etl.management.commands.refresh_calendar.timezone.localdate", lambda: day
    )


def _serve(month: str, payload: dict[str, Any]) -> None:
    responses.get(
        FMTQIK_URL,
        json=payload,
        match=[matchers.query_param_matcher({"date": f"{month}01", "response": "json"})],
    )


def _without(payload: dict[str, Any], *roc_days: str) -> dict[str, Any]:
    return {**payload, "data": [row for row in payload["data"] if row[0] not in roc_days]}


def _with(payload: dict[str, Any], roc_day: str) -> dict[str, Any]:
    return {**payload, "data": [*payload["data"], [roc_day, *payload["data"][0][1:]]]}


def _flags() -> dict[date, tuple[bool, str]]:
    rows = TradingCalendar.objects.values_list("date", "is_trading_day", "note")
    return {day: (is_open, note) for day, is_open, note in rows}


def test_a_day_missing_from_the_traded_days_before_the_last_one_is_closed() -> None:
    stored = {date(2026, 10, 28): (True, ""), date(2026, 10, 30): (True, "")}
    traded = {date(2026, 10, 29)}
    assert corrections(traded, stored) == {date(2026, 10, 28): (False, UNLABELLED_CLOSURE)}


def test_a_listed_weekend_or_holiday_is_opened_with_its_note() -> None:
    stored = {date(2026, 10, 24): (False, ""), date(2026, 10, 26): (False, "補假")}
    traded = {date(2026, 10, 24), date(2026, 10, 26)}
    assert corrections(traded, stored) == {
        date(2026, 10, 24): (True, WEEKEND_TRADING),
        date(2026, 10, 26): (True, ""),
    }


@responses.activate
def test_a_typhoon_day_is_corrected_and_recorded(
    monkeypatch: pytest.MonkeyPatch,
    load_fixture: Callable[[str], Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _today(monkeypatch, date(2026, 10, 31))
    _serve("202610", _without(load_fixture(OCTOBER), "115/10/28"))
    before = _flags()
    call_command("refresh_calendar")
    after = _flags()
    assert after[date(2026, 10, 28)] == (False, UNLABELLED_CLOSURE)
    assert {day: flag for day, flag in after.items() if before[day] != flag} == {
        date(2026, 10, 28): (False, UNLABELLED_CLOSURE)
    }
    run = EtlRun.objects.get(dataset="trading_calendar_refresh")
    assert (run.status, run.rows, run.checkpoint) == (
        "success",
        1,
        {"corrected": {"2026-10-28": False}},
    )
    assert "2026-10-28 is now closed" in capsys.readouterr().out


@responses.activate
def test_the_last_day_is_not_closed_until_a_later_day_is_listed(
    monkeypatch: pytest.MonkeyPatch, load_fixture: Callable[[str], Any]
) -> None:
    _today(monkeypatch, date(2026, 10, 31))
    _serve("202610", _without(load_fixture(OCTOBER), "115/10/30"))
    call_command("refresh_calendar")
    assert _flags()[date(2026, 10, 30)] == (True, "")


@responses.activate
def test_a_day_listed_as_traded_is_opened(
    monkeypatch: pytest.MonkeyPatch, load_fixture: Callable[[str], Any]
) -> None:
    _today(monkeypatch, date(2026, 10, 31))
    _serve("202610", _with(load_fixture(OCTOBER), "115/10/24"))
    call_command("refresh_calendar")
    assert _flags()[date(2026, 10, 24)] == (True, WEEKEND_TRADING)


@responses.activate
def test_today_and_later_days_are_left_to_the_holiday_schedule(
    monkeypatch: pytest.MonkeyPatch, load_fixture: Callable[[str], Any]
) -> None:
    _today(monkeypatch, date(2026, 10, 29))
    _serve("202610", _without(load_fixture(OCTOBER), "115/10/29", "115/10/30"))
    call_command("refresh_calendar")
    assert _flags()[date(2026, 10, 29)] == (True, "")
    assert _flags()[date(2026, 10, 30)] == (True, "")
    assert not EtlRun.objects.get().checkpoint["corrected"]


@responses.activate
def test_a_month_in_progress_may_list_nothing_yet(
    monkeypatch: pytest.MonkeyPatch, load_fixture: Callable[[str], Any]
) -> None:
    TradingCalendar.objects.bulk_create(
        TradingCalendar(
            date=date(2026, 11, 1) + timedelta(days=n), is_trading_day=n % 7 not in (0, 6)
        )
        for n in range(30)
    )
    _today(monkeypatch, date(2026, 11, 2))
    _serve("202610", load_fixture(OCTOBER))
    _serve("202611", load_fixture("twse/reference/fmtqik_202611.json"))
    call_command("refresh_calendar")
    assert EtlRun.objects.get().status == "success"
    assert _flags()[date(2026, 10, 30)] == (True, "")


@responses.activate
def test_a_broken_table_changes_nothing_and_fails_the_run(
    monkeypatch: pytest.MonkeyPatch, load_fixture: Callable[[str], Any]
) -> None:
    _today(monkeypatch, date(2026, 10, 31))
    _serve("202610", {**load_fixture(OCTOBER), "fields": ["成交股數"]})
    before = _flags()
    with pytest.raises(CommandError, match="fields changed"):
        call_command("refresh_calendar")
    assert _flags() == before
    assert EtlRun.objects.get().status == "failed"
