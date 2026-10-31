import json
from collections.abc import Callable
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
import responses
from django.core.management import call_command
from django.core.management.base import CommandError
from responses import matchers

from apps.etl.models import EtlRun, RawFinmind, RawTwse
from apps.market.calendar import TradingDays
from apps.market.models import TradingCalendar


@pytest.mark.django_db
def test_loads_every_day_with_flags_and_notes(fixture_path: Callable[[str], Path]) -> None:
    out = StringIO()
    call_command("load_calendar", file=str(fixture_path("calendar/twse_2026_10.json")), stdout=out)
    assert TradingCalendar.objects.count() == 31
    assert TradingCalendar.objects.filter(is_trading_day=True).count() == 19
    assert TradingCalendar.objects.get(date=date(2026, 10, 9)).note == "國慶日補假"
    assert "31 days" in out.getvalue()
    assert TradingDays.from_db().shift(date(2026, 10, 22), 1) == date(2026, 10, 27)


@pytest.mark.django_db
def test_reloading_updates_in_place(fixture_path: Callable[[str], Path], tmp_path: Path) -> None:
    source = fixture_path("calendar/twse_2026_10.json")
    call_command("load_calendar", file=str(source))
    data = json.loads(source.read_text(encoding="utf-8"))
    data["trading_days"].remove("2026-10-30")
    data["notes"]["2026-10-30"] = "颱風假"
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    call_command("load_calendar", file=str(edited))
    row = TradingCalendar.objects.get(date=date(2026, 10, 30))
    assert (row.is_trading_day, row.note) == (False, "颱風假")
    assert TradingCalendar.objects.count() == 31


@pytest.mark.django_db
@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"start": "2026-10-01", "trading_days": []}',
        '{"start": "2026-10-10", "end": "2026-10-31", "trading_days": ["2026-10-01"]}',
        '{"start": "2026-10-01", "end": "2026-10-31", "trading_days": ["2026/10/01"]}',
        "[]",
        '{"start": "2026-10-01", "end": "2026-10-31", "trading_days": null}',
    ],
)
def test_bad_files_are_command_errors(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(CommandError, match="cannot read calendar file"):
        call_command("load_calendar", file=str(path))
    assert TradingCalendar.objects.count() == 0


def test_missing_file_is_a_command_error(tmp_path: Path) -> None:
    with pytest.raises(CommandError, match="cannot read calendar file"):
        call_command("load_calendar", file=str(tmp_path / "absent.json"))


def _calendar_file(tmp_path: Path, start: str, end: str, trading_days: list[str]) -> Path:
    path = tmp_path / f"{start}.json"
    payload = {"start": start, "end": end, "trading_days": trading_days}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.django_db
def test_file_leaving_a_gap_is_refused(fixture_path: Callable[[str], Path], tmp_path: Path) -> None:
    call_command("load_calendar", file=str(fixture_path("calendar/twse_2026_10.json")))
    later = _calendar_file(tmp_path, "2026-11-02", "2026-11-03", ["2026-11-02", "2026-11-03"])
    with pytest.raises(CommandError, match="gap"):
        call_command("load_calendar", file=str(later))
    earlier = _calendar_file(tmp_path, "2026-09-28", "2026-09-29", ["2026-09-28"])
    with pytest.raises(CommandError, match="gap"):
        call_command("load_calendar", file=str(earlier))
    assert TradingCalendar.objects.count() == 31


@pytest.mark.django_db
def test_file_touching_the_loaded_range_extends_it(
    fixture_path: Callable[[str], Path], tmp_path: Path
) -> None:
    call_command("load_calendar", file=str(fixture_path("calendar/twse_2026_10.json")))
    after = _calendar_file(tmp_path, "2026-11-01", "2026-11-02", ["2026-11-02"])
    before = _calendar_file(tmp_path, "2026-09-29", "2026-09-30", ["2026-09-29", "2026-09-30"])
    call_command("load_calendar", file=str(after))
    call_command("load_calendar", file=str(before))
    calendar = TradingDays.from_db()
    assert (calendar.start, calendar.end) == (date(2026, 9, 29), date(2026, 11, 2))
    assert calendar.shift(date(2026, 10, 30), 1) == date(2026, 11, 2)


FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
SCHEDULE_URL = "https://www.twse.com.tw/rwd/zh/holidaySchedule/holidaySchedule"
TODAY = date(2026, 11, 1)


@pytest.fixture
def frozen_today(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "apps.etl.management.commands.load_calendar.timezone.localdate", lambda: TODAY
    )
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)
    monkeypatch.setattr("apps.etl.adapters.finmind.time.sleep", lambda seconds: None)


@pytest.fixture
def october_sources(load_fixture: Callable[[str], Any]) -> None:
    october = load_fixture("calendar/twse_2026_10.json")["trading_days"]
    responses.get(
        FINMIND_URL,
        json={"msg": "success", "status": 200, "data": [{"date": d} for d in october]},
        match=[
            matchers.query_param_matcher(
                {
                    "dataset": "TaiwanStockTradingDate",
                    "start_date": "2026-10-01",
                    "end_date": "2026-10-31",
                }
            )
        ],
    )
    for year in (2026, 2027):
        responses.get(
            SCHEDULE_URL,
            json=load_fixture(f"twse/reference/holiday_schedule_{year}.json"),
            match=[matchers.query_param_matcher({"date": f"{year}0101", "response": "json"})],
        )


@responses.activate
@pytest.mark.django_db
@pytest.mark.usefixtures("frozen_today", "october_sources")
def test_from_rebuilds_past_and_scheduled_days() -> None:
    out = StringIO()
    call_command("load_calendar", "--from", "2026-10-01", stdout=out)
    assert TradingCalendar.objects.count() == 92
    calendar = TradingDays.from_db()
    assert (calendar.start, calendar.end) == (date(2026, 10, 1), date(2026, 12, 31))
    assert len(calendar.between(date(2026, 10, 1), date(2026, 10, 31))) == 19
    assert len(calendar.between(date(2026, 11, 1), date(2026, 11, 30))) == 21
    assert len(calendar.between(date(2026, 12, 1), date(2026, 12, 31))) == 22
    assert TradingCalendar.objects.get(date=date(2026, 12, 25)).note == "行憲紀念日"
    run = EtlRun.objects.get()
    assert (run.status, run.rows, run.end_date) == ("success", 92, date(2026, 12, 31))
    assert RawFinmind.objects.filter(dataset="trading_date").count() == 1
    assert RawTwse.objects.filter(dataset="holiday_schedule").count() == 2
    assert "92 days 2026-10-01..2026-12-31" in out.getvalue()


@responses.activate
@pytest.mark.django_db
@pytest.mark.usefixtures("frozen_today", "october_sources")
def test_from_twice_is_idempotent() -> None:
    call_command("load_calendar", "--from", "2026-10-01")
    call_command("load_calendar", "--from", "2026-10-01")
    assert TradingCalendar.objects.count() == 92
    assert EtlRun.objects.filter(status="success").count() == 2


@pytest.mark.django_db
@pytest.mark.usefixtures("frozen_today")
@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--from", "2026-11-01"], "before today"),
        (["--from", "2026/10/01"], "ISO date"),
        (["--from", "2026-10-01", "--file", "x.json"], "not allowed with"),
        ([], "one of the arguments"),
    ],
)
def test_bad_arguments_are_command_errors(args: list[str], message: str) -> None:
    with pytest.raises(CommandError, match=message):
        call_command("load_calendar", *args)


@responses.activate
@pytest.mark.django_db
@pytest.mark.usefixtures("frozen_today")
def test_source_failure_leaves_the_calendar_alone(fixture_path: Callable[[str], Path]) -> None:
    call_command("load_calendar", file=str(fixture_path("calendar/twse_2026_10.json")))
    responses.get(FINMIND_URL, json={"msg": "dataset not found", "status": 400})
    with pytest.raises(CommandError, match="SourceRequestError"):
        call_command("load_calendar", "--from", "2026-10-01")
    assert TradingCalendar.objects.count() == 31
    assert EtlRun.objects.get().status == "failed"
