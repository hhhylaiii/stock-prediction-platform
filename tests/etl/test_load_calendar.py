import json
from collections.abc import Callable
from datetime import date
from io import StringIO
from pathlib import Path

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

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
