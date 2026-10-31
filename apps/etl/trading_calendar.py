"""trading_calendar rows from traded dates plus the TWSE holiday schedule (SPEC §5, §8.7)."""

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, timedelta
from typing import Any

from apps.etl.adapters.base import SchemaChangedError
from apps.market.calendar import TradingDays

SCHEDULE_FIELDS = ["日期", "名稱", "說明"]
NO_TRADING = "無交易"
TRADING_REMINDER = "交易日"
NOTE_LIMIT = 64
ScheduledDay = tuple[bool, str]


def _iso_date(value: Any, what: str) -> date:
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        raise SchemaChangedError(f"{what} has a bad date {value!r}") from None


def schedule_year(payload: Mapping[str, Any]) -> int:
    year = payload.get("queryYear")
    if not isinstance(year, int):
        raise SchemaChangedError(f"TWSE holiday schedule has no queryYear: {year!r}")
    return year


def schedule_days(payload: Mapping[str, Any]) -> dict[date, ScheduledDay]:
    """Each listed date as (is_trading_day, note); 無交易 anywhere in a row means closed."""
    if payload.get("stat") != "ok":
        raise SchemaChangedError(f"TWSE holiday schedule stat is {payload.get('stat')!r}")
    if payload.get("fields") != SCHEDULE_FIELDS:
        raise SchemaChangedError(f"TWSE holiday schedule fields changed: {payload.get('fields')}")
    grouped: dict[date, list[tuple[str, str]]] = {}
    for row in payload.get("data") or []:
        day = _iso_date(row[0], "TWSE holiday schedule")
        grouped.setdefault(day, []).append((str(row[1]).strip(), " ".join(str(row[2]).split())))
    days: dict[date, ScheduledDay] = {}
    for day, rows in grouped.items():
        if any(NO_TRADING in name + detail for name, detail in rows):
            days[day] = (False, rows[0][0][:NOTE_LIMIT])
        elif any(TRADING_REMINDER in name for name, _ in rows):
            days[day] = (True, "")
        else:
            days[day] = (False, rows[0][0][:NOTE_LIMIT])
    return days


def _traded_dates(rows: Iterable[Mapping[str, Any]], start: date) -> set[date]:
    traded = set()
    for row in rows:
        if "date" not in row:
            raise SchemaChangedError("FinMind trading date rows have no date")
        traded.add(_iso_date(row["date"], "FinMind TaiwanStockTradingDate"))
    traded = {day for day in traded if day >= start}
    if not traded:
        raise SchemaChangedError(f"FinMind returned no trading dates since {start}")
    return traded


def _schedule_through(
    covered: date, schedules: Sequence[Mapping[str, Any]]
) -> tuple[date, dict[date, ScheduledDay]]:
    """Extend coverage year by year while schedules are published, refusing gaps."""
    scheduled: dict[date, ScheduledDay] = {}
    for payload in sorted(schedules, key=schedule_year):
        year, days = schedule_year(payload), schedule_days(payload)
        if not days:
            break
        if date(year, 1, 1) > covered + timedelta(days=1):
            raise SchemaChangedError(f"no TWSE holiday schedule for {covered.year + 1}")
        scheduled.update(days)
        covered = max(covered, date(year, 12, 31))
    return covered, scheduled


def build_calendar(
    traded_rows: Iterable[Mapping[str, Any]],
    schedules: Sequence[Mapping[str, Any]],
    start: date,
) -> tuple[TradingDays, dict[str, str]]:
    """Days up to the last traded date follow the traded dates; later days follow the schedule."""
    traded = _traded_dates(traded_rows, start)
    last_traded = max(traded)
    end, scheduled = _schedule_through(last_traded, schedules)
    trading: list[date] = []
    notes: dict[str, str] = {}
    day = start
    while day <= end:
        listed = scheduled.get(day)
        if day <= last_traded:
            is_open = day in traded
        else:
            is_open = listed[0] if listed is not None else day.weekday() < 5
        if is_open:
            trading.append(day)
        elif listed is not None and not listed[0]:
            notes[day.isoformat()] = listed[1]
        day += timedelta(days=1)
    return TradingDays(trading, start, end), notes
