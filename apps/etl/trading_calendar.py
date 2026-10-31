"""trading_calendar rows from TWSE's monthly trading days plus its holiday schedule (SPEC §8.7)."""

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, timedelta
from typing import Any

from apps.etl.errors import SchemaChangedError
from apps.etl.roc import parse_roc_date
from apps.market.calendar import TradingDays

SCHEDULE_FIELDS = ["日期", "名稱", "說明"]
MONTHLY_DATE_FIELD = "日期"
MONTHLY_OK = "OK"
MONTHLY_EMPTY = "很抱歉，沒有符合條件的資料!"
FIRST_SCHEDULE_YEAR = 2021
SCHEDULE_YEARS_AHEAD = 1
UNLABELLED_CLOSURE = "非交易日（原因未標註）"
WEEKEND_TRADING = "週末補行交易日"
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


def schedule_years(start: date, today: date) -> range:
    """Published schedules that can label the range, through next year."""
    first = max(start.year, FIRST_SCHEDULE_YEAR)
    return range(first, today.year + SCHEDULE_YEARS_AHEAD + 1)


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


def traded_days(payload: Mapping[str, Any], month: date, allow_empty: bool = False) -> set[date]:
    """Days listed in TWSE's FMTQIK table for one month; only a month in progress may be empty."""
    if payload.get("stat") == MONTHLY_EMPTY and allow_empty:
        return set()
    if payload.get("stat") != MONTHLY_OK:
        raise SchemaChangedError(
            f"TWSE monthly trading table for {month:%Y-%m} stat is {payload.get('stat')!r}"
        )
    fields = payload.get("fields") or []
    if not fields or fields[0] != MONTHLY_DATE_FIELD:
        raise SchemaChangedError(f"TWSE monthly trading table fields changed: {fields}")
    days = {parse_roc_date(row[0]) for row in payload.get("data") or []}
    if not days or any((day.year, day.month) != (month.year, month.month) for day in days):
        raise SchemaChangedError(f"TWSE monthly trading table for {month:%Y-%m} has wrong dates")
    return days


def _schedule_through(
    covered: date, schedules: Sequence[Mapping[str, Any]]
) -> tuple[date, dict[date, ScheduledDay]]:
    """Extend coverage year by year while schedules are published, refusing gaps."""
    scheduled: dict[date, ScheduledDay] = {}
    for payload in sorted(schedules, key=schedule_year):
        year, days = schedule_year(payload), schedule_days(payload)
        if not days:
            if year <= covered.year:
                raise SchemaChangedError(f"TWSE holiday schedule for {year} is empty")
            break
        if date(year, 1, 1) > covered + timedelta(days=1):
            raise SchemaChangedError(f"no TWSE holiday schedule for {covered.year + 1}")
        scheduled.update(days)
        covered = max(covered, date(year, 12, 31))
    return covered, scheduled


def _note(day: date, is_open: bool, listed: ScheduledDay | None) -> str:
    weekend = day.weekday() >= 5
    if is_open:
        return WEEKEND_TRADING if weekend else ""
    if listed is not None and not listed[0]:
        return listed[1]
    return "" if weekend else UNLABELLED_CLOSURE


def build_calendar(
    traded: Iterable[date],
    through: date,
    schedules: Sequence[Mapping[str, Any]],
    start: date,
) -> tuple[TradingDays, dict[str, str]]:
    """Days up to through follow the traded days; later days follow the holiday schedule."""
    trading_set = set(traded)
    end, scheduled = _schedule_through(through, schedules)
    trading: list[date] = []
    notes: dict[str, str] = {}
    day = start
    while day <= end:
        listed = scheduled.get(day)
        if day <= through:
            is_open = day in trading_set
        else:
            is_open = listed[0] if listed is not None else day.weekday() < 5
        if is_open:
            trading.append(day)
        note = _note(day, is_open, listed)
        if note:
            notes[day.isoformat()] = note
        day += timedelta(days=1)
    return TradingDays(trading, start, end), notes
