"""trading_calendar rows from TWSE's monthly trading days plus its holiday schedule (SPEC §8.7)."""

import re
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
REFRESH_SOURCE = "trading_calendar"
REFRESH_DATASET = "trading_calendar_refresh"
TRADING_REMINDER = "交易日"
NOTE_LIMIT = 64
ROC_YEAR = r"(?:(\d{2,3})年)?"
ROC_OFFSET = 1911
BRACKETED = r"[（(][^）)]*[）)]"
WEEKDAY = rf"(?:{BRACKETED})?"
ONE_DAY = rf"{ROC_YEAR}(\d{{1,2}})月(\d{{1,2}})日"
NAMED_DAY = re.compile(ONE_DAY)
DAY_RANGE = re.compile(rf"{ONE_DAY}{WEEKDAY}至{ONE_DAY}")
LEADING_DAYS = re.compile(rf"^(?:(?:\d{{2,3}}年)?\d{{1,2}}月\d{{1,2}}日{WEEKDAY}[及、與和至]?)+")
HALF_YEAR = timedelta(days=183)
SUBJECT_FILLER = re.compile(rf"[及、與和至\s]|市場|{BRACKETED}")
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


def _no_trading(day: date, name: str, detail: str) -> dict[date, str]:
    """Days a row says do not trade, each with the clause saying so, less its leading dates.

    A clause naming days before 無交易 is about those days: the last trading day before a holiday
    lists the settlement-only days after it. A clause naming none is about the row's own day.
    Anything else before 無交易 than days, their joiners and 市場 is refused, not guessed at.
    """
    closed: dict[date, str] = {}
    for text in (name, *detail.split("。")):
        if NO_TRADING not in text:
            continue
        note = LEADING_DAYS.sub("", text.strip()).strip().rstrip("。")
        subject, rest = text.split(NO_TRADING, 1)
        if NO_TRADING in rest or SUBJECT_FILLER.sub("", NAMED_DAY.sub("", subject)):
            raise SchemaChangedError(
                f"TWSE holiday schedule has a no-trading clause that cannot be read: {text.strip()}"
            )
        for target in _named_days(day, subject) or [day]:
            closed.setdefault(target, note)
    return closed


def _named_days(day: date, text: str) -> list[date]:
    """Every day the text names, the inside of each 至 range included."""
    named = {_named(day, *found) for found in NAMED_DAY.findall(text)}
    for found in DAY_RANGE.findall(text):
        first, last = _named(day, *found[:3]), _named(day, *found[3:])
        if last < first:
            raise SchemaChangedError(
                f"TWSE holiday schedule names the range {found[1]}月{found[2]}日"
                f"至{found[4]}月{found[5]}日, which runs backwards"
            )
        named.update(first + timedelta(days=n) for n in range((last - first).days))
    return sorted(named)


def _named(day: date, roc_year: str, month: str, of_month: str) -> date:
    """The named day in its 民國 year when one is written, else in the year nearest the row."""
    try:
        if roc_year:
            return date(int(roc_year) + ROC_OFFSET, int(month), int(of_month))
        found = date(day.year, int(month), int(of_month))
        if abs(found - day) > HALF_YEAR:
            found = found.replace(year=day.year + (1 if found < day else -1))
    except ValueError:
        written = f"{roc_year}年" if roc_year else ""
        raise SchemaChangedError(
            f"TWSE holiday schedule names {written}{month}月{of_month}日, which does not exist"
        ) from None
    return found


def schedule_days(payload: Mapping[str, Any]) -> dict[date, ScheduledDay]:
    """Each listed date as (is_trading_day, note); a day a 無交易 clause is about is closed."""
    if payload.get("stat") != "ok":
        raise SchemaChangedError(f"TWSE holiday schedule stat is {payload.get('stat')!r}")
    if payload.get("fields") != SCHEDULE_FIELDS:
        raise SchemaChangedError(f"TWSE holiday schedule fields changed: {payload.get('fields')}")
    grouped: dict[date, list[str]] = {}
    closed: dict[date, str] = {}
    own: set[date] = set()
    for row in payload.get("data") or []:
        day = _iso_date(row[0], "TWSE holiday schedule")
        name, detail = str(row[1]).strip(), " ".join(str(row[2]).split())
        grouped.setdefault(day, []).append(name)
        for target, note in _no_trading(day, name, detail).items():
            closed.setdefault(target, note)
            own.update({target} & {day})
    days: dict[date, ScheduledDay] = {
        day: (False, note[:NOTE_LIMIT]) for day, note in closed.items()
    }
    for day, names in grouped.items():
        reminded = any(TRADING_REMINDER in name for name in names)
        if day in closed:
            if reminded and day not in own:
                raise SchemaChangedError(
                    f"TWSE holiday schedule calls {day} a trading day, yet another row closes it"
                )
            continue
        if reminded:
            days[day] = (True, "")
        else:
            days[day] = (False, names[0][:NOTE_LIMIT])
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


def corrections(
    traded: set[date], stored: Mapping[date, ScheduledDay], priced: frozenset[date] = frozenset()
) -> dict[date, ScheduledDay]:
    """Stored days FMTQIK contradicts; an unpriced missing day closes before the last traded one."""
    last = max(traded, default=None)
    changed: dict[date, ScheduledDay] = {}
    for day, (is_open, _) in sorted(stored.items()):
        opened = day in traded
        if opened == is_open:
            continue
        if not opened and (last is None or day > last or day in priced):
            continue
        changed[day] = (opened, _note(day, opened, None))
    return changed
