"""Attention, disposition and full-delivery lists of both exchanges as disposition_event rows."""

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from typing import Any

import pandas as pd

from apps.etl.adapters.exchange_tables import is_foreign_currency_etf
from apps.etl.errors import SchemaChangedError
from apps.etl.exchange_rows import stock_code
from apps.etl.roc import parse_roc_date
from apps.etl.upsert import upsert
from apps.market.calendar import OutsideCalendarError, TradingDays
from apps.market.models import DispositionEvent, DispositionType

EVENT_COLUMNS = ["stock_id", "start_date", "type", "end_date"]
ANNOUNCED = "announced"
PERIOD = re.compile(r"(.+?)\s*[~～]\s*(.+)")
PLACEHOLDER = "本日無"
ATTENTION = DispositionType.ATTENTION.value
DISPOSITION = DispositionType.DISPOSITION.value
FULL_DELIVERY = DispositionType.FULL_DELIVERY.value


def previous_session(calendar: TradingDays, day: date) -> date:
    """The trading day before day, or day itself at the start of the calendar."""
    try:
        return calendar.shift(day, -1)
    except OutsideCalendarError:
        return day


def list_rows(body: Mapping[str, Any], label: str, needed: Sequence[str]) -> list[dict[str, Any]]:
    """The list's rows keyed by field name; the row count must match total when given."""
    fields = [str(name).strip() for name in body.get("fields") or []]
    missing = sorted(set(needed) - set(fields))
    if missing:
        raise SchemaChangedError(f"{label} lost columns {missing}")
    rows = body.get("data")
    if not isinstance(rows, list):
        raise SchemaChangedError(f"{label} rows are malformed")
    total = body.get("total", body.get("totalCount"))
    if total is not None and total != len(rows):
        raise SchemaChangedError(f"{label} sent {len(rows)} of {total} rows")
    if not all(isinstance(row, list) and len(row) == len(fields) for row in rows):
        raise SchemaChangedError(f"{label} rows do not match its fields")
    return [dict(zip(fields, row, strict=True)) for row in rows]


def listed_code(record: Mapping[str, Any], field: str, label: str) -> str | None:
    """The row's code; None for a 'nothing listed today' placeholder or a foreign-currency line."""
    text = str(record[field]).strip()
    if text:
        code = stock_code(text)
        return None if is_foreign_currency_etf(code) else code
    if any(PLACEHOLDER in str(value) for value in record.values()):
        return None
    raise SchemaChangedError(f"{label} has a row without a code: {record}")


def disposition_period(text: str) -> tuple[date, date]:
    match = PERIOD.fullmatch(str(text).strip())
    if match is None:
        raise SchemaChangedError(f"unexpected disposition period {text!r}")
    first, last = (parse_roc_date(part) for part in match.groups())
    if last < first:
        raise SchemaChangedError(f"disposition period {text!r} ends before it starts")
    return first, last


def announced_events(
    records: Iterable[Mapping[str, Any]],
    fields: tuple[str, str],
    label: str,
    sessions: Sequence[date],
) -> list[dict[str, Any]]:
    """One-day attention events; each must be announced on a trading day of the query."""
    code_field, date_field = fields
    allowed = set(sessions)
    events = []
    for record in records:
        code = listed_code(record, code_field, label)
        if code is None:
            continue
        day = parse_roc_date(str(record[date_field]))
        if day not in allowed:
            raise SchemaChangedError(
                f"{label} lists {code} on {day}, not a trading day of {sessions[0]}..{sessions[-1]}"
            )
        events.append(_event(code, day, ATTENTION, day, day))
    return events


def period_events(
    records: Iterable[Mapping[str, Any]],
    fields: tuple[str, str, str],
    label: str,
    window: tuple[date, date],
) -> list[dict[str, Any]]:
    """Periods overlapping the window or announced in it, with their announcement day."""
    code_field, announced_field, period_field = fields
    events = []
    for record in records:
        code = listed_code(record, code_field, label)
        if code is None:
            continue
        first, last = disposition_period(str(record[period_field]))
        announced = parse_roc_date(str(record[announced_field]))
        outside = last < window[0] or first > window[1]
        if outside and not window[0] <= announced <= window[1]:
            raise SchemaChangedError(
                f"{label} lists {code} for {first}..{last}, outside {window[0]}..{window[1]}"
            )
        events.append(_event(code, first, DISPOSITION, last, announced))
    return events


def day_events(codes: Iterable[str], day: date, kind: str) -> list[dict[str, Any]]:
    return [_event(code, day, kind, day, day) for code in codes]


def _event(code: str, start: date, kind: str, end: date, announced: date) -> dict[str, Any]:
    return {
        "stock_id": code,
        "start_date": start,
        "type": kind,
        "end_date": end,
        ANNOUNCED: announced,
    }


def event_frame(events: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(list(events), columns=[*EVENT_COLUMNS, ANNOUNCED], dtype=object)


def standard_events(raw_df: pd.DataFrame) -> pd.DataFrame:
    """One row per event key; a period announced again keeps the latest announcement's end."""
    if raw_df.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS, dtype=object)
    ordered = raw_df.sort_values(ANNOUNCED, kind="stable")
    latest = ordered.drop_duplicates(subset=["stock_id", "start_date", "type"], keep="last")
    return latest[EVENT_COLUMNS].reset_index(drop=True)


def upsert_events(frame: pd.DataFrame) -> int:
    return upsert(
        DispositionEvent,
        frame,
        unique_fields=("stock_id", "start_date", "type"),
        update_fields=("end_date",),
    )
