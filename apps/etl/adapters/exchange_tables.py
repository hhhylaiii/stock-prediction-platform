"""Daily whole-market tables of TWSE and TPEx mapped onto standard rows (SPEC §4, §5)."""

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from typing import Any, NamedTuple

import pandas as pd
from django.db import models

from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.exchange_rows import parse_count, parse_percent, parse_signed, stock_code
from apps.market.models import DailyChip, ShareChangeReason

logger = logging.getLogger(__name__)
NO_DATA_STAT = "沒有符合條件的資料"
AFTER_LATEST_STAT = "查詢日期大於可查詢最大日期"
FOREIGN_CURRENCY_CODE_SUFFIXES = frozenset("KMSC")
SHARES_PER_LOT = 1000
TOTAL_ROW_NAME = "合計"
SHARE_COLUMNS = ["stock_id", "effective_date", "shares", "reason"]
BLANK_MARKERS = frozenset({"-", "--", ""})
PERCENT = 100.0
Converter = Callable[[Sequence[Any]], int | float | None]
AnswerCheck = Callable[[dict[str, Any], date], None]


def is_foreign_currency_etf(code: str) -> bool:
    """TWSE: an ETF code whose sixth character is K, M, S or C trades in a foreign currency."""
    return len(code) == 6 and code.startswith("00") and code[5] in FOREIGN_CURRENCY_CODE_SUFFIXES


def find_table(payload: dict[str, Any], title_part: str) -> dict[str, Any]:
    for table in payload.get("tables", []):
        if title_part in (table.get("title") or ""):
            return dict(table)
    raise SchemaChangedError(f"no table titled *{title_part}* in TWSE response")


def check_published(payload: dict[str, Any], label: str, day: date) -> None:
    """TWSE says 'no data', 'after the latest date' or sends only empty tables before publishing."""
    stat = str(payload.get("stat", ""))
    if NO_DATA_STAT in stat or AFTER_LATEST_STAT in stat:
        raise SourceNotReadyError(f"{label} for {day} not available: {stat}")
    bodies = payload.get("tables", [payload])
    if stat.lower() == "ok" and isinstance(bodies, list) and bodies and all(map(_blank, bodies)):
        raise SourceNotReadyError(f"{label} for {day} has no rows yet")


def _blank(body: Any) -> bool:
    """An empty placeholder table; a missing or renamed data key is a layout change instead."""
    return isinstance(body, dict) and (not body or body.get("data") == [])


def net_shares(values: Sequence[Any]) -> int:
    return sum(parse_signed(value) for value in values)


def shares(values: Sequence[Any]) -> int:
    return sum(parse_count(value) for value in values)


def lots_in_shares(values: Sequence[Any]) -> int:
    return shares(values) * SHARES_PER_LOT


def percent(values: Sequence[Any]) -> float:
    return sum(parse_percent(value) for value in values)


def ratio(values: Sequence[Any]) -> float | None:
    """A published ratio such as P/E; TWSE prints '-' where none is published."""
    (text,) = (str(value).replace(",", "").strip() for value in values)
    if text in BLANK_MARKERS:
        return None
    try:
        number = float(text)
    except ValueError:
        raise SchemaChangedError(f"unexpected ratio value {text!r}") from None
    if not math.isfinite(number):
        raise SchemaChangedError(f"unexpected ratio value {text!r}")
    return number


def yield_fraction(values: Sequence[Any]) -> float | None:
    published = ratio(values)
    return None if published is None else published / PERCENT


def is_percent(value: float) -> bool:
    return 0 <= value <= PERCENT


def is_fraction(value: float) -> bool:
    return 0 <= value < 1


def is_positive(value: float) -> bool:
    return value > 0


class DailyTable(NamedTuple):
    """One whole-market table and the standard columns it fills, each summing its sources."""

    label: str
    path: str
    params: Mapping[str, str]
    title: str | None
    hints: str | None
    code: str
    name: str
    columns: Mapping[str, tuple[str, ...]]
    convert: Converter
    model: type[models.Model] = DailyChip
    valid: Mapping[str, Callable[[float], bool]] = {}
    check: Callable[[dict[str, Any]], None] | None = None
    pending_title: str | None = None
    converters: Mapping[str, Converter] = {}
    keys: tuple[str, str] = ("stock_id", "date")


def labelled_fields(body: Mapping[str, Any], label: str) -> list[str]:
    """Field names prefixed by their column group, so 融資今日餘額 and 融券今日餘額 differ."""
    fields = [str(name) for name in body.get("fields") or []]
    groups = body.get("groups") or []
    if groups:
        titles = [
            str(group.get("title") or "")
            for group in groups
            for _ in range(int(group.get("span") or 0))
        ]
        if len(titles) != len(fields):
            raise SchemaChangedError(f"TWSE {label} column groups no longer match its fields")
        fields = [f"{title}{name}" for title, name in zip(titles, fields, strict=True)]
    if len(set(fields)) != len(fields):
        raise SchemaChangedError(f"TWSE {label} has repeated columns: {fields}")
    return fields


def source_rows(
    payload: dict[str, Any], day: date, table: DailyTable, answered: AnswerCheck
) -> pd.DataFrame:
    """The table's rows for the day under labelled field names, after layout checks."""
    answered(payload, day)
    if table.title is not None and table.pending_title is not None:
        titles = [str(listed.get("title") or "") for listed in payload.get("tables", [])]
        if not any(table.title in title for title in titles) and any(
            title.endswith(table.pending_title) for title in titles
        ):
            raise SourceNotReadyError(
                f"TWSE {table.label} for {day} lists {table.pending_title} only"
            )
    body = payload if table.title is None else find_table(payload, table.title)
    if table.hints is not None and body.get("hints") != table.hints:
        raise SchemaChangedError(f"TWSE {table.label} units changed: {body.get('hints')!r}")
    if table.check is not None:
        table.check(payload)
    fields = labelled_fields(body, table.label)
    needed = {table.code, table.name, *(name for names in table.columns.values() for name in names)}
    missing = sorted(needed - set(fields))
    if missing:
        raise SchemaChangedError(f"TWSE {table.label} lost columns {missing}")
    try:
        frame = pd.DataFrame(body["data"], columns=fields)
    except (KeyError, ValueError) as exc:
        raise SchemaChangedError(f"TWSE {table.label} table is malformed: {exc}") from None
    if frame.empty:
        raise SourceNotReadyError(f"TWSE {table.label} for {day} has no rows yet")
    check_complete(payload, body, table, len(frame))
    return frame.assign(date=day)


def check_complete(
    payload: Mapping[str, Any], body: Mapping[str, Any], table: DailyTable, rows: int
) -> None:
    """The row count and the selectType TWSE echoes back must match what was asked for."""
    total = body.get("total", body.get("totalCount"))
    if total is not None and total != rows:
        raise SchemaChangedError(f"TWSE {table.label} sent {rows} of {total} rows")
    answered = payload.get("selectType")
    asked = table.params.get("selectType")
    if answered is not None and asked is not None and answered != asked:
        raise SchemaChangedError(
            f"TWSE {table.label} answered selectType {answered!r}, not {asked!r}"
        )


def standard_rows(raw_df: pd.DataFrame, table: DailyTable) -> pd.DataFrame:
    """One row per local-currency security; the total row is skipped, repeats are refused."""
    records = []
    seen: set[tuple[str, date]] = set()
    for row in raw_df.to_dict("records"):
        if not str(row[table.code]).strip() and str(row[table.name]).strip() == TOTAL_ROW_NAME:
            continue
        code = stock_code(row[table.code])
        if is_foreign_currency_etf(code):
            continue
        key = (code, row["date"])
        if key in seen:
            raise SchemaChangedError(f"TWSE {table.label} repeated {code} on {row['date']}")
        seen.add(key)
        values = {
            column: table.converters.get(column, table.convert)([row[name] for name in names])
            for column, names in table.columns.items()
        }
        records.append({"stock_id": code, "date": row["date"], **_validated(values, table, key)})
    if not records:
        raise SourceNotReadyError(f"TWSE {table.label} has no stock rows yet")
    return pd.DataFrame(records, columns=["stock_id", "date", *table.columns], dtype=object)


def _validated(
    values: dict[str, int | float | None], table: DailyTable, key: tuple[str, date]
) -> dict[str, int | float | None]:
    kept = dict(values)
    for column, accepts in table.valid.items():
        value = values[column]
        if value is not None and not accepts(value):
            logger.warning(
                "TWSE %s %s %s: %s %s out of range, dropped",
                table.label,
                key[1],
                key[0],
                column,
                values[column],
            )
            kept[column] = None
    return kept


def quoted_only(
    frame: pd.DataFrame,
    quoted: Mapping[date, set[str]],
    label: str,
    zero_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Rows of codes quoted that day; with zero_columns, quoted codes left out get zeros."""
    parts = []
    for day, codes in sorted(quoted.items()):
        rows = frame.loc[frame["date"] == day]
        listed = rows["stock_id"].isin(codes)
        skipped = sorted(rows.loc[~listed, "stock_id"])
        if skipped:
            logger.info(
                "%s %s: %d codes not quoted that day skipped: %s", label, day, len(skipped), skipped
            )
        parts.append(rows.loc[listed])
        if zero_columns:
            quiet = sorted(codes - set(rows["stock_id"]))
            logger.info("%s %s: %d quoted stocks without rows, set to 0", label, day, len(quiet))
            zeros = [
                {"stock_id": code, "date": day, **dict.fromkeys(zero_columns, 0)} for code in quiet
            ]
            parts.append(pd.DataFrame(zeros, columns=frame.columns, dtype=object))
    kept = [part for part in parts if not part.empty]
    return pd.concat(kept, ignore_index=True) if kept else frame.iloc[0:0]


def share_changes(frame: pd.DataFrame, latest: Mapping[str, int]) -> pd.DataFrame:
    """Days on which shares issued differ from the last known count, in date order."""
    known = dict(latest)
    records = []
    for row in frame.sort_values(["date", "stock_id"]).to_dict("records"):
        code, count = str(row["stock_id"]), row["shares"]
        if count is None or known.get(code) == count:
            continue
        known[code] = count
        records.append(
            {
                "stock_id": code,
                "effective_date": row["date"],
                "shares": count,
                "reason": ShareChangeReason.UNKNOWN.value,
            }
        )
    return pd.DataFrame(records, columns=SHARE_COLUMNS, dtype=object)
