"""Daily whole-market tables of TWSE and TPEx mapped onto standard rows (SPEC §4, §5)."""

import logging
import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from datetime import date
from typing import Any, NamedTuple

import pandas as pd
from django.db import models

from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.exchange_rows import (
    parse_count,
    parse_percent,
    parse_price,
    parse_signed,
    stock_code,
)
from apps.etl.upsert import upsert
from apps.market.adjust import refresh_where_later_events
from apps.market.models import DailyChip, DailyPrice, IndexDaily, ShareChangeReason

logger = logging.getLogger(__name__)
NO_DATA_STAT = "沒有符合條件的資料"
AFTER_LATEST_STAT = "查詢日期大於可查詢最大日期"
FOREIGN_CURRENCY_CODE_SUFFIXES = frozenset("KMSC")
SHARES_PER_LOT = 1000
TOTAL_ROW_NAME = "合計"
SHARE_COLUMNS = ["stock_id", "effective_date", "shares", "reason"]
BLANK_MARKERS = frozenset({"-", "--", ""})
PERCENT = 100.0
PRICE_COLUMNS = ("open", "high", "low", "close")
COUNT_COLUMNS = ("volume", "turnover")
Converter = Callable[[Sequence[Any]], int | float | None]
AnswerCheck = Callable[[dict[str, Any], date], None]


def is_foreign_currency_etf(code: str) -> bool:
    """On both exchanges an ETF code whose sixth character is K, M, S or C is foreign-currency."""
    return len(code) == 6 and code.startswith("00") and code[5] in FOREIGN_CURRENCY_CODE_SUFFIXES


def priced_codes(day: date, quotes_kept_from: str | None) -> set[str]:
    """Local-currency codes with a daily_price bar, for days before any quote page was kept."""
    if quotes_kept_from is not None and f"{day:%Y%m%d}" >= quotes_kept_from:
        return set()
    codes = DailyPrice.objects.filter(date=day).values_list("stock_id", flat=True)
    found = {code for code in codes if not is_foreign_currency_etf(code)}
    if found:
        logger.info(
            "%s: before the first kept quote page %s, %d priced codes stand in",
            day,
            quotes_kept_from,
            len(found),
        )
    return found


def require_kept(frame: pd.DataFrame, kept: pd.DataFrame, label: str) -> None:
    """A day whose table lists codes but keeps none has not had this market's quotes or prices."""
    empty = sorted(set(frame["date"]) - set(kept["date"]))
    if empty:
        raise SourceNotReadyError(f"{label} {empty[0]}: none of its codes is quoted or priced")


def find_table(payload: dict[str, Any], title_part: str) -> dict[str, Any]:
    for table in payload.get("tables", []):
        if title_part in (table.get("title") or ""):
            return dict(table)
    raise SchemaChangedError(f"no table titled *{title_part}* in the response")


def check_answered(payload: dict[str, Any], day: date, exchange: str, ok: str) -> None:
    """The payload must be a successful answer for the requested day itself."""
    stat = str(payload.get("stat", ""))
    if stat != ok:
        raise SchemaChangedError(f"{exchange} answered with unexpected stat {stat!r}")
    requested = day.strftime("%Y%m%d")
    answered = str(payload.get("date", ""))
    if answered == requested:
        return
    if len(answered) == len(requested) and answered.isdigit() and answered < requested:
        raise SourceNotReadyError(f"{exchange} still serves {answered} when asked for {day}")
    raise SchemaChangedError(f"{exchange} answered for {answered!r} instead of {requested}")


def row_of_day(by_day: Mapping[date, list[Any]], day: date, label: str) -> list[Any]:
    """The day's row of a monthly table; a day after the last row is not published yet."""
    if any(listed.replace(day=1) != day.replace(day=1) for listed in by_day):
        raise SchemaChangedError(f"{label} answered {day:%Y-%m} with days of another month")
    if day in by_day:
        return by_day[day]
    if not by_day or day > max(by_day):
        raise SourceNotReadyError(f"{label} has no row for {day} yet")
    raise SchemaChangedError(f"{label} skips trading day {day}; the calendar may be wrong")


def check_published(payload: dict[str, Any], label: str, day: date) -> None:
    """Before publishing, TWSE may say 'no data' and either exchange may send only empty tables."""
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
    """A published ratio such as P/E; TWSE prints '-' or a JSON null where none is published."""
    (value,) = values
    text = "" if value is None else str(value).replace(",", "").strip()
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
    exchange: str = "TWSE"
    positional: tuple[tuple[str, str], ...] = ()
    consistent: Callable[[Mapping[Hashable, Any]], bool] | None = None
    locate: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    @property
    def where(self) -> str:
        return f"{self.exchange} {self.label}"


def labelled_fields(body: Mapping[str, Any], table: DailyTable) -> list[str]:
    """Field names prefixed by their column group, so 融資今日餘額 and 融券今日餘額 differ."""
    fields = [str(name).strip() for name in body.get("fields") or []]
    if table.positional:
        if fields != [source for source, _ in table.positional]:
            raise SchemaChangedError(f"{table.where} fields changed: {fields}")
        return [label for _, label in table.positional]
    groups = body.get("groups") or []
    if groups:
        titles = [
            str(group.get("title") or "")
            for group in groups
            for _ in range(int(group.get("span") or 0))
        ]
        if len(titles) != len(fields):
            raise SchemaChangedError(f"{table.where} column groups no longer match its fields")
        fields = [f"{title}{name}" for title, name in zip(titles, fields, strict=True)]
    if len(set(fields)) != len(fields):
        raise SchemaChangedError(f"{table.where} has repeated columns: {fields}")
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
            raise SourceNotReadyError(f"{table.where} for {day} lists {table.pending_title} only")
    if table.locate is not None:
        body = table.locate(payload)
    else:
        body = payload if table.title is None else find_table(payload, table.title)
    if table.hints is not None and body.get("hints") != table.hints:
        raise SchemaChangedError(f"{table.where} units changed: {body.get('hints')!r}")
    if table.check is not None:
        table.check(payload)
    fields = labelled_fields(body, table)
    needed = {table.code, table.name, *(name for names in table.columns.values() for name in names)}
    missing = sorted(needed - set(fields))
    if missing:
        raise SchemaChangedError(f"{table.where} lost columns {missing}")
    try:
        frame = pd.DataFrame(body["data"], columns=fields)
    except (KeyError, ValueError) as exc:
        raise SchemaChangedError(f"{table.where} table is malformed: {exc}") from None
    if frame.empty:
        raise SourceNotReadyError(f"{table.where} for {day} has no rows yet")
    check_complete(payload, body, table, len(frame))
    return frame.assign(date=day)


def check_complete(
    payload: Mapping[str, Any], body: Mapping[str, Any], table: DailyTable, rows: int
) -> None:
    """The row count and the selectType echoed back must match what was asked for."""
    total = body.get("total", body.get("totalCount"))
    if total is not None and total != rows:
        raise SchemaChangedError(f"{table.where} sent {rows} of {total} rows")
    answered = payload.get("selectType")
    asked = table.params.get("selectType")
    if answered is not None and asked is not None and answered != asked:
        raise SchemaChangedError(f"{table.where} answered selectType {answered!r}, not {asked!r}")


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
            raise SchemaChangedError(f"{table.where} repeated {code} on {row['date']}")
        seen.add(key)
        if table.consistent is not None and not table.consistent(row):
            raise SchemaChangedError(
                f"{table.where} {code} on {row['date']}: columns do not add up"
            )
        values = {
            column: table.converters.get(column, table.convert)([row[name] for name in names])
            for column, names in table.columns.items()
        }
        records.append({"stock_id": code, "date": row["date"], **_validated(values, table, key)})
    if not records:
        raise SourceNotReadyError(f"{table.where} has no stock rows yet")
    return pd.DataFrame(records, columns=["stock_id", "date", *table.columns], dtype=object)


def _validated(
    values: dict[str, int | float | None], table: DailyTable, key: tuple[str, date]
) -> dict[str, int | float | None]:
    kept = dict(values)
    for column, accepts in table.valid.items():
        value = values[column]
        if value is not None and not accepts(value):
            logger.warning(
                "%s %s %s: %s %s out of range, dropped",
                table.where,
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


def parsed_quotes(frame: pd.DataFrame) -> pd.DataFrame:
    """Price and count text of daily_price columns as numbers; no-trade marks become None."""
    converters: dict[str, Callable[[str], object]] = {
        **dict.fromkeys(PRICE_COLUMNS, parse_price),
        **dict.fromkeys(COUNT_COLUMNS, parse_count),
    }
    return frame.assign(**{col: frame[col].map(convert) for col, convert in converters.items()})


def upsert_daily_prices(frame: pd.DataFrame) -> int:
    rows = upsert(
        DailyPrice,
        frame,
        unique_fields=("stock_id", "date"),
        update_fields=(*PRICE_COLUMNS, *COUNT_COLUMNS),
    )
    if not frame.empty:
        refresh_where_later_events(frame["stock_id"].unique(), frame["date"].min())
    return rows


def upsert_index_rows(frame: pd.DataFrame) -> int:
    return upsert(
        IndexDaily, frame, unique_fields=("index_id", "date"), update_fields=("close", "volume")
    )
