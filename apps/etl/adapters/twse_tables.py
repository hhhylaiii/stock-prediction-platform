"""TWSE whole-market daily tables mapped onto standard rows (SPEC §4, §5)."""

import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from typing import Any, NamedTuple

import pandas as pd
from django.db import models

from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.exchange_rows import parse_count, parse_percent, parse_price, parse_signed, stock_code
from apps.etl.roc import parse_roc_date
from apps.market.models import (
    DailyChip,
    DailyValuation,
    IndexId,
    ShareChangeReason,
    SharesOutstanding,
)

logger = logging.getLogger(__name__)
NO_DATA_STAT = "沒有符合條件的資料"
AFTER_LATEST_STAT = "查詢日期大於可查詢最大日期"
FOREIGN_CURRENCY_CODE_SUFFIXES = frozenset("KMSC")
SHARES_PER_LOT = 1000
TOTAL_ROW_NAME = "合計"
MARGIN_SUMMARY_TITLE = "信用交易統計"
MARGIN_LOT_ROWS = ("融資(交易單位)", "融券(交易單位)")
MONTHLY_UNIT_HINT = "單位：元、股"
MONTHLY_FIELDS = ("日期", "成交股數", "成交金額", "成交筆數", "發行量加權股價指數", "漲跌點數")
INDEX_COLUMNS = ["index_id", "date", "close", "volume"]
SHARE_COLUMNS = ["stock_id", "effective_date", "shares", "reason"]
BLANK_MARKERS = frozenset({"-", "--", ""})
PERCENT = 100.0
Converter = Callable[[Sequence[Any]], int | float | None]


def is_foreign_currency_etf(code: str) -> bool:
    """TWSE: an ETF code whose sixth character is K, M, S or C trades in a foreign currency."""
    return len(code) == 6 and code.startswith("00") and code[5] in FOREIGN_CURRENCY_CODE_SUFFIXES


def find_table(payload: dict[str, Any], title_part: str) -> dict[str, Any]:
    for table in payload.get("tables", []):
        if title_part in (table.get("title") or ""):
            return dict(table)
    raise SchemaChangedError(f"no table titled *{title_part}* in TWSE response")


def check_answer(payload: dict[str, Any], day: date) -> None:
    """The payload must be a successful answer for the requested day itself."""
    stat = str(payload.get("stat", ""))
    if stat != "OK":
        raise SchemaChangedError(f"TWSE answered with unexpected stat {stat!r}")
    requested = day.strftime("%Y%m%d")
    answered = str(payload.get("date", ""))
    if answered == requested:
        return
    if len(answered) == len(requested) and answered.isdigit() and answered < requested:
        raise SourceNotReadyError(f"TWSE still serves {answered} when asked for {day}")
    raise SchemaChangedError(f"TWSE answered for {answered!r} instead of {requested}")


def check_published(payload: dict[str, Any], label: str, day: date) -> None:
    """TWSE says 'no data', 'after the latest date' or sends only empty tables before publishing."""
    stat = str(payload.get("stat", ""))
    if NO_DATA_STAT in stat or AFTER_LATEST_STAT in stat:
        raise SourceNotReadyError(f"TWSE {label} for {day} not available: {stat}")
    bodies = payload.get("tables", [payload])
    if stat == "OK" and isinstance(bodies, list) and bodies and all(map(_blank, bodies)):
        raise SourceNotReadyError(f"TWSE {label} for {day} has no rows yet")


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
        return float(text)
    except ValueError:
        raise SchemaChangedError(f"unexpected ratio value {text!r}") from None


def yield_fraction(values: Sequence[Any]) -> float | None:
    published = ratio(values)
    return None if published is None else published / PERCENT


def is_percent(value: float) -> bool:
    return 0 <= value <= PERCENT


def is_fraction(value: float) -> bool:
    return 0 <= value < 1


def is_positive(value: float) -> bool:
    return value > 0


def check_margin_lots(payload: dict[str, Any]) -> None:
    """MI_MARGN counts stock margin in 交易單位 (1,000 shares); its summary table says so."""
    summary = find_table(payload, MARGIN_SUMMARY_TITLE)
    labels = {str(row[0]) for row in summary.get("data") or [] if row}
    if not set(MARGIN_LOT_ROWS) <= labels:
        raise SchemaChangedError(f"TWSE MI_MARGN no longer counts in 交易單位: {sorted(labels)}")


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


TABLES: dict[str, DailyTable] = {
    "institutional": DailyTable(
        label="T86",
        path="/rwd/zh/fund/T86",
        params={"selectType": "ALLBUT0999"},
        title=None,
        hints="單位：股",
        code="證券代號",
        name="證券名稱",
        columns={
            "fi_net": ("外陸資買賣超股數(不含外資自營商)", "外資自營商買賣超股數"),
            "it_net": ("投信買賣超股數",),
            "dealer_self_net": ("自營商買賣超股數(自行買賣)",),
            "dealer_hedge_net": ("自營商買賣超股數(避險)",),
        },
        convert=net_shares,
    ),
    "margin": DailyTable(
        label="MI_MARGN",
        path="/rwd/zh/marginTrading/MI_MARGN",
        params={"selectType": "ALL"},
        title="融資融券彙總",
        hints=None,
        code="股票代號",
        name="股票名稱",
        columns={
            "margin_balance": ("融資今日餘額",),
            "margin_limit": ("融資次一營業日限額",),
            "short_balance": ("融券今日餘額",),
        },
        convert=lots_in_shares,
        check=check_margin_lots,
    ),
    "sbl": DailyTable(
        label="TWT93U",
        path="/rwd/zh/marginTrading/TWT93U",
        params={},
        title=None,
        hints="單位：股",
        code="股票代號",
        name="股票名稱",
        columns={"sbl_short_balance": ("借券賣出當日餘額",)},
        convert=shares,
    ),
    "foreign_holding": DailyTable(
        label="MI_QFIIS",
        path="/rwd/zh/fund/MI_QFIIS",
        params={"selectType": "ALLBUT0999"},
        title=None,
        hints="單位:股",
        code="證券代號",
        name="證券名稱",
        columns={"fi_holding_pct": ("全體外資及陸資持股比率",)},
        convert=percent,
        valid={"fi_holding_pct": is_percent},
    ),
    "day_trade": DailyTable(
        label="TWTB4U",
        path="/rwd/zh/dayTrading/TWTB4U",
        params={"selectType": "All"},
        title="當日沖銷交易標的及成交量值",
        hints="單位：元、股",
        code="證券代號",
        name="證券名稱",
        columns={"day_trade_volume": ("當日沖銷交易成交股數",)},
        convert=shares,
        pending_title="當日沖銷交易標的",
    ),
    "daily_valuation": DailyTable(
        label="BWIBBU_d",
        path="/rwd/zh/afterTrading/BWIBBU_d",
        params={"selectType": "ALL"},
        title=None,
        hints=None,
        code="證券代號",
        name="證券名稱",
        columns={"pe": ("本益比",), "pb": ("股價淨值比",), "dividend_yield": ("殖利率(%)",)},
        convert=ratio,
        model=DailyValuation,
        valid={"pe": is_positive, "pb": is_positive, "dividend_yield": is_fraction},
        converters={"dividend_yield": yield_fraction},
    ),
    "shares_outstanding": DailyTable(
        label="MI_QFIIS",
        path="/rwd/zh/fund/MI_QFIIS",
        params={"selectType": "ALLBUT0999"},
        title=None,
        hints="單位:股",
        code="證券代號",
        name="證券名稱",
        columns={"shares": ("發行股數",)},
        convert=shares,
        model=SharesOutstanding,
        valid={"shares": is_positive},
        keys=("stock_id", "effective_date"),
    ),
}


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


def source_rows(payload: dict[str, Any], day: date, table: DailyTable) -> pd.DataFrame:
    """The table's rows for the day under labelled field names, after layout checks."""
    check_answer(payload, day)
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
    total = body.get("total")
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


def taiex_rows(payload: dict[str, Any], day: date) -> pd.DataFrame:
    """The FMTQIK row of the day; a day after the last row is not published yet."""
    if str(payload.get("stat", "")) != "OK":
        raise SchemaChangedError(f"TWSE FMTQIK answered with stat {payload.get('stat')!r}")
    if payload.get("hints") != MONTHLY_UNIT_HINT:
        raise SchemaChangedError(f"TWSE FMTQIK units changed: {payload.get('hints')!r}")
    if tuple(payload.get("fields") or ()) != MONTHLY_FIELDS:
        raise SchemaChangedError(f"TWSE FMTQIK fields changed: {payload.get('fields')}")
    rows = payload.get("data")
    if not isinstance(rows, list) or not all(
        isinstance(row, list) and len(row) == len(MONTHLY_FIELDS) for row in rows
    ):
        raise SchemaChangedError("TWSE FMTQIK rows are malformed")
    by_day = {parse_roc_date(str(row[0])): row for row in rows}
    if any(listed.replace(day=1) != day.replace(day=1) for listed in by_day):
        raise SchemaChangedError(f"TWSE FMTQIK answered {day:%Y-%m} with days of another month")
    if day in by_day:
        return pd.DataFrame([by_day[day]], columns=list(MONTHLY_FIELDS)).assign(date=day)
    if not by_day or day > max(by_day):
        raise SourceNotReadyError(f"TWSE FMTQIK has no row for {day} yet")
    raise SchemaChangedError(f"TWSE FMTQIK skips trading day {day}; the calendar may be wrong")


def taiex_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in raw_df.to_dict("records"):
        close = parse_price(row["發行量加權股價指數"])
        if close is None:
            raise SchemaChangedError(f"TWSE FMTQIK has no TAIEX close on {row['date']}")
        records.append(
            {
                "index_id": IndexId.TAIEX.value,
                "date": row["date"],
                "close": close,
                "volume": parse_count(row["成交股數"]),
            }
        )
    return pd.DataFrame(records, columns=INDEX_COLUMNS, dtype=object)
