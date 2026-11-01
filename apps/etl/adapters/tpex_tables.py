"""TPEx-specific tables and checks on top of the shared exchange table parsing (SPEC §8.6)."""

import math
import re
from collections.abc import Callable, Hashable, Mapping, Sequence
from datetime import date
from typing import Any

import pandas as pd

from apps.etl.adapters.dispositions import announced_events, list_rows, period_events
from apps.etl.adapters.exchange_tables import (
    BLANK_MARKERS,
    PERCENT,
    DailyTable,
    check_answered,
    find_table,
    is_foreign_currency_etf,
    is_fraction,
    is_percent,
    is_positive,
    lots_in_shares,
    net_shares,
    ratio,
    row_of_day,
    shares,
)
from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.exchange_rows import (
    parse_count,
    parse_percent,
    parse_price,
    parse_signed,
    stock_code,
)
from apps.market.models import DailyValuation, IndexId, SharesOutstanding

TPEX = "TPEx"
OK_STAT = "ok"
QUOTED_TITLE = "上櫃股票每日收盤行情(不含定價)"
MANAGED_TITLE = "管理股票"
QUOTE_TITLES = ("上櫃股票行情", MANAGED_TITLE)
CODE_FIELD = "代號"
NAME_FIELD = "名稱"
QUOTE_COLUMNS = {
    "開盤": "open",
    "最高": "high",
    "最低": "low",
    "收盤": "close",
    "成交股數": "volume",
    "成交金額(元)": "turnover",
}
INDEX_TITLE = "櫃買指數"
INDEX_FIELDS = ("日期", "開市", "最高", "最低", "收市", "漲/跌")
INDEX_COLUMNS = ["index_id", "date", "close", "volume"]
MARGIN_TITLE = "上櫃股票融資融券餘額"
MARGIN_LOT_FIELDS = ("前資餘額(張)", "前券餘額(張)")
TRADE_FIELDS = ("買進股數", "賣出股數", "買賣超股數")
FI = "外資及陸資(不含外資自營商)"
FI_DEALER = "外資自營商"
FI_ALL = "外資及陸資合計"
IT = "投信"
DEALER_SELF = "自營商(自行買賣)"
DEALER_HEDGE = "自營商(避險)"
DEALER_ALL = "自營商合計"
INSTITUTIONS = (FI, FI_DEALER, FI_ALL, IT, DEALER_SELF, DEALER_HEDGE, DEALER_ALL)
INSTITUTIONAL_TOTAL = "三大法人買賣超股數合計"
SBL_SOURCE_FIELDS = (
    "股票代號",
    "股票名稱",
    "前日餘額",
    "賣出",
    "買進",
    "現券",
    "當日餘額",
    "限額",
    "前日餘額",
    "當日賣出",
    "當日還券",
    "當日調整數額",
    "當日餘額",
    "次一營業日可借券賣出限額",
    "備註",
)
SBL_GROUPS = ("", "", *("融券",) * 6, *("借券賣出",) * 6, "")
SECURITY_CODE = "證券代號"
ATTENTION_TITLE = "上櫃公布注意有價證券"
ATTENTION_FIELDS = (SECURITY_CODE, "公告日期")
DISPOSAL_TITLE = "上櫃處置有價證券"
DISPOSAL_FIELDS = (SECURITY_CODE, "公布日期", "處置起訖時間")
ALTERED_TITLE = "變更交易、分盤交易、管理股票與停止交易資訊"
ALTERED_FLAGS = ("變更交易", "屬管理股票")
FLAG = "Ｙ"
NOT_AVAILABLE = "N/A"
PERCENT_SIGN = "%"
QFII_TITLE = "櫃檯買賣股票僑外資及陸資持股比例排行表"
DAY_TRADE_SUMMARY_TITLE = "現股當沖交易統計資訊"
DAY_TRADE_TOTAL = "當日沖銷交易總成交股數"
DAY_TRADE_VOLUME = "當日沖銷交易成交股數"
LISTED_COUNT = re.compile(r"總累計次數:\s*(\d+)")


def check_answer(payload: dict[str, Any], day: date) -> None:
    check_answered(payload, day, TPEX, OK_STAT)


def _rows(table: Mapping[str, Any], label: str) -> list[list[Any]]:
    """The table's data, which must hold as many rows as its totalCount says."""
    rows = table.get("data")
    if not isinstance(rows, list) or not all(isinstance(row, list) for row in rows):
        raise SchemaChangedError(f"TPEx {label} rows are malformed")
    if table.get("totalCount") != len(rows):
        raise SchemaChangedError(f"TPEx {label} sent {len(rows)} of {table.get('totalCount')} rows")
    return rows


def quoted_codes(payload: dict[str, Any], day: date) -> set[str]:
    """Local-currency codes of the 上櫃股票 (EW) closing quotes; warrants are not in this list."""
    return set(quoted_names(payload, day))


def quoted_names(payload: dict[str, Any], day: date) -> dict[str, str]:
    """Local-currency codes of the EW closing quotes with the short names they are quoted under."""
    check_answer(payload, day)
    table = find_table(payload, QUOTED_TITLE)
    fields = [str(name).strip() for name in table.get("fields") or []]
    if fields[:2] != [CODE_FIELD, NAME_FIELD]:
        raise SchemaChangedError(f"TPEx otc quotes fields changed: {fields}")
    rows = _rows(table, "otc quotes")
    if not rows:
        raise SchemaChangedError(f"TPEx otc quotes for {day} have no rows")
    names = {stock_code(row[0]): str(row[1]).strip() for row in rows}
    return {code: name for code, name in names.items() if not is_foreign_currency_etf(code)}


def quote_rows(otc: dict[str, Any], quotes: dict[str, Any], day: date) -> pd.DataFrame:
    """dailyQuotes rows, whose volume includes odd lots, for the codes the EW list quotes."""
    codes = quoted_codes(otc, day)
    check_answer(quotes, day)
    found: dict[str, dict[str, Any]] = {}
    for title in QUOTE_TITLES:
        table = find_table(quotes, title)
        fields = [str(name).strip() for name in table.get("fields") or []]
        missing = sorted({CODE_FIELD, *QUOTE_COLUMNS} - set(fields))
        if missing:
            raise SchemaChangedError(f"TPEx {title} lost columns {missing}")
        for row in _rows(table, title):
            if len(row) != len(fields):
                raise SchemaChangedError(f"TPEx {title} rows do not match its fields")
            record = dict(zip(fields, row, strict=True))
            code = str(record[CODE_FIELD]).strip()
            if title == MANAGED_TITLE and code not in codes:
                raise SchemaChangedError(
                    f"TPEx managed stock {code} on {day} is not in the EW list"
                )
            if code in codes and code in found:
                raise SchemaChangedError(f"TPEx daily quotes repeat {code} on {day}")
            found[code] = record
    absent = sorted(codes - set(found))
    if absent:
        raise SchemaChangedError(f"TPEx daily quotes on {day} lack quoted codes {absent}")
    records = [
        {
            "stock_id": code,
            "date": day,
            **{column: found[code][source] for source, column in QUOTE_COLUMNS.items()},
        }
        for code in sorted(codes)
    ]
    return pd.DataFrame(records, columns=["stock_id", "date", *QUOTE_COLUMNS.values()])


def index_rows(payload: dict[str, Any], day: date) -> pd.DataFrame:
    """The 櫃買指數 row of the day from the month's table."""
    check_answered(payload, day.replace(day=1), TPEX, OK_STAT)
    table = find_table(payload, INDEX_TITLE)
    fields = tuple(str(name).strip() for name in table.get("fields") or ())
    if fields != INDEX_FIELDS:
        raise SchemaChangedError(f"TPEx index fields changed: {fields}")
    rows = _rows(table, "index")
    if not all(len(row) == len(INDEX_FIELDS) for row in rows):
        raise SchemaChangedError("TPEx index rows are malformed")
    by_day = {_gregorian(str(row[0])): row for row in rows}
    row = row_of_day(by_day, day, "TPEx index")
    return pd.DataFrame([row], columns=list(INDEX_FIELDS)).assign(date=day)


def _gregorian(text: str) -> date:
    try:
        return date.fromisoformat(text.strip().replace("/", "-"))
    except ValueError:
        raise SchemaChangedError(f"unexpected TPEx index date {text!r}") from None


def index_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in raw_df.to_dict("records"):
        close = parse_price(row["收市"])
        if close is None:
            raise SchemaChangedError(f"TPEx index has no close on {row['date']}")
        records.append(
            {"index_id": IndexId.TPEX.value, "date": row["date"], "close": close, "volume": None}
        )
    return pd.DataFrame(records, columns=INDEX_COLUMNS, dtype=object)


def attention_events(payload: dict[str, Any], sessions: Sequence[date]) -> list[dict[str, Any]]:
    table = find_table(payload, ATTENTION_TITLE)
    records = list_rows(table, "TPEx attention", ATTENTION_FIELDS)
    events = announced_events(records, ATTENTION_FIELDS, "TPEx attention", sessions)
    check_listed_count(table, records, "TPEx attention")
    return events


def disposal_events(payload: dict[str, Any], window: tuple[date, date]) -> list[dict[str, Any]]:
    table = find_table(payload, DISPOSAL_TITLE)
    records = list_rows(table, "TPEx disposal", DISPOSAL_FIELDS)
    events = period_events(records, DISPOSAL_FIELDS, "TPEx disposal", window)
    check_listed_count(table, records, "TPEx disposal")
    return events


def check_listed_count(
    table: Mapping[str, Any], records: Sequence[Mapping[str, Any]], label: str
) -> None:
    """TPEx lists give their row count only in the subtitle (總累計次數), placeholders aside."""
    listed = sum(1 for record in records if str(record[SECURITY_CODE]).strip())
    subtitle = str(table.get("subtitle") or "")
    match = LISTED_COUNT.search(subtitle)
    if match is None:
        if listed:
            raise SchemaChangedError(f"{label} subtitle no longer gives 總累計次數: {subtitle!r}")
        return
    if int(match.group(1)) != listed:
        raise SchemaChangedError(f"{label} sent {listed} of {match.group(1)} rows")


def full_delivery_codes(payload: dict[str, Any], day: date) -> list[str]:
    """Codes under 變更交易 or listed as 管理股票 that day; suspended-only rows are left out."""
    check_answer(payload, day)
    table = find_table(payload, ALTERED_TITLE)
    codes = []
    for record in list_rows(table, "TPEx chtm", (SECURITY_CODE, *ALTERED_FLAGS)):
        code = stock_code(record[SECURITY_CODE])
        if is_foreign_currency_etf(code):
            continue
        flags = {str(record[name]).strip() for name in ALTERED_FLAGS}
        if not flags <= {"", FLAG}:
            raise SchemaChangedError(f"TPEx chtm {code} has an unexpected flag {sorted(flags)}")
        if FLAG in flags:
            codes.append(code)
    return codes


def check_margin_lots(payload: dict[str, Any]) -> None:
    """Margin balances count in 張 (1,000 shares); the previous-balance headers say so."""
    fields = {str(name).strip() for name in find_table(payload, MARGIN_TITLE).get("fields") or []}
    if not set(MARGIN_LOT_FIELDS) <= fields:
        raise SchemaChangedError(f"TPEx margin balances no longer count in 張: {sorted(fields)}")


def institutional_adds_up(row: Mapping[Hashable, Any]) -> bool:
    """Each group nets buys against sells, and the subtotals and total add up."""
    net = {}
    for group in INSTITUTIONS:
        bought, sold, balance = (row[f"{group}{field}"] for field in TRADE_FIELDS)
        net[group] = parse_signed(balance)
        if parse_count(bought) - parse_count(sold) != net[group]:
            return False
    return (
        net[FI_ALL] == net[FI] + net[FI_DEALER]
        and net[DEALER_ALL] == net[DEALER_SELF] + net[DEALER_HEDGE]
        and parse_signed(row[INSTITUTIONAL_TOTAL]) == net[FI_ALL] + net[IT] + net[DEALER_ALL]
    )


def sbl_rolls_forward(row: Mapping[Hashable, Any]) -> bool:
    """Both balances equal the previous one plus the day's additions minus its reductions."""
    short = [parse_count(row[f"融券{field}"]) for field in ("前日餘額", "賣出", "買進", "現券")]
    lent = [parse_count(row[f"借券賣出{field}"]) for field in ("前日餘額", "當日賣出", "當日還券")]
    adjusted = parse_signed(row["借券賣出當日調整數額"])
    short_holds = short[0] + short[1] - short[2] - short[3] == parse_count(row["融券當日餘額"])
    lent_holds = lent[0] + lent[1] - lent[2] + adjusted == parse_count(row["借券賣出當日餘額"])
    return short_holds and lent_holds


def published_ratio(values: Sequence[Any]) -> float | None:
    """A ratio such as P/E; N/A is TPEx's only mark for a missing one."""
    (text,) = (str(value).strip() for value in values)
    if text == NOT_AVAILABLE:
        return None
    if text in BLANK_MARKERS:
        raise SchemaChangedError(f"unexpected ratio value {text!r}")
    return ratio(values)


def published_yield(values: Sequence[Any]) -> float | None:
    """A dividend yield printed in percent, under the same missing-value rule as P/E."""
    published = published_ratio(values)
    return None if published is None else published / PERCENT


def percent_text(values: Sequence[Any]) -> float:
    """A percentage printed with its sign, such as 30.29%."""
    (text,) = (str(value).strip() for value in values)
    if not text.endswith(PERCENT_SIGN):
        raise SchemaChangedError(f"unexpected percent value {text!r}")
    number = parse_percent(text.removesuffix(PERCENT_SIGN))
    if not math.isfinite(number):
        raise SchemaChangedError(f"unexpected percent value {text!r}")
    return number


def check_day_trade_total(payload: dict[str, Any]) -> None:
    """The stock rows must add up to the market total, which is zero before publishing."""
    summary = find_table(payload, DAY_TRADE_SUMMARY_TITLE)
    fields = [str(name).strip() for name in summary.get("fields") or []]
    rows = summary.get("data") or []
    if DAY_TRADE_TOTAL not in fields or len(rows) > 1:
        raise SchemaChangedError(f"TPEx {DAY_TRADE_SUMMARY_TITLE} lost {DAY_TRADE_TOTAL}")
    total = parse_count(rows[0][fields.index(DAY_TRADE_TOTAL)]) if rows else 0
    if total == 0:
        raise SourceNotReadyError(f"TPEx {DAY_TRADE_TOTAL} is 0; the day is not published yet")
    stocks = table_with(SECURITY_CODE)(payload)
    stock_fields = [str(name).strip() for name in stocks.get("fields") or []]
    if DAY_TRADE_VOLUME not in stock_fields:
        raise SchemaChangedError(f"TPEx 當日沖銷 rows lost {DAY_TRADE_VOLUME}")
    column = stock_fields.index(DAY_TRADE_VOLUME)
    listed = sum(parse_count(row[column]) for row in stocks.get("data") or [])
    if listed != total:
        raise SchemaChangedError(f"TPEx 當日沖銷 rows add up to {listed}, not the total {total}")


def table_with(field: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Locate the response table listing the given field, for tables without a title."""

    def locate(payload: dict[str, Any]) -> dict[str, Any]:
        for table in payload.get("tables", []):
            if field in (str(name).strip() for name in table.get("fields") or []):
                return dict(table)
        raise SchemaChangedError(f"no TPEx table lists {field}")

    return locate


def _same(*names: str) -> tuple[tuple[str, str], ...]:
    return tuple((name, name) for name in names)


TABLES: dict[str, DailyTable] = {
    "institutional": DailyTable(
        label="insti/dailyTrade",
        path="/www/zh-tw/insti/dailyTrade",
        params={"type": "Daily", "sect": "EW"},
        title="三大法人買賣明細資訊",
        hints=None,
        code=CODE_FIELD,
        name="名稱",
        columns={
            "fi_net": (f"{FI}買賣超股數", f"{FI_DEALER}買賣超股數"),
            "it_net": (f"{IT}買賣超股數",),
            "dealer_self_net": (f"{DEALER_SELF}買賣超股數",),
            "dealer_hedge_net": (f"{DEALER_HEDGE}買賣超股數",),
        },
        convert=net_shares,
        exchange=TPEX,
        positional=(
            *_same(CODE_FIELD, "名稱"),
            *((field, f"{group}{field}") for group in INSTITUTIONS for field in TRADE_FIELDS),
            *_same(INSTITUTIONAL_TOTAL),
        ),
        consistent=institutional_adds_up,
    ),
    "margin": DailyTable(
        label="margin/balance",
        path="/www/zh-tw/margin/balance",
        params={},
        title=MARGIN_TITLE,
        hints=None,
        code=CODE_FIELD,
        name="名稱",
        columns={
            "margin_balance": ("資餘額",),
            "margin_limit": ("資限額",),
            "short_balance": ("券餘額",),
        },
        convert=lots_in_shares,
        check=check_margin_lots,
        exchange=TPEX,
    ),
    "sbl": DailyTable(
        label="margin/sbl",
        path="/www/zh-tw/margin/sbl",
        params={},
        title="信用額度總量管制餘額表",
        hints=None,
        code="股票代號",
        name="股票名稱",
        columns={"sbl_short_balance": ("借券賣出當日餘額",)},
        convert=shares,
        exchange=TPEX,
        positional=tuple(
            (field, f"{group}{field}")
            for group, field in zip(SBL_GROUPS, SBL_SOURCE_FIELDS, strict=True)
        ),
        consistent=sbl_rolls_forward,
    ),
    "foreign_holding": DailyTable(
        label="insti/qfii",
        path="/www/zh-tw/insti/qfii",
        params={},
        title=QFII_TITLE,
        hints=None,
        code=CODE_FIELD,
        name="名稱",
        columns={"fi_holding_pct": ("僑外資及陸資持股比率(E=C/A)",)},
        convert=percent_text,
        valid={"fi_holding_pct": is_percent},
        exchange=TPEX,
    ),
    "day_trade": DailyTable(
        label="intraday/stat",
        path="/www/zh-tw/intraday/stat",
        params={"type": "Daily"},
        title=None,
        hints=None,
        code=SECURITY_CODE,
        name="證券名稱",
        columns={"day_trade_volume": (DAY_TRADE_VOLUME,)},
        convert=shares,
        check=check_day_trade_total,
        exchange=TPEX,
        locate=table_with(SECURITY_CODE),
    ),
    "daily_valuation": DailyTable(
        label="afterTrading/peQryDate",
        path="/www/zh-tw/afterTrading/peQryDate",
        params={},
        title=None,
        hints=None,
        code="股票代號",
        name="公司名稱",
        columns={"pe": ("本益比",), "pb": ("股價淨值比",), "dividend_yield": ("殖利率(%)",)},
        convert=published_ratio,
        model=DailyValuation,
        valid={"pe": is_positive, "pb": is_positive, "dividend_yield": is_fraction},
        converters={"dividend_yield": published_yield},
        exchange=TPEX,
        locate=table_with("股票代號"),
    ),
    "shares_outstanding": DailyTable(
        label="insti/qfii",
        path="/www/zh-tw/insti/qfii",
        params={},
        title=QFII_TITLE,
        hints=None,
        code=CODE_FIELD,
        name="名稱",
        columns={"shares": ("發行股數(A)",)},
        convert=shares,
        model=SharesOutstanding,
        valid={"shares": is_positive},
        exchange=TPEX,
        keys=("stock_id", "effective_date"),
    ),
}
