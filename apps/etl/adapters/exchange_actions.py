"""The exchanges' ex-rights results and restart reference price tables, parsed into actions."""

from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal
from typing import Any, NamedTuple

from apps.etl.adapters.corporate_actions import (
    CAPITAL_REDUCTION,
    EX_RIGHTS,
    PAR_VALUE_CHANGE,
    Action,
    in_span,
    price_factor,
    stock_value,
)
from apps.etl.adapters.dispositions import list_rows
from apps.etl.errors import SchemaChangedError
from apps.etl.exchange_rows import parse_price, stock_code
from apps.etl.roc import parse_roc_day

Span = tuple[date, date]
ZERO = Decimal(0)
CASH_ONLY = "息"


class RestartTable(NamedTuple):
    """A table of days trading restarts at a reference price after a reduction or par change."""

    label: str
    path: str
    raw: str
    day: str
    code: str
    before: str
    after: str


class ExRight(NamedTuple):
    """A TWT49U row; 權值+息值 is the cash dividend only when the kind is 息."""

    stock_id: str
    ex_date: date
    factor: float | None
    kind: str
    value: Decimal


TWSE_EX_RIGHTS_FIELDS = (
    "資料日期",
    "股票代號",
    "除權息前收盤價",
    "除權息參考價",
    "權值+息值",
    "權/息",
)
DETAIL_CASH = "(每股配發現金股利)除息"
DETAIL_BONUS = "A. 按普通股股東持股比例每千股無償配股"
TWSE_DETAIL_FIELDS = ("股票代號", DETAIL_CASH, DETAIL_BONUS)
PER_SHARE = "元／股"
SHARES = "股"
TPEX_EX_RIGHTS_FIELDS = (
    "除權息日期",
    "代號",
    "除權息前收盤價",
    "除權息參考價",
    "現金股利",
    "每仟股無償配股",
)
TWSE_RESTARTS = (
    RestartTable(
        "TWSE TWTAUU",
        "/rwd/zh/reducation/TWTAUU",
        CAPITAL_REDUCTION,
        "恢復買賣日期",
        "股票代號",
        "停止買賣前收盤價格",
        "恢復買賣參考價",
    ),
    RestartTable(
        "TWSE TWTB8U",
        "/rwd/zh/change/TWTB8U",
        PAR_VALUE_CHANGE,
        "恢復買賣日期",
        "股票代號",
        "停止買賣前收盤價格",
        "恢復買賣參考價",
    ),
)
TPEX_RESTARTS = (
    RestartTable(
        "TPEx revivt",
        "/www/zh-tw/bulletin/revivt",
        CAPITAL_REDUCTION,
        "恢復買賣日期",
        "股票代號",
        "最後交易日之收盤價格",
        "減資恢復買賣開始日參考價格",
    ),
    RestartTable(
        "TPEx pvChgRslt",
        "/www/zh-tw/bulletin/pvChgRslt",
        PAR_VALUE_CHANGE,
        "恢復買賣日期",
        "證券代號",
        "最後交易日之收盤價格",
        "恢復買賣開始參考價",
    ),
)


def twse_ex_rights(body: Mapping[str, Any], span: Span) -> list[ExRight]:
    """TWT49U rows in the span; the detail page splits any row that is not cash only."""
    label = "TWSE TWT49U"
    found = []
    for record in list_rows(body, label, TWSE_EX_RIGHTS_FIELDS):
        code = stock_code(record["股票代號"])
        day = in_span(parse_roc_day(record["資料日期"]), span, label)
        where = f"{label} {code} {day}"
        found.append(
            ExRight(
                code,
                day,
                price_factor(
                    parse_price(record["除權息參考價"]),
                    parse_price(record["除權息前收盤價"]),
                    where,
                ),
                str(record["權/息"]).strip(),
                _amount(record["權值+息值"], where),
            )
        )
    return found


def twse_action(row: ExRight, detail: tuple[Decimal, Decimal] | None) -> Action:
    """A cash-only row pays its 權值+息值; any other takes cash and bonus shares from the detail."""
    if row.kind == CASH_ONLY:
        return Action(row.stock_id, row.ex_date, row.value, ZERO, row.factor, EX_RIGHTS)
    if detail is None:
        raise SchemaChangedError(f"TWSE TWT49U {row.stock_id} {row.ex_date} needs its detail")
    cash, per_thousand = detail
    return Action(row.stock_id, row.ex_date, cash, stock_value(per_thousand), row.factor, EX_RIGHTS)


def twse_detail(body: Mapping[str, Any], code: str) -> tuple[Decimal, Decimal]:
    """Cash per share and bonus shares per thousand held, from a TWT49UDetail page."""
    label = f"TWSE TWT49UDetail {code}"
    records = list_rows(body, label, TWSE_DETAIL_FIELDS)
    if len(records) != 1 or stock_code(records[0]["股票代號"]) != code:
        raise SchemaChangedError(f"{label} should list {code} once")
    record = records[0]
    return (
        _amount(_without_unit(record[DETAIL_CASH], PER_SHARE, label), label),
        _amount(_without_unit(record[DETAIL_BONUS], SHARES, label), label),
    )


def tpex_ex_rights(body: Mapping[str, Any], span: Span) -> list[Action]:
    """exDailyQ rows of the span with their cash dividend and bonus shares."""
    label = "TPEx exDailyQ"
    actions = []
    for record in list_rows(_first_table(body, label), label, TPEX_EX_RIGHTS_FIELDS):
        code = stock_code(record["代號"])
        day = in_span(parse_roc_day(record["除權息日期"]), span, label)
        where = f"{label} {code} {day}"
        actions.append(
            Action(
                code,
                day,
                _amount(record["現金股利"], where),
                stock_value(_amount(record["每仟股無償配股"], where)),
                price_factor(
                    parse_price(record["除權息參考價"]),
                    parse_price(record["除權息前收盤價"]),
                    where,
                ),
                EX_RIGHTS,
            )
        )
    return actions


def restart_actions(body: Mapping[str, Any], table: RestartTable, span: Span) -> list[Action]:
    """Factor-only actions: restart reference price over the last close before the halt."""
    actions = []
    for record in list_rows(body, table.label, (table.day, table.code, table.before, table.after)):
        code = stock_code(record[table.code])
        day = in_span(parse_roc_day(record[table.day]), span, table.label)
        factor = price_factor(
            parse_price(record[table.after]),
            parse_price(record[table.before]),
            f"{table.label} {code} {day}",
        )
        actions.append(Action(code, day, ZERO, ZERO, factor, table.raw))
    return actions


def tpex_table(body: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    return _first_table(body, label)


def _first_table(body: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    tables = body.get("tables")
    if not isinstance(tables, Sequence) or len(tables) != 1 or not isinstance(tables[0], Mapping):
        raise SchemaChangedError(f"{label} should answer one table")
    return tables[0]


def _without_unit(text: Any, unit: str, where: str) -> str:
    """'1.5 元／股' read as '1.5'; a figure printed without its unit is refused."""
    cleaned = str(text).strip()
    if not cleaned.endswith(unit):
        raise SchemaChangedError(f"{where} printed {text!r} without {unit}")
    return cleaned.removesuffix(unit).strip()


def _amount(text: Any, where: str) -> Decimal:
    """A non-negative amount; a dash or blank counts as none paid."""
    value = parse_price(text)
    if value is None:
        return ZERO
    if value < 0:
        raise SchemaChangedError(f"{where} has a negative amount {text!r}")
    return value
