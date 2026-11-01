"""TWSE-specific tables and checks on top of the shared exchange table parsing (SPEC §8.6)."""

from datetime import date
from typing import Any

import pandas as pd

from apps.etl.adapters.exchange_tables import (
    DailyTable,
    find_table,
    is_fraction,
    is_percent,
    is_positive,
    lots_in_shares,
    net_shares,
    percent,
    ratio,
    shares,
    yield_fraction,
)
from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.exchange_rows import parse_count, parse_price
from apps.etl.roc import parse_roc_date
from apps.market.models import DailyValuation, IndexId, SharesOutstanding

MARGIN_SUMMARY_TITLE = "信用交易統計"
MARGIN_LOT_ROWS = ("融資(交易單位)", "融券(交易單位)")
MONTHLY_UNIT_HINT = "單位：元、股"
MONTHLY_FIELDS = ("日期", "成交股數", "成交金額", "成交筆數", "發行量加權股價指數", "漲跌點數")
INDEX_COLUMNS = ["index_id", "date", "close", "volume"]


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


def check_margin_lots(payload: dict[str, Any]) -> None:
    """MI_MARGN counts stock margin in 交易單位 (1,000 shares); its summary table says so."""
    summary = find_table(payload, MARGIN_SUMMARY_TITLE)
    labels = {str(row[0]) for row in summary.get("data") or [] if row}
    if not set(MARGIN_LOT_ROWS) <= labels:
        raise SchemaChangedError(f"TWSE MI_MARGN no longer counts in 交易單位: {sorted(labels)}")


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
