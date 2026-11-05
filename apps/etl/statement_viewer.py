"""MOPS XBRL viewer (t164sb01): one company's quarter under the summaries' names (plan K17b)."""

import re
from datetime import date

from apps.etl.errors import SchemaChangedError
from apps.etl.filings import quarter_end
from apps.etl.html_cells import TableCells
from apps.etl.statement_summary import (
    ASSETS,
    BALANCE,
    CASH_FLOW,
    EPS,
    EQUITY,
    INCOME,
    LIABILITIES,
    NET_INCOME,
    OPERATING_CF,
    PARENT_EQUITY,
    PARENT_NET_INCOME,
)

TITLES = {BALANCE: "資產負債表", INCOME: "綜合損益表", CASH_FLOW: "現金流量表"}
NOT_FILED = "檔案不存在"
THOUSANDS = "單位：新臺幣仟元"
TITLE_HEADER = "會計項目"
CODE_HEADER = "代號Code"
OLD_CODE = re.compile(r"name\s*=\s*'CO_ID'\s+id\s*=\s*'CO_ID'\s+value\s*=\s*'([0-9A-Z]*)'")
NEW_CODE = re.compile(r"<title>\s*([0-9A-Z]+)\s+\d{4}Q\d\s")
DAY = re.compile(r"^(\d{4})年(\d{1,2})月(\d{1,2})日")
YEAR_TO_DATE = re.compile(r"^(\d{4})年0?1月0?1日至(?:\d{4}年)?(\d{1,2})月(\d{1,2})日")
WHOLE_YEAR = re.compile(r"^(\d{4})年度")
NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
MISSING = {"", "-", "--"}
NAMES = {
    INCOME: {
        "營業收入合計": "營業收入",
        "淨收益": "淨收益",
        "收益合計": "收入",
        "營業毛利（毛損）": "營業毛利（毛損）",
        "營業利益（損失）": "營業利益（損失）",
        "營業利益": "營業利益",
        "本期淨利（淨損）": "本期淨利（淨損）",
        "本期稅後淨利（淨損）": "本期稅後淨利（淨損）",
        "母公司業主（淨利／損）": "淨利（淨損）歸屬於母公司業主",
        "本期稅後淨利（損）歸屬於母公司業主": "淨利（損）歸屬於母公司業主",
        "基本每股盈餘合計": "基本每股盈餘（元）",
    },
    BALANCE: {
        "資產總計": "資產總計",
        "資產總額": "資產總額",
        "負債總計": "負債總計",
        "負債總額": "負債總額",
        "歸屬於母公司業主之權益合計": "歸屬於母公司業主之權益合計",
        "權益總計": "權益總計",
        "權益總額": "權益總額",
        "非控制權益合計": "非控制權益",
    },
    CASH_FLOW: {
        "營業活動之淨現金流入（流出）": "營業活動之淨現金流入（流出）",
        "營運產生之現金流入（流出）": "營運產生之現金流入（流出）",
    },
}
REQUIRED = {
    INCOME: (EPS, NET_INCOME + PARENT_NET_INCOME),
    BALANCE: (ASSETS, LIABILITIES, EQUITY + PARENT_EQUITY),
    CASH_FLOW: (OPERATING_CF,),
}
Table = list[list[str]]


def parse_viewer(page: str, code: str, period: str) -> dict[str, dict[str, float | None]] | None:
    """Year-to-date thousands of TWD (EPS in TWD) per statement; a report never filed is None."""
    if NOT_FILED in page and TITLE_HEADER not in page and CODE_HEADER not in page:
        return None
    shown = OLD_CODE.search(page) or NEW_CODE.search(page)
    if shown is None or shown[1] != code:
        raise SchemaChangedError(f"MOPS viewer page is {shown[1] if shown else '?'}, is not {code}")
    if THOUSANDS not in page:
        raise SchemaChangedError(f"MOPS viewer {code} {period} is not in thousands of TWD")
    tables = _tables(page)
    return {report: _figures(tables, report, code, period) for report in TITLES}


def _tables(page: str) -> dict[int, Table]:
    parser = TableCells()
    parser.feed(page)
    tables: dict[int, Table] = {}
    for where, cells in parser.rows:
        tables.setdefault(where, []).append(cells)
    return tables


def _figures(
    tables: dict[int, Table], report: str, code: str, period: str
) -> dict[str, float | None]:
    """The first filled row of each known title, read in the period's own column."""
    rows = _statement(tables, report, code)
    header = next(row for row in rows if row and row[0] in (TITLE_HEADER, CODE_HEADER))
    label = 1 if header[0] == CODE_HEADER else 0
    column = label + 1 + _column(header[label + 1 :], report, period, code)
    figures: dict[str, float | None] = {}
    for row in rows:
        if len(row) <= column:
            continue
        name = NAMES[report].get(_title(row[label]))
        if name is not None and figures.get(name) is None:
            figures[name] = _number(row[column], code)
    for names in REQUIRED[report]:
        if all(figures.get(name) is None for name in names):
            raise SchemaChangedError(f"MOPS viewer {code} {period} lacks {names[0]}")
    return figures


def _title(text: str) -> str:
    """The Chinese title; the 2019 layout adds the English one after it."""
    cleaned = text.strip(" \u3000")
    end = next((i for i, ch in enumerate(cleaned) if ch.isascii() and ch.isalpha()), len(cleaned))
    return cleaned[:end].strip(" \u3000")


def _statement(tables: dict[int, Table], report: str, code: str) -> Table:
    for rows in tables.values():
        if any(row and row[0].startswith(TITLES[report]) for row in rows[:3]):
            return rows
    raise SchemaChangedError(f"MOPS viewer {code} has no {TITLES[report]}")


def _column(headers: list[str], report: str, period: str, code: str) -> int:
    """The balance at quarter end, or the flows from January 1 to it."""
    end = quarter_end(period)
    for index, text in enumerate(headers):
        if report == BALANCE:
            day = DAY.match(text)
            if day and date(int(day[1]), int(day[2]), int(day[3])) == end:
                return index
            continue
        span = YEAR_TO_DATE.match(text)
        if span and (int(span[1]), int(span[2]), int(span[3])) == (end.year, end.month, end.day):
            return index
        year = WHOLE_YEAR.match(text)
        if year and end.month == 12 and int(year[1]) == end.year:
            return index
    raise SchemaChangedError(f"MOPS viewer {code} {TITLES[report]} has no column for {period}")


def _number(text: str, code: str) -> float | None:
    cleaned = text.replace(",", "").strip()
    if cleaned in MISSING:
        return None
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = f"-{cleaned[1:-1]}"
    if not NUMBER.match(cleaned):
        raise SchemaChangedError(f"MOPS viewer {code} has a non-numeric value {text!r}")
    return float(cleaned)
