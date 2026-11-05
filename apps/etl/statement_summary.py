"""MOPS market-wide statement summaries turned into quarterly financial_statement rows (SPEC §4)."""

import re
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from decimal import Decimal
from html.parser import HTMLParser
from typing import Any

from apps.etl.deadlines import statement_deadline
from apps.etl.errors import SchemaChangedError
from apps.etl.filings import quarter_end
from apps.market.models import FinType

INCOME, BALANCE, CASH_FLOW = "t163sb04", "t163sb05", "t163sb20"
REPORTS = (INCOME, BALANCE, CASH_FLOW)
MARKET_LABELS = {"sii": "上市", "otc": "上櫃"}
MARKETS = tuple(MARKET_LABELS)
TYPE_MISMATCH_CHECK = "statement_type_mismatch"
QUARTERS = (1, 2, 3, 4)
SEASON_LABELS = {1: "第一季", 2: "第二季", 3: "第三季", 4: "第四季"}
NO_FILINGS = "查詢無資料"
THOUSANDS_NOTE = "單位：新台幣仟元"
CODE_HEADER = "公司代號"
TITLE = re.compile(r"<h2>(.*?)</h2>", re.S)
NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
COMPANY_CODE = re.compile(r"^[0-9A-Z]{4,6}$")
MISSING = {"", "--"}
THOUSAND = 1000
EPS_PLACES = Decimal("0.0001")
EPS_CENTS = 2
STEADY_SHARES = 0.05
RELIABLE_EPS = 0.10

REVENUE = ("營業收入", "淨收益", "收入")
GROSS_PROFIT = ("營業毛利（毛損）",)
OPERATING_INCOME = ("營業利益（損失）", "營業利益")
PARENT_NET_INCOME = ("淨利（淨損）歸屬於母公司業主", "淨利（損）歸屬於母公司業主")
NET_INCOME = ("本期淨利（淨損）", "本期稅後淨利（淨損）")
EPS = ("基本每股盈餘（元）", "基本每股盈餘")
ASSETS = ("資產總計", "資產總額")
LIABILITIES = ("負債總計", "負債總額")
PARENT_EQUITY = ("歸屬於母公司業主權益合計", "歸屬於母公司業主之權益合計")
EQUITY = ("權益總計", "權益總額")
NONCONTROLLING = ("非控制權益",)
OPERATING_CF = ("營業活動之淨現金流入（流出）",)

VALUE_COLUMNS = [
    "eps",
    "revenue",
    "gross_margin",
    "op_margin",
    "net_income",
    "roe",
    "debt_ratio",
    "operating_cf",
]

Row = Mapping[str, float | None]
Summary = Mapping[str, Mapping[str, Row]]


class _Cells(HTMLParser):
    """Rows of cell text, each tagged with its table; a cell ends at either </td> or </th>."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[int, list[str]]] = []
        self._tables: list[int] = []
        self._opened = 0
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._opened += 1
            self._tables.append(self._opened)
        elif tag == "tr":
            self.rows.append((self._tables[-1] if self._tables else 0, []))
        elif tag in ("td", "th") and self.rows:
            self._close()
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self._close()
        elif tag == "table" and self._tables:
            self._tables.pop()

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def _close(self) -> None:
        if self._cell is not None:
            self.rows[-1][1].append("".join(self._cell).strip())
            self._cell = None


def parse_summary(page: str, market: str, season: int) -> dict[str, dict[str, float | None]]:
    """Each company's named values in thousands of TWD; no filings yet is an empty answer."""
    titles = [re.sub(r"\s+", "", title) for title in TITLE.findall(page)]
    if not titles:
        if NO_FILINGS in page:
            return {}
        raise SchemaChangedError(f"MOPS {market} season {season} summary has no title")
    expected = f"{MARKET_LABELS[market]}公司{SEASON_LABELS[season]}資料"
    if titles[0] != expected or THOUSANDS_NOTE not in page:
        raise SchemaChangedError(f"MOPS summary {titles[0]!r} is not {expected!r} in thousands")
    parser = _Cells()
    parser.feed(page)
    companies: dict[str, dict[str, float | None]] = {}
    header: list[str] | None = None
    table = -1
    for where, cells in parser.rows:
        if cells and cells[0] == CODE_HEADER:
            header, table = cells, where
            continue
        if not cells:
            continue
        if header is None or where != table:
            if COMPANY_CODE.match(cells[0]):
                raise SchemaChangedError(f"MOPS {market} lists {cells[0]} outside a report table")
            continue
        if len(cells) != len(header):
            raise SchemaChangedError(f"MOPS {market} row {cells[:2]} has {len(cells)} cells")
        code = cells[0]
        if code in companies:
            raise SchemaChangedError(f"MOPS {market} lists {code} twice")
        pairs = zip(header[2:], cells[2:], strict=True)
        companies[code] = {name: _number(value, code) for name, value in pairs}
    if not companies:
        raise SchemaChangedError(f"MOPS {market} season {season} summary lists no company")
    return companies


def _number(text: str, code: str) -> float | None:
    cleaned = text.replace(",", "").strip()
    if cleaned in MISSING:
        return None
    if not NUMBER.match(cleaned):
        raise SchemaChangedError(f"MOPS {code} has a non-numeric value {text!r}")
    return float(cleaned)


def open_periods(day: date, days_after: int) -> list[str]:
    """Quarters being filed: from the day after quarter end to days past the latest deadline."""
    candidates = [f"{year}-Q{quarter}" for year in (day.year - 1, day.year) for quarter in QUARTERS]
    return [
        period
        for period in candidates
        if quarter_end(period) < day <= statement_deadline(period) + timedelta(days=days_after)
    ]


def previous_period(period: str) -> str:
    year, quarter = int(period[:4]), int(period[-1])
    return f"{year - 1}-Q4" if quarter == 1 else f"{year}-Q{quarter - 1}"


def incomplete(current: Summary) -> list[str]:
    """Companies missing from some of the quarter's summaries, as while a filing uploads."""
    listed = [set(current[report]) for report in REPORTS]
    return sorted(set[str]().union(*listed) - listed[0].intersection(*listed[1:]))


def statement_rows(current: Summary, previous: Summary, season: int) -> list[dict[str, Any]]:
    """Single-quarter rows of companies found in all three current summaries."""
    rows = []
    complete = current[BALANCE].keys() & current[CASH_FLOW].keys()
    for code, income in sorted(current[INCOME].items()):
        if code not in complete:
            continue
        before = previous[INCOME].get(code)
        consolidated = _first(income, PARENT_NET_INCOME) is not None
        same = (
            before is not None and (_first(before, PARENT_NET_INCOME) is not None) == consolidated
        )
        rows.append(
            {
                "stock_id": code,
                "fin_type": (FinType.CONSOLIDATED if consolidated else FinType.INDIVIDUAL).value,
                **_values(code, current, previous, season, consolidated, same),
            }
        )
    return rows


def _values(
    code: str, current: Summary, previous: Summary, season: int, consolidated: bool, same: bool
) -> dict[str, Any]:
    flows = season == 1 or same
    net_names = PARENT_NET_INCOME if consolidated else NET_INCOME

    def quarter(report: str, names: Iterable[str]) -> float | None:
        now = _first(current[report].get(code, {}), names)
        if season == 1:
            return now
        earlier = _first(previous[report].get(code, {}), names) if same else None
        return None if now is None or earlier is None else now - earlier

    revenue, net_income = quarter(INCOME, REVENUE), quarter(INCOME, net_names)
    eps = None
    if flows and net_income is not None:
        before = previous[INCOME].get(code, {})
        eps = _quarter_eps(
            season,
            net_income,
            (_first(current[INCOME][code], net_names), _first(current[INCOME][code], EPS)),
            (_first(before, net_names), _first(before, EPS)),
        )
    sheet = current[BALANCE].get(code, {})
    equity, equity_before = _parent_equity(sheet), _parent_equity(previous[BALANCE].get(code, {}))
    roe = None
    if same and net_income is not None and equity is not None and equity_before is not None:
        roe = net_income / ((equity + equity_before) / 2) if equity + equity_before > 0 else None
    return {
        "eps": None if eps is None else Decimal(str(round(eps, EPS_CENTS))).quantize(EPS_PLACES),
        "revenue": _money(revenue),
        "gross_margin": _share(quarter(INCOME, GROSS_PROFIT), revenue),
        "op_margin": _share(quarter(INCOME, OPERATING_INCOME), revenue),
        "net_income": _money(net_income),
        "roe": roe,
        "debt_ratio": _share(_first(sheet, LIABILITIES), _first(sheet, ASSETS)),
        "operating_cf": _money(quarter(CASH_FLOW, OPERATING_CF)),
    }


def _quarter_eps(
    season: int,
    net_income: float,
    now: tuple[float | None, float | None],
    before: tuple[float | None, float | None],
) -> float | None:
    """Difference of year-to-date EPS unless implied shares moved; then net income at YTD shares."""
    (net_now, eps_now), (net_before, eps_before) = now, before
    if eps_now is None:
        return None
    if season == 1:
        return eps_now
    reliable = abs(eps_now) >= RELIABLE_EPS
    if eps_before is None:
        return net_income * eps_now / net_now if net_now and reliable else None
    if net_now and net_before and reliable and abs(eps_before) >= RELIABLE_EPS:
        shares_now, shares_before = net_now / eps_now, net_before / eps_before
        if abs(shares_now / shares_before - 1) > STEADY_SHARES:
            return net_income * eps_now / net_now
    return eps_now - eps_before


def _parent_equity(sheet: Row) -> float | None:
    parent = _first(sheet, PARENT_EQUITY)
    if parent is not None:
        return parent
    equity = _first(sheet, EQUITY)
    return None if equity is None else equity - (_first(sheet, NONCONTROLLING) or 0.0)


def _first(row: Row, names: Iterable[str]) -> float | None:
    return next((row[name] for name in names if row.get(name) is not None), None)


def _share(part: float | None, whole: float | None) -> float | None:
    return None if part is None or whole is None or whole <= 0 else part / whole


def _money(thousands: float | None) -> int | None:
    return None if thousands is None else round(thousands * THOUSAND)
