"""Monthly revenue from the daily MOPS scan, inserted once and dated by the scan that saw it."""

import re
from datetime import date, timedelta
from typing import Any

import pandas as pd
from django.db import transaction

from apps.etl.deadlines import revenue_deadline
from apps.etl.errors import SchemaChangedError
from apps.etl.html_cells import TableCells
from apps.etl.models import DataQualityLog
from apps.etl.quality import replace_day_log
from apps.etl.statement_summary import COMPANY_CODE, MARKET_LABELS
from apps.market.models import MonthlyRevenue

ORIGINS = {0: "國內", 1: "國外"}
TITLE = re.compile(r"<b>\s*(\S+?)公司(\d{2,3})年(\d{1,2})月份\(累計與當月\)營業收入統計表\s*</b>")
NO_FILINGS = "查無資料"
THOUSANDS_NOTE = "單位：千元"
CODE_HEADER = "公司代號"
THIS_MONTH, LAST_MONTH, LAST_YEAR = "當月營收", "上月營收", "去年當月營收"
FIGURES = (THIS_MONTH, LAST_MONTH, LAST_YEAR)
AMOUNT = re.compile(r"^-?\d+$")
GENERATED_AT = re.compile(r"<!--\d{2}:\d{2}:\d{2}-->")
THOUSAND = 1000
ROC_OFFSET = 1911
MONTHS = 12
REVENUE_COLUMNS = ["stock_id", "period", "revenue", "yoy", "mom"]
REVENUE_MISMATCH_CHECK = "revenue_source_mismatch"
GROWTH_TOLERANCE = 1e-6
Figures = tuple[int, int, int]


def open_months(day: date, days_after: int) -> list[str]:
    """Months being filed: from the 1st of the next month to days past the latest deadline."""
    index = day.year * MONTHS + day.month - 1
    months = []
    for back in range(1, MONTHS + 1):
        period = _period(index - back)
        if revenue_deadline(period) + timedelta(days=days_after) < day:
            break
        months.append(period)
    return sorted(months)


def parse_revenue_page(page: str, market: str, origin: int, period: str) -> dict[str, Figures]:
    """This month, last month and the same month last year, in thousands, for each filer."""
    label = MARKET_LABELS[market]
    expected = (label, str(int(period[:4]) - ROC_OFFSET), str(int(period[5:])))
    titles = TITLE.findall(page)
    if titles != [expected]:
        raise SchemaChangedError(f"MOPS t21sc03 {market} {period} has the title {titles}")
    parser = TableCells()
    parser.feed(page)
    parser.close()
    headers = [cells for _, cells in parser.rows if cells and cells[0] == CODE_HEADER]
    if not headers:
        if NO_FILINGS in page:
            return {}
        raise SchemaChangedError(f"MOPS t21sc03 {market} {period} has no revenue table")
    for note in (THOUSANDS_NOTE, f"全部{ORIGINS[origin]}{label}公司合計"):
        if note not in page:
            raise SchemaChangedError(f"MOPS t21sc03 {market} {period} lacks {note}")
    return _companies(parser.rows, f"{market} {period}")


def revenue_rows(companies: dict[str, Figures], period: str) -> list[dict[str, Any]]:
    """Revenue in TWD; growth against the page's own earlier months, as FinMind's is defined."""
    return [
        {
            "stock_id": code,
            "period": period,
            "revenue": now * THOUSAND,
            "yoy": _growth(now, last_year),
            "mom": _growth(now, last_month),
        }
        for code, (now, last_month, last_year) in sorted(companies.items())
    ]


def insert_new_revenues(frame: pd.DataFrame, day: date, session: date) -> int:
    """First sight wins; a stored month keeps its figures and date, and differences are logged."""
    stored = _stored(frame)
    new: list[MonthlyRevenue] = []
    differing: list[dict[str, Any]] = []
    for raw in frame.to_dict("records"):
        record = {str(name): value for name, value in raw.items()}
        row = stored.get((record["stock_id"], record["period"]))
        if row is None:
            new.append(MonthlyRevenue(**record, announce_date=session))
            continue
        fields = _differences(row, record)
        if fields:
            differing.append({"stock_id": row.stock_id, "period": row.period, "fields": fields})
    detail = {"revenues": differing} if differing else None
    with transaction.atomic():
        MonthlyRevenue.objects.bulk_create(new)
        replace_day_log(day, REVENUE_MISMATCH_CHECK, DataQualityLog.Severity.INFO, detail)
    return len(new)


def without_generation_time(page: str) -> str:
    """The page as it would be hashed: the hourly regeneration stamp is not content."""
    return GENERATED_AT.sub("", page)


def _companies(rows: list[tuple[int, list[str]]], where: str) -> dict[str, Figures]:
    found: dict[str, Figures] = {}
    header: list[str] | None = None
    table = -1
    for at, cells in rows:
        if cells and cells[0] == CODE_HEADER:
            missing = [name for name in FIGURES if name not in cells]
            if missing:
                raise SchemaChangedError(f"MOPS t21sc03 {where} header lacks {missing[0]}")
            header, table = cells, at
            continue
        if not cells or not COMPANY_CODE.match(cells[0]):
            continue
        code = cells[0]
        if header is None or at != table:
            raise SchemaChangedError(f"MOPS t21sc03 {where} lists {code} outside a revenue table")
        if len(cells) != len(header) + 1:
            raise SchemaChangedError(f"MOPS t21sc03 {where} {code} has {len(cells)} cells")
        if code in found:
            raise SchemaChangedError(f"MOPS t21sc03 {where} lists {code} twice")
        now, last_month, last_year = (
            _amount(cells[header.index(name)], code, where) for name in FIGURES
        )
        found[code] = (now, last_month, last_year)
    return found


def _amount(text: str, code: str, where: str) -> int:
    plain = text.replace(",", "")
    if not AMOUNT.match(plain):
        raise SchemaChangedError(f"MOPS t21sc03 {where} {code} has the amount {text!r}")
    return int(plain)


def _growth(now: int, earlier: int) -> float | None:
    return now / earlier - 1 if earlier > 0 else None


def _period(index: int) -> str:
    return f"{index // MONTHS:04d}-{index % MONTHS + 1:02d}"


def _stored(frame: pd.DataFrame) -> dict[tuple[str, str], MonthlyRevenue]:
    if frame.empty:
        return {}
    rows = MonthlyRevenue.objects.filter(
        stock_id__in=frame["stock_id"].unique().tolist(),
        period__in=frame["period"].unique().tolist(),
    )
    return {(row.stock_id, row.period): row for row in rows}


def _differences(row: MonthlyRevenue, record: dict[str, Any]) -> dict[str, list[Any]]:
    """Figures the page gives differently from the stored row (SPEC §8.5)."""
    fields: dict[str, list[Any]] = {}
    if row.revenue != record["revenue"]:
        fields["revenue"] = [row.revenue, record["revenue"]]
    for name in ("yoy", "mom"):
        stored, found = getattr(row, name), record[name]
        if stored is None and found is None:
            continue
        if stored is None or found is None or abs(stored - found) > GROWTH_TOLERANCE:
            fields[name] = [stored, found]
    return fields
