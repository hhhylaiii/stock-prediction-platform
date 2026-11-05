"""TWSE e-document listings: when each financial report was filed (SPEC §3 MOPS, §8.1)."""

import re
from collections.abc import Iterator
from datetime import date, datetime, time
from html.parser import HTMLParser
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from django.conf import settings

from apps.etl.adapters.base import BaseAdapter
from apps.etl.client import SourceClient
from apps.etl.errors import SchemaChangedError
from apps.etl.filings import FILING_COLUMNS, Filing, apply_statement_filings
from apps.etl.models import EtlRun, RawMops
from apps.etl.ratelimit import RateLimiter
from apps.market.calendar import TradingDays
from apps.market.models import FinancialStatement, FinType

MOPS_MIN_INTERVAL = 3.0
STATEMENT_FILING = "statement_filing"
LISTING_PATH = "/server-java/t57sb01"
REPORT_KIND = "A"
PAGE_ENCODING = "cp950"
NO_REPORTS = "查無所需資料"
CODE, YEAR, DETAIL, UPLOADED, CORRECTION = (
    "證券代號",
    "資料年度",
    "資料細節說明",
    "上傳日期",
    "財務報告更(補)正",
)
HEADER = (CODE, YEAR, DETAIL, UPLOADED, CORRECTION)
ENGLISH = "英文版"
GROUP_REPORT = "合併"
NOT_CORRECTED = "無"
ROC_OFFSET = 1911
SEASONS = {"一": 1, "二": 2, "三": 3, "四": 4}
SEASON = re.compile(r"^(\d{2,3})\s*年\s*第([一二三四])季$")
UPLOAD = re.compile(r"^(\d{2,3})/(\d{2})/(\d{2}) (\d{2}):(\d{2}):(\d{2})$")
TAIPEI = ZoneInfo("Asia/Taipei")


class MopsClient(SourceClient):
    source = "mops"
    label = "doc.twse.com.tw"
    raw_model = RawMops
    min_interval = MOPS_MIN_INTERVAL

    @classmethod
    def build(cls, limiter: RateLimiter, session: requests.Session | None = None) -> "MopsClient":
        return cls(limiter, str(settings.MOPS_DOC_BASE_URL), float(settings.MOPS_TIMEOUT), session)

    def listing(self, code: str, roc_year: int) -> tuple[dict[str, str], str]:
        """The year's financial report files of one company, as the Big5 page decodes."""
        params = {
            "step": "1",
            "colorchg": "1",
            "co_id": code,
            "year": str(roc_year),
            "seamon": "",
            "mtype": REPORT_KIND,
        }
        body = self.get_content(f"{self.base_url}{LISTING_PATH}", params)
        return params, body.decode(PAGE_ENCODING, errors="replace")


class _Rows(HTMLParser):
    """Text of every table cell, row by row."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th") and self.rows:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None:
            self.rows[-1].append("".join(self._cell).strip())
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def parse_listing(page: str, code: str, roc_year: int) -> list[Filing]:
    """Chinese-language reports of the company's year; no reports is a valid empty answer."""
    if NO_REPORTS in page:
        return []
    parser = _Rows()
    parser.feed(page)
    starts = [i for i, row in enumerate(parser.rows) if all(name in row for name in HEADER)]
    if not starts:
        raise SchemaChangedError(f"doc.twse {code} {roc_year} has no report table")
    header = parser.rows[starts[0]]
    position = {name: header.index(name) for name in HEADER}
    filings = []
    for row in parser.rows[starts[0] + 1 :]:
        if len(row) != len(header):
            continue
        cells = {name: row[index] for name, index in position.items()}
        if ENGLISH not in cells[DETAIL]:
            filings.append(_filing(cells, code, roc_year))
    return filings


def _filing(cells: dict[str, str], code: str, roc_year: int) -> Filing:
    season = SEASON.match(cells[YEAR])
    if cells[CODE] != code or season is None or int(season[1]) != roc_year:
        raise SchemaChangedError(
            f"doc.twse {code} {roc_year} listed {cells[CODE]!r} {cells[YEAR]!r}"
        )
    uploaded = UPLOAD.match(cells[UPLOADED])
    if uploaded is None:
        raise SchemaChangedError(f"doc.twse {code} {roc_year} upload time {cells[UPLOADED]!r}")
    year, month, day, hour, minute, second = (int(part) for part in uploaded.groups())
    fin_type = FinType.CONSOLIDATED if GROUP_REPORT in cells[DETAIL] else FinType.INDIVIDUAL
    return Filing(
        stock_id=code,
        period=f"{roc_year + ROC_OFFSET}-Q{SEASONS[season[2]]}",
        fin_type=fin_type.value,
        uploaded_at=datetime(year + ROC_OFFSET, month, day, hour, minute, second, tzinfo=TAIPEI),
        corrected=cells[CORRECTION] != NOT_CORRECTED,
    )


class MopsAdapter(BaseAdapter):
    source = "mops"
    raw_model = RawMops
    datasets = frozenset({STATEMENT_FILING})
    per_stock = True

    def __init__(
        self,
        calendar: TradingDays,
        limiter: RateLimiter,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(calendar, limiter)
        self.client = MopsClient.build(limiter, session)

    def run(self, dataset: str, start: date, end: date, stock_id: str | None = None) -> EtlRun:
        raise ValueError(f"{self.source}/{dataset} goes stock by stock; use manage.py backfill")

    def units(self, dataset: str, start: date, end: date) -> Iterator[tuple[date, date]]:
        yield start, end

    def fetch(
        self, dataset: str, start: date, end: date, stock_id: str | None = None
    ) -> pd.DataFrame:
        self.require_dataset(dataset)
        if stock_id is None:
            raise ValueError(f"{self.source}/{dataset} goes stock by stock; give a stock")
        filings: list[Filing] = []
        for year in _statement_years(stock_id, start, end):
            params, page = self.client.listing(stock_id, year - ROC_OFFSET)
            self.save_raw(dataset, params, {"html": page})
            filings.extend(parse_listing(page, stock_id, year - ROC_OFFSET))
        return pd.DataFrame(filings, columns=FILING_COLUMNS)

    def normalize(self, dataset: str, raw_df: pd.DataFrame) -> pd.DataFrame:
        """The first upload of each report type is the announcement; a correction is noted."""
        self.require_dataset(dataset)
        if raw_df.empty:
            return raw_df
        grouped = raw_df.groupby(["stock_id", "period", "fin_type"], as_index=False, sort=True)
        return grouped.agg(uploaded_at=("uploaded_at", "min"), corrected=("corrected", "any"))

    def upsert(self, dataset: str, frame: pd.DataFrame) -> int:
        self.require_dataset(dataset)
        if frame.empty:
            return 0
        cutoff = time.fromisoformat(settings.FILING_VISIBLE_BEFORE)
        return apply_statement_filings(frame, self.calendar, cutoff)


def _statement_years(stock_id: str, start: date, end: date) -> list[int]:
    """Years with stored statements; a report of year Y is listed under Y even if filed in Y+1."""
    periods = FinancialStatement.objects.filter(
        stock_id=stock_id, period__gte=f"{start.year}", period__lt=f"{end.year + 1}"
    ).values_list("period", flat=True)
    return sorted({int(period[:4]) for period in periods})
