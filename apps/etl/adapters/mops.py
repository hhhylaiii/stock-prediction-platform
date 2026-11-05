"""MOPS: when each financial report was filed, and the daily scans of new filings (SPEC §3, §5)."""

import logging
import re
from collections.abc import Callable, Iterator, Sequence
from datetime import date, datetime, time
from html.parser import HTMLParser
from typing import NamedTuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from apps.etl.adapters.base import BaseAdapter
from apps.etl.client import SourceClient
from apps.etl.deadlines import (
    revenue_deadline,
    revenue_visible_from,
    statement_deadline,
    visible_from,
)
from apps.etl.errors import SchemaChangedError, SourceError
from apps.etl.filings import FILING_COLUMNS, Filing, apply_statement_filings
from apps.etl.models import EtlRun, RawMops
from apps.etl.mops_history import (
    HISTORIES,
    IFRS_FROM,
    REVENUE_HISTORY,
    STATEMENT_HISTORY,
    insert_missing_history,
    months_between,
    quarters_between,
    unsettled,
)
from apps.etl.ratelimit import RateLimiter, RetryableError
from apps.etl.revenue_scan import (
    ORIGINS,
    REVENUE_COLUMNS,
    Figures,
    insert_new_revenues,
    open_months,
    parse_revenue_page,
    revenue_rows,
    without_generation_time,
)
from apps.etl.statement_scan import SCAN_COLUMNS, insert_new_statements, scan_session
from apps.etl.statement_summary import (
    MARKETS,
    REPORTS,
    VALUE_COLUMNS,
    Summary,
    incomplete,
    open_periods,
    parse_summary,
    previous_period,
    statement_rows,
)
from apps.market.calendar import TradingDays
from apps.market.models import FinancialStatement, FinType

MOPS_MIN_INTERVAL = 5.0
STATEMENT_FILING = "statement_filing"
STATEMENT_SCAN = "statement_scan"
REVENUE_SCAN = "revenue_scan"
SCANS = frozenset({STATEMENT_SCAN, REVENUE_SCAN})
REVENUE_PATH = "/nas/t21"
SUMMARY_PATH = "/mops/web/ajax_"
SUMMARY_ENCODING = "utf-8"
EXPORT_NAME = re.compile(r"name='filename' value='[^']*'")
LISTING_PATH = "/server-java/t57sb01"
REPORT_KIND = "A"
PAGE_ENCODING = "cp950"
NO_REPORTS = "查無所需資料"
TOO_MANY_QUERIES = "查詢過量"
FISCAL_YEAR = re.compile(r"會計期間：\s*([^<\s]+)")
CALENDAR_YEAR = "曆年制"
CODE, YEAR, KIND, DETAIL, UPLOADED, CORRECTION = (
    "證券代號",
    "資料年度",
    "資料類型",
    "資料細節說明",
    "上傳日期",
    "財務報告更(補)正",
)
HEADER = (CODE, YEAR, KIND, DETAIL, UPLOADED, CORRECTION)
FINANCIAL_REPORT = "財務報告書"
ENGLISH = "英文版"
SUPPLEMENTS = frozenset({"關係企業合併報表"})
RESTATED_MARKS = ("(重編)", "重編")
REPORT_TYPES = {
    "IFRSs合併財報": FinType.CONSOLIDATED,
    "母子公司合併報表": FinType.CONSOLIDATED,
    "IFRSs個體財報": FinType.INDIVIDUAL,
    "IFRSs個別財報": FinType.INDIVIDUAL,
    "母公司財報": FinType.INDIVIDUAL,
}
NOT_CORRECTED = "無"
ROC_OFFSET = 1911
SEASONS = {"一": 1, "二": 2, "三": 3, "四": 4}
SEASON = re.compile(r"^(\d{2,3})\s*年\s*第([一二三四])季$")
UPLOAD = re.compile(r"^(\d{2,3})/(\d{2})/(\d{2}) (\d{2}):(\d{2}):(\d{2})$")
TAIPEI = ZoneInfo("Asia/Taipei")
logger = logging.getLogger(__name__)


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
        url = f"{self.base_url}{LISTING_PATH}"
        return params, self.limiter.call(lambda: self._page(url, params))

    def _page(self, url: str, params: dict[str, str]) -> str:
        """A page asking to query less is a failure the limiter backs off from and retries."""
        page = self._receive("GET", url, params=params).content.decode(PAGE_ENCODING, "replace")
        if TOO_MANY_QUERIES in page:
            raise RetryableError(f"{self.label} asked to query less for {params['co_id']}")
        return page


class MopsSummaryClient(SourceClient):
    source = "mops"
    label = "mopsov.twse.com.tw"
    raw_model = RawMops
    min_interval = MOPS_MIN_INTERVAL

    @classmethod
    def build(
        cls, limiter: RateLimiter, session: requests.Session | None = None
    ) -> "MopsSummaryClient":
        return cls(limiter, str(settings.MOPS_BASE_URL), float(settings.MOPS_TIMEOUT), session)

    def summary(
        self, report: str, market: str, roc_year: int, season: int
    ) -> tuple[dict[str, str], str]:
        """One market's year-to-date figures of a quarter for every company that has filed."""
        form = {
            "encodeURIComponent": "1",
            "step": "1",
            "firstin": "1",
            "off": "1",
            "isQuery": "Y",
            "TYPEK": market,
            "year": str(roc_year),
            "season": f"{season:02d}",
        }
        body = self.post_content(f"{self.base_url}{SUMMARY_PATH}{report}", form)
        return {"report": report, **form}, body.decode(SUMMARY_ENCODING, errors="replace")

    def revenue(
        self, market: str, roc_year: int, month: int, origin: int
    ) -> tuple[dict[str, str], str]:
        """One market's revenue page of a month, for domestic (0) or foreign (1) companies."""
        name = f"t21sc03_{roc_year}_{month}_{origin}.html"
        body = self.get_content(f"{self.base_url}{REVENUE_PATH}/{market}/{name}", {})
        params = {"market": market, "year": str(roc_year), "month": str(month)}
        return {**params, "origin": str(origin)}, body.decode(PAGE_ENCODING, errors="replace")


class _Rows(HTMLParser):
    """Text of every table cell, row by row, each row tagged with the table it sits in."""

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
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self._tables:
            self._tables.pop()
        elif tag in ("td", "th") and self._cell is not None:
            self.rows[-1][1].append("".join(self._cell).strip())
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def parse_listing(page: str, code: str, roc_year: int) -> list[Filing]:
    """Chinese-language reports of the company's year; only 查無所需資料 may stand for none."""
    parser = _Rows()
    parser.feed(page)
    starts = [i for i, (_, row) in enumerate(parser.rows) if all(name in row for name in HEADER)]
    if not starts:
        if NO_REPORTS in page:
            return []
        raise SchemaChangedError(f"doc.twse {code} {roc_year} has no report table")
    fiscal = FISCAL_YEAR.search(page)
    if fiscal is None or fiscal[1] != CALENDAR_YEAR:
        shown = fiscal[1] if fiscal else "missing"
        raise SchemaChangedError(f"doc.twse {code} {roc_year} fiscal year is {shown}")
    table, header = parser.rows[starts[0]]
    position = {name: header.index(name) for name in HEADER}
    filings = []
    for where, row in parser.rows[starts[0] + 1 :]:
        if where != table:
            continue
        if len(row) != len(header):
            raise SchemaChangedError(f"doc.twse {code} {roc_year} row has {len(row)} cells")
        cells = {name: row[index] for name, index in position.items()}
        if ENGLISH not in cells[DETAIL] and _report(cells[DETAIL])[0] not in SUPPLEMENTS:
            filings.append(_filing(cells, code, roc_year))
    return filings


def _report(detail: str) -> tuple[str, bool]:
    """The report named by a detail, and whether that upload restates it."""
    mark = next((mark for mark in RESTATED_MARKS if detail.endswith(mark)), "")
    return detail.removesuffix(mark), bool(mark)


def _filing(cells: dict[str, str], code: str, roc_year: int) -> Filing:
    season = SEASON.match(cells[YEAR])
    if cells[CODE] != code or season is None or int(season[1]) != roc_year:
        raise SchemaChangedError(
            f"doc.twse {code} {roc_year} listed {cells[CODE]!r} {cells[YEAR]!r}"
        )
    report, restated = _report(cells[DETAIL])
    fin_type = REPORT_TYPES.get(report)
    if cells[KIND] != FINANCIAL_REPORT or fin_type is None:
        raise SchemaChangedError(
            f"doc.twse {code} {roc_year} has an unknown report {cells[KIND]!r} {cells[DETAIL]!r}"
        )
    return Filing(
        stock_id=code,
        period=f"{roc_year + ROC_OFFSET}-Q{SEASONS[season[2]]}",
        fin_type=fin_type.value,
        uploaded_at=_uploaded_at(cells[UPLOADED], code, roc_year),
        corrected=cells[CORRECTION] != NOT_CORRECTED,
        restated=restated,
    )


def _uploaded_at(text: str, code: str, roc_year: int) -> datetime:
    uploaded = UPLOAD.match(text)
    try:
        if uploaded is None:
            raise ValueError(text)
        year, month, day, hour, minute, second = (int(part) for part in uploaded.groups())
        return datetime(year + ROC_OFFSET, month, day, hour, minute, second, tzinfo=TAIPEI)
    except ValueError as exc:
        raise SchemaChangedError(f"doc.twse {code} {roc_year} upload time {text!r}") from exc


class MopsAdapter(BaseAdapter):
    source = "mops"
    raw_model = RawMops
    datasets = frozenset({STATEMENT_FILING, *SCANS, *HISTORIES})
    per_stock = True

    def __init__(
        self,
        calendar: TradingDays,
        limiter: RateLimiter,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(calendar, limiter)
        self.client = MopsClient.build(limiter, session)
        self.summaries = MopsSummaryClient.build(limiter, session)

    def run(self, dataset: str, start: date, end: date, stock_id: str | None = None) -> EtlRun:
        """Scans read today's pages; filing dates go stock by stock through backfill."""
        self.require_dataset(dataset)
        if dataset == STATEMENT_FILING:
            raise ValueError(f"{self.source}/{dataset} goes stock by stock; use manage.py backfill")
        today = timezone.localdate()
        if dataset in SCANS and (start, end) != (today, today):
            raise ValueError(f"{self.source}/{dataset} sees only today {today}, not {start}..{end}")
        if dataset in HISTORIES:
            days_after = int(
                settings.REVENUE_SCAN_DAYS_AFTER
                if dataset == REVENUE_HISTORY
                else settings.STATEMENT_SCAN_DAYS_AFTER
            )
            pending = unsettled(dataset, start, end, today, days_after)
            if pending:
                raise ValueError(f"{self.source}/{dataset} leaves {pending} to the daily scan")
        return super().run(dataset, start, end, stock_id)

    def units(self, dataset: str, start: date, end: date) -> Iterator[tuple[date, date]]:
        """History covers the periods inside the whole span at once, whatever day they end on."""
        if dataset in HISTORIES:
            yield start, end
        else:
            yield from super().units(dataset, start, end)

    def fetch(
        self, dataset: str, start: date, end: date, stock_id: str | None = None
    ) -> pd.DataFrame:
        self.require_dataset(dataset)
        if dataset in SCANS | HISTORIES:
            if stock_id is not None:
                raise ValueError(f"{self.source}/{dataset} covers the whole market, not {stock_id}")
            if dataset == REVENUE_HISTORY:
                return self._revenue_history(start, end)
            if dataset == STATEMENT_HISTORY:
                return self._statement_history(start, end)
            return self._scan(end) if dataset == STATEMENT_SCAN else self._revenue_scan(end)
        if stock_id is None:
            raise ValueError(f"{self.source}/{dataset} goes stock by stock; give a stock")
        filings: list[Filing] = []
        for year in _statement_years(stock_id, start, end):
            params, page = self.client.listing(stock_id, year - ROC_OFFSET)
            kept = self.save_raw(dataset, params, {"html": page})
            RawMops.objects.filter(pk=kept.pk).update(fetched_at=timezone.now())
            filings.extend(parse_listing(page, stock_id, year - ROC_OFFSET))
        return pd.DataFrame(filings, columns=FILING_COLUMNS)

    def normalize(self, dataset: str, raw_df: pd.DataFrame) -> pd.DataFrame:
        """The first upload of each report type, or its last restatement, is the announcement."""
        self.require_dataset(dataset)
        if raw_df.empty:
            return raw_df
        if dataset in (STATEMENT_SCAN, STATEMENT_HISTORY):
            kept = raw_df[raw_df[VALUE_COLUMNS].notna().any(axis=1)].reset_index(drop=True)
            kept.attrs.update(raw_df.attrs)
            return kept
        if dataset in (REVENUE_SCAN, REVENUE_HISTORY):
            return raw_df
        records = []
        for key, group in raw_df.groupby(["stock_id", "period", "fin_type"], sort=True):
            restated = group.loc[group["restated"], "uploaded_at"]
            uploaded = restated.max() if len(restated) else group["uploaded_at"].min()
            records.append((*key, uploaded, bool(group["corrected"].any()), bool(len(restated))))
        return pd.DataFrame(records, columns=FILING_COLUMNS)

    def upsert(self, dataset: str, frame: pd.DataFrame) -> int:
        self.require_dataset(dataset)
        if dataset in SCANS:
            session = scan_session(timezone.now(), self.calendar)
            insert = insert_new_statements if dataset == STATEMENT_SCAN else insert_new_revenues
            return insert(frame, timezone.localdate(), session)
        if dataset in HISTORIES:
            span = (frame.attrs["start"], frame.attrs["end"])
            waiting = frame.attrs.get("incomplete", {})
            return insert_missing_history(
                dataset, frame, self._conservative(dataset), span, waiting
            )
        if frame.empty:
            return 0
        cutoff = time.fromisoformat(settings.FILING_VISIBLE_BEFORE)
        return apply_statement_filings(frame, self.calendar, cutoff)

    def _scan(self, day: date) -> pd.DataFrame:
        periods = open_periods(day, int(settings.STATEMENT_SCAN_DAYS_AFTER))
        return self._quarters(periods, day, STATEMENT_SCAN)

    def _statement_history(self, start: date, end: date) -> pd.DataFrame:
        frame = self._quarters(
            quarters_between(start, end), timezone.localdate(), STATEMENT_HISTORY
        )
        frame.attrs.update(start=start, end=end)
        return frame

    def _quarters(self, periods: list[str], day: date, dataset: str) -> pd.DataFrame:
        """Each quarter against the one before, fetched once; IFRS has nothing before 2013-Q1."""
        fetched: dict[str, Summary] = {}

        def summaries(period: str) -> Summary:
            if period < IFRS_FROM:
                return {report: {} for report in REPORTS}
            if period not in fetched:
                filed = statement_deadline(period) < day
                fetched[period] = self._summaries(period, filed, dataset)
            return fetched[period]

        records = []
        waits: dict[str, list[str]] = {}
        for period in periods:
            current, previous = summaries(period), summaries(previous_period(period))
            waits[period] = incomplete(current)
            if waits[period]:
                logger.warning("mops %s scan waits for %s", period, ", ".join(waits[period]))
            for row in statement_rows(current, previous, int(period[-1])):
                records.append({**row, "period": period})
        frame = pd.DataFrame(records, columns=SCAN_COLUMNS, dtype=object)
        frame.attrs["incomplete"] = waits
        return frame

    def _summaries(self, period: str, filed: bool, dataset: str) -> Summary:
        """Both markets' three summaries, kept as raw first; a filed quarter cannot be empty."""
        year, season = int(period[:4]), int(period[-1])
        found: dict[str, dict[str, dict[str, float | None]]] = {}
        for report in REPORTS:
            found[report] = {}
            for market in MARKETS:
                params, page = self.summaries.summary(report, market, year - ROC_OFFSET, season)
                identity = {"html": EXPORT_NAME.sub("", page)}
                self.save_raw(dataset, params, {"html": page}, identity)
                companies = parse_summary(page, market, season, report)
                if filed and not companies:
                    raise SchemaChangedError(f"MOPS {report} {market} {period} came back empty")
                if companies.keys() & found[report].keys():
                    raise SchemaChangedError(f"MOPS {report} {period} lists a company twice")
                found[report] |= companies
        return found

    def _revenue_scan(self, day: date) -> pd.DataFrame:
        periods = open_months(day, int(settings.REVENUE_SCAN_DAYS_AFTER))
        return self._months(periods, day, REVENUE_SCAN)

    def _revenue_history(self, start: date, end: date) -> pd.DataFrame:
        frame = self._months(months_between(start, end), timezone.localdate(), REVENUE_HISTORY)
        frame.attrs.update(start=start, end=end)
        return frame

    def _months(self, periods: list[str], day: date, dataset: str) -> pd.DataFrame:
        """Each month in both markets; a month past its deadline must have filers."""
        records = []
        for period in periods:
            listed: set[str] = set()
            for market in MARKETS:
                companies = self._revenue_pages(market, period, listed, dataset)
                if not companies and revenue_deadline(period) < day:
                    raise SchemaChangedError(f"MOPS t21sc03 {market} {period} came back empty")
                listed |= companies.keys()
                records.extend(revenue_rows(companies, period))
        return pd.DataFrame(records, columns=REVENUE_COLUMNS, dtype=object)

    def _revenue_pages(
        self, market: str, period: str, listed: set[str], dataset: str
    ) -> dict[str, Figures]:
        """Domestic and foreign pages, kept as raw first; a company may appear only once."""
        year, month = int(period[:4]), int(period[5:])
        found: dict[str, Figures] = {}
        for origin in ORIGINS:
            params, page = self.summaries.revenue(market, year - ROC_OFFSET, month, origin)
            identity = {"html": without_generation_time(page)}
            self.save_raw(dataset, params, {"html": page}, identity)
            companies = parse_revenue_page(page, market, origin, period)
            if companies.keys() & (found.keys() | listed):
                raise SchemaChangedError(f"MOPS t21sc03 {period} lists a company twice")
            found |= companies
        return found

    def _conservative(self, dataset: str) -> Callable[[str], date]:
        """The deadline-based date FinMind rows carry, for periods first seen long after."""
        if dataset == REVENUE_HISTORY:
            return lambda period: revenue_visible_from(period, self.calendar)
        return lambda period: visible_from(statement_deadline(period), self.calendar)


def _statement_years(stock_id: str, start: date, end: date) -> list[int]:
    """Years with stored statements; a report of year Y is listed under Y even if filed in Y+1."""
    periods = FinancialStatement.objects.filter(
        stock_id=stock_id, period__gte=f"{start.year}", period__lt=f"{end.year + 1}"
    ).values_list("period", flat=True)
    return sorted({int(period[:4]) for period in periods})


class Reapplied(NamedTuple):
    changed: int
    stocks: int
    rejected: dict[str, str]


def reapply_filings(adapter: MopsAdapter, stock_ids: Sequence[str] | None = None) -> Reapplied:
    """Rerun parsing and the date rules on the page last seen for each company and year."""
    rows = RawMops.objects.filter(dataset=STATEMENT_FILING)
    if stock_ids:
        rows = rows.filter(request_params__co_id__in=list(stock_ids))
    latest: dict[tuple[str, str], int] = {}
    ordered = rows.order_by("fetched_at", "id").values_list("id", "request_params")
    for raw_id, params in ordered.iterator():
        latest[(params["co_id"], params["year"])] = raw_id
    by_stock: dict[str, list[int]] = {}
    for (code, _), raw_id in sorted(latest.items()):
        by_stock.setdefault(code, []).append(raw_id)
    changed, rejected = 0, {}
    for code, raw_ids in by_stock.items():
        try:
            changed += _reapply_stock(adapter, code, raw_ids)
        except SourceError as exc:
            rejected[code] = f"{type(exc).__name__}: {exc}"
    return Reapplied(changed, len(by_stock), rejected)


def _reapply_stock(adapter: MopsAdapter, code: str, raw_ids: list[int]) -> int:
    pages = RawMops.objects.filter(id__in=raw_ids).values_list("request_params", "payload")
    filings = [
        filing
        for params, payload in pages
        for filing in parse_listing(payload["html"], code, int(params["year"]))
    ]
    frame = adapter.normalize(STATEMENT_FILING, pd.DataFrame(filings, columns=FILING_COLUMNS))
    with transaction.atomic():
        return adapter.upsert(STATEMENT_FILING, frame)
