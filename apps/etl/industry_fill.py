"""Industries the stock master lacks, from the exchanges' quotes listed by industry (plan K17c)."""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, NamedTuple

from django.db import transaction
from django.db.models import Q

from apps.etl.adapters.exchange_tables import NO_DATA_STAT, find_table
from apps.etl.adapters.tpex import OTC_QUOTES_PATH, TpexAdapter
from apps.etl.adapters.tpex_tables import QUOTED_TITLE, _rows
from apps.etl.adapters.tpex_tables import check_answer as tpex_answered
from apps.etl.adapters.twse import DAILY_QUOTES_PATH, DAILY_QUOTES_TITLE, TwseAdapter
from apps.etl.adapters.twse_tables import check_answer as twse_answered
from apps.etl.errors import SchemaChangedError
from apps.etl.exchange_rows import stock_code
from apps.etl.models import DataQualityLog, EtlRun
from apps.etl.runs import failing_on_termination, tracked_run
from apps.market.calendar import CalendarError, TradingDays
from apps.market.models import StockMaster

INDUSTRY_CHECK = "industry_fill"
RAW_DATASET = "industry_quotes"
INDUSTRY_TYPES = tuple(f"{number:02d}" for number in range(1, 39) if number not in (7, 13))
QUARTER_ENDS = ((3, 31), (6, 30), (9, 30), (12, 31))
MARKETS = {"twse": "TWSE", "tpex": "TPEx"}
TITLED = re.compile(rf"{DAILY_QUOTES_TITLE}\((.+)\)")


class Listed(NamedTuple):
    """The industry a type's page names, if any, and the codes it lists."""

    page: str | None
    codes: list[str]


def sample_days(calendar: TradingDays, start: date, end: date) -> list[date]:
    """Each quarter's last session in the span, the end standing in for a quarter not over."""
    end = min(end, calendar.end)
    days: list[date] = []
    for year in range(start.year, end.year + 1):
        for month, day in QUARTER_ENDS:
            close = date(year, month, day)
            if date(year, month - 2, 1) > end or close < start:
                continue
            try:
                session = calendar.latest_on_or_before(min(close, end))
            except CalendarError:
                continue
            if session >= start and session not in days:
                days.append(session)
    return days


def named(listed: Mapping[str, Listed], known: Mapping[str, str]) -> dict[str, list[Any]]:
    """Each type's industry: the page's own name when the master uses it, else the majority's.

    Each entry is [industry or None, page name, majority industry, its votes, stocks with one].
    """
    names = set(known.values())
    found = {}
    for kind, (page, codes) in listed.items():
        votes = Counter(known[code] for code in codes if known.get(code))
        top, count = votes.most_common(1)[0] if votes else (None, 0)
        found[kind] = [page if page in names else top, page, top, count, sum(votes.values())]
    return found


def fill_industries(
    adapter: TwseAdapter | TpexAdapter, days: Sequence[date], since: date | None = None
) -> EtlRun:
    """Stocks without an industry take the last one they were listed under on the sampled days."""
    if not days:
        raise ValueError("no sampled days to read industries from")
    blank = set(StockMaster.objects.filter(industry="").values_list("stock_id", flat=True))
    known = dict(StockMaster.objects.exclude(industry="").values_list("stock_id", "industry"))
    seen: dict[str, list[str]] = {}
    types: dict[str, dict[str, list[Any]]] = {}
    first, last = days[0], days[-1]
    with failing_on_termination(), tracked_run(adapter.source, "fill:industry", first, last) as run:
        for day in days:
            listed = listed_by_type(adapter, day)
            names = named(listed, known)
            types = {**types, day.isoformat(): names}
            seen = _seen(seen, listed, names, blank, day)
            run.checkpoint = {"last_date": day.isoformat(), "seen": seen, "types": types}
            run.save(update_fields=["checkpoint"])
        with transaction.atomic():
            written = {
                code: entry
                for code, entry in seen.items()
                if StockMaster.objects.filter(stock_id=code, industry="").update(industry=entry[0])
            }
            run.rows = len(written)
            _log(adapter, since or first, last, types, written, blank)
    return run


def listed_by_type(adapter: TwseAdapter | TpexAdapter, day: date) -> dict[str, Listed]:
    """What each industry type lists that day, each answer kept as raw first."""
    found = {}
    for kind in INDUSTRY_TYPES:
        if isinstance(adapter, TwseAdapter):
            listed = _twse_type(adapter, day, kind)
        else:
            listed = _otc_type(adapter, day, kind)
        if listed.codes:
            found[kind] = listed
    if not found:
        raise SchemaChangedError(f"{adapter.source} {day}: no industry type lists any stock")
    return found


def _seen(
    seen: Mapping[str, list[str]],
    listed: Mapping[str, Listed],
    names: Mapping[str, list[Any]],
    blank: set[str],
    day: date,
) -> dict[str, list[str]]:
    """Blank stocks with the industry seen that day; first seen counts from the last change."""
    today = day.isoformat()
    after = dict(seen)
    for kind, (_, codes) in listed.items():
        industry = names[kind][0]
        if industry is None:
            continue
        for code in codes:
            if code not in blank:
                continue
            before = seen.get(code)
            since = before[2] if before and before[0] == industry else today
            after[code] = [industry, kind, since, today]
    return after


def _twse_type(adapter: TwseAdapter, day: date, kind: str) -> Listed:
    params = {"date": f"{day:%Y%m%d}", "type": kind, "response": "json"}
    payload: dict[str, Any] = adapter.client.get_json(DAILY_QUOTES_PATH, params)
    adapter.save_raw(RAW_DATASET, params, payload)
    if NO_DATA_STAT in str(payload.get("stat", "")):
        return Listed(None, [])
    twse_answered(payload, day)
    tables = [
        (match.group(1), table["data"])
        for table in payload.get("tables") or []
        if (match := TITLED.search(str(table.get("title") or ""))) and table.get("data")
    ]
    if len(tables) > 1:
        raise SchemaChangedError(
            f"TWSE MI_INDEX type {kind} on {day} has {len(tables)} quote tables"
        )
    if not tables:
        return Listed(None, [])
    page, rows = tables[0]
    return Listed(page, [stock_code(row[0]) for row in rows])


def _otc_type(adapter: TpexAdapter, day: date, kind: str) -> Listed:
    form = {"date": f"{day:%Y/%m/%d}", "type": kind, "response": "json"}
    payload: dict[str, Any] = adapter.client.post_json(OTC_QUOTES_PATH, form)
    adapter.save_raw(RAW_DATASET, form, payload)
    tpex_answered(payload, day)
    table = find_table(payload, QUOTED_TITLE)
    rows = _rows(table, f"otc type {kind}")
    page = str(table.get("category") or "").strip() or None
    return Listed(page, [stock_code(row[0]) for row in rows])


def _log(
    adapter: TwseAdapter | TpexAdapter,
    since: date,
    last: date,
    types: Mapping[str, Mapping[str, list[Any]]],
    written: Mapping[str, list[str]],
    blank: set[str],
) -> None:
    """Each sampled day's industry per type, the stocks filled, and this market's still blank."""
    alive = StockMaster.objects.filter(
        Q(delist_date__isnull=True) | Q(delist_date__gte=since),
        market=MARKETS[adapter.source],
        stock_id__in=blank,
        industry="",
    )
    DataQualityLog.objects.create(
        date=last,
        check_name=INDUSTRY_CHECK,
        severity=DataQualityLog.Severity.INFO,
        detail={
            "source": adapter.source,
            "types": types,
            "filled": written,
            "unfilled": sorted(alive.values_list("stock_id", flat=True)),
        },
    )
