"""Industries the stock master lacks, from the exchanges' quotes listed by industry (plan K17c)."""

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from django.db import transaction

from apps.etl.adapters.tpex import TpexAdapter
from apps.etl.adapters.twse import TwseAdapter
from apps.etl.exchange_rows import stock_code
from apps.etl.models import DataQualityLog, EtlRun
from apps.etl.runs import failing_on_termination, tracked_run
from apps.market.calendar import CalendarError, TradingDays
from apps.market.models import StockMaster

INDUSTRY_CHECK = "industry_fill"
RAW_DATASET = "industry_quotes"
TWSE_QUOTES = "/rwd/zh/afterTrading/MI_INDEX"
TPEX_QUOTES = "/www/zh-tw/afterTrading/otc"
QUOTES_TITLE = "每日收盤行情"
INDUSTRY_TYPES = tuple(f"{number:02d}" for number in range(1, 39) if number not in (7, 13))
QUARTER_ENDS = ((3, 31), (6, 30), (9, 30), (12, 31))


def sample_days(calendar: TradingDays, start: date, end: date) -> list[date]:
    """Each quarter's last session in the span, the end standing in for a quarter not over."""
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


def named(listed: Mapping[str, Sequence[str]], known: Mapping[str, str]) -> dict[str, str]:
    """Each type's name: the industry most of its stocks with a known one carry."""
    names = {}
    for kind, codes in listed.items():
        votes = Counter(known[code] for code in codes if known.get(code))
        if votes:
            names[kind] = votes.most_common(1)[0][0]
    return names


def fill_industries(adapter: TwseAdapter | TpexAdapter, days: Sequence[date]) -> EtlRun:
    """Stocks without an industry take the last one they were listed under on the sampled days."""
    if not days:
        raise ValueError("no sampled days to read industries from")
    blank = set(StockMaster.objects.filter(industry="").values_list("stock_id", flat=True))
    known = dict(StockMaster.objects.exclude(industry="").values_list("stock_id", "industry"))
    seen: dict[str, list[str]] = {}
    types: dict[str, dict[str, str]] = {}
    first, last = days[0], days[-1]
    with failing_on_termination(), tracked_run(adapter.source, "fill:industry", first, last) as run:
        for day in days:
            listed = listed_by_type(adapter, day)
            names = named(listed, known)
            types[day.isoformat()] = names
            for kind, codes in listed.items():
                for code in codes:
                    if code in blank and kind in names:
                        since = seen[code][2] if code in seen else day.isoformat()
                        seen[code] = [names[kind], kind, since, day.isoformat()]
            run.checkpoint = {"last_date": day.isoformat(), "seen": seen}
            run.save(update_fields=["checkpoint"])
        with transaction.atomic():
            for code, (industry, *_) in seen.items():
                StockMaster.objects.filter(stock_id=code, industry="").update(industry=industry)
            run.rows = len(seen)
            _log(adapter.source, last, types, seen)
    return run


def listed_by_type(adapter: TwseAdapter | TpexAdapter, day: date) -> dict[str, list[str]]:
    """The codes each industry type lists that day, each answer kept as raw first."""
    found = {}
    for kind in INDUSTRY_TYPES:
        if isinstance(adapter, TwseAdapter):
            codes = _twse_codes(adapter, day, kind)
        else:
            codes = _otc(adapter, day, kind)
        if codes:
            found[kind] = codes
    return found


def _twse_codes(adapter: TwseAdapter, day: date, kind: str) -> list[str]:
    params = {"date": f"{day:%Y%m%d}", "type": kind, "response": "json"}
    payload: dict[str, Any] = adapter.client.get_json(TWSE_QUOTES, params)
    adapter.save_raw(RAW_DATASET, params, payload)
    tables = [
        table
        for table in payload.get("tables") or []
        if QUOTES_TITLE in str(table.get("title") or "") and table.get("data")
    ]
    return [stock_code(row[0]) for table in tables for row in table["data"]]


def _otc(adapter: TpexAdapter, day: date, kind: str) -> list[str]:
    form = {"date": f"{day:%Y/%m/%d}", "type": kind, "response": "json"}
    payload: dict[str, Any] = adapter.client.post_json(TPEX_QUOTES, form)
    adapter.save_raw(RAW_DATASET, form, payload)
    tables = payload.get("tables") or []
    rows = (tables[0].get("data") or []) if tables else []
    return [stock_code(row[0]) for row in rows]


def _log(
    source: str, last: date, types: dict[str, dict[str, str]], seen: dict[str, list[str]]
) -> None:
    """The industry each type stood for on each sampled day, and the stocks filled."""
    DataQualityLog.objects.create(
        date=last,
        check_name=INDUSTRY_CHECK,
        severity=DataQualityLog.Severity.INFO,
        detail={"source": source, "types": types, "filled": seen},
    )
