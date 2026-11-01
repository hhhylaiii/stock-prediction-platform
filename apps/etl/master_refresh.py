"""stock_master kept current from each day's exchange quotes and the official delisting lists."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd
from django.db import transaction
from django.db.models import Max, QuerySet

from apps.etl.adapters import limiter_for
from apps.etl.adapters.tpex import TpexClient
from apps.etl.adapters.tpex_tables import quoted_names
from apps.etl.adapters.twse import TwseClient, is_foreign_currency_etf, quotes_frame
from apps.etl.alerts import dispatch
from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.models import DataQualityLog, EtlRun, RawRecord, RawTpex, RawTwse
from apps.etl.runs import tracked_run
from apps.etl.stock_master import MASTER_COLUMNS, tpex_delistings, twse_delistings
from apps.etl.upsert import upsert
from apps.market.calendar import (
    CalendarError,
    NotATradingDayError,
    OutsideCalendarError,
    TradingDays,
)
from apps.market.models import DailyPrice, Market, StockMaster

VANISHED_SESSIONS = 5
DELISTING_WINDOW = timedelta(days=90)
CHECK_NAME = "stock_master"
DAILY_DATASET = "stock_master_daily"
UPDATED_FIELDS = ("name", "market", "delist_date")
Listing = tuple[str, str]
MasterRow = tuple[str, str, date | None]


@dataclass(frozen=True)
class MasterChanges:
    rows: list[dict[str, Any]]
    added: list[str]
    moved: list[str]
    relisted: list[str]
    delisted: list[str]


def quoted_listings(day: date) -> dict[str, Listing]:
    """Local-currency codes quoted on each exchange that day, with name and market."""
    twse = _latest(RawTwse.objects.filter(payload__date=f"{day:%Y%m%d}"), "TWSE", day)
    tpex_rows = RawTpex.objects.filter(request_params__type="EW", payload__date=f"{day:%Y%m%d}")
    tpex = _latest(tpex_rows, "TPEx", day)
    frame = quotes_frame(twse.payload, day)
    if not {"證券代號", "證券名稱"} <= set(frame.columns):
        raise SchemaChangedError(f"TWSE daily quotes fields changed: {list(frame.columns)}")
    codes = frame["證券代號"].astype(str).str.strip()
    names = frame["證券名稱"].astype(str).str.strip()
    listed = {
        code: (name, Market.TWSE.value)
        for code, name in zip(codes, names, strict=True)
        if not is_foreign_currency_etf(code)
    }
    for code, name in quoted_names(tpex.payload, day).items():
        if code in listed:
            raise SchemaChangedError(f"{code} is quoted on both exchanges on {day}")
        listed[code] = (name, Market.TPEX.value)
    return listed


def _latest(rows: QuerySet[Any], exchange: str, day: date) -> RawRecord:
    stored: RawRecord | None = rows.filter(dataset="daily_price").order_by("-fetched_at").first()
    if stored is None:
        raise SourceNotReadyError(f"{exchange} daily_price for {day} is not loaded")
    return stored


def master_changes(
    master: Mapping[str, MasterRow],
    quoted: Mapping[str, Listing],
    delistings: pd.DataFrame,
    day: date,
) -> MasterChanges:
    """Quoted codes listed under their quoted name and market; recent official delistings apply."""
    rows: list[dict[str, Any]] = []
    added: list[str] = []
    moved: list[str] = []
    relisted: list[str] = []
    for code, (name, market) in sorted(quoted.items()):
        known = master.get(code)
        if known == (name, market, None):
            continue
        if known is None:
            added.append(code)
        elif known[1] != market:
            moved.append(code)
        gone = known[2] if known is not None else None
        if gone is not None and gone <= day:
            relisted.append(code)
        rows.append(_row(code, name, market, gone if gone is not None and gone > day else None))
    recent = _recent_delistings(delistings, day)
    delisted = [
        code
        for code, (_, market, gone) in sorted(master.items())
        if gone is None and code not in quoted and (code, market) in recent
    ]
    rows.extend(
        _row(code, master[code][0], master[code][1], recent[(code, master[code][1])])
        for code in delisted
    )
    return MasterChanges(rows, added, moved, relisted, delisted)


def _row(code: str, name: str, market: str, delisted: date | None) -> dict[str, Any]:
    return {
        "stock_id": code,
        "name": name,
        "market": market,
        "industry": "",
        "delist_date": delisted,
    }


def _recent_delistings(delistings: pd.DataFrame, day: date) -> dict[tuple[str, str], date]:
    """Latest official delisting per code and market within the window."""
    recent = delistings.loc[
        (delistings["delist_date"] > day - DELISTING_WINDOW) & (delistings["delist_date"] <= day)
    ]
    latest = recent.sort_values("delist_date").drop_duplicates(["stock_id", "market"], keep="last")
    keys = zip(latest["stock_id"], latest["market"], strict=True)
    return dict(zip(keys, latest["delist_date"], strict=True))


def newly_vanished(
    codes: Iterable[str], day: date, calendar: TradingDays, after: date | None = None
) -> list[str]:
    """Codes whose last prices went five sessions quiet since the refresh after which this runs."""
    try:
        upper = calendar.shift(day, -VANISHED_SESSIONS)
        if after is not None and after < day:
            lower = calendar.shift(calendar.latest_on_or_before(after), -VANISHED_SESSIONS)
        else:
            lower = calendar.shift(upper, -1)
    except OutsideCalendarError:
        return []
    prices = DailyPrice.objects.filter(stock_id__in=list(codes), date__gt=lower, date__lte=day)
    last = prices.values("stock_id").annotate(last=Max("date")).values_list("stock_id", "last")
    return sorted(code for code, seen in last if seen <= upper)


def refresh_stock_master(day: date, calendar: TradingDays) -> EtlRun:
    """Apply the day's quotes and recent delistings, then log and alert on vanished codes."""
    if not calendar.is_trading_day(day):
        raise NotATradingDayError(f"{day} is not a trading day")
    last = (
        EtlRun.objects.filter(dataset=DAILY_DATASET, status=EtlRun.Status.SUCCESS)
        .order_by("-start_date")
        .first()
    )
    after = last.start_date if last is not None else None
    if after is not None and day < after:
        raise CalendarError(f"{day} is before the last refresh on {after}")
    with tracked_run("stock_master", DAILY_DATASET, day, day) as run:
        quoted = quoted_listings(day)
        delistings = pd.concat(
            [
                twse_delistings(TwseClient.build(limiter_for("twse")).suspend_listing()),
                tpex_delistings(TpexClient.build(limiter_for("tpex")).delisted()),
            ],
            ignore_index=True,
        )
        rows = StockMaster.objects.values_list("stock_id", "name", "market", "delist_date")
        changes = master_changes({row[0]: row[1:] for row in rows}, quoted, delistings, day)
        if changes.rows:
            frame = pd.DataFrame(changes.rows, columns=MASTER_COLUMNS)
            run.rows = upsert(StockMaster, frame, ("stock_id",), UPDATED_FIELDS)
        listed = StockMaster.objects.filter(delist_date__isnull=True).values_list(
            "stock_id", flat=True
        )
        vanished = newly_vanished(set(listed) - set(quoted), day, calendar, after)
        run.checkpoint = {
            "listed": len(listed),
            "added": changes.added,
            "moved": changes.moved,
            "relisted": changes.relisted,
            "delisted": changes.delisted,
            "vanished": vanished,
        }
    dispatch([_log(day, run.checkpoint)])
    return run


def _log(day: date, found: dict[str, Any]) -> DataQualityLog:
    """One row per day: warning when codes vanished without a delisting, else info."""
    severity = (
        DataQualityLog.Severity.WARNING if found["vanished"] else DataQualityLog.Severity.INFO
    )
    detail = {key: value for key, value in found.items() if key != "listed"}
    with transaction.atomic():
        DataQualityLog.objects.filter(date=day, check_name=CHECK_NAME).delete()
        return DataQualityLog.objects.create(
            date=day, check_name=CHECK_NAME, severity=severity, detail=detail
        )
