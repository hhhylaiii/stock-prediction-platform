"""Bars the price backfill lacks, added from the exchange's quotes of that one day."""

from collections.abc import Sequence
from datetime import date
from typing import Any

from django.db import transaction

from apps.etl.adapters import build_adapter
from apps.etl.adapters.tpex import TpexAdapter
from apps.etl.runs import tracked_run
from apps.etl.upsert import insert_missing
from apps.market.adjust import refresh_where_later_events
from apps.market.calendar import NotATradingDayError, TradingDays
from apps.market.models import DailyPrice, StockMaster

FILL_DATASET = "daily_price_fill"
FILLABLE = ("tpex",)
KEYS = ("stock_id", "date")
Bar = dict[str, Any]


def fill_price_bars(
    source: str, day: date, stocks: Sequence[str], calendar: TradingDays
) -> list[Bar]:
    """Insert the chosen stocks' bars of one day, all of them or none; returns the bars.

    The answers are kept as daily_price_fill, so the first kept quote page stays where it is.
    """
    codes = sorted(set(stocks))
    adapter = build_adapter(source, calendar)
    if not isinstance(adapter, TpexAdapter):
        raise ValueError(f"cannot fill {source!r}: choose from {list(FILLABLE)}")
    _refuse_unfit(adapter, day, codes, calendar)
    with tracked_run(source, FILL_DATASET, day, day) as run:
        quotes = adapter.day_quotes(day, FILL_DATASET)
        bars = quotes.loc[quotes["stock_id"].isin(codes)].reset_index(drop=True)
        absent = sorted(set(codes) - set(bars["stock_id"]))
        if absent:
            raise ValueError(f"{source} quotes of {day} do not list: {', '.join(absent)}")
        with transaction.atomic():
            run.rows = insert_missing(DailyPrice, bars, KEYS)
            refresh_where_later_events(codes, day)
        run.checkpoint = {"stocks": codes}
    return [{str(name): value for name, value in bar.items()} for bar in bars.to_dict("records")]


def _refuse_unfit(adapter: TpexAdapter, day: date, codes: list[str], calendar: TradingDays) -> None:
    """Everything that can be told before asking the exchange."""
    if not codes:
        raise ValueError("no stock chosen")
    if not calendar.is_trading_day(day):
        raise NotATradingDayError(f"{day} is not a trading day")
    adapter.require_before_kept_quotes(day)
    known = {row.stock_id: row for row in StockMaster.objects.filter(stock_id__in=codes)}
    unknown = [code for code in codes if code not in known]
    if unknown:
        raise ValueError(f"not in stock_master: {', '.join(unknown)}")
    unlisted = [code for code in codes if not _listed(known[code], day)]
    if unlisted:
        raise ValueError(f"not listed on {day}: {', '.join(unlisted)}")
    stored = DailyPrice.objects.filter(stock_id__in=codes, date=day)
    have = sorted(stored.values_list("stock_id", flat=True))
    if have:
        raise ValueError(f"already have a bar on {day}: {', '.join(have)}")


def _listed(stock: StockMaster, day: date) -> bool:
    listed = stock.list_date is None or day >= stock.list_date
    return listed and (stock.delist_date is None or day < stock.delist_date)
