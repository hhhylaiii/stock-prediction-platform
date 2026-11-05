"""Stock header and price-series endpoints over stock_master and daily_price."""

from datetime import date
from typing import Any

from django.db.models import Max, Min
from django.shortcuts import get_object_or_404
from rest_framework.exceptions import NotFound
from rest_framework.request import Request
from rest_framework.response import Response

from apps.api.envelope import envelope
from apps.api.params import PriceQuery, parse
from apps.api.sessions import load_calendar, trailing_sessions
from apps.api.views import ReadOnlyView
from apps.market.adjust import PRICE_COLUMNS, adjusted
from apps.market.models import DailyPrice, StockMaster

PRICE_DIGITS = 4
RATIO_DIGITS = 6
QUOTE_FIELDS = ("date", "close", "adj_factor", "volume", "turnover")
BAR_FIELDS = (*PRICE_COLUMNS, "date", "volume", "turnover", "adj_factor")


def master_fields(stock: StockMaster) -> dict[str, Any]:
    return {
        "stock_id": stock.stock_id,
        "name": stock.name,
        "market": stock.market,
        "market_label": stock.get_market_display(),
        "industry": stock.industry,
        "list_date": stock.list_date,
        "delist_date": stock.delist_date,
    }


def quote(bars: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Latest traded close and its change against the previous close carried across ex-dates."""
    if not bars:
        return None
    latest = bars[0]
    close = float(latest["close"])
    head = {"date": latest["date"], "close": close}
    tail = {"volume": latest["volume"], "turnover": latest["turnover"]}
    if len(bars) == 1:
        empty = dict.fromkeys(
            ("previous_date", "previous_close", "reference_price", "change", "change_ratio")
        )
        return {**head, **empty, **tail}
    previous = bars[1]
    reference = float(previous["close"]) * previous["adj_factor"] / latest["adj_factor"]
    return {
        **head,
        "previous_date": previous["date"],
        "previous_close": float(previous["close"]),
        "reference_price": round(reference, PRICE_DIGITS),
        "change": round(close - reference, PRICE_DIGITS),
        "change_ratio": round(close / reference - 1, RATIO_DIGITS),
        **tail,
    }


def bar(day: date, row: dict[str, Any] | None, anchor_factor: float) -> dict[str, Any]:
    """One session; prices scaled so the series ends on the latest raw price."""
    if row is None:
        return {"date": day, **dict.fromkeys((*PRICE_COLUMNS, "volume", "turnover"))}
    scale = row["adj_factor"] / anchor_factor
    prices = {}
    for column in PRICE_COLUMNS:
        value = adjusted(row[column], scale)
        prices[column] = None if value is None else round(value, PRICE_DIGITS)
    return {"date": day, **prices, "volume": row["volume"], "turnover": row["turnover"]}


class StockDetailView(ReadOnlyView):
    def get(self, request: Request, stock_id: str) -> Response:
        stock = get_object_or_404(StockMaster, stock_id=stock_id)
        bars = list(
            DailyPrice.objects.filter(stock_id=stock_id, close__isnull=False)
            .order_by("-date")
            .values(*QUOTE_FIELDS)[:2]
        )
        return envelope({**master_fields(stock), "quote": quote(bars)})


class StockPricesView(ReadOnlyView):
    def get(self, request: Request, stock_id: str) -> Response:
        days = parse(PriceQuery, request)["days"]
        if not StockMaster.objects.filter(stock_id=stock_id).exists():
            raise NotFound(f"找不到股票 {stock_id}。")
        prices = DailyPrice.objects.filter(stock_id=stock_id)
        span = prices.aggregate(first=Min("date"), last=Max("date"))
        if span["last"] is None:
            meta = {"days": days, "start_date": None, "end_date": None, "adjusted_to": None}
            return envelope([], meta)
        sessions = trailing_sessions(load_calendar(), span["last"], days)
        rows = {
            row["date"]: row
            for row in prices.filter(date__range=(sessions[0], span["last"])).values(*BAR_FIELDS)
        }
        anchor_factor = rows[span["last"]]["adj_factor"]
        window = [day for day in sessions if day >= span["first"]]
        meta = {
            "days": days,
            "start_date": window[0] if window else None,
            "end_date": window[-1] if window else None,
            "adjusted_to": span["last"],
        }
        return envelope([bar(day, rows.get(day), anchor_factor) for day in window], meta)
