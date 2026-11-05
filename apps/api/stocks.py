"""Stock header and price-series endpoints over stock_master and daily_price."""

from datetime import date
from typing import Any

from django.shortcuts import get_object_or_404
from rest_framework.exceptions import NotFound
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.api.envelope import envelope
from apps.api.params import PriceQuery, parse
from apps.api.sessions import load_calendar, trailing_sessions
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
            ("previous_date", "previous_close", "reference_price", "change", "change_pct")
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
        "change_pct": round(close / reference - 1, RATIO_DIGITS),
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


class StockDetailView(APIView):
    def get(self, request: Request, stock_id: str) -> Response:
        stock = get_object_or_404(StockMaster, stock_id=stock_id)
        bars = list(
            DailyPrice.objects.filter(stock_id=stock_id, close__isnull=False)
            .order_by("-date")
            .values(*QUOTE_FIELDS)[:2]
        )
        return envelope({**master_fields(stock), "quote": quote(bars)})


class StockPricesView(APIView):
    def get(self, request: Request, stock_id: str) -> Response:
        days = parse(PriceQuery, request)["days"]
        if not StockMaster.objects.filter(stock_id=stock_id).exists():
            raise NotFound(f"stock {stock_id} is not in stock_master")
        prices = DailyPrice.objects.filter(stock_id=stock_id)
        anchor = prices.order_by("-date").values("date", "adj_factor").first()
        if anchor is None:
            meta = {"days": days, "start_date": None, "end_date": None, "adjusted_to": None}
            return envelope([], meta)
        sessions = trailing_sessions(load_calendar(), anchor["date"], days)
        rows = {
            row["date"]: row
            for row in prices.filter(date__range=(sessions[0], sessions[-1])).values(*BAR_FIELDS)
        }
        first = min(rows, default=anchor["date"])
        window = [day for day in sessions if rows and day >= first]
        meta = {
            "days": days,
            "start_date": window[0] if window else None,
            "end_date": window[-1] if window else None,
            "adjusted_to": anchor["date"],
        }
        return envelope([bar(day, rows.get(day), anchor["adj_factor"]) for day in window], meta)
