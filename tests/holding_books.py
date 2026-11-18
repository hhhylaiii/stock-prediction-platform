"""Main and other shadow book rows for API and page tests, with TAIEX_TR closes."""

from collections.abc import Mapping
from datetime import date
from typing import Any

from apps.api.holding import MAIN_BOOK
from apps.market.models import IndexDaily
from apps.research.models import (
    BookAction,
    BookPhase,
    BookStatus,
    HoldingBookAction,
    HoldingBookDay,
)

FILL_FIELDS = {"value": 1.0, "cash": 1.0, "paid": 0.0, "traded": 0.0, "held": 0}


def add_list(
    signal: date,
    trade: date,
    actions: Mapping[str, tuple[str, int, float | None]] | None = None,
    book: str = MAIN_BOOK,
    status: str = BookStatus.LISTED,
    backfilled: bool = False,
    universe: list[str] | None = None,
) -> None:
    """A list and its lines; its universe is the listed stocks, or empty without predictions."""
    lines = actions or {}
    listed = sorted(lines) or ["2330"]
    HoldingBookDay.objects.create(
        book=book,
        signal_date=signal,
        phase=BookPhase.LIST,
        trade_date=trade,
        status=status,
        universe=(listed if status == BookStatus.LISTED else []) if universe is None else universe,
        run_id="rank-2026-11.h5",
        held_before=sum(
            action in (BookAction.KEEP, BookAction.SELL) for action, _, _ in lines.values()
        ),
        sells=sum(action == BookAction.SELL for action, _, _ in lines.values()),
        buys=sum(action == BookAction.BUY for action, _, _ in lines.values()),
        backups=sum(action == BookAction.BACKUP for action, _, _ in lines.values()),
        backfilled=backfilled,
    )
    HoldingBookAction.objects.bulk_create(
        HoldingBookAction(
            book=book,
            signal_date=signal,
            trade_date=trade,
            stock_id=stock,
            phase=BookPhase.LIST,
            action=action,
            priority=priority,
            score=0.5,
            pct=pct,
            run_id="rank-2026-11.h5",
        )
        for stock, (action, priority, pct) in lines.items()
    )


def add_fill(
    signal: date,
    trade: date,
    value: float,
    universe: float | None = 0.0,
    book: str = MAIN_BOOK,
    stuck: tuple[str, ...] = (),
    **fields: Any,
) -> None:
    HoldingBookDay.objects.create(
        book=book,
        signal_date=signal,
        phase=BookPhase.FILL,
        trade_date=trade,
        run_id="rank-2026-11.h5",
        **{
            "status": BookStatus.TRADED,
            **FILL_FIELDS,
            "value": value,
            "universe_return": universe,
            **fields,
        },
    )
    HoldingBookAction.objects.bulk_create(
        HoldingBookAction(
            book=book,
            signal_date=signal,
            trade_date=trade,
            stock_id=stock,
            phase=BookPhase.FILL,
            action=BookAction.STUCK,
            value=0.03,
        )
        for stock in stuck
    )


def add_closes(closes: dict[date, float]) -> None:
    IndexDaily.objects.bulk_create(
        IndexDaily(index_id="TAIEX_TR", date=day, close=close) for day, close in closes.items()
    )
