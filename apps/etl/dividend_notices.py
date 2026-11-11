"""Coming ex-dates from the exchanges' notice tables, dated by the session that first saw them."""

from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, NamedTuple

import pandas as pd
from django.db import transaction

from apps.etl.adapters.exchange_notices import (
    DIVIDEND_NOTICE,
    NOTICE_COLUMNS,
    Notice,
    tpex_notices,
    twse_notices,
)
from apps.etl.adapters.finmind_datasets import DIVIDEND_PLACES
from apps.etl.alerts import dispatch
from apps.etl.dividend_dates import session_seen
from apps.etl.errors import SchemaChangedError
from apps.etl.models import DataQualityLog, DividendNotice, RawRecord, RawTpex, RawTwse
from apps.market.calendar import TradingDays
from apps.market.models import DividendEvent, StockMaster

NOTICE_CHECK = DIVIDEND_NOTICE
MAX_GONE = 5
MAX_GONE_SHARE = 0.2
UNSETTLED_DAYS = timedelta(days=7)
AMOUNTS = ("cash_dividend", "stock_dividend")
Key = tuple[str, date]
SOURCES: dict[str, tuple[type[RawRecord], Callable[[Mapping[str, Any]], list[Notice]]]] = {
    "twse": (RawTwse, twse_notices),
    "tpex": (RawTpex, tpex_notices),
}
Severity = DataQualityLog.Severity


class Replayed(NamedTuple):
    lists: int
    rows: int
    skipped: int


class Written(NamedTuple):
    """What one list changed in dividend_event."""

    created: int
    dated: int
    revised: int


def write_notices(source: str, frame: pd.DataFrame, calendar: TradingDays) -> int:
    """The list as the session of its fetch saw it: that evening's if read before the cutoff."""
    notices = [Notice(*row) for row in frame[NOTICE_COLUMNS].itertuples(index=False)]
    return apply_notices(source, notices, session_seen(calendar, frame.attrs["seen"]))


def replay_notices(source: str, calendar: TradingDays) -> Replayed:
    """Every stored list again, oldest fetch first; one that no longer parses is skipped."""
    raw_model, parse = SOURCES[source]
    lists = rows = skipped = 0
    for record in raw_model._default_manager.filter(dataset=DIVIDEND_NOTICE).order_by("fetched_at"):
        try:
            notices = parse(record.payload)
        except SchemaChangedError:
            notices = []
        if not notices:
            skipped += 1
            continue
        rows += apply_notices(source, notices, session_seen(calendar, record.fetched_at))
        lists += 1
    return Replayed(lists, rows, skipped)


def apply_notices(source: str, notices: Sequence[Notice], session: date) -> int:
    """Write the list as the session saw it, then drop events two sound lists in a row lack."""
    coming = {(notice.stock_id, notice.ex_date): notice for notice in notices}
    coming = {key: notice for key, notice in coming.items() if key[1] > session}
    with transaction.atomic():
        before = _fetch_before(source, session)
        gone = _gone(source, before, session, coming)
        sound = bool(coming) and (before is None or len(gone) <= _allowed(source, before, session))
        seen = _see(source, coming, session)
        written = _write(coming, seen)
        removed = _withdraw(source, before, session) if sound else []
        detail = {
            "source": source,
            "listed": len(coming),
            "sound": sound,
            "created": written.created,
            "dated": written.dated,
            "revised": written.revised,
            "gone": [_label(key) for key in sorted(gone)],
            "removed": [*_removed_earlier(source, session), *removed],
            "unsettled": _unsettled(source, session),
        }
        row = _log(source, session, detail)
    dispatch([row])
    return sum(written) + len(removed)


def _fetch_before(source: str, session: date) -> DataQualityLog | None:
    """The latest list of an earlier session, with whether it looked sound."""
    return (
        DataQualityLog.objects.filter(
            check_name=NOTICE_CHECK, detail__source=source, date__lt=session
        )
        .order_by("-date")
        .first()
    )


def _expected(source: str, before: date, session: date) -> Any:
    """Events the earlier list held that are still to come, so this list should hold them too."""
    return DividendNotice.objects.filter(
        source=source, first_seen__lte=before, last_seen__gte=before, ex_date__gt=session
    )


def _gone(
    source: str, before: DataQualityLog | None, session: date, coming: dict[Key, Notice]
) -> list[Key]:
    if before is None:
        return []
    keys = _expected(source, before.date, session).values_list("stock_id", "ex_date")
    return [key for key in keys if key not in coming]


def _allowed(source: str, before: DataQualityLog, session: date) -> float:
    """A handful of events may leave a list at once; more than that reads as a broken list."""
    expected: int = _expected(source, before.date, session).count()
    return max(MAX_GONE, MAX_GONE_SHARE * expected)


def _see(source: str, coming: dict[Key, Notice], session: date) -> dict[Key, DividendNotice]:
    """Each listed event with its first and last session, widened by this one."""
    stored = {
        (row.stock_id, row.ex_date): row
        for row in DividendNotice.objects.filter(
            source=source, stock_id__in={key[0] for key in coming}
        )
        if (row.stock_id, row.ex_date) in coming
    }
    fresh = [
        DividendNotice(
            source=source, stock_id=key[0], ex_date=key[1], first_seen=session, last_seen=session
        )
        for key in coming
        if key not in stored
    ]
    for row in stored.values():
        row.first_seen, row.last_seen = min(row.first_seen, session), max(row.last_seen, session)
    DividendNotice.objects.bulk_update(stored.values(), ["first_seen", "last_seen"])
    DividendNotice.objects.bulk_create(fresh)
    return {**stored, **{(row.stock_id, row.ex_date): row for row in fresh}}


def _write(coming: dict[Key, Notice], seen: dict[Key, DividendNotice]) -> Written:
    """New events with their first session; blank dates filled; own unsettled amounts follow."""
    listed = set(
        StockMaster.objects.filter(stock_id__in={key[0] for key in coming}).values_list(
            "stock_id", flat=True
        )
    )
    stored = {
        (row.stock_id, row.ex_date): row
        for row in DividendEvent.objects.filter(
            stock_id__in=listed, ex_date__in={key[1] for key in coming}
        )
    }
    new: list[DividendEvent] = []
    dated: list[DividendEvent] = []
    revised: list[DividendEvent] = []
    owned: list[DividendNotice] = []
    for key, notice in coming.items():
        if key[0] not in listed:
            continue
        sighting, event = seen[key], stored.get(key)
        if event is None:
            new.append(_event(notice, sighting.first_seen))
            owned.append(sighting)
            continue
        if event.announce_date is None:
            event.announce_date = sighting.first_seen
            dated.append(event)
        if sighting.created_event and event.adj_factor is None and _differs(event, notice):
            event.cash_dividend, event.stock_dividend = _amounts(notice)
            revised.append(event)
    for sighting in owned:
        sighting.created_event = True
    DividendEvent.objects.bulk_create(new)
    DividendEvent.objects.bulk_update(dated, ["announce_date"])
    DividendEvent.objects.bulk_update(revised, list(AMOUNTS))
    DividendNotice.objects.bulk_update(owned, ["created_event"])
    return Written(len(new), len(dated), len(revised))


def _withdraw(source: str, before: DataQualityLog | None, session: date) -> list[dict[str, Any]]:
    """Unsettled events this table made that neither this list nor the sound one before holds.

    One the other exchange has listed since stays: the stock moved markets, not its ex-date.
    """
    if before is None or not before.detail.get("sound"):
        return []
    lost = DividendNotice.objects.filter(
        source=source, created_event=True, last_seen__lt=before.date, ex_date__gt=session
    )
    removed = []
    for sighting in lost.order_by("ex_date", "stock_id"):
        elsewhere = DividendNotice.objects.exclude(source=source).filter(
            stock_id=sighting.stock_id, ex_date=sighting.ex_date, last_seen__gt=sighting.last_seen
        )
        if elsewhere.exists():
            continue
        events = DividendEvent.objects.filter(
            stock_id=sighting.stock_id, ex_date=sighting.ex_date, adj_factor__isnull=True
        )
        for event in events:
            removed.append(_record(event, sighting, session))
        events.delete()
    return removed


def _removed_earlier(source: str, session: date) -> list[dict[str, Any]]:
    """Removals this session already logged, kept when the list is read again."""
    row = DataQualityLog.objects.filter(
        check_name=NOTICE_CHECK, detail__source=source, date=session
    ).first()
    return [] if row is None else list(row.detail.get("removed", []))


def _unsettled(source: str, session: date) -> list[str]:
    """Events this table made whose ex-date came in the past week without a result."""
    made = DividendNotice.objects.filter(
        source=source,
        created_event=True,
        ex_date__lte=session,
        ex_date__gte=session - UNSETTLED_DAYS,
    ).values_list("stock_id", "ex_date")
    keys = sorted(
        key
        for key in made
        if DividendEvent.objects.filter(
            stock_id=key[0], ex_date=key[1], adj_factor__isnull=True
        ).exists()
    )
    return [_label(key) for key in keys]


def _log(source: str, session: date, detail: dict[str, Any]) -> DataQualityLog:
    """One row per session and source; anything a person should look at makes it a warning."""
    alarming = not detail["sound"] or detail["removed"] or detail["unsettled"]
    DataQualityLog.objects.filter(
        check_name=NOTICE_CHECK, detail__source=source, date=session
    ).delete()
    return DataQualityLog.objects.create(
        date=session,
        check_name=NOTICE_CHECK,
        severity=Severity.WARNING if alarming else Severity.INFO,
        detail=detail,
    )


def _event(notice: Notice, first_seen: date) -> DividendEvent:
    cash, shares = _amounts(notice)
    return DividendEvent(
        stock_id=notice.stock_id,
        ex_date=notice.ex_date,
        cash_dividend=cash,
        stock_dividend=shares,
        adj_factor=None,
        announce_date=first_seen,
    )


def _amounts(notice: Notice) -> list[Decimal]:
    due = (notice.cash_dividend, notice.stock_dividend)
    return [value.quantize(DIVIDEND_PLACES) for value in due]


def _differs(event: DividendEvent, notice: Notice) -> bool:
    stored = [getattr(event, name).quantize(DIVIDEND_PLACES) for name in AMOUNTS]
    return stored != _amounts(notice)


def _record(event: DividendEvent, sighting: DividendNotice, session: date) -> dict[str, Any]:
    """Everything needed to put a removed event back by hand."""
    return {
        "stock_id": event.stock_id,
        "ex_date": event.ex_date.isoformat(),
        "cash_dividend": format(event.cash_dividend, "f"),
        "stock_dividend": format(event.stock_dividend, "f"),
        "announce_date": None if event.announce_date is None else event.announce_date.isoformat(),
        "first_seen": sighting.first_seen.isoformat(),
        "last_seen": sighting.last_seen.isoformat(),
        "removed_on": session.isoformat(),
    }


def _label(key: Key) -> str:
    return f"{key[0]}@{key[1].isoformat()}"
