"""Coming ex-dates from the exchanges' notice tables, dated by the session that first saw them."""

from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, NamedTuple

import pandas as pd
from django.conf import settings
from django.db import transaction
from django.db.models import Min
from django.utils import timezone

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
UNSETTLED_DAYS = timedelta(days=30)
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


class Outcome(NamedTuple):
    """What one list did, or in a preview would do, to dividend_event."""

    session: date
    detail: dict[str, Any]
    created: list[dict[str, Any]]
    dated: list[dict[str, Any]]
    revised: list[dict[str, Any]]

    @property
    def rows(self) -> int:
        return sum(len(part) for part in (self.created, self.dated, self.revised)) + len(
            self.detail["removed_now"]
        )


def write_notices(source: str, frame: pd.DataFrame, calendar: TradingDays) -> int:
    """The list as the session of its fetch saw it: that evening's if read before the cutoff."""
    return _read(source, frame, calendar, quiet=False).rows


def preview_notices(source: str, frame: pd.DataFrame, calendar: TradingDays) -> Outcome:
    """What the list would write, by the same code as a real run; the caller rolls it back."""
    return _read(source, frame, calendar, quiet=True)


def _read(source: str, frame: pd.DataFrame, calendar: TradingDays, quiet: bool) -> Outcome:
    seen = frame.attrs["seen"]
    notices = [Notice(*row) for row in frame[NOTICE_COLUMNS].itertuples(index=False)]
    today = timezone.localtime(seen).date()
    return apply_notices(source, notices, session_seen(calendar, seen), today, quiet)


def replay_notices(source: str, calendar: TradingDays) -> Replayed:
    """Every stored list again, oldest fetch first and without alerts; one refused is skipped."""
    raw_model, parse = SOURCES[source]
    today = timezone.localdate()
    lists = rows = skipped = 0
    for record in raw_model._default_manager.filter(dataset=DIVIDEND_NOTICE).order_by("fetched_at"):
        try:
            notices = parse(record.payload)
        except SchemaChangedError:
            notices = []
        if not notices:
            skipped += 1
            continue
        session = session_seen(calendar, record.fetched_at)
        rows += apply_notices(source, notices, session, today, quiet=True).rows
        lists += 1
    return Replayed(lists, rows, skipped)


def apply_notices(
    source: str,
    notices: Sequence[Notice],
    session: date,
    today: date | None = None,
    quiet: bool = False,
) -> Outcome:
    """Write the list as the session saw it, then withdraw what two sound lists in a row lack.

    today, the day the list is applied, differs from the session only when replaying old lists.
    """
    today = today or session
    coming = {(notice.stock_id, notice.ex_date): notice for notice in notices}
    coming = {key: notice for key, notice in coming.items() if key[1] > session}
    with transaction.atomic():
        before = _fetch_before(source, session)
        gone = _gone(source, session, coming)
        sound = bool(coming) and len(gone) <= int(settings.DIVIDEND_NOTICE_MAX_GONE)
        seen = _see(source, coming, session)
        created, dated, revised = _write(coming, seen, session, today)
        again = sound and before is not None and bool(before.detail.get("sound"))
        removed = _withdraw(source, gone, before.date, session) if again and before else []
        detail = {
            "source": source,
            "listed": len(coming),
            "sound": sound,
            "created": len(created),
            "dated": len(dated),
            "revised": len(revised),
            "gone": [_label(key) for key in sorted(_keys(gone))],
            "removed": [*_removed_earlier(source, session), *removed],
            "removed_now": removed,
            "unsettled": _unsettled(source, min(session, today)),
        }
        row = _log(source, session, {k: v for k, v in detail.items() if k != "removed_now"})
    if not quiet:
        dispatch([row])
    return Outcome(session, detail, created, dated, revised)


def _fetch_before(source: str, session: date) -> DataQualityLog | None:
    """The latest list of an earlier session, with whether it looked sound."""
    return (
        DataQualityLog.objects.filter(
            check_name=NOTICE_CHECK, detail__source=source, date__lt=session
        )
        .order_by("-date")
        .first()
    )


def _gone(source: str, session: date, coming: dict[Key, Notice]) -> list[DividendNotice]:
    """Events an earlier list held that are still to come, not yet withdrawn, and not on this one.

    They keep counting until withdrawn, so a list cut short night after night never reads sound.
    More than DIVIDEND_NOTICE_MAX_GONE of them make the list unsound; at 0 nothing is ever removed.
    """
    live = DividendNotice.objects.filter(
        source=source, withdrawn__isnull=True, first_seen__lt=session, ex_date__gt=session
    )
    return [row for row in live if (row.stock_id, row.ex_date) not in coming]


def _see(source: str, coming: dict[Key, Notice], session: date) -> dict[Key, DividendNotice]:
    """Each listed event with its first and last session, widened by this one.

    An event listed again after it was withdrawn is live again; an older list replayed is not.
    """
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
        if row.withdrawn is not None and session >= row.withdrawn:
            row.withdrawn = None
    DividendNotice.objects.bulk_update(stored.values(), ["first_seen", "last_seen", "withdrawn"])
    DividendNotice.objects.bulk_create(fresh)
    return {**stored, **{(row.stock_id, row.ex_date): row for row in fresh}}


def _write(
    coming: dict[Key, Notice], seen: dict[Key, DividendNotice], session: date, today: date
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """New events with their first session; blank dates filled; own unsettled amounts follow.

    A withdrawn event is left alone, no event is made for an ex-date that has already come,
    and only the latest list seen sets an amount, so an older one replayed changes nothing.
    """
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
    first = _first_seen_anywhere(coming)
    new: list[DividendEvent] = []
    dated: list[DividendEvent] = []
    revised: list[tuple[DividendEvent, list[Decimal]]] = []
    owned: list[DividendNotice] = []
    for key, notice in coming.items():
        sighting = seen[key]
        if key[0] not in listed or sighting.withdrawn is not None:
            continue
        event = stored.get(key)
        if event is None:
            if notice.ex_date > today:
                new.append(_event(notice, first[key]))
                owned.append(sighting)
            continue
        if event.announce_date is None:
            event.announce_date = first[key]
            dated.append(event)
        ours = sighting.created_event and sighting.last_seen == session
        if ours and event.adj_factor is None and _differs(event, notice):
            held = [getattr(event, name) for name in AMOUNTS]
            event.cash_dividend, event.stock_dividend = _amounts(notice)
            revised.append((event, held))
    for sighting in owned:
        sighting.created_event = True
    DividendEvent.objects.bulk_create(new)
    DividendEvent.objects.bulk_update(dated, ["announce_date"])
    DividendEvent.objects.bulk_update([event for event, _ in revised], list(AMOUNTS))
    DividendNotice.objects.bulk_update(owned, ["created_event"])
    return (
        [_shown(event) for event in new],
        [_shown(event) for event in dated],
        [
            {**_shown(event), "was": [format(value, "f") for value in held]}
            for event, held in revised
        ],
    )


def _first_seen_anywhere(coming: dict[Key, Notice]) -> dict[Key, date]:
    """The earliest session either exchange listed each event: a stock may move between them."""
    rows = (
        DividendNotice.objects.filter(
            stock_id__in={key[0] for key in coming}, ex_date__in={key[1] for key in coming}
        )
        .values_list("stock_id", "ex_date")
        .annotate(first=Min("first_seen"))
    )
    return {(code, ex_date): first for code, ex_date, first in rows}


def _withdraw(
    source: str, gone: list[DividendNotice], before: date, session: date
) -> list[dict[str, Any]]:
    """Mark what this list and the sound one before both lack; drop its unsettled event if ours.

    An event the other exchange has listed since stays: the stock moved markets, not its ex-date.
    """
    removed: list[dict[str, Any]] = []
    lost = [row for row in gone if row.last_seen < before]
    for sighting in sorted(lost, key=lambda row: (row.ex_date, row.stock_id)):
        sighting.withdrawn = session
        elsewhere = DividendNotice.objects.exclude(source=source).filter(
            stock_id=sighting.stock_id, ex_date=sighting.ex_date, last_seen__gt=sighting.last_seen
        )
        if not sighting.created_event or elsewhere.exists():
            continue
        events = DividendEvent.objects.filter(
            stock_id=sighting.stock_id, ex_date=sighting.ex_date, adj_factor__isnull=True
        )
        removed.extend(_record(event, sighting, session) for event in events)
        events.delete()
    DividendNotice.objects.bulk_update(lost, ["withdrawn"])
    return removed


def _removed_earlier(source: str, session: date) -> list[dict[str, Any]]:
    """Removals this session already logged, kept when the list is read again."""
    row = DataQualityLog.objects.filter(
        check_name=NOTICE_CHECK, detail__source=source, date=session
    ).first()
    return [] if row is None else list(row.detail.get("removed", []))


def _unsettled(source: str, day: date) -> list[str]:
    """Events this table made whose ex-date came in the past 30 days without a result."""
    made = DividendNotice.objects.filter(
        source=source, created_event=True, ex_date__lte=day, ex_date__gte=day - UNSETTLED_DAYS
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


def _shown(event: DividendEvent) -> dict[str, Any]:
    return {
        "stock_id": event.stock_id,
        "ex_date": event.ex_date.isoformat(),
        "cash_dividend": format(event.cash_dividend, "f"),
        "stock_dividend": format(event.stock_dividend, "f"),
        "announce_date": None if event.announce_date is None else event.announce_date.isoformat(),
    }


def _record(event: DividendEvent, sighting: DividendNotice, session: date) -> dict[str, Any]:
    """Everything needed to put a removed event back by hand."""
    return {
        **_shown(event),
        "first_seen": sighting.first_seen.isoformat(),
        "last_seen": sighting.last_seen.isoformat(),
        "removed_on": session.isoformat(),
    }


def _keys(rows: list[DividendNotice]) -> list[Key]:
    return [(row.stock_id, row.ex_date) for row in rows]


def _label(key: Key) -> str:
    return f"{key[0]}@{key[1].isoformat()}"
