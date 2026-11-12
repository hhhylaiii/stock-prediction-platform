"""When each ex-date became public, from FinMind's dividend announcements; blanks only."""

from collections.abc import Callable, Mapping
from datetime import date, datetime, time
from typing import NamedTuple

from django.conf import settings

from apps.etl.errors import SchemaChangedError
from apps.etl.filings import filing_session
from apps.market.calendar import CalendarError, TradingDays
from apps.market.models import DividendEvent

Session = Callable[[datetime], date]


def announcement_session(calendar: TradingDays, cutoff: time) -> Session:
    """Announcements before the evening cutoff count for that session, as filings do."""

    def session(moment: datetime) -> date:
        try:
            return filing_session(moment, calendar, cutoff)
        except CalendarError as exc:
            raise SchemaChangedError(f"announcement {moment} is off the trading calendar") from exc

    return session


def session_seen(calendar: TradingDays, moment: datetime) -> date:
    """The session a list read at this moment counts for, by the same cutoff as filings."""
    cutoff = time.fromisoformat(settings.FILING_VISIBLE_BEFORE)
    return announcement_session(calendar, cutoff)(moment)


class Redated(NamedTuple):
    """An event with the date it holds and the one its announcement gives."""

    stock_id: str
    ex_date: date
    stored: date | None
    announced: date


def compare_announce_dates(stock_id: str, announced: Mapping[date, date | None]) -> list[Redated]:
    """Each stored event whose announcement is known, beside the date it holds now."""
    known = {ex: day for ex, day in announced.items() if day is not None}
    rows = DividendEvent.objects.filter(stock_id=stock_id, ex_date__in=list(known))
    return [
        Redated(stock_id, row.ex_date, row.announce_date, known[row.ex_date])
        for row in rows.order_by("ex_date")
    ]


def exchanges_lead_from() -> date:
    """The first ex-date the exchanges' own tables settle; FinMind's backfill ends before it."""
    return date.fromisoformat(str(settings.DIVIDEND_EXCHANGE_PRIMARY_FROM))


def moves(found: Redated, earlier: bool) -> bool:
    """A blank date is filled; with earlier, a first sight moves back. None ever moves forward.

    Only events the exchanges lead hold a first sight; dates of the FinMind years never move.
    """
    if found.stored is None:
        return True
    return earlier and found.ex_date >= exchanges_lead_from() and found.announced < found.stored


def fill_announce_dates(
    stock_id: str, announced: Mapping[date, date | None], earlier: bool = False
) -> int:
    """Dates only where none is stored; the exchanges' notice tables (plan 2.7b) come first.

    With earlier, an announcement before the stored date replaces it: a first sight only trails.
    """
    changes = [row for row in compare_announce_dates(stock_id, announced) if moves(row, earlier)]
    for row in changes:
        DividendEvent.objects.filter(stock_id=stock_id, ex_date=row.ex_date).update(
            announce_date=row.announced
        )
    return len(changes)
