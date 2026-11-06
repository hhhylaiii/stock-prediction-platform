"""When each ex-date became public, from FinMind's dividend announcements; blanks only."""

from collections.abc import Callable, Mapping
from datetime import date, datetime, time

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


def fill_announce_dates(stock_id: str, announced: Mapping[date, date | None]) -> int:
    """Dates only where none is stored; the exchanges' notice tables (plan 2.7) come first."""
    known = {ex: day for ex, day in announced.items() if day is not None}
    rows = list(
        DividendEvent.objects.filter(
            stock_id=stock_id, ex_date__in=list(known), announce_date__isnull=True
        )
    )
    for row in rows:
        row.announce_date = known[row.ex_date]
    DividendEvent.objects.bulk_update(rows, ["announce_date"])
    return len(rows)
