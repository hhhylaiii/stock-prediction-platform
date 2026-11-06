"""When each ex-date became public, from FinMind's dividend announcements; blanks only."""

from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from typing import Any, NamedTuple

from django.db import transaction

from apps.etl.adapters.finmind_datasets import Session, ex_date_announcements
from apps.etl.errors import SchemaChangedError, SourceError
from apps.etl.filings import filing_session
from apps.etl.models import RawFinmind
from apps.market.calendar import CalendarError, TradingDays
from apps.market.models import DividendEvent

POLICY = "dividend_policy"


class Reapplied(NamedTuple):
    filled: int
    stocks: int
    rejected: dict[str, str]


def announcement_session(calendar: TradingDays, cutoff: time) -> Session:
    """Announcements before the evening cutoff count for that session, as filings do."""

    def session(moment: datetime) -> date:
        try:
            return filing_session(moment, calendar, cutoff)
        except CalendarError as exc:
            raise SchemaChangedError(f"announcement {moment} is off the trading calendar") from exc

    return session


def fill_announce_dates(stock_id: str, announced: Mapping[date, date | None]) -> int:
    """Dates only where no source has given one; FinMind stays secondary (SPEC §3)."""
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


def reapply_announce_dates(session: Session, stock_ids: Sequence[str] | None = None) -> Reapplied:
    """Every stored dividend answer read again, without fetching."""
    answers = RawFinmind.objects.filter(dataset=POLICY)
    if stock_ids:
        answers = answers.filter(request_params__data_id__in=list(stock_ids))
    by_stock: dict[str, list[Any]] = {}
    for params, payload in answers.values_list("request_params", "payload").iterator():
        by_stock.setdefault(str(params["data_id"]), []).extend(payload.get("data", []))
    filled, rejected = 0, {}
    for code, rows in sorted(by_stock.items()):
        try:
            with transaction.atomic():
                filled += fill_announce_dates(code, ex_date_announcements(rows, code, session))
        except SourceError as exc:
            rejected[code] = f"{type(exc).__name__}: {exc}"
    return Reapplied(filled, len(by_stock), rejected)
