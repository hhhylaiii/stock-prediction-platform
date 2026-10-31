"""Single entry point for trading-day arithmetic over the trading_calendar table (SPEC §8.7)."""

from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from datetime import date, timedelta


class CalendarError(ValueError):
    """Trading calendar is misused or incomplete."""


class NotATradingDayError(CalendarError):
    """An operation that needs a trading day received another date."""


class OutsideCalendarError(CalendarError):
    """A date or a shift result falls outside the loaded calendar range."""


class TradingDays:
    """Immutable sorted trading days plus the [start, end] range the calendar covers."""

    __slots__ = ("_days", "_end", "_lookup", "_start")

    def __init__(self, days: Iterable[date], start: date, end: date) -> None:
        ordered = tuple(sorted(set(days)))
        if start > end:
            raise CalendarError(f"calendar start {start} is after end {end}")
        if ordered and (ordered[0] < start or ordered[-1] > end):
            raise CalendarError("trading days fall outside the calendar range")
        self._days = ordered
        self._lookup = frozenset(ordered)
        self._start = start
        self._end = end

    @property
    def start(self) -> date:
        return self._start

    @property
    def end(self) -> date:
        return self._end

    @classmethod
    def from_db(cls) -> "TradingDays":
        """Load the whole trading_calendar table; it must hold every day of its range."""
        from apps.market.models import TradingCalendar

        rows = list(TradingCalendar.objects.order_by("date").values_list("date", "is_trading_day"))
        if not rows:
            raise CalendarError("trading_calendar is empty")
        start, end = rows[0][0], rows[-1][0]
        expected = (end - start).days + 1
        if len(rows) != expected:
            raise CalendarError(f"trading_calendar has {expected - len(rows)} missing day(s)")
        return cls((day for day, is_open in rows if is_open), start, end)

    def _require_covered(self, day: date) -> None:
        if not self.start <= day <= self.end:
            raise OutsideCalendarError(f"{day} is outside {self.start}..{self.end}")

    def is_trading_day(self, day: date) -> bool:
        self._require_covered(day)
        return day in self._lookup

    def shift(self, day: date, n: int) -> date:
        """Return the trading day n sessions after (n > 0) or before (n < 0) a trading day."""
        if not self.is_trading_day(day):
            raise NotATradingDayError(f"{day} is not a trading day")
        index = bisect_left(self._days, day) + n
        if not 0 <= index < len(self._days):
            raise OutsideCalendarError(f"shifting {day} by {n} leaves the calendar")
        return self._days[index]

    def between(self, start: date, end: date) -> tuple[date, ...]:
        """Trading days in [start, end], both ends inclusive and inside the calendar."""
        self._require_covered(start)
        self._require_covered(end)
        if start > end:
            raise CalendarError(f"start {start} is after end {end}")
        return self._days[bisect_left(self._days, start) : bisect_right(self._days, end)]

    def latest_on_or_before(self, day: date) -> date:
        """Most recent trading day not later than day."""
        self._require_covered(day)
        index = bisect_right(self._days, day) - 1
        if index < 0:
            raise OutsideCalendarError(f"no trading day on or before {day}")
        return self._days[index]

    def all_days(self) -> Iterable[tuple[date, bool]]:
        """Every calendar day in range with its trading flag, for loading the table."""
        day = self.start
        while day <= self.end:
            yield day, day in self._lookup
            day += timedelta(days=1)
