from datetime import date, timedelta

import pytest

from apps.etl.deadlines import (
    revenue_deadline,
    revenue_visible_from,
    statement_deadline,
    visible_from,
)
from apps.market.calendar import TradingDays

HOLIDAYS = {date(2024, 4, 4), date(2024, 4, 5)}
DAYS = TradingDays(
    (
        date(2023, 1, 1) + timedelta(days=n)
        for n in range(800)
        if (date(2023, 1, 1) + timedelta(days=n)).weekday() < 5
        and date(2023, 1, 1) + timedelta(days=n) not in HOLIDAYS
    ),
    date(2023, 1, 1),
    date(2025, 3, 10),
)


@pytest.mark.parametrize(
    ("period", "deadline"),
    [
        ("2024-01", date(2024, 2, 10)),
        ("2024-12", date(2025, 1, 10)),
        ("2025-11", date(2025, 12, 10)),
        ("2025-12", date(2026, 1, 15)),
        ("2026-03", date(2026, 4, 15)),
    ],
)
def test_revenue_is_due_on_the_10th_and_the_15th_from_2026(period: str, deadline: date) -> None:
    assert revenue_deadline(period) == deadline


@pytest.mark.parametrize(
    ("period", "deadline"),
    [
        ("2024-Q1", date(2024, 5, 31)),
        ("2024-Q2", date(2024, 8, 31)),
        ("2024-Q3", date(2024, 11, 30)),
        ("2024-Q4", date(2025, 3, 31)),
        ("2011-Q4", date(2012, 3, 31)),
        ("2010-Q4", date(2011, 4, 30)),
        ("2012-Q1", date(2012, 5, 31)),
    ],
)
def test_statements_take_the_latest_deadline_of_any_issuer(period: str, deadline: date) -> None:
    assert statement_deadline(period) == deadline


@pytest.mark.parametrize("period", ["2024-13", "2024-Q5", "24-01", "2024-1"])
def test_malformed_periods_are_rejected(period: str) -> None:
    with pytest.raises(ValueError, match=period):
        revenue_deadline(period) if "Q" not in period else statement_deadline(period)


@pytest.mark.parametrize(
    ("deadline", "visible"),
    [
        (date(2024, 4, 10), date(2024, 4, 11)),
        (date(2024, 2, 10), date(2024, 2, 13)),
        (date(2024, 4, 4), date(2024, 4, 9)),
        (date(2024, 5, 30), date(2024, 5, 31)),
    ],
)
def test_data_is_visible_the_trading_day_after_the_effective_deadline(
    deadline: date, visible: date
) -> None:
    assert visible_from(deadline, DAYS) == visible


CLOSURES = {
    date(2019, 2, 1),
    *(date(2019, 2, d) for d in range(4, 9)),
    date(2022, 1, 27),
    date(2022, 1, 28),
    date(2022, 1, 31),
    *(date(2022, 2, d) for d in range(1, 5)),
    *(date(2023, 4, d) for d in range(3, 6)),
    date(2024, 10, 10),
}
SESSIONS = TradingDays(
    (
        date(2019, 1, 1) + timedelta(days=n)
        for n in range(2200)
        if (date(2019, 1, 1) + timedelta(days=n)).weekday() < 5
        and date(2019, 1, 1) + timedelta(days=n) not in CLOSURES
    ),
    date(2019, 1, 1),
    date(2025, 1, 31),
)


@pytest.mark.parametrize(
    ("period", "announced_deadline"),
    [
        ("2019-01", date(2019, 2, 15)),
        ("2022-01", date(2022, 2, 14)),
        ("2023-03", date(2023, 4, 12)),
    ],
)
def test_revenue_stays_hidden_through_every_announced_extension(
    period: str, announced_deadline: date
) -> None:
    assert revenue_visible_from(period, SESSIONS) > announced_deadline


@pytest.mark.parametrize(
    ("period", "visible"),
    [
        ("2023-03", date(2023, 4, 14)),
        ("2024-09", date(2024, 10, 14)),
        ("2024-05", date(2024, 6, 11)),
    ],
)
def test_holidays_before_the_10th_push_revenue_back_by_as_many_sessions(
    period: str, visible: date
) -> None:
    assert revenue_visible_from(period, SESSIONS) == visible
