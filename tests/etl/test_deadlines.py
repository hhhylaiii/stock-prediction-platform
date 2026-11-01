from datetime import date, timedelta

import pytest

from apps.etl.deadlines import revenue_deadline, statement_deadline, visible_from
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
        ("2024-Q1", date(2024, 5, 30)),
        ("2024-Q2", date(2024, 8, 31)),
        ("2024-Q3", date(2024, 11, 29)),
        ("2024-Q4", date(2025, 3, 31)),
        ("2011-Q4", date(2012, 3, 31)),
        ("2010-Q4", date(2011, 4, 30)),
        ("2012-Q1", date(2012, 5, 30)),
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
