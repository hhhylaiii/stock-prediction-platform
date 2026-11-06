import math
from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from apps.market.models import DividendEvent
from apps.research.features.dividend import days_to_ex_date, load_ex_dates
from tests.research.conftest import weekday_calendar

CALENDAR = weekday_calendar(date(2016, 1, 1), date(2016, 12, 31))


def d(text: str) -> date:
    return date.fromisoformat(text)


def panel(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame({"stock_id": [s for s, _ in rows], "date": [d(t) for _, t in rows]})


def events(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_id": [s for s, _, _ in rows],
            "announce_date": [d(a) for _, a, _ in rows],
            "ex_date": [d(e) for _, _, e in rows],
        }
    )


def test_trading_days_to_the_nearest_ex_date_announced_by_the_day() -> None:
    days = ["2016-03-04", "2016-03-07", "2016-03-09", "2016-03-11", "2016-03-14", "2016-03-21"]
    rows = panel([("2330", day) for day in days] + [("1101", "2016-03-09")])
    found = events(
        [
            ("2330", "2016-03-07", "2016-03-14"),
            ("2330", "2016-03-09", "2016-03-21"),
            ("2317", "2016-03-01", "2016-03-10"),
        ]
    )

    counted = days_to_ex_date(rows, found, CALENDAR)

    assert counted.index.equals(rows.index)
    assert math.isnan(counted.iloc[0])
    assert counted.iloc[1:5].tolist() == [5.0, 3.0, 1.0, 5.0]
    assert math.isnan(counted.iloc[5])
    assert math.isnan(counted.iloc[6])


def test_an_ex_date_past_the_calendar_or_announced_after_it_is_not_counted() -> None:
    rows = panel([("2330", "2016-12-28"), ("1101", "2016-06-08")])
    found = events([("2330", "2016-12-20", "2017-01-05"), ("1101", "2016-06-15", "2016-06-10")])

    counted = days_to_ex_date(rows, found, CALENDAR)

    assert counted.isna().all()


def test_no_events_leave_every_day_missing() -> None:
    rows = panel([("2330", "2016-03-07")])

    assert days_to_ex_date(rows, events([]), CALENDAR).isna().all()


@pytest.mark.django_db
def test_only_announced_cash_dividends_that_go_ex_after_the_start_are_loaded() -> None:
    def event(ex: str, cash: str, announce: str | None, stock: str = "0") -> DividendEvent:
        return DividendEvent(
            stock_id="2330",
            ex_date=d(ex),
            cash_dividend=Decimal(cash),
            stock_dividend=Decimal(stock),
            announce_date=d(announce) if announce else None,
        )

    DividendEvent.objects.bulk_create(
        [
            event("2016-03-14", "3", "2016-03-01"),
            event("2016-06-14", "0", "2016-06-01", stock="1"),
            event("2016-09-14", "2", None),
            event("2016-01-04", "1", "2015-12-20"),
            event("2016-12-20", "1", "2016-12-01"),
            event("2016-02-15", "1", "2016-02-01"),
            event("2016-12-21", "1", "2016-11-30"),
        ]
    )

    loaded = load_ex_dates(d("2016-02-15"), d("2016-11-30"))

    assert loaded.to_dict("records") == [
        {"stock_id": "2330", "announce_date": d("2016-03-01"), "ex_date": d("2016-03-14")},
        {"stock_id": "2330", "announce_date": d("2016-11-30"), "ex_date": d("2016-12-21")},
    ]
