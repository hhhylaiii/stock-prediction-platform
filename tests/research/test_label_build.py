import math
from datetime import date, timedelta

import pandas as pd
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.market.models import DailyPrice, StockMaster, TradingCalendar
from apps.research.labels import LABEL_COLUMNS, build_labels, write_labels
from apps.research.models import LabelDaily
from tests.research.conftest import weekday_calendar

CALENDAR = weekday_calendar(date(2016, 1, 1), date(2016, 12, 31))
pytestmark = pytest.mark.django_db


def d(text: str) -> date:
    return date.fromisoformat(text)


def listed(stock_id: str, **fields: object) -> None:
    defaults: dict[str, object] = {
        "name": stock_id,
        "market": "TWSE",
        "industry": "水泥工業",
        "list_date": d("2016-01-04"),
    }
    StockMaster.objects.create(stock_id=stock_id, **{**defaults, **fields})


def prices(
    stock_id: str,
    start: str,
    end: str,
    close: float = 100.0,
    special: dict[date, tuple[float, float]] | None = None,
) -> None:
    """Flat bars from start to end; special maps a day to its (open, close)."""
    chosen = special or {}
    DailyPrice.objects.bulk_create(
        DailyPrice(
            stock_id=stock_id,
            date=day,
            open=chosen.get(day, (close, close))[0],
            high=max(chosen.get(day, (close, close))),
            low=min(chosen.get(day, (close, close))),
            close=chosen.get(day, (close, close))[1],
            volume=1_000_000,
            turnover=100_000_000,
        )
        for day in CALENDAR.between(d(start), d(end))
    )


def one(labels: pd.DataFrame, stock_id: str, day: str, horizon: int) -> pd.Series:
    picked = labels[
        (labels["stock_id"] == stock_id)
        & (labels["date"] == d(day))
        & (labels["horizon"] == horizon)
    ]
    assert len(picked) == 1
    return picked.iloc[0]


@pytest.mark.spec8
def test_delisted_stock_settles_at_its_last_close() -> None:
    listed("1101", delist_date=d("2016-03-14"))
    prices("1101", "2016-01-04", "2016-03-09", close=50.0)
    DailyPrice.objects.filter(stock_id="1101", date=d("2016-03-09")).update(close=55.0)
    listed("2330")
    prices("2330", "2016-01-04", "2016-04-29")

    labels = build_labels(CALENDAR, d("2016-03-01"), d("2016-03-11"))
    settled = one(labels, "1101", "2016-03-04", 5)
    beyond = one(labels, "1101", "2016-03-04", 20)
    final = one(labels, "1101", "2016-03-09", 20)

    assert settled["fwd_ret"] == pytest.approx(55.0 / 50.0 - 1)
    assert settled["fwd_ret_open"] == pytest.approx(55.0 / 50.0 - 1)
    assert beyond["fwd_ret_open"] == pytest.approx(55.0 / 50.0 - 1)
    assert final["fwd_ret"] == pytest.approx(0.0)
    assert final["untradable_t1"] is True


@pytest.mark.spec8
def test_untradable_t1_on_limit_up_open() -> None:
    listed("1101")
    prices("1101", "2016-01-04", "2016-03-31", special={d("2016-03-02"): (110.0, 110.0)})
    listed("1102")
    prices("1102", "2016-01-04", "2016-03-31", special={d("2016-03-02"): (109.5, 110.0)})

    labels = build_labels(CALENDAR, d("2016-03-01"), d("2016-03-01"))
    blocked = one(labels, "1101", "2016-03-01", 5)
    open_below = one(labels, "1102", "2016-03-01", 5)

    assert blocked["untradable_t1"] is True
    assert math.isnan(blocked["fwd_rank_pct"])
    assert open_below["untradable_t1"] is False


def test_labels_are_limited_to_the_requested_trading_days() -> None:
    listed("2330")
    prices("2330", "2016-01-04", "2016-04-29")

    labels = build_labels(CALENDAR, d("2016-02-27"), d("2016-03-04"))

    assert list(labels.columns) == LABEL_COLUMNS
    assert sorted(set(labels["date"])) == list(CALENDAR.between(d("2016-02-29"), d("2016-03-04")))
    assert labels.groupby("date").size().eq(2).all()


def test_labels_past_the_data_end_stay_open() -> None:
    listed("2330")
    prices("2330", "2016-01-04", "2016-03-31")

    labels = build_labels(CALENDAR, d("2016-03-28"), d("2016-04-15"))

    assert set(labels["date"]) == set(CALENDAR.between(d("2016-03-28"), d("2016-03-31")))
    assert labels["fwd_ret_open"].isna().all()


@pytest.mark.parametrize(
    ("start", "end"), [("2016-05-02", "2016-05-31"), ("2016-03-05", "2016-03-06")]
)
def test_no_labels_without_days_or_prices(start: str, end: str) -> None:
    listed("2330")
    prices("2330", "2016-01-04", "2016-03-31")

    assert build_labels(CALENDAR, d(start), d(end)).empty


def test_horizon_past_the_calendar_end_is_left_open() -> None:
    listed("2330")
    prices("2330", "2016-11-01", "2016-12-30")

    labels = build_labels(CALENDAR, d("2016-12-26"), d("2016-12-30"))

    assert len(labels) == 10
    assert labels["fwd_ret_open"].isna().all()


def test_no_labels_on_an_empty_price_table() -> None:
    assert build_labels(CALENDAR, d("2016-03-01"), d("2016-03-31")).empty


def test_writing_twice_changes_nothing_and_a_new_price_updates_the_row() -> None:
    listed("2330")
    prices("2330", "2016-01-04", "2016-04-29")
    labels = build_labels(CALENDAR, d("2016-03-01"), d("2016-03-31"))

    assert write_labels(labels) == len(labels)
    write_labels(labels)
    DailyPrice.objects.filter(stock_id="2330", date=d("2016-03-08")).update(close=120.0)
    write_labels(build_labels(CALENDAR, d("2016-03-01"), d("2016-03-31")))

    assert LabelDaily.objects.count() == len(labels)
    row = LabelDaily.objects.get(stock_id="2330", date=d("2016-03-01"), horizon=5)
    assert row.fwd_ret == pytest.approx(0.2)
    assert str(row) == "2330 2016-03-01 5d"


def calendar_rows(start: str, end: str) -> None:
    first, last = d(start), d(end)
    TradingCalendar.objects.bulk_create(
        TradingCalendar(
            date=first + timedelta(days=n), is_trading_day=(first + timedelta(days=n)).weekday() < 5
        )
        for n in range((last - first).days + 1)
    )


def test_command_rebuilds_the_recent_days_by_default() -> None:
    calendar_rows("2016-01-01", "2016-12-31")
    listed("2330")
    prices("2330", "2016-01-04", "2016-04-29")

    call_command("build_labels", recent=5)

    days = set(LabelDaily.objects.values_list("date", flat=True))
    assert days == set(CALENDAR.between(d("2016-04-25"), d("2016-04-29")))


def test_command_takes_a_range_across_years() -> None:
    calendar_rows("2015-12-01", "2016-12-31")
    listed("2330", list_date=d("2015-12-01"))
    DailyPrice.objects.bulk_create(
        DailyPrice(
            stock_id="2330", date=day, open=100, high=100, low=100, close=100, volume=1, turnover=1
        )
        for day in weekday_calendar(d("2015-12-01"), d("2016-02-29")).between(
            d("2015-12-01"), d("2016-02-29")
        )
    )

    call_command("build_labels", start=d("2015-12-28"), end=d("2016-01-05"))

    assert LabelDaily.objects.filter(horizon=5).count() == 7


def test_command_refuses_an_empty_price_table_and_a_reversed_range() -> None:
    calendar_rows("2016-01-01", "2016-12-31")
    with pytest.raises(CommandError, match="empty"):
        call_command("build_labels")
    listed("2330")
    prices("2330", "2016-01-04", "2016-03-31")
    with pytest.raises(CommandError, match="after the end"):
        call_command("build_labels", start=d("2016-04-04"))
