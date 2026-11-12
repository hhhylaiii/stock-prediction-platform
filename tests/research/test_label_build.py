import math
from datetime import date, timedelta
from decimal import Decimal

import pandas as pd
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.etl.adapters.exchange_notices import Notice
from apps.etl.dividend_notices import apply_notices
from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice, MarketHistory, StockMaster, TradingCalendar
from apps.research.labels import LABEL_COLUMNS, build_labels, write_labels
from apps.research.models import LabelDaily
from apps.research.ranges import years
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
    stock = StockMaster.objects.create(stock_id=stock_id, **{**defaults, **fields})
    MarketHistory.objects.create(
        stock_id=stock_id, start_date=stock.list_date or d("2016-01-04"), market=stock.market
    )


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
    prices("1101", "2016-01-04", "2016-05-31", special={d("2016-04-05"): (110.0, 110.0)})
    listed("1102")
    prices("1102", "2016-01-04", "2016-05-31", special={d("2016-04-05"): (109.5, 110.0)})

    labels = build_labels(CALENDAR, d("2016-04-04"), d("2016-04-04"))
    blocked = one(labels, "1101", "2016-04-04", 5)
    open_below = one(labels, "1102", "2016-04-04", 5)

    assert blocked["untradable_t1"] is True
    assert math.isnan(blocked["fwd_rank_pct"])
    assert open_below["untradable_t1"] is False
    assert open_below["fwd_rank_pct"] == 1.0
    assert one(labels, "1102", "2016-04-04", 20)["fwd_rank_pct"] == 1.0


def test_labels_are_limited_to_the_requested_trading_days() -> None:
    listed("2330")
    prices("2330", "2016-01-04", "2016-04-29")

    labels = build_labels(CALENDAR, d("2016-02-27"), d("2016-03-04"))

    assert list(labels.columns) == LABEL_COLUMNS
    assert sorted(set(labels["date"])) == list(CALENDAR.between(d("2016-02-29"), d("2016-03-04")))
    assert labels.groupby("date").size().eq(2).all()


def test_an_event_a_notice_wrote_ahead_changes_no_label() -> None:
    listed("2330")
    prices("2330", "2016-01-04", "2016-04-29")
    before = build_labels(CALENDAR, d("2016-02-01"), d("2016-03-31"))

    coming = Notice("2330", d("2016-03-07"), Decimal(1), Decimal(0))
    assert apply_notices("twse", [coming], d("2016-02-24")).rows == 1

    pd.testing.assert_frame_equal(build_labels(CALENDAR, d("2016-02-01"), d("2016-03-31")), before)
    assert not before.empty


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
    march = (d("2016-03-01"), d("2016-03-31"))
    labels = build_labels(CALENDAR, *march)

    assert write_labels(labels, *march) == len(labels)
    write_labels(labels, *march)
    DailyPrice.objects.filter(stock_id="2330", date=d("2016-03-08")).update(close=120.0)
    write_labels(build_labels(CALENDAR, *march), *march)

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


def test_command_without_a_range_rebuilds_the_last_sixty_trading_days() -> None:
    calendar_rows("2016-01-01", "2016-12-31")
    listed("2330")
    prices("2330", "2016-01-04", "2016-04-29")

    call_command("build_labels")

    days = set(LabelDaily.objects.values_list("date", flat=True))
    assert days == set(CALENDAR.between(CALENDAR.shift(d("2016-04-29"), -59), d("2016-04-29")))


def test_rebuilding_a_range_drops_rows_no_longer_built() -> None:
    listed("2330")
    prices("2330", "2016-01-04", "2016-04-29")
    march = (d("2016-03-01"), d("2016-03-31"))
    write_labels(build_labels(CALENDAR, *march), *march)
    write_labels(
        build_labels(CALENDAR, d("2016-04-01"), d("2016-04-05")), d("2016-04-01"), d("2016-04-05")
    )
    LabelDaily.objects.filter(date=d("2016-03-02")).update(fwd_rank_pct=0.5)
    StockMaster.objects.filter(stock_id="2330").update(list_date=d("2016-03-03"))

    write_labels(build_labels(CALENDAR, *march), *march)

    assert not LabelDaily.objects.filter(date__lt=d("2016-03-03")).exists()
    assert LabelDaily.objects.filter(date=d("2016-04-01")).exists()
    with pytest.raises(ValueError, match="outside"):
        write_labels(build_labels(CALENDAR, *march), d("2016-03-10"), d("2016-03-31"))


def test_command_takes_a_range_across_years(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[date, date]] = []

    def recorded(calendar: TradingDays, start: date, end: date) -> pd.DataFrame:
        calls.append((start, end))
        return build_labels(calendar, start, end)

    monkeypatch.setattr("apps.research.management.commands.build_labels.build_labels", recorded)
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
    assert calls == [(d("2015-12-28"), d("2015-12-31")), (d("2016-01-01"), d("2016-01-05"))]


def test_command_refuses_an_empty_price_table_and_a_reversed_range() -> None:
    calendar_rows("2016-01-01", "2016-12-31")
    with pytest.raises(CommandError, match="empty"):
        call_command("build_labels")
    listed("2330")
    prices("2330", "2016-01-04", "2016-03-31")
    with pytest.raises(CommandError, match="after the end"):
        call_command("build_labels", start=d("2016-04-04"))


def test_untraded_next_day_with_an_open_cannot_be_entered() -> None:
    listed("4923")
    prices("4923", "2016-01-04", "2016-05-31")
    DailyPrice.objects.filter(stock_id="4923", date=d("2016-04-05")).update(
        close=None, high=None, low=None
    )

    labels = build_labels(CALENDAR, d("2016-04-04"), d("2016-04-04"))
    blocked = one(labels, "4923", "2016-04-04", 5)

    assert blocked["untradable_t1"] is True
    assert math.isnan(blocked["fwd_ret_open"])


def test_ranges_are_rebuilt_one_calendar_year_at_a_time() -> None:
    assert list(years(d("2015-12-28"), d("2017-01-05"))) == [
        (d("2015-12-28"), d("2015-12-31")),
        (d("2016-01-01"), d("2016-12-31")),
        (d("2017-01-01"), d("2017-01-05")),
    ]
