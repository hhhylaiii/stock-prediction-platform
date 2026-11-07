from datetime import date

import pandas as pd
import pytest

from apps.market.models import DailyPrice, MarketHistory, StockMaster
from apps.research.panel import (
    PANEL_COLUMNS,
    REUSE_GAP,
    listing_segments,
    load_price_panel,
    require_sorted,
)
from tests.research.conftest import weekday_calendar

CALENDAR = weekday_calendar(date(2013, 1, 1), date(2014, 12, 31))
pytestmark = pytest.mark.django_db


def d(text: str) -> date:
    return date.fromisoformat(text)


def days(start: str, end: str) -> tuple[date, ...]:
    return CALENDAR.between(d(start), d(end))


def stock(stock_id: str, **fields: object) -> None:
    defaults: dict[str, object] = {"name": stock_id, "market": "TWSE", "industry": "半導體業"}
    StockMaster.objects.create(stock_id=stock_id, **{**defaults, **fields})


def bars(stock_id: str, when: tuple[date, ...], close: float = 10.0, adj: float = 1.0) -> None:
    DailyPrice.objects.bulk_create(
        DailyPrice(
            stock_id=stock_id,
            date=day,
            open=close - 0.5,
            high=close + 1,
            low=close - 1,
            close=close,
            volume=1_000,
            turnover=int(close * 1_000),
            adj_factor=adj,
        )
        for day in when
    )


def panel_of(stock_id: str, frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["stock_id"] == stock_id].reset_index(drop=True)


def test_each_row_takes_its_market_and_listing_on_the_day_from_the_market_history() -> None:
    stock("3092")
    stock("2330")
    for stock_id in ("3092", "2330"):
        bars(stock_id, days("2013-01-02", "2013-03-29"))
    MarketHistory.objects.bulk_create(
        [
            MarketHistory(
                stock_id="3092", start_date=d("2010-01-04"), end_date=d("2013-02-28"), market="TPEx"
            ),
            MarketHistory(stock_id="3092", start_date=d("2013-03-01"), market="TWSE"),
        ]
    )

    panel = load_price_panel(CALENDAR, d("2013-01-02"), d("2013-03-29"))

    moved = panel_of("3092", panel)
    switched = moved["date"].ge(d("2013-03-01"))
    assert moved.loc[~switched, "market"].eq("TPEx").all()
    assert moved.loc[switched, "market"].eq("TWSE").all()
    assert switched.any()
    assert not switched.all()
    assert moved["listed"].all()
    unknown = panel_of("2330", panel)
    assert unknown["market"].eq("TWSE").all()
    assert not unknown["listed"].any()


def test_adjusted_prices_are_floats_times_the_factor() -> None:
    stock("2330")
    bars("2330", days("2014-03-03", "2014-03-07"), close=600.5, adj=0.8)
    panel = load_price_panel(CALENDAR, d("2014-03-03"), d("2014-03-07"))
    assert list(panel.columns) == list(PANEL_COLUMNS)
    assert panel["close"].dtype == "float64"
    assert panel["adj_close"].tolist() == pytest.approx([600.5 * 0.8] * 5)
    assert panel["adj_open"].tolist() == pytest.approx([600.0 * 0.8] * 5)
    assert panel["adj_high"].iloc[0] == pytest.approx(601.5 * 0.8)
    assert panel["adj_low"].iloc[0] == pytest.approx(599.5 * 0.8)


def test_missing_days_inside_a_listing_are_filled_as_halted() -> None:
    stock("2330")
    bars("2330", days("2014-03-03", "2014-03-04") + days("2014-03-10", "2014-03-11"))
    panel = load_price_panel(CALENDAR, d("2014-01-01"), d("2014-12-31"))
    assert panel["date"].tolist() == list(days("2014-03-03", "2014-03-11"))
    assert panel["has_bar"].tolist() == [True, True, False, False, False, True, True]
    assert panel.loc[~panel["has_bar"], ["close", "volume", "adj_factor"]].isna().all().all()


def test_a_day_without_a_close_keeps_no_open_high_or_low() -> None:
    stock("4923")
    bars("4923", days("2014-03-03", "2014-03-05"))
    DailyPrice.objects.filter(stock_id="4923", date=d("2014-03-04")).update(
        close=None, high=None, low=None
    )
    panel = load_price_panel(CALENDAR, d("2014-03-03"), d("2014-03-05"))
    untraded = panel.loc[panel["date"] == d("2014-03-04")].iloc[0]
    assert untraded["has_bar"]
    assert untraded[["open", "high", "low", "adj_open", "adj_high", "adj_low"]].isna().all()
    assert panel["open"].notna().sum() == 2


def test_gap_longer_than_the_reuse_gap_starts_a_new_listing() -> None:
    stock("5450")
    stock("2330")
    first = days("2013-01-02", "2013-01-31")
    reused_from = CALENDAR.shift(first[-1], REUSE_GAP + 2)
    halted_from = CALENDAR.shift(first[-1], REUSE_GAP + 1)
    second = CALENDAR.between(reused_from, CALENDAR.shift(reused_from, 9))
    bars("5450", first + second)
    bars("2330", first + CALENDAR.between(halted_from, second[-1]))
    panel = load_price_panel(CALENDAR, d("2013-01-01"), d("2014-12-31"))
    reused, halted = panel_of("5450", panel), panel_of("2330", panel)
    assert reused["segment"].unique().tolist() == [1, 2]
    assert reused.loc[reused["date"] == second[0], "age"].item() == 0
    assert reused.loc[reused["date"] == second[1], "age"].item() == 1
    assert len(reused) == len(first) + len(second)
    assert halted["segment"].unique().tolist() == [1]
    assert (~halted["has_bar"]).sum() == REUSE_GAP


def test_halted_days_after_the_last_bar_run_to_delisting_or_the_data_end() -> None:
    stock("1111", delist_date=d("2014-03-20"))
    stock("2222")
    stock("2330")
    bars("1111", days("2014-03-03", "2014-03-05"))
    bars("2222", days("2014-03-03", "2014-03-05"))
    bars("2330", days("2014-03-03", "2014-03-31"))
    panel = load_price_panel(CALENDAR, d("2014-03-01"), d("2014-04-30"))
    delisted, suspended = panel_of("1111", panel), panel_of("2222", panel)
    assert delisted["date"].tolist() == list(days("2014-03-03", "2014-03-19"))
    assert suspended["date"].tolist() == list(days("2014-03-03", "2014-03-31"))
    assert suspended["has_bar"].sum() == 3


def test_halted_days_before_the_first_bar_start_at_the_listing_day() -> None:
    stock("6488", list_date=d("2014-03-03"))
    stock("2330", list_date=d("2012-06-01"))
    bars("2330", days("2014-02-03", "2014-03-14"))
    bars("6488", days("2014-03-10", "2014-03-14"))
    panel = load_price_panel(CALENDAR, d("2014-01-01"), d("2014-12-31"))
    late = panel_of("6488", panel)
    assert late["date"].tolist() == list(days("2014-03-03", "2014-03-14"))
    assert late["age"].tolist() == list(range(10))
    assert panel_of("2330", panel)["date"].iloc[0] == d("2014-02-03")


def test_age_counts_trading_days_from_the_listing_day() -> None:
    stock("1101")
    stock("6488", list_date=d("2014-03-05"))
    stock("2330", list_date=d("2013-01-02"))
    stock("3008")
    bars("1101", days("2013-06-03", "2014-03-12"))
    bars("6488", days("2014-03-05", "2014-03-12"))
    bars("2330", days("2014-03-03", "2014-03-12"))
    bars("3008", days("2014-03-10", "2014-03-12"))
    panel = load_price_panel(CALENDAR, d("2014-03-10"), d("2014-03-12"))
    since_listing = len(days("2013-01-02", "2014-03-10")) - 1
    since_calendar_start = len(days("2013-01-01", "2014-03-10")) - 1
    assert panel_of("6488", panel)["age"].tolist() == [3, 4, 5]
    assert panel_of("2330", panel)["age"].tolist() == [since_listing + n for n in range(3)]
    assert panel_of("3008", panel)["age"].tolist() == [0, 1, 2]
    assert panel_of("1101", panel)["age"].tolist() == [since_calendar_start + n for n in range(3)]


def test_rows_outside_the_listing_window_are_left_out() -> None:
    stock("6488", list_date=d("2014-03-05"), delist_date=d("2014-03-10"))
    bars("6488", days("2014-03-03", "2014-03-12"))
    panel = load_price_panel(CALENDAR, d("2014-01-01"), d("2014-12-31"))
    assert panel["date"].tolist() == list(days("2014-03-05", "2014-03-07"))


@pytest.mark.parametrize(
    ("stock_id", "industry"),
    [
        ("0050", "ETF"),
        ("9188", "存託憑證"),
        ("9151", ""),
        ("2881A", "金融保險"),
        ("910322", "存託憑證"),
        ("01001T", "受益證券"),
        ("4444", "存託憑證"),
    ],
)
def test_only_common_stocks_are_loaded(stock_id: str, industry: str) -> None:
    stock(stock_id, industry=industry)
    stock("2330")
    bars(stock_id, days("2014-03-03", "2014-03-04"))
    bars("2330", days("2014-03-03", "2014-03-04"))
    panel = load_price_panel(CALENDAR, d("2014-03-03"), d("2014-03-04"))
    assert set(panel["stock_id"]) == {"2330"}


def test_window_and_stock_filter_trim_the_output_only() -> None:
    stock("2330")
    stock("2317", market="TPEx", industry="", list_date=d("2014-01-02"))
    bars("2330", days("2014-01-02", "2014-03-31"))
    bars("2317", days("2014-01-02", "2014-03-31"))
    panel = load_price_panel(CALENDAR, d("2014-03-24"), d("2014-03-31"), stock_ids=["2317"])
    assert set(panel["stock_id"]) == {"2317"}
    assert panel["date"].tolist() == list(days("2014-03-24", "2014-03-31"))
    assert panel["age"].iloc[0] == len(days("2014-01-02", "2014-03-24")) - 1
    assert panel["market"].unique().tolist() == ["TPEx"]
    assert panel["industry"].unique().tolist() == [""]


def test_empty_range_gives_an_empty_frame_with_the_columns() -> None:
    panel = load_price_panel(CALENDAR, d("2014-03-03"), d("2014-03-07"))
    assert panel.empty
    assert list(panel.columns) == list(PANEL_COLUMNS)


def test_segments_report_first_and_last_day_per_listing() -> None:
    stock("5450")
    bars("5450", days("2013-01-02", "2013-01-10") + days("2014-06-02", "2014-06-06"))
    segments = listing_segments(CALENDAR)
    assert segments[["stock_id", "segment", "first_day", "last_day"]].to_dict("records") == [
        {
            "stock_id": "5450",
            "segment": 1,
            "first_day": d("2013-01-02"),
            "last_day": d("2013-01-10"),
        },
        {
            "stock_id": "5450",
            "segment": 2,
            "first_day": d("2014-06-02"),
            "last_day": d("2014-06-06"),
        },
    ]


def test_only_the_latest_listing_takes_the_master_industry() -> None:
    stock("5450", market="TPEx", industry="光電業")
    bars("5450", days("2013-01-02", "2013-01-10") + days("2014-06-02", "2014-06-06"))
    panel = load_price_panel(CALENDAR, d("2013-01-01"), d("2014-12-31"))
    old, new = panel[panel["segment"] == 1], panel[panel["segment"] == 2]
    assert set(old["market"]) == {"TPEx"}
    assert set(old["industry"]) == {""}
    assert set(new["market"]) == {"TPEx"}
    assert set(new["industry"]) == {"光電業"}
    assert listing_segments(CALENDAR)["industry"].tolist() == ["", "光電業"]


def test_bar_on_a_closed_day_is_refused() -> None:
    stock("2330")
    bars("2330", (d("2014-03-08"),))
    with pytest.raises(ValueError, match="not a trading day"):
        load_price_panel(CALENDAR, d("2014-03-03"), d("2014-03-10"))


def test_listings_outside_the_window_are_skipped() -> None:
    stock("5450")
    bars("5450", days("2013-01-02", "2013-01-10") + days("2014-06-02", "2014-06-06"))
    panel = load_price_panel(CALENDAR, d("2014-03-03"), d("2014-06-03"))
    assert panel["date"].tolist() == list(days("2014-06-02", "2014-06-03"))
    assert panel["segment"].unique().tolist() == [2]


def test_unsorted_or_repeated_rows_are_refused() -> None:
    frame = pd.DataFrame({"stock_id": ["2330", "2330"], "date": [d("2014-03-04"), d("2014-03-03")]})
    with pytest.raises(ValueError, match="sorted"):
        require_sorted(frame)
    with pytest.raises(ValueError, match="unique"):
        require_sorted(frame.assign(date=d("2014-03-03")))
    require_sorted(frame.iloc[::-1])
