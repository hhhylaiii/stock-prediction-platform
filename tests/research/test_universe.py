from datetime import date

import numpy as np
import pandas as pd
import pytest

from apps.market.models import DailyPrice, DispositionEvent, StockMaster
from apps.research.universe import (
    EXCLUSIONS,
    MIN_MEDIAN_TURNOVER,
    apply_universe,
    event_days,
    load_disposition_events,
    load_universe_panel,
)
from tests.research.conftest import weekday_calendar

CALENDAR = weekday_calendar(date(2014, 1, 1), date(2014, 12, 31))
DAYS = CALENDAR.between(date(2014, 3, 3), date(2014, 4, 11))
NO_EVENTS = pd.DataFrame(columns=["stock_id", "type", "start_date", "end_date"])


def d(text: str) -> date:
    return date.fromisoformat(text)


def panel(stock_id: str = "2330", **columns: object) -> pd.DataFrame:
    """Thirty liquid, long-listed trading days; keyword columns override the defaults."""
    n = len(DAYS)
    base: dict[str, object] = {
        "stock_id": stock_id,
        "date": list(DAYS),
        "segment": 1,
        "age": np.arange(n) + 500,
        "has_bar": True,
        "close": 10.0,
        "turnover": 2e7,
    }
    return pd.DataFrame({**base, **columns})


def events(*rows: tuple[str, str, str, str | None]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            (stock, kind, d(start), None if end is None else d(end))
            for stock, kind, start, end in rows
        ],
        columns=["stock_id", "type", "start_date", "end_date"],
    )


def excluded(frame: pd.DataFrame, flag: str) -> list[date]:
    return list(frame.loc[frame[flag], "date"])


def test_a_liquid_listed_stock_is_in_the_universe_after_twenty_days() -> None:
    result = apply_universe(panel(), NO_EVENTS, CALENDAR)
    assert result.loc[19:, "in_universe"].all()
    assert not result.loc[:18, "in_universe"].any()
    assert result.loc[:18, "illiquid"].all()
    assert set(EXCLUSIONS) <= set(result.columns)


def test_halted_day_is_excluded() -> None:
    has_bar = [day != DAYS[25] for day in DAYS]
    close = [10.0 if bar else np.nan for bar in has_bar]
    result = apply_universe(panel(has_bar=has_bar, close=close), NO_EVENTS, CALENDAR)
    assert excluded(result, "halted") == [DAYS[25]]
    assert excluded(result, "no_trade") == []
    assert not result.loc[25, "in_universe"]


def test_day_without_a_trade_is_excluded() -> None:
    close = [np.nan if day == DAYS[25] else 10.0 for day in DAYS]
    result = apply_universe(panel(close=close), NO_EVENTS, CALENDAR)
    assert excluded(result, "no_trade") == [DAYS[25]]
    assert excluded(result, "halted") == []
    assert not result.loc[25, "in_universe"]


def test_first_sixty_trading_days_after_listing_are_excluded() -> None:
    age = np.arange(len(DAYS)) + 40
    result = apply_universe(panel(age=age), NO_EVENTS, CALENDAR)
    assert excluded(result, "new_listing") == list(DAYS[:20])
    assert result.loc[20:, "in_universe"].all()


@pytest.mark.parametrize(
    ("kind", "flag"), [("disposition", "disposition"), ("full_delivery", "full_delivery")]
)
def test_disposition_and_full_delivery_periods_are_excluded(kind: str, flag: str) -> None:
    periods = events(("2330", kind, "2014-03-31", "2014-04-02"), ("2317", kind, "2014-03-03", None))
    result = apply_universe(panel(), periods, CALENDAR)
    assert excluded(result, flag) == [d("2014-03-31"), d("2014-04-01"), d("2014-04-02")]
    assert not result.loc[result[flag], "in_universe"].any()


def test_open_ended_period_runs_through_the_last_day() -> None:
    periods = events(("2330", "full_delivery", "2014-04-09", None))
    result = apply_universe(panel(), periods, CALENDAR)
    assert excluded(result, "full_delivery") == [d("2014-04-09"), d("2014-04-10"), d("2014-04-11")]


def test_attention_stocks_stay_in_the_universe_with_a_flag() -> None:
    result = apply_universe(
        panel(), events(("2330", "attention", "2014-04-07", "2014-04-07")), CALENDAR
    )
    assert excluded(result, "attention") == [d("2014-04-07")]
    assert result.loc[result["date"] == d("2014-04-07"), "in_universe"].item()
    assert "attention" not in EXCLUSIONS


def test_median_turnover_below_ten_million_is_excluded() -> None:
    turnover = [MIN_MEDIAN_TURNOVER - 1] * 10 + [MIN_MEDIAN_TURNOVER] * 20
    result = apply_universe(panel(turnover=turnover), NO_EVENTS, CALENDAR)
    assert result.loc[19, "turnover_med_20d"] == pytest.approx(MIN_MEDIAN_TURNOVER - 0.5)
    assert result.loc[19, "illiquid"]
    assert result.loc[20, "turnover_med_20d"] == MIN_MEDIAN_TURNOVER
    assert not result.loc[20, "illiquid"]


def test_halted_days_count_as_zero_turnover() -> None:
    has_bar = [i < 15 or i >= 26 for i in range(len(DAYS))]
    turnover = [2e7 if bar else np.nan for bar in has_bar]
    close = [10.0 if bar else np.nan for bar in has_bar]
    result = apply_universe(
        panel(has_bar=has_bar, close=close, turnover=turnover), NO_EVENTS, CALENDAR
    )
    assert result.loc[27, "turnover_med_20d"] == 0
    assert result.loc[27, "illiquid"]


def test_turnover_window_restarts_at_a_new_listing() -> None:
    segment = [1] * 20 + [2] * 10
    result = apply_universe(panel(segment=segment), NO_EVENTS, CALENDAR)
    assert result.loc[19, "turnover_med_20d"] == 2e7
    assert result.loc[20:, "turnover_med_20d"].isna().all()


def test_stocks_do_not_share_flags_or_windows() -> None:
    frame = pd.concat([panel("2330"), panel("2317", turnover=1.0)], ignore_index=True)
    periods = events(("2317", "disposition", "2014-03-03", "2014-04-11"))
    result = apply_universe(frame, periods, CALENDAR)
    assert result.loc[result["stock_id"] == "2330", "disposition"].sum() == 0
    assert result.loc[(result["stock_id"] == "2330") & (result.index >= 19), "in_universe"].all()
    assert not result.loc[result["stock_id"] == "2317", "in_universe"].any()


def test_input_panel_is_not_modified() -> None:
    frame = panel()
    before = frame.copy()
    apply_universe(frame, NO_EVENTS, CALENDAR)
    pd.testing.assert_frame_equal(frame, before)


def test_event_days_clamp_to_the_calendar_and_last_day() -> None:
    days = event_days(
        events(
            ("2330", "disposition", "2013-12-30", "2014-01-02"),
            ("2330", "attention", "2014-12-30", None),
        ),
        CALENDAR,
        last_day=d("2014-12-31"),
    )
    assert days["date"].tolist() == [
        d("2014-01-01"),
        d("2014-01-02"),
        d("2014-12-30"),
        d("2014-12-31"),
    ]


@pytest.mark.django_db
def test_load_disposition_events_keeps_only_overlapping_periods() -> None:
    DispositionEvent.objects.bulk_create(
        [
            DispositionEvent(
                stock_id="2330",
                type="disposition",
                start_date=d("2014-03-03"),
                end_date=d("2014-03-14"),
            ),
            DispositionEvent(
                stock_id="2330",
                type="attention",
                start_date=d("2014-05-02"),
                end_date=d("2014-05-02"),
            ),
            DispositionEvent(
                stock_id="2317", type="full_delivery", start_date=d("2013-01-02"), end_date=None
            ),
            DispositionEvent(
                stock_id="2454",
                type="disposition",
                start_date=d("2014-01-02"),
                end_date=d("2014-01-10"),
            ),
        ]
    )
    loaded = load_disposition_events(d("2014-03-10"), d("2014-04-30"))
    assert sorted(loaded["stock_id"]) == ["2317", "2330"]
    assert list(loaded.columns) == ["stock_id", "type", "start_date", "end_date"]


def test_events_entirely_before_the_calendar_are_ignored() -> None:
    days = event_days(
        events(("2330", "disposition", "2012-01-02", "2012-02-01")), CALENDAR, d("2014-12-31")
    )
    assert days.empty


def test_unsorted_panel_is_refused() -> None:
    with pytest.raises(ValueError, match="sorted"):
        apply_universe(panel().iloc[::-1], NO_EVENTS, CALENDAR)


LONG = weekday_calendar(date(2013, 1, 1), date(2014, 12, 31))
CHECKED = ["in_universe", "illiquid", "turnover_med_20d", "reference", "limit_up", "limit_up_close"]


def seed_prices() -> None:
    StockMaster.objects.create(stock_id="2330", name="2330", market="TWSE", industry="半導體業")
    trading = LONG.between(d("2013-01-02"), d("2014-06-30"))
    closes = [100.0 * 1.001**i for i in range(len(trading))]
    closes[-10] = round(closes[-11] * 1.1 - 0.4, 0)
    DailyPrice.objects.bulk_create(
        DailyPrice(
            stock_id="2330",
            date=day,
            open=close,
            high=close,
            low=close,
            close=close,
            volume=100_000,
            turnover=int(close * 100_000) if i % 7 else 1_000,
            adj_factor=0.9 if day < d("2014-04-01") else 1.0,
        )
        for i, (day, close) in enumerate(zip(trading, closes, strict=True))
    )


@pytest.mark.django_db
def test_flags_do_not_depend_on_where_the_window_starts() -> None:
    seed_prices()
    full = load_universe_panel(LONG, d("2013-01-01"), d("2014-06-30"))
    late = load_universe_panel(LONG, d("2014-05-05"), d("2014-06-30"))
    expected = full.loc[full["date"] >= d("2014-05-05"), CHECKED].reset_index(drop=True)
    assert late["date"].iloc[0] == d("2014-05-05")
    pd.testing.assert_frame_equal(late[CHECKED], expected)
    assert late["in_universe"].all()
    assert late["limit_up_close"].sum() == 1


@pytest.mark.django_db
def test_window_near_the_calendar_start_warms_up_on_what_exists() -> None:
    seed_prices()
    early = load_universe_panel(LONG, d("2013-02-01"), d("2013-03-29"))
    assert early["date"].iloc[0] == d("2013-02-01")
    assert early["in_universe"].iloc[0] == (early["age"].iloc[0] >= 60)


@pytest.mark.django_db
def test_window_without_trading_days_is_empty() -> None:
    seed_prices()
    assert load_universe_panel(LONG, d("2014-07-05"), d("2014-07-06")).empty
