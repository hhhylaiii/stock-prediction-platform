import math
from datetime import date, timedelta
from decimal import Decimal
from io import StringIO

import pandas as pd
import pytest
from django.core.management import call_command

from apps.etl.models import DataQualityLog
from apps.market.common_stock import DEPOSITARY_INDUSTRY
from apps.market.models import DividendEvent, StockMaster, TradingCalendar
from apps.research.features.dividend import days_to_ex_date, load_ex_dates, past_calendar
from apps.research.management.commands import build_features
from apps.research.management.commands.build_features import EX_DATE_CHECK, flag_ex_dates
from tests.research.conftest import weekday_calendar

CALENDAR = weekday_calendar(date(2016, 1, 1), date(2016, 12, 31))
Severity = DataQualityLog.Severity


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
    rows = panel([("2330", "2016-12-28"), ("1101", "2016-06-08"), ("2317", "2016-12-29")])
    found = events(
        [
            ("2330", "2016-12-20", "2017-01-05"),
            ("1101", "2016-06-15", "2016-06-10"),
            ("2317", "2016-12-20", "2016-12-31"),
        ]
    )

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


def cash_event(stock: str, ex: str, announce: str | None, cash: str = "1") -> DividendEvent:
    return DividendEvent(
        stock_id=stock,
        ex_date=d(ex),
        cash_dividend=Decimal(cash),
        stock_dividend=Decimal(0),
        announce_date=d(announce) if announce else None,
    )


@pytest.fixture
def alerts(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    sent: list[dict[str, object]] = []
    monkeypatch.setattr(
        build_features, "dispatch", lambda rows: sent.extend(row.detail for row in rows)
    )
    return sent


@pytest.mark.django_db
def test_the_ex_dates_listed_as_past_the_calendar_are_the_ones_the_feature_drops() -> None:
    DividendEvent.objects.bulk_create(
        [
            cash_event("2330", "2017-01-05", "2016-12-20"),
            cash_event("1101", "2016-12-31", "2016-12-20"),
            cash_event("2317", "2016-12-30", "2016-12-20"),
            cash_event("2454", "2017-01-06", "2016-12-20", cash="0"),
            cash_event("2412", "2017-01-09", None),
            cash_event("3008", "2017-01-10", "2016-12-29"),
            cash_event("0050", "2017-01-11", "2016-12-20"),
            cash_event("00878", "2017-01-12", "2016-12-20"),
            cash_event("9105", "2017-01-13", "2016-12-20"),
            cash_event("1262", "2017-01-16", "2016-12-20"),
        ]
    )
    StockMaster.objects.create(
        stock_id="1262", name="存託憑證", market="TWSE", industry=DEPOSITARY_INDUSTRY
    )
    day = "2016-12-28"

    assert past_calendar(d(day), CALENDAR) == [d("2016-12-31"), d("2017-01-05")]

    rows = panel([("2330", day), ("1101", day), ("2317", day)])
    counted = days_to_ex_date(rows, load_ex_dates(d("2016-12-01"), d(day)), CALENDAR)
    assert counted.isna().tolist() == [True, True, False]
    assert counted.iloc[2] == 2.0


@pytest.mark.django_db
def test_ex_dates_past_the_calendar_make_one_warning_a_day_and_alert_only_when_it_changes(
    alerts: list[dict[str, object]],
) -> None:
    day = d("2016-12-28")
    DividendEvent.objects.bulk_create([cash_event("2330", "2017-01-05", "2016-12-20")])
    first = {"count": 1, "latest_ex_date": "2017-01-05", "calendar_end": "2016-12-31"}
    second = {"count": 2, "latest_ex_date": "2017-01-12", "calendar_end": "2016-12-31"}

    assert flag_ex_dates(day, CALENDAR) == first
    assert flag_ex_dates(day, CALENDAR) == first
    DividendEvent.objects.bulk_create([cash_event("1101", "2017-01-12", "2016-12-27")])
    assert flag_ex_dates(day, CALENDAR) == second

    row = DataQualityLog.objects.get()
    assert (row.date, row.check_name, row.severity) == (day, EX_DATE_CHECK, Severity.WARNING)
    assert row.detail == second
    assert alerts == [first, second]


@pytest.mark.django_db
def test_the_next_session_writes_its_own_row_and_alerts_only_on_news(
    alerts: list[dict[str, object]],
) -> None:
    DividendEvent.objects.bulk_create([cash_event("2330", "2017-01-05", "2016-12-20")])
    first = {"count": 1, "latest_ex_date": "2017-01-05", "calendar_end": "2016-12-31"}
    second = {"count": 2, "latest_ex_date": "2017-01-12", "calendar_end": "2016-12-31"}

    flag_ex_dates(d("2016-12-26"), CALENDAR)
    flag_ex_dates(d("2016-12-27"), CALENDAR)
    assert alerts == [first]
    DividendEvent.objects.bulk_create([cash_event("1101", "2017-01-12", "2016-12-28")])
    flag_ex_dates(d("2016-12-28"), CALENDAR)

    assert alerts == [first, second]
    rows = DataQualityLog.objects.order_by("date").values_list("date", "detail")
    assert list(rows) == [
        (d("2016-12-26"), first),
        (d("2016-12-27"), first),
        (d("2016-12-28"), second),
    ]


@pytest.mark.django_db
def test_the_same_warning_alerts_again_after_a_session_without_a_row(
    alerts: list[dict[str, object]],
) -> None:
    DividendEvent.objects.bulk_create([cash_event("2330", "2017-01-05", "2016-12-20")])
    first = {"count": 1, "latest_ex_date": "2017-01-05", "calendar_end": "2016-12-31"}

    flag_ex_dates(d("2016-12-23"), CALENDAR)
    flag_ex_dates(d("2016-12-26"), CALENDAR)
    flag_ex_dates(d("2016-12-28"), CALENDAR)

    assert alerts == [first, first]


@pytest.mark.django_db
def test_a_rerun_once_the_calendar_reaches_the_ex_dates_removes_the_day_s_warning(
    alerts: list[dict[str, object]],
) -> None:
    day = d("2016-12-28")
    DividendEvent.objects.bulk_create([cash_event("2330", "2017-01-05", "2016-12-20")])
    other = DataQualityLog.objects.create(
        date=day, check_name="research_evening", severity=Severity.INFO, detail={}
    )
    earlier = DataQualityLog.objects.create(
        date=d("2016-12-27"), check_name=EX_DATE_CHECK, severity=Severity.WARNING, detail={}
    )
    flag_ex_dates(day, CALENDAR)

    longer = weekday_calendar(date(2016, 1, 1), date(2017, 12, 31))
    assert flag_ex_dates(day, longer) is None

    assert set(DataQualityLog.objects.values_list("pk", flat=True)) == {other.pk, earlier.pk}
    assert len(alerts) == 1


@pytest.mark.django_db
def test_no_ex_date_past_the_calendar_writes_nothing(alerts: list[dict[str, object]]) -> None:
    DividendEvent.objects.bulk_create([cash_event("2330", "2016-12-30", "2016-12-20")])

    assert flag_ex_dates(d("2016-12-28"), CALENDAR) is None

    assert not DataQualityLog.objects.exists()
    assert alerts == []


@pytest.mark.django_db
def test_the_command_flags_the_ex_dates_past_the_calendar_on_the_last_day_it_built(
    monkeypatch: pytest.MonkeyPatch, alerts: list[dict[str, object]]
) -> None:
    first = date(2016, 12, 1)
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=day.weekday() < 5)
        for day in (first + timedelta(days=n) for n in range(31))
    )
    DividendEvent.objects.bulk_create([cash_event("2330", "2017-01-05", "2016-12-20")])
    monkeypatch.setattr(build_features, "latest_price_date", lambda: d("2016-12-28"))
    monkeypatch.setattr(build_features, "build_feature_frame", lambda *args: pd.DataFrame())
    monkeypatch.setattr(build_features, "write_features", lambda frame, low, high: 7)
    monkeypatch.setattr(build_features, "export_year", lambda *args, **kwargs: None)
    out = StringIO()

    call_command("build_features", start=d("2016-12-26"), end=d("2016-12-28"), stdout=out)

    row = DataQualityLog.objects.get(check_name=EX_DATE_CHECK)
    assert row.date == d("2016-12-28")
    assert row.detail == {"count": 1, "latest_ex_date": "2017-01-05", "calendar_end": "2016-12-31"}
    assert out.getvalue().strip() == (
        "feature_daily: 7 rows for 2016-12-26..2016-12-28; "
        f"1 announced ex-date(s) after the trading calendar, see {EX_DATE_CHECK}"
    )
    assert len(alerts) == 1


@pytest.mark.django_db
def test_the_command_s_output_is_unchanged_when_the_calendar_reaches_every_ex_date(
    monkeypatch: pytest.MonkeyPatch, alerts: list[dict[str, object]]
) -> None:
    first = date(2016, 12, 1)
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=day.weekday() < 5)
        for day in (first + timedelta(days=n) for n in range(31))
    )
    monkeypatch.setattr(build_features, "latest_price_date", lambda: d("2016-12-28"))
    monkeypatch.setattr(build_features, "build_feature_frame", lambda *args: pd.DataFrame())
    monkeypatch.setattr(build_features, "write_features", lambda frame, low, high: 7)
    monkeypatch.setattr(build_features, "export_year", lambda *args, **kwargs: None)
    out = StringIO()

    call_command("build_features", date=d("2016-12-28"), stdout=out)

    assert out.getvalue().strip() == "feature_daily: 7 rows for 2016-12-28..2016-12-28"
    assert not DataQualityLog.objects.exists()
