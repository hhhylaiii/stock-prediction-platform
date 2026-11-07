import copy
from collections.abc import Callable, Iterable
from datetime import date, timedelta
from io import StringIO
from typing import Any

import pytest
from django.core.management import call_command
from django.utils import timezone

from apps.etl.market_history import REBUILT_CHECK, rebuild_market_history
from apps.etl.models import DataQualityLog, RawTpex, RawTwse
from apps.market.calendar import TradingDays
from apps.market.models import Market, MarketHistory, StockMaster, TradingCalendar

pytestmark = pytest.mark.django_db

TWSE_PAGE = "twse/foreign_holding/mi_qfiis_20261028.json"
TPEX_PAGE = "tpex/foreign_holding/qfii_20261028.json"
DAYS = [date(2013, 7, 29), date(2013, 7, 30), date(2013, 7, 31), date(2013, 8, 1)]
CALENDAR = TradingDays([*DAYS, date(2013, 8, 2)], DAYS[0], date(2013, 8, 2))
TWSE = Market.TWSE.value
TPEX = Market.TPEX.value


def _twse_page(
    load_fixture: Callable[[str], Any], day: date, codes: Iterable[str], version: int = 0
) -> int:
    payload = copy.deepcopy(load_fixture(TWSE_PAGE))
    template = payload["data"][2]
    payload["date"] = f"{day:%Y%m%d}"
    payload["data"] = [[code, code, *template[2:]] for code in codes]
    payload["total"] = len(payload["data"])
    params = {"date": f"{day:%Y%m%d}", "selectType": "ALLBUT0999", "response": "json"}
    raw = RawTwse.objects.create(
        dataset="foreign_holding",
        request_params=params,
        payload=payload,
        source_hash=f"{day}/{version}",
    )
    return raw.pk


def _tpex_page(load_fixture: Callable[[str], Any], day: date, codes: Iterable[str]) -> int:
    payload = copy.deepcopy(load_fixture(TPEX_PAGE))
    payload["date"] = f"{day:%Y%m%d}"
    table = payload["tables"][0]
    template = table["data"][0]
    table["date"] = f"{day.year - 1911}/{day:%m/%d}"
    table["data"] = [[str(rank), code, code, *template[3:]] for rank, code in enumerate(codes, 1)]
    table["totalCount"] = len(table["data"])
    params = {"date": f"{day:%Y/%m/%d}", "response": "json"}
    raw = RawTpex.objects.create(
        dataset="foreign_holding", request_params=params, payload=payload, source_hash=str(day)
    )
    return raw.pk


def _master(*codes: str) -> None:
    StockMaster.objects.bulk_create(
        StockMaster(stock_id=code, name=code, market=TWSE) for code in codes
    )


def _spans() -> list[tuple[str, date, date | None, str]]:
    rows = MarketHistory.objects.order_by("stock_id", "start_date")
    return list(rows.values_list("stock_id", "start_date", "end_date", "market"))


@pytest.fixture
def transfer(load_fixture: Callable[[str], Any]) -> None:
    _master("2330", "6177", "5555")
    MarketHistory.objects.create(stock_id="5555", start_date=DAYS[0], market=TPEX)
    for day in DAYS:
        on_tpex = ["3293", "5555", *(["6177"] if day <= DAYS[2] else [])]
        on_twse = ["2330", "1101", "5555", *(["6177"] if day >= DAYS[2] else [])]
        _twse_page(load_fixture, day, on_twse)
        _tpex_page(load_fixture, day, on_tpex)


@pytest.mark.usefixtures("transfer")
def test_spans_follow_the_holding_pages_and_the_shared_day_stays_on_the_old_market() -> None:
    report = rebuild_market_history(CALENDAR)
    assert _spans() == [
        ("2330", DAYS[0], None, TWSE),
        ("5555", DAYS[0], None, TPEX),
        ("6177", DAYS[0], date(2013, 7, 31), TPEX),
        ("6177", DAYS[3], None, TWSE),
    ]
    assert (report.spans, report.stocks, report.moved) == (3, 2, ["6177"])
    assert (report.shared_days, report.unsettled) == (5, ["5555"])
    assert report.pages == {"twse": 4, "tpex": 4}
    assert report.missing == {"twse": [], "tpex": []}


@pytest.mark.usefixtures("transfer")
def test_a_second_rebuild_writes_the_same_spans() -> None:
    rebuild_market_history(CALENDAR)
    first = _spans()
    rebuild_market_history(CALENDAR)
    assert _spans() == first


@pytest.mark.usefixtures("transfer")
def test_spans_of_listed_codes_are_replaced_and_others_kept() -> None:
    MarketHistory.objects.create(stock_id="6177", start_date=DAYS[1], market=TWSE)
    MarketHistory.objects.create(stock_id="9999", start_date=DAYS[0], market=TPEX)
    rebuild_market_history(CALENDAR)
    assert MarketHistory.objects.filter(stock_id="6177").count() == 2
    assert list(MarketHistory.objects.filter(stock_id="9999").values_list("market", flat=True)) == [
        TPEX
    ]


def test_the_latest_page_of_a_day_is_read(load_fixture: Callable[[str], Any]) -> None:
    _master("6177")
    _tpex_page(load_fixture, DAYS[0], ["6177"])
    _twse_page(load_fixture, DAYS[0], ["2330"])
    _tpex_page(load_fixture, DAYS[1], ["3293"])
    _twse_page(load_fixture, DAYS[1], ["6177"])
    older = _twse_page(load_fixture, DAYS[1], ["2330"], version=1)
    RawTwse.objects.filter(pk=older).update(fetched_at=timezone.now() - timedelta(hours=4))
    rebuild_market_history(CALENDAR)
    assert _spans() == [
        ("6177", DAYS[0], DAYS[0], TPEX),
        ("6177", DAYS[1], None, TWSE),
    ]


def test_an_unreadable_page_is_skipped_and_reported(
    load_fixture: Callable[[str], Any], caplog: pytest.LogCaptureFixture
) -> None:
    _master("2330")
    for day in DAYS[:2]:
        _tpex_page(load_fixture, day, ["3293"])
    _twse_page(load_fixture, DAYS[0], ["2330"])
    broken = _twse_page(load_fixture, DAYS[1], ["2330"])
    payload = RawTwse.objects.get(pk=broken).payload
    payload["fields"] = [f"{name}x" for name in payload["fields"]]
    RawTwse.objects.filter(pk=broken).update(payload=payload)
    report = rebuild_market_history(CALENDAR)
    assert report.unreadable == [("twse", DAYS[1])]
    assert report.pages == {"twse": 1, "tpex": 2}
    assert _spans() == [("2330", DAYS[0], None, TWSE)]
    assert str(broken) in caplog.text


@pytest.mark.usefixtures("transfer")
def test_the_command_rebuilds_and_prints_the_report() -> None:
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=is_open) for day, is_open in CALENDAR.all_days()
    )
    out = StringIO()
    call_command("rebuild_market_history", stdout=out)
    assert MarketHistory.objects.count() == 4
    text = out.getvalue()
    assert "3 spans for 2 stocks" in text
    assert "moved: 6177" in text


def test_trading_days_without_a_page_are_reported(load_fixture: Callable[[str], Any]) -> None:
    _master("2330")
    for day in (DAYS[0], DAYS[3]):
        _twse_page(load_fixture, day, ["2330"])
    _tpex_page(load_fixture, DAYS[1], ["3293"])
    report = rebuild_market_history(CALENDAR)
    assert report.missing == {"twse": DAYS[1:3], "tpex": [DAYS[0], DAYS[2], DAYS[3]]}


@pytest.mark.usefixtures("transfer")
def test_spans_a_rebuild_changes_are_logged_and_an_unchanged_rebuild_logs_nothing() -> None:
    MarketHistory.objects.create(stock_id="6177", start_date=DAYS[0], market=TPEX)
    MarketHistory.objects.create(stock_id="2330", start_date=DAYS[1], market=TWSE)
    rebuild_market_history(CALENDAR)
    rebuild_market_history(CALENDAR)
    (log,) = DataQualityLog.objects.filter(check_name=REBUILT_CHECK)
    assert (log.severity, log.detail["changed"], log.detail["start_shifted"]) == ("info", 1, 1)
    assert log.detail["sample"] == [
        [
            "6177",
            [["2013-07-29", None, TPEX]],
            [["2013-07-29", "2013-07-31", TPEX], ["2013-08-01", None, TWSE]],
        ]
    ]


def test_a_day_with_one_exchanges_page_does_not_end_a_span(
    load_fixture: Callable[[str], Any],
) -> None:
    _master("6177")
    for day in DAYS[:3]:
        _tpex_page(load_fixture, day, ["6177"])
    for day in DAYS[2:]:
        _twse_page(load_fixture, day, ["6177"])
    _twse_page(load_fixture, DAYS[0], ["2330"])
    report = rebuild_market_history(CALENDAR)
    assert _spans() == [("6177", DAYS[0], None, TPEX)]
    assert report.missing == {"twse": [DAYS[1]], "tpex": [DAYS[3]]}


@pytest.mark.usefixtures("transfer")
def test_a_stored_move_after_the_last_page_day_is_kept() -> None:
    MarketHistory.objects.create(stock_id="2330", start_date=DAYS[0], end_date=DAYS[3], market=TWSE)
    MarketHistory.objects.create(stock_id="2330", start_date=date(2013, 8, 2), market=TPEX)
    report = rebuild_market_history(CALENDAR)
    assert [row for row in _spans() if row[0] == "2330"] == [
        ("2330", DAYS[0], DAYS[3], TWSE),
        ("2330", date(2013, 8, 2), None, TPEX),
    ]
    assert "2330" not in report.changed
