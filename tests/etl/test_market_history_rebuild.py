import copy
from collections.abc import Callable, Iterable
from datetime import date, timedelta
from io import StringIO
from typing import Any

import pytest
from django.core.management import call_command
from django.utils import timezone

from apps.etl.market_history import rebuild_market_history
from apps.etl.models import RawTpex, RawTwse
from apps.market.models import Market, MarketHistory, StockMaster

pytestmark = pytest.mark.django_db

TWSE_PAGE = "twse/foreign_holding/mi_qfiis_20261028.json"
TPEX_PAGE = "tpex/foreign_holding/qfii_20261028.json"
DAYS = [date(2013, 7, 29), date(2013, 7, 30), date(2013, 7, 31), date(2013, 8, 1)]
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
    _master("2330", "6177")
    for day in DAYS:
        on_tpex = ["3293", *(["6177"] if day <= DAYS[2] else [])]
        on_twse = ["2330", "1101", *(["6177"] if day >= DAYS[2] else [])]
        _twse_page(load_fixture, day, on_twse)
        _tpex_page(load_fixture, day, on_tpex)


@pytest.mark.usefixtures("transfer")
def test_spans_follow_the_holding_pages_and_the_shared_day_stays_on_the_old_market() -> None:
    report = rebuild_market_history()
    assert _spans() == [
        ("2330", DAYS[0], None, TWSE),
        ("6177", DAYS[0], date(2013, 7, 31), TPEX),
        ("6177", DAYS[3], None, TWSE),
    ]
    assert (report.spans, report.stocks, report.moved, report.shared_days) == (3, 2, ["6177"], 1)
    assert report.pages == {"twse": 4, "tpex": 4}


@pytest.mark.usefixtures("transfer")
def test_a_second_rebuild_writes_the_same_spans() -> None:
    rebuild_market_history()
    first = _spans()
    rebuild_market_history()
    assert _spans() == first


@pytest.mark.usefixtures("transfer")
def test_spans_of_listed_codes_are_replaced_and_others_kept() -> None:
    MarketHistory.objects.create(stock_id="6177", start_date=DAYS[1], market=TWSE)
    MarketHistory.objects.create(stock_id="9999", start_date=DAYS[0], market=TPEX)
    rebuild_market_history()
    assert MarketHistory.objects.filter(stock_id="6177").count() == 2
    assert list(MarketHistory.objects.filter(stock_id="9999").values_list("market", flat=True)) == [
        TPEX
    ]


def test_the_latest_page_of_a_day_is_read(load_fixture: Callable[[str], Any]) -> None:
    _master("6177")
    _tpex_page(load_fixture, DAYS[0], ["6177"])
    _twse_page(load_fixture, DAYS[1], ["6177"])
    older = _twse_page(load_fixture, DAYS[1], [], version=1)
    RawTwse.objects.filter(pk=older).update(fetched_at=timezone.now() - timedelta(hours=4))
    rebuild_market_history()
    assert _spans() == [
        ("6177", DAYS[0], DAYS[0], TPEX),
        ("6177", DAYS[1], None, TWSE),
    ]


def test_an_unreadable_page_is_skipped_and_reported(
    load_fixture: Callable[[str], Any], caplog: pytest.LogCaptureFixture
) -> None:
    _master("2330")
    _twse_page(load_fixture, DAYS[0], ["2330"])
    broken = _twse_page(load_fixture, DAYS[1], ["2330"])
    payload = RawTwse.objects.get(pk=broken).payload
    payload["fields"] = [f"{name}x" for name in payload["fields"]]
    RawTwse.objects.filter(pk=broken).update(payload=payload)
    report = rebuild_market_history()
    assert report.unreadable == [("twse", DAYS[1])]
    assert report.pages == {"twse": 1, "tpex": 0}
    assert _spans() == [("2330", DAYS[0], None, TWSE)]
    assert str(broken) in caplog.text


@pytest.mark.usefixtures("transfer")
def test_the_command_rebuilds_and_prints_the_report() -> None:
    out = StringIO()
    call_command("rebuild_market_history", stdout=out)
    assert MarketHistory.objects.count() == 3
    text = out.getvalue()
    assert "3 spans for 2 stocks" in text
    assert "moved: 6177" in text
