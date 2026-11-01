from collections.abc import Callable
from datetime import date
from typing import Any

import pytest
import responses
from django.core.management import call_command
from django.core.management.base import CommandError
from responses import matchers

from apps.etl.adapters import limiter_for
from apps.etl.adapters.tpex import TpexClient
from apps.etl.adapters.twse import TwseClient
from apps.etl.errors import SchemaChangedError
from apps.etl.listing import Listing, first_listed_day
from apps.etl.models import EtlRun
from apps.market.models import (
    DailyChip,
    DailyPrice,
    DailyValuation,
    Market,
    ShareChangeReason,
    SharesOutstanding,
    StockMaster,
)

TWSE_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
TPEX_URL = "https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock"
EMPTY_TPEX = "tpex/trading_stock/6488_201508"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)


@pytest.fixture
def twse_month(load_fixture: Callable[[str], Any]) -> Callable[[str, str, str], None]:
    def _mock(stock: str, month: str, name: str) -> None:
        query = {"date": f"{month}01", "stockNo": stock, "response": "json"}
        responses.get(
            TWSE_URL,
            json=load_fixture(f"{name}.json"),
            match=[matchers.query_param_matcher(query)],
        )

    return _mock


@pytest.fixture
def tpex_month(load_fixture: Callable[[str], Any]) -> Callable[[str, str, str], None]:
    def _mock(stock: str, month: str, name: str) -> None:
        form = {"code": stock, "date": f"{month[:4]}/{month[4:]}/01", "response": "json"}
        responses.post(
            TPEX_URL,
            json=load_fixture(f"{name}.json"),
            match=[matchers.urlencoded_params_matcher(form)],
        )

    return _mock


@responses.activate
@pytest.mark.django_db
def test_twse_months_list_their_trading_days(twse_month: Callable[..., None]) -> None:
    twse_month("6446", "202401", "twse/stock_day/6446_202401")
    twse_month("6446", "202312", "twse/stock_day/6446_202312")
    client = TwseClient.build(limiter_for("twse"))
    days = client.stock_month("6446", date(2024, 1, 1))
    assert days[:2] == [date(2024, 1, 25), date(2024, 1, 26)]
    assert client.stock_month("6446", date(2023, 12, 1)) == []


@responses.activate
@pytest.mark.django_db
def test_an_unexpected_twse_answer_is_a_schema_change() -> None:
    responses.get(TWSE_URL, json={"stat": "查詢日期大於今日，請重新查詢!"})
    with pytest.raises(SchemaChangedError, match="查詢日期"):
        TwseClient.build(limiter_for("twse")).stock_month("6446", date(2030, 1, 1))


@responses.activate
@pytest.mark.django_db
def test_tpex_months_drop_the_first_day_marker(tpex_month: Callable[..., None]) -> None:
    tpex_month("6488", "201509", "tpex/trading_stock/6488_201509")
    tpex_month("6488", "201508", EMPTY_TPEX)
    client = TpexClient.build(limiter_for("tpex"))
    assert client.stock_month("6488", date(2015, 9, 1)) == [date(2015, 9, 25), date(2015, 9, 30)]
    assert client.stock_month("6488", date(2015, 8, 1)) == []


def _probe_from(first: date, asked: list[date]) -> Callable[[date], list[date]]:
    def probe(month: date) -> list[date]:
        asked.append(month)
        return [max(month, first)] if month >= first.replace(day=1) else []

    return probe


def test_a_stock_also_trading_the_month_before_the_hint_was_listed_earlier() -> None:
    asked: list[date] = []
    probe = _probe_from(date(2008, 3, 1), asked)
    found = first_listed_day(date(2010, 1, 4), date(2026, 10, 30), probe)
    assert found == Listing(None, before_hint=True)
    assert asked == [date(2010, 1, 1), date(2009, 12, 1)]


def test_a_stock_first_trading_in_the_hint_month_is_listed_then() -> None:
    probe = _probe_from(date(2012, 3, 5), [])
    found = first_listed_day(date(2012, 3, 5), date(2026, 10, 30), probe)
    assert found == Listing(date(2012, 3, 5))


def test_emerging_market_months_are_skipped_with_few_probes() -> None:
    asked: list[date] = []
    probe = _probe_from(date(2015, 9, 25), asked)
    found = first_listed_day(date(2014, 10, 28), date(2026, 10, 30), probe)
    assert found == Listing(date(2015, 9, 25))
    assert len(asked) <= 10


def test_no_official_month_up_to_the_last_price_gives_none() -> None:
    asked: list[date] = []
    probe = _probe_from(date(2030, 1, 1), asked)
    assert first_listed_day(date(2014, 10, 28), date(2016, 3, 1), probe) == Listing(None)
    assert asked[-1] == date(2016, 3, 1)


def _prices(stock: str, *days: date) -> None:
    DailyPrice.objects.bulk_create(
        DailyPrice(stock_id=stock, date=day, close=10, volume=1, turnover=10) for day in days
    )


@responses.activate
@pytest.mark.django_db
def test_listing_dates_drop_earlier_market_rows_but_keep_shares(
    tpex_month: Callable[..., None],
) -> None:
    StockMaster.objects.create(stock_id="6488", name="環球晶", market=Market.TPEX)
    _prices("6488", date(2015, 8, 31), date(2015, 9, 25), date(2015, 9, 30))
    DailyChip.objects.create(stock_id="6488", date=date(2015, 8, 31), fi_net=1)
    DailyValuation.objects.create(stock_id="6488", date=date(2015, 8, 31))
    SharesOutstanding.objects.create(
        stock_id="6488",
        effective_date=date(2015, 8, 31),
        shares=1,
        reason=ShareChangeReason.UNKNOWN,
    )
    tpex_month("6488", "201508", EMPTY_TPEX)
    tpex_month("6488", "201509", "tpex/trading_stock/6488_201509")
    call_command("load_listing_dates", "--stock", "6488")
    assert StockMaster.objects.get(stock_id="6488").list_date == date(2015, 9, 25)
    assert list(DailyPrice.objects.values_list("date", flat=True).order_by("date")) == [
        date(2015, 9, 25),
        date(2015, 9, 30),
    ]
    assert not DailyChip.objects.exists()
    assert not DailyValuation.objects.exists()
    assert SharesOutstanding.objects.count() == 1
    run = EtlRun.objects.get(dataset="list_date")
    assert (run.status, run.rows) == ("success", 1)


@responses.activate
@pytest.mark.django_db
def test_a_listed_stock_that_moved_from_tpex_keeps_its_tpex_years(
    twse_month: Callable[..., None], tpex_month: Callable[..., None]
) -> None:
    StockMaster.objects.create(stock_id="6446", name="藥華藥", market=Market.TWSE)
    _prices("6446", date(2016, 9, 1))
    twse_month("6446", "201609", "twse/stock_day/6446_202312")
    tpex_month("6446", "201609", "tpex/trading_stock/6446_201609")
    twse_month("6446", "201608", "twse/stock_day/6446_202312")
    tpex_month("6446", "201608", EMPTY_TPEX)
    call_command("load_listing_dates", "--stock", "6446")
    assert StockMaster.objects.get(stock_id="6446").list_date == date(2016, 9, 1)


@responses.activate
@pytest.mark.django_db
def test_a_bad_answer_for_one_stock_is_recorded_and_the_rest_continue(
    tpex_month: Callable[..., None],
) -> None:
    for code in ("6488", "6489"):
        StockMaster.objects.create(stock_id=code, name=code, market=Market.TPEX)
        _prices(code, date(2015, 9, 25))
    responses.post(
        TPEX_URL,
        json={"stat": "error"},
        match=[
            matchers.urlencoded_params_matcher(
                {"code": "6489", "date": "2015/09/01", "response": "json"}
            )
        ],
    )
    tpex_month("6488", "201509", "tpex/trading_stock/6488_201509")
    tpex_month("6488", "201508", EMPTY_TPEX)
    with pytest.raises(CommandError, match="1 stock rejected: 6489"):
        call_command("load_listing_dates")
    assert StockMaster.objects.get(stock_id="6488").list_date == date(2015, 9, 25)
    run = EtlRun.objects.get(dataset="list_date")
    assert run.status == "failed"
    assert run.checkpoint["rejected"] == {
        "6489": "SchemaChangedError: TPEx stock month stat is 'error'"
    }


@pytest.mark.django_db
def test_stocks_without_prices_or_with_a_date_are_left_alone() -> None:
    StockMaster.objects.create(stock_id="1101", name="台泥", market=Market.TWSE)
    StockMaster.objects.create(
        stock_id="2330", name="台積電", market=Market.TWSE, list_date=date(1994, 9, 5)
    )
    _prices("2330", date(2010, 1, 4))
    call_command("load_listing_dates")
    assert EtlRun.objects.get(dataset="list_date").rows == 0


@responses.activate
@pytest.mark.django_db
def test_a_stock_never_found_on_either_market_is_rejected(
    twse_month: Callable[..., None], tpex_month: Callable[..., None]
) -> None:
    StockMaster.objects.create(stock_id="6446", name="藥華藥", market=Market.TWSE)
    _prices("6446", date(2023, 12, 1))
    twse_month("6446", "202312", "twse/stock_day/6446_202312")
    tpex_month("6446", "202312", EMPTY_TPEX)
    with pytest.raises(CommandError, match="1 stock rejected: 6446"):
        call_command("load_listing_dates", "--stock", "6446")
    run = EtlRun.objects.get(dataset="list_date")
    assert run.checkpoint["rejected"] == {"6446": "no official trading day up to the last price"}


@responses.activate
@pytest.mark.django_db
def test_a_listing_older_than_the_prices_leaves_the_date_empty(
    twse_month: Callable[..., None],
) -> None:
    StockMaster.objects.create(stock_id="6446", name="藥華藥", market=Market.TWSE)
    _prices("6446", date(2024, 2, 1))
    twse_month("6446", "202402", "twse/stock_day/6446_202401")
    twse_month("6446", "202401", "twse/stock_day/6446_202401")
    call_command("load_listing_dates", "--stock", "6446")
    assert StockMaster.objects.get(stock_id="6446").list_date is None
    run = EtlRun.objects.get(dataset="list_date")
    assert (run.status, run.rows) == ("success", 0)
    assert run.checkpoint["listed_before_prices"] == ["6446"]


@responses.activate
@pytest.mark.django_db
def test_twse_stocks_trading_in_january_2010_count_as_listed_earlier(
    twse_month: Callable[..., None], tpex_month: Callable[..., None]
) -> None:
    StockMaster.objects.create(stock_id="2330", name="台積電", market=Market.TWSE)
    _prices("2330", date(2010, 1, 4))
    twse_month("2330", "201001", "twse/stock_day/6446_202401")
    tpex_month("2330", "200912", EMPTY_TPEX)
    call_command("load_listing_dates", "--stock", "2330")
    assert StockMaster.objects.get(stock_id="2330").list_date is None
    assert EtlRun.objects.get(dataset="list_date").checkpoint["listed_before_prices"] == ["2330"]
