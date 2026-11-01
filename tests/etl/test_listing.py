import os
import signal
from collections.abc import Callable
from datetime import date, timedelta
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
from apps.etl.listing import Listing, Outcome, Probe, first_listed_day
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
EMPTY_TWSE = "twse/stock_day/6446_202312"
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
    twse_month("6446", "202312", EMPTY_TWSE)
    client = TwseClient.build(limiter_for("twse"))
    days = client.stock_month("6446", date(2024, 1, 1))
    assert days is not None
    assert days[:2] == [date(2024, 1, 25), date(2024, 1, 26)]
    assert client.stock_month("6446", date(2023, 12, 1)) == []


@pytest.mark.django_db
def test_twse_cannot_answer_months_before_2010() -> None:
    assert TwseClient.build(limiter_for("twse")).stock_month("2330", date(2009, 12, 1)) is None


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"stat": "查詢日期大於今日，請重新查詢!"}, "查詢日期"),
        ({"stat": "OK"}, "rows are malformed"),
        ({"stat": "OK", "data": [{"date": "115/10/01"}]}, "rows are malformed"),
    ],
)
def test_unexpected_twse_months_are_schema_changes(payload: dict[str, Any], message: str) -> None:
    responses.get(TWSE_URL, json=payload)
    with pytest.raises(SchemaChangedError, match=message):
        TwseClient.build(limiter_for("twse")).stock_month("6446", date(2026, 10, 1))


@responses.activate
@pytest.mark.django_db
def test_a_month_answered_with_other_days_is_a_schema_change(
    twse_month: Callable[..., None],
) -> None:
    twse_month("6446", "202402", "twse/stock_day/6446_202401")
    with pytest.raises(SchemaChangedError, match="2024-02 with days of another month"):
        TwseClient.build(limiter_for("twse")).stock_month("6446", date(2024, 2, 1))


@responses.activate
@pytest.mark.django_db
def test_tpex_months_drop_the_first_day_marker(tpex_month: Callable[..., None]) -> None:
    tpex_month("6488", "201509", "tpex/trading_stock/6488_201509")
    tpex_month("6488", "201508", EMPTY_TPEX)
    client = TpexClient.build(limiter_for("tpex"))
    assert client.stock_month("6488", date(2015, 9, 1)) == [date(2015, 9, 25), date(2015, 9, 30)]
    assert client.stock_month("6488", date(2015, 8, 1)) == []


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"stat": "error"}, "stat is 'error'"),
        ({"stat": "ok", "tables": []}, "has no table"),
        ({"stat": "ok", "tables": [{"data": None}]}, "rows are malformed"),
    ],
)
def test_unexpected_tpex_months_are_schema_changes(payload: dict[str, Any], message: str) -> None:
    responses.post(TPEX_URL, json=payload)
    with pytest.raises(SchemaChangedError, match=message):
        TpexClient.build(limiter_for("tpex")).stock_month("6488", date(2015, 9, 1))


def _months(first: date, last: date, skip: tuple[date, date] | None = None) -> list[date]:
    """The first day of every month in [first, last], minus an optional span without prices."""
    days, day = [], first
    while day <= last:
        if skip is None or not skip[0] <= day <= skip[1]:
            days.append(day)
        day = (day.replace(day=1) + timedelta(days=32)).replace(day=1)
    return days


def _probe_from(first: date, asked: list[date], silent: tuple[date, date] | None = None) -> Probe:
    def probe(month: date, every: bool = False) -> list[date] | None:
        asked.append(month)
        if silent is not None and silent[0] <= month <= silent[1]:
            return None
        return [max(month, first)] if month >= first.replace(day=1) else []

    return probe


def test_a_stock_also_trading_the_month_before_its_prices_was_listed_earlier() -> None:
    asked: list[date] = []
    probe = _probe_from(date(2008, 3, 3), asked)
    found = first_listed_day(_months(date(2010, 1, 4), date(2026, 10, 1)), probe)
    assert found == Listing(Outcome.EARLIER)
    assert asked == [date(2010, 1, 1), date(2009, 12, 1)]


def test_prices_before_the_first_official_day_of_their_month_are_emerging() -> None:
    asked: list[date] = []
    probe = _probe_from(date(2012, 3, 5), asked)
    found = first_listed_day(_months(date(2012, 3, 1), date(2026, 10, 1)), probe)
    assert found == Listing(Outcome.LISTED, date(2012, 3, 5))
    assert asked == [date(2012, 3, 1), date(2012, 3, 1)]


def test_a_first_price_on_the_first_official_day_is_not_proof_of_listing() -> None:
    probe = _probe_from(date(2012, 3, 1), [])
    found = first_listed_day(_months(date(2012, 3, 1), date(2026, 10, 1)), probe)
    assert found == Listing(Outcome.FIRST_PRICE, date(2012, 3, 1))


def test_a_market_switch_in_the_first_month_is_not_mistaken_for_a_listing() -> None:
    asked: list[tuple[date, bool]] = []

    def probe(month: date, every: bool = False) -> list[date] | None:
        asked.append((month, every))
        if month == date(2024, 1, 1):
            return [date(2024, 1, 2), date(2024, 1, 25)] if every else [date(2024, 1, 25)]
        return [date(2023, 12, 1)]

    found = first_listed_day([date(2024, 1, 2), date(2024, 2, 1)], probe)
    assert found == Listing(Outcome.EARLIER)
    assert asked[1] == (date(2024, 1, 1), True)


def test_emerging_market_months_are_skipped_with_few_probes() -> None:
    asked: list[date] = []
    probe = _probe_from(date(2015, 9, 25), asked)
    found = first_listed_day(_months(date(2014, 10, 28), date(2026, 10, 1)), probe)
    assert found == Listing(Outcome.LISTED, date(2015, 9, 25))
    assert len(asked) <= 10


def test_months_without_prices_are_never_probed() -> None:
    asked: list[date] = []
    halted = (date(2013, 9, 1), date(2025, 1, 1))
    firsts = _months(date(2012, 1, 2), date(2026, 10, 1), skip=halted)
    found = first_listed_day(firsts, _probe_from(date(2013, 7, 15), asked))
    assert found == Listing(Outcome.LISTED, date(2013, 7, 15))
    assert not [month for month in asked if halted[0] <= month <= halted[1]]


def test_a_month_no_market_can_answer_leaves_the_listing_unknown() -> None:
    probe = _probe_from(date(2015, 9, 25), [], silent=(date(2015, 1, 1), date(2015, 12, 1)))
    found = first_listed_day(_months(date(2014, 10, 28), date(2026, 10, 1)), probe)
    assert found == Listing(Outcome.UNKNOWN)


def test_no_official_month_among_the_prices_is_missing() -> None:
    asked: list[date] = []
    probe = _probe_from(date(2030, 1, 1), asked)
    assert first_listed_day(_months(date(2014, 10, 1), date(2016, 3, 1)), probe) == Listing(
        Outcome.MISSING
    )
    assert asked[-1] == date(2016, 3, 1)
    assert first_listed_day([date(2016, 3, 1)], probe) == Listing(Outcome.MISSING)


def _prices(stock: str, *days: date) -> None:
    DailyPrice.objects.bulk_create(
        DailyPrice(stock_id=stock, date=day, close=10, volume=1, turnover=10) for day in days
    )


def _dates(stock: str) -> list[date]:
    return list(DailyPrice.objects.filter(stock_id=stock).values_list("date", flat=True))


@pytest.fixture
def listed_6488(
    twse_month: Callable[..., None], tpex_month: Callable[..., None]
) -> Callable[[], None]:
    def _mock() -> None:
        tpex_month("6488", "201508", EMPTY_TPEX)
        twse_month("6488", "201508", EMPTY_TWSE)
        tpex_month("6488", "201509", "tpex/trading_stock/6488_201509")
        twse_month("6488", "201509", EMPTY_TWSE)

    return _mock


@responses.activate
@pytest.mark.django_db
def test_listing_dates_drop_earlier_market_rows_but_keep_shares(
    listed_6488: Callable[[], None],
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
    listed_6488()
    call_command("load_listing_dates", "--stock", "6488")
    assert StockMaster.objects.get(stock_id="6488").list_date == date(2015, 9, 25)
    assert sorted(_dates("6488")) == [date(2015, 9, 25), date(2015, 9, 30)]
    assert not DailyChip.objects.exists()
    assert not DailyValuation.objects.exists()
    assert SharesOutstanding.objects.count() == 1
    run = EtlRun.objects.get(dataset="list_date")
    assert (run.status, run.rows) == ("success", 1)
    assert run.checkpoint["listed"] == {"6488": ["2015-09-25", 3]}


@responses.activate
@pytest.mark.django_db
def test_a_tpex_stock_that_traded_on_twse_first_keeps_its_twse_rows(
    twse_month: Callable[..., None], tpex_month: Callable[..., None]
) -> None:
    StockMaster.objects.create(stock_id="6423", name="億而得", market=Market.TPEX)
    _prices("6423", date(2025, 6, 2), date(2025, 12, 1), date(2026, 2, 2))
    tpex_month("6423", "202506", "tpex/trading_stock/6423_202506")
    twse_month("6423", "202506", "twse/stock_day/6423_202506")
    tpex_month("6423", "202505", "tpex/trading_stock/6423_202505")
    twse_month("6423", "202505", "twse/stock_day/6423_202505")
    call_command("load_listing_dates", "--stock", "6423")
    assert StockMaster.objects.get(stock_id="6423").list_date is None
    assert len(_dates("6423")) == 3
    settled = EtlRun.objects.get(dataset="list_date").checkpoint["settled"]
    assert settled == {"6423": ["2025-06-02", "listed_before_prices"]}


@responses.activate
@pytest.mark.django_db
def test_a_twse_stock_that_traded_on_tpex_first_keeps_its_tpex_rows(
    twse_month: Callable[..., None], tpex_month: Callable[..., None]
) -> None:
    StockMaster.objects.create(stock_id="6446", name="藥華藥", market=Market.TWSE)
    _prices("6446", date(2016, 9, 1), date(2024, 1, 25))
    twse_month("6446", "201609", EMPTY_TWSE)
    tpex_month("6446", "201609", "tpex/trading_stock/6446_201609")
    twse_month("6446", "201608", EMPTY_TWSE)
    tpex_month("6446", "201608", "tpex/trading_stock/6446_201608")
    call_command("load_listing_dates", "--stock", "6446")
    assert StockMaster.objects.get(stock_id="6446").list_date is None
    assert len(_dates("6446")) == 2


@responses.activate
@pytest.mark.django_db
def test_prices_starting_in_its_twse_month_find_the_earlier_tpex_listing(
    twse_month: Callable[..., None], tpex_month: Callable[..., None]
) -> None:
    StockMaster.objects.create(stock_id="6446", name="藥華藥", market=Market.TWSE)
    _prices("6446", date(2024, 1, 25))
    twse_month("6446", "202401", "twse/stock_day/6446_202401")
    twse_month("6446", "202312", EMPTY_TWSE)
    tpex_month("6446", "202312", "tpex/trading_stock/6446_202312")
    call_command("load_listing_dates", "--stock", "6446")
    assert StockMaster.objects.get(stock_id="6446").list_date is None
    assert EtlRun.objects.get().checkpoint["settled"]["6446"][1] == "listed_before_prices"


@responses.activate
@pytest.mark.django_db
def test_twse_stocks_trading_in_january_2010_stay_unknown(
    twse_month: Callable[..., None], tpex_month: Callable[..., None]
) -> None:
    StockMaster.objects.create(stock_id="2330", name="台積電", market=Market.TWSE)
    _prices("2330", date(2010, 1, 4))
    twse_month("2330", "201001", "twse/stock_day/2330_201001")
    tpex_month("2330", "200912", EMPTY_TPEX)
    call_command("load_listing_dates", "--stock", "2330")
    assert StockMaster.objects.get(stock_id="2330").list_date is None
    run = EtlRun.objects.get()
    assert (run.status, run.checkpoint["settled"]) == (
        "success",
        {"2330": ["2010-01-04", "unanswerable"]},
    )
    call_command("load_listing_dates")
    call_command("load_listing_dates", "--refresh")
    assert len(responses.calls) == 2
    _prices("2330", date(2009, 12, 31))
    call_command("load_listing_dates")
    assert len(responses.calls) == 3
    latest = EtlRun.objects.latest("run_id").checkpoint["settled"]
    assert latest == {"2330": ["2009-12-31", "unanswerable"]}


@responses.activate
@pytest.mark.django_db
def test_refresh_clears_a_listing_date_the_months_no_longer_confirm(
    twse_month: Callable[..., None], tpex_month: Callable[..., None]
) -> None:
    StockMaster.objects.create(
        stock_id="6423", name="億而得", market=Market.TPEX, list_date=date(2025, 6, 2)
    )
    _prices("6423", date(2025, 6, 2))
    call_command("load_listing_dates")
    assert not responses.calls
    tpex_month("6423", "202506", "tpex/trading_stock/6423_202506")
    twse_month("6423", "202506", "twse/stock_day/6423_202506")
    tpex_month("6423", "202505", "tpex/trading_stock/6423_202505")
    twse_month("6423", "202505", "twse/stock_day/6423_202505")
    call_command("load_listing_dates", "--refresh")
    assert StockMaster.objects.get(stock_id="6423").list_date is None
    assert EtlRun.objects.latest("run_id").checkpoint["cleared"] == ["6423"]


@responses.activate
@pytest.mark.django_db
def test_a_bad_answer_for_one_stock_is_recorded_and_the_rest_continue(
    listed_6488: Callable[[], None],
) -> None:
    for code in ("6488", "6489"):
        StockMaster.objects.create(stock_id=code, name=code, market=Market.TPEX)
        _prices(code, date(2015, 8, 31), date(2015, 9, 25))
    form = {"code": "6489", "date": "2015/08/01", "response": "json"}
    responses.post(
        TPEX_URL, json={"stat": "error"}, match=[matchers.urlencoded_params_matcher(form)]
    )
    listed_6488()
    with pytest.raises(CommandError, match="1 stock rejected: 6489"):
        call_command("load_listing_dates")
    assert StockMaster.objects.get(stock_id="6488").list_date == date(2015, 9, 25)
    run = EtlRun.objects.get(dataset="list_date")
    assert run.status == "failed"
    reason = "SchemaChangedError: TPEx stock month stat is 'error'"
    assert run.checkpoint["listed"] == {"6488": ["2015-09-25", 1]}
    assert run.checkpoint["rejected"] == {"6489": reason}


@responses.activate
@pytest.mark.django_db
def test_a_dead_source_stops_the_run_but_keeps_the_rejections(
    tpex_month: Callable[..., None],
) -> None:
    for code in ("6488", "6489"):
        StockMaster.objects.create(stock_id=code, name=code, market=Market.TPEX)
        _prices(code, date(2015, 9, 25))
    form = {"code": "6488", "date": "2015/09/01", "response": "json"}
    responses.post(
        TPEX_URL, json={"stat": "error"}, match=[matchers.urlencoded_params_matcher(form)]
    )
    with pytest.raises(CommandError, match="RetryableError"):
        call_command("load_listing_dates")
    run = EtlRun.objects.get()
    assert run.status == "failed"
    assert list(run.checkpoint["rejected"]) == ["6488"]


@responses.activate
@pytest.mark.django_db
def test_termination_marks_the_run_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    StockMaster.objects.create(stock_id="6488", name="環球晶", market=Market.TPEX)
    _prices("6488", date(2015, 9, 25))

    def terminate(firsts: list[date], probe: Any) -> Listing:
        os.kill(os.getpid(), signal.SIGTERM)
        return Listing(Outcome.UNKNOWN)

    monkeypatch.setattr(
        "apps.etl.management.commands.load_listing_dates.first_listed_day", terminate
    )
    with pytest.raises(SystemExit):
        call_command("load_listing_dates")
    assert EtlRun.objects.get().status == "failed"


@responses.activate
@pytest.mark.django_db
def test_stocks_without_prices_or_with_a_date_are_left_alone() -> None:
    StockMaster.objects.create(stock_id="1101", name="台泥", market=Market.TWSE)
    StockMaster.objects.create(
        stock_id="2330", name="台積電", market=Market.TWSE, list_date=date(1994, 9, 5)
    )
    _prices("2330", date(2010, 1, 4))
    call_command("load_listing_dates")
    assert EtlRun.objects.get(dataset="list_date").rows == 0
    assert StockMaster.objects.get(stock_id="2330").list_date == date(1994, 9, 5)


@responses.activate
@pytest.mark.django_db
def test_rows_from_the_delisting_day_on_are_dropped() -> None:
    StockMaster.objects.create(
        stock_id="5371",
        name="中光電",
        market=Market.TPEX,
        list_date=date(2001, 1, 1),
        delist_date=date(2026, 9, 3),
    )
    _prices("5371", date(2026, 8, 21), date(2026, 9, 3), date(2026, 9, 10))
    DailyChip.objects.create(stock_id="5371", date=date(2026, 9, 10), fi_net=1)
    call_command("load_listing_dates", "--stock", "5371")
    assert _dates("5371") == [date(2026, 8, 21)]
    assert not DailyChip.objects.exists()
    assert EtlRun.objects.get().checkpoint["delisting_dropped"] == {"5371": 3}


@responses.activate
@pytest.mark.django_db
def test_a_stock_never_found_on_either_market_is_rejected(
    twse_month: Callable[..., None], tpex_month: Callable[..., None]
) -> None:
    StockMaster.objects.create(stock_id="6446", name="藥華藥", market=Market.TWSE)
    _prices("6446", date(2023, 12, 1))
    twse_month("6446", "202312", EMPTY_TWSE)
    tpex_month("6446", "202312", EMPTY_TPEX)
    with pytest.raises(CommandError, match="1 stock rejected: 6446"):
        call_command("load_listing_dates", "--stock", "6446")
    run = EtlRun.objects.get(dataset="list_date")
    assert run.checkpoint["rejected"] == {"6446": "no official trading day up to the last price"}


@responses.activate
@pytest.mark.django_db
def test_prices_starting_in_the_month_it_moved_to_twse_keep_the_tpex_days(
    twse_month: Callable[..., None], tpex_month: Callable[..., None]
) -> None:
    StockMaster.objects.create(stock_id="6446", name="藥華藥", market=Market.TWSE)
    _prices("6446", date(2024, 1, 2), date(2024, 1, 25))
    twse_month("6446", "202401", "twse/stock_day/6446_202401")
    tpex_month("6446", "202401", "tpex/trading_stock/6446_202401")
    twse_month("6446", "202312", EMPTY_TWSE)
    tpex_month("6446", "202312", "tpex/trading_stock/6446_202312")
    call_command("load_listing_dates", "--stock", "6446")
    assert StockMaster.objects.get(stock_id="6446").list_date is None
    assert sorted(_dates("6446")) == [date(2024, 1, 2), date(2024, 1, 25)]
    assert len(responses.calls) == 4


@responses.activate
@pytest.mark.django_db
def test_refresh_keeps_a_date_whose_earlier_rows_are_already_gone(
    twse_month: Callable[..., None], tpex_month: Callable[..., None]
) -> None:
    StockMaster.objects.create(
        stock_id="6488", name="環球晶", market=Market.TPEX, list_date=date(2015, 9, 25)
    )
    _prices("6488", date(2015, 9, 25))
    tpex_month("6488", "201509", "tpex/trading_stock/6488_201509")
    tpex_month("6488", "201508", EMPTY_TPEX)
    twse_month("6488", "201508", EMPTY_TWSE)
    call_command("load_listing_dates", "--refresh")
    assert StockMaster.objects.get(stock_id="6488").list_date == date(2015, 9, 25)
    checkpoint = EtlRun.objects.get().checkpoint
    assert checkpoint["settled"] == {"6488": ["2015-09-25", "first_price_after_empty_month"]}
    assert checkpoint["cleared"] == []


@responses.activate
@pytest.mark.django_db
def test_a_run_of_rejections_stops_asking(settings: Any) -> None:
    settings.ETL_MAX_CONSECUTIVE_REJECTIONS = 2
    for code in ("6488", "6489", "6490"):
        StockMaster.objects.create(stock_id=code, name=code, market=Market.TPEX)
        _prices(code, date(2015, 9, 25))
    responses.post(TPEX_URL, body="<html>blocked</html>")
    with pytest.raises(CommandError, match="2 stocks rejected in a row; last: 6489"):
        call_command("load_listing_dates")
    run = EtlRun.objects.get()
    assert (run.status, sorted(run.checkpoint["rejected"])) == ("failed", ["6488", "6489"])
    assert len(responses.calls) == 2
