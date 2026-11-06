import io
import json
from collections.abc import Callable
from datetime import date
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
import responses
from django.core.management import call_command

from apps.etl.adapters.tpex import TpexAdapter
from apps.etl.adapters.twse import TwseAdapter
from apps.etl.errors import SchemaChangedError
from apps.etl.industry_fill import (
    INDUSTRY_CHECK,
    INDUSTRY_TYPES,
    Listed,
    fill_industries,
    named,
    sample_days,
)
from apps.etl.models import DataQualityLog, RawTpex, RawTwse
from apps.etl.ratelimit import RateLimiter
from apps.market.calendar import TradingDays
from apps.market.models import StockMaster

TWSE_QUOTES = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
TPEX_QUOTES = "https://www.tpex.org.tw/www/zh-tw/afterTrading/otc"
DAY = date(2012, 6, 1)
LATER = date(2012, 9, 28)
NO_DATA = {"stat": "很抱歉，沒有符合條件的資料!"}

pytestmark = pytest.mark.django_db


@pytest.fixture
def calendar() -> TradingDays:
    return TradingDays([DAY, LATER], date(2012, 1, 1), date(2012, 12, 31))


def master(**industries: str) -> None:
    StockMaster.objects.bulk_create(
        StockMaster(stock_id=code.lstrip("_"), name=code, market="TWSE", industry=industry)
        for code, industry in industries.items()
    )


def twse_answers(pages: dict[tuple[str, str], dict[str, Any]]) -> None:
    def answer(request: Any) -> tuple[int, dict[str, str], str]:
        query = {key: values[0] for key, values in parse_qs(urlsplit(request.url).query).items()}
        payload = pages.get((query["date"], query["type"]), NO_DATA)
        return 200, {}, json.dumps(payload)

    responses.add_callback(responses.GET, TWSE_QUOTES, callback=answer)


def cement(load_fixture: Callable[[str], Any]) -> dict[str, Any]:
    return dict(load_fixture("twse/industry/mi_index_01_20120601.json"))


def test_a_type_takes_its_page_s_name_and_else_most_of_its_known_stocks() -> None:
    known = {"1101": "水泥工業", "1102": "水泥工業", "1103": "其他", "2330": "半導體業"}
    listed = {
        "01": Listed("新產業", ["1103", "1101", "1102", "1104"]),
        "20": Listed("其他", ["1101", "1102"]),
        "24": Listed(None, ["9999"]),
    }
    assert named(listed, known) == {
        "01": ["水泥工業", "新產業", "水泥工業", 2, 3],
        "20": ["其他", "其他", "水泥工業", 2, 2],
    }


def test_the_sampled_days_are_each_quarter_s_last_session(calendar: TradingDays) -> None:
    assert sample_days(calendar, date(2012, 1, 1), date(2012, 12, 31)) == [DAY, LATER]
    assert sample_days(calendar, date(2012, 6, 2), date(2012, 12, 31)) == [LATER]


@responses.activate
def test_a_stock_without_an_industry_takes_the_one_it_was_listed_under(
    calendar: TradingDays, twse_limiter: RateLimiter, load_fixture: Callable[[str], Any]
) -> None:
    twse_answers({("20120601", "01"): cement(load_fixture)})
    master(**{"_1101": "水泥工業", "_1102": "水泥工業", "_1103": "", "_1104": "觀光餐旅"})
    fill_industries(TwseAdapter(calendar, twse_limiter), [DAY])
    industries = dict(StockMaster.objects.values_list("stock_id", "industry"))
    assert industries == {
        "1101": "水泥工業",
        "1102": "水泥工業",
        "1103": "水泥工業",
        "1104": "觀光餐旅",
    }
    assert len(responses.calls) == len(INDUSTRY_TYPES)
    assert RawTwse.objects.filter(dataset="industry_quotes").count() == len(INDUSTRY_TYPES)
    (log,) = DataQualityLog.objects.filter(check_name=INDUSTRY_CHECK)
    assert log.detail["filled"] == {"1103": ["水泥工業", "01", "2012-06-01", "2012-06-01"]}
    assert log.detail["types"] == {"2012-06-01": {"01": ["水泥工業", "水泥工業", "水泥工業", 2, 3]}}


@responses.activate
def test_every_listed_stock_stays_in_raw_with_its_type(
    calendar: TradingDays, twse_limiter: RateLimiter, load_fixture: Callable[[str], Any]
) -> None:
    page = cement(load_fixture)
    twse_answers({("20120601", "01"): page})
    master(**{"_1101": "水泥工業"})
    fill_industries(TwseAdapter(calendar, twse_limiter), [DAY])
    (kept,) = RawTwse.objects.filter(dataset="industry_quotes", request_params__type="01")
    assert kept.request_params["date"] == "20120601"
    assert kept.payload == page


@responses.activate
def test_the_last_industry_seen_wins(
    calendar: TradingDays, twse_limiter: RateLimiter, load_fixture: Callable[[str], Any]
) -> None:
    other = {
        "stat": "OK",
        "date": "20120928",
        "tables": [{"title": "每日收盤行情(其他)", "data": [["1103", "嘉泥"], ["9907", "統一實"]]}],
    }
    twse_answers({("20120601", "01"): cement(load_fixture), ("20120928", "20"): other})
    master(**{"_1101": "水泥工業", "_1102": "水泥工業", "_1103": "", "_9907": "其他", "_9999": ""})
    run = fill_industries(TwseAdapter(calendar, twse_limiter), [DAY, LATER])
    assert StockMaster.objects.get(stock_id="1103").industry == "其他"
    assert (run.rows, run.checkpoint["last_date"]) == (1, "2012-09-28")
    (log,) = DataQualityLog.objects.filter(check_name=INDUSTRY_CHECK)
    assert log.detail["filled"] == {"1103": ["其他", "20", "2012-09-28", "2012-09-28"]}
    assert log.detail["unfilled"] == ["9999"]


@responses.activate
def test_a_twse_answer_other_than_quotes_or_no_data_stops_the_run(
    calendar: TradingDays, twse_limiter: RateLimiter
) -> None:
    twse_answers({("20120601", "01"): {"stat": "查詢日期大於可查詢最大日期，請重新查詢!"}})
    with pytest.raises(SchemaChangedError, match="unexpected stat"):
        fill_industries(TwseAdapter(calendar, twse_limiter), [DAY])


@responses.activate
def test_a_day_no_type_lists_a_stock_on_stops_the_run(
    calendar: TradingDays, twse_limiter: RateLimiter
) -> None:
    twse_answers({})
    with pytest.raises(SchemaChangedError, match="no industry type lists any stock"):
        fill_industries(TwseAdapter(calendar, twse_limiter), [DAY])


@responses.activate
def test_the_otc_market_lists_its_types_by_post(
    calendar: TradingDays, twse_limiter: RateLimiter, load_fixture: Callable[[str], Any]
) -> None:
    food = load_fixture("tpex/industry/otc_02_20120601.json")
    empty = load_fixture("tpex/industry/otc_01_20120601.json")

    def answer(request: Any) -> tuple[int, dict[str, str], str]:
        form = {key: values[0] for key, values in parse_qs(str(request.body)).items()}
        return 200, {}, json.dumps(food if form["type"] == "02" else empty)

    responses.add_callback(responses.POST, TPEX_QUOTES, callback=answer)
    master(**{"_1258": "食品工業", "_4205": "食品工業", "_4207": ""})
    fill_industries(TpexAdapter(calendar, twse_limiter), [DAY])
    assert StockMaster.objects.get(stock_id="4207").industry == "食品工業"
    assert RawTpex.objects.filter(dataset="industry_quotes").exists()


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_the_command_fills_from_the_sampled_days(
    load_fixture: Callable[[str], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)
    twse_answers({("20261030", "01"): {**cement(load_fixture), "date": "20261030"}})
    master(**{"_1103": ""})
    out = io.StringIO()
    call_command(
        "fill_industry",
        "--source",
        "twse",
        "--start",
        "2026-10-01",
        "--end",
        "2026-10-31",
        stdout=out,
    )
    assert "0 industries filled from 1 sampled days" in out.getvalue()
    assert len(responses.calls) == len(INDUSTRY_TYPES)


@responses.activate
def test_an_otc_table_short_of_its_count_stops_the_run(
    calendar: TradingDays, twse_limiter: RateLimiter, load_fixture: Callable[[str], Any]
) -> None:
    food = load_fixture("tpex/industry/otc_02_20120601.json")
    short = {**food, "tables": [{**food["tables"][0], "totalCount": 5}]}
    responses.post(TPEX_QUOTES, json=short)
    with pytest.raises(SchemaChangedError, match="sent 4 of 5 rows"):
        fill_industries(TpexAdapter(calendar, twse_limiter), [DAY])
