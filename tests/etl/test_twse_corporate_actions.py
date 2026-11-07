import copy
from collections import Counter
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import pandas as pd
import pytest
import responses
from responses import matchers

from apps.etl.adapters.twse import TwseAdapter
from apps.etl.models import RawTwse
from apps.etl.ratelimit import RateLimiter, RetryableError
from apps.etl.replay import replay_adapter
from apps.market.calendar import TradingDays
from apps.market.models import DividendEvent, StockMaster

BASE = "https://www.twse.com.tw/rwd/zh"
TWT49U = f"{BASE}/exRight/TWT49U"
DETAIL = f"{BASE}/exRight/TWT49UDetail"
RESTARTS = (f"{BASE}/reducation/TWTAUU", f"{BASE}/change/TWTB8U", f"{BASE}/split/TWTCAU")
NO_DATA = {"stat": "很抱歉，沒有符合條件的資料!"}
OCT_6 = date(2026, 10, 6)
OCT_7 = date(2026, 10, 7)
FIXTURES = "twse/dividend_event"


@pytest.fixture
def adapter(october_2026: TradingDays, twse_limiter: RateLimiter) -> TwseAdapter:
    return TwseAdapter(october_2026, twse_limiter)


def _sent(call_index: int) -> dict[str, str]:
    url = responses.calls[call_index].request.url or ""
    return dict(parse_qsl(urlsplit(url).query))


def _detail_of(code: str, day: str, payload: Any) -> None:
    query = {"STK_NO": code, "T1": day, "response": "json"}
    responses.get(DETAIL, json=payload, match=[matchers.query_param_matcher(query)])


def _with_dividends(load_fixture: Callable[[str], Any]) -> dict[str, Any]:
    """The 8021 detail page relabelled as 2614 paying cash and bonus shares."""
    body: dict[str, Any] = copy.deepcopy(
        load_fixture(f"{FIXTURES}/twt49u_detail_8021_20261006.json")
    )
    body["data"][0][0] = "2614  "
    body["data"][0][2] = "0.5 元／股"
    body["data"][0][4] = "30 股"
    return body


def _one_day(load_fixture: Callable[[str], Any], day: date) -> dict[str, Any]:
    """The recorded two-day TWT49U answer cut to the rows of one day, as TWSE sends a day."""
    body: dict[str, Any] = copy.deepcopy(load_fixture(f"{FIXTURES}/twt49u_20261006_07.json"))
    written = f"{day.year - 1911}年{day:%m月%d日}"
    body["data"] = [row for row in body["data"] if row[0] == written]
    return body


def _answer(load_fixture: Callable[[str], Any], day: date, detail_2614: Any = None) -> None:
    responses.get(TWT49U, json=_one_day(load_fixture, day))
    _detail_of("2614", "20261006", detail_2614 or _with_dividends(load_fixture))
    _detail_of("8021", "20261006", load_fixture(f"{FIXTURES}/twt49u_detail_8021_20261006.json"))
    _detail_of("2890", "20261007", load_fixture(f"{FIXTURES}/twt49u_detail_2890_20261007.json"))
    for url in RESTARTS:
        responses.get(url, json=NO_DATA)


def _events(adapter: TwseAdapter, day: date) -> pd.DataFrame:
    return adapter.normalize("dividend_event", adapter.fetch("dividend_event", day, day))


@responses.activate
@pytest.mark.django_db
def test_dividend_events_of_the_session_with_details_for_rows_not_cash_only(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    _answer(load_fixture, OCT_6)
    by_code = _events(adapter, OCT_6).set_index("stock_id")
    assert _sent(0) == {"startDate": "20261006", "endDate": "20261006", "response": "json"}
    assert sorted(by_code.index) == ["2614", "8021"]
    assert by_code.loc["2614", "cash_dividend"] == Decimal("0.5")
    assert by_code.loc["2614", "stock_dividend"] == Decimal("0.3")
    assert by_code.loc["2614", "adj_factor"] == pytest.approx(16.13 / 19.10)
    assert by_code.loc["8021", "cash_dividend"] == Decimal("0")
    assert by_code.loc["8021", "adj_factor"] == pytest.approx(555.45 / 560.00)
    assert Counter(RawTwse.objects.values_list("dataset", flat=True)) == {
        "ex_rights": 1,
        "ex_rights_detail": 2,
        "capital_reduction": 1,
        "par_value_change": 1,
        "split": 1,
    }


@responses.activate
@pytest.mark.django_db
def test_cash_only_rows_need_no_detail(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    _answer(load_fixture, OCT_7)
    by_code = _events(adapter, OCT_7).set_index("stock_id")
    assert len(by_code) == 7
    assert by_code.loc["6108", "cash_dividend"] == Decimal("1")
    assert by_code.loc["2890", "cash_dividend"] == Decimal("0")
    assert by_code.loc["2890", "adj_factor"] == pytest.approx(44.97 / 45.40)
    assert RawTwse.objects.filter(dataset="ex_rights_detail").count() == 1


@responses.activate
@pytest.mark.django_db
def test_dividend_events_replay_from_the_list_details_and_restart_tables_kept(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    _answer(load_fixture, OCT_6)
    fetched = _events(adapter, OCT_6)
    calls = len(responses.calls)
    replay = replay_adapter("twse", "dividend_event", adapter.calendar)
    replayed = replay.normalize("dividend_event", replay.fetch("dividend_event", OCT_6, OCT_6))
    assert len(responses.calls) == calls
    assert replayed.equals(fetched)


@responses.activate
@pytest.mark.django_db
def test_dividend_events_are_written_for_listed_stocks_only(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    StockMaster.objects.bulk_create(
        StockMaster(stock_id=code, name=code, market="TWSE") for code in ("2890", "6108")
    )
    _answer(load_fixture, OCT_7)
    assert adapter.upsert("dividend_event", _events(adapter, OCT_7)) == 2
    assert dict(DividendEvent.objects.values_list("stock_id", "ex_date")) == {
        "2890": OCT_7,
        "6108": OCT_7,
    }


@responses.activate
@pytest.mark.django_db
def test_a_busy_detail_page_is_retried_later_and_not_kept(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    _answer(load_fixture, OCT_6, load_fixture(f"{FIXTURES}/twt49u_detail_busy.json"))
    with pytest.raises(RetryableError, match="busy"):
        adapter.fetch("dividend_event", OCT_6, OCT_6)
    assert not RawTwse.objects.filter(dataset="ex_rights_detail").exists()


@responses.activate
@pytest.mark.django_db
def test_a_quiet_session_has_no_events(adapter: TwseAdapter) -> None:
    responses.get(TWT49U, json=NO_DATA)
    for url in RESTARTS:
        responses.get(url, json=NO_DATA)
    assert _events(adapter, OCT_6).empty
    assert len(responses.calls) == 4
