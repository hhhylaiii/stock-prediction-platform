from collections import Counter
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any
from urllib.parse import parse_qs

import pytest
import responses
from responses import matchers

from apps.etl.adapters.tpex import TpexAdapter
from apps.etl.errors import SchemaChangedError
from apps.etl.models import RawTpex
from apps.etl.ratelimit import RateLimiter
from apps.etl.replay import replay_adapter
from apps.market.calendar import TradingDays

BASE = "https://www.tpex.org.tw/www/zh-tw/bulletin"
EX_DAILY = f"{BASE}/exDailyQ"
REDUCTION = f"{BASE}/revivt"
PAR_VALUE = f"{BASE}/pvChgRslt"
OCT_6 = date(2026, 10, 6)
OCT_7 = date(2026, 10, 7)
FIXTURES = "tpex/dividend_event"


@pytest.fixture
def adapter(october_2026: TradingDays, twse_limiter: RateLimiter) -> TpexAdapter:
    return TpexAdapter(october_2026, twse_limiter)


def _sent(call_index: int) -> dict[str, str]:
    body = responses.calls[call_index].request.body or ""
    text = body.decode() if isinstance(body, bytes) else body
    return {key: values[0] for key, values in parse_qs(text, keep_blank_values=True).items()}


def _day(url: str, day: str, payload: Any) -> None:
    form = {"startDate": day, "endDate": day, "response": "json"}
    responses.post(url, json=payload, match=[matchers.urlencoded_params_matcher(form)])


def _answer_the_session(load_fixture: Callable[[str], Any]) -> None:
    _day(EX_DAILY, "2026/10/07", load_fixture(f"{FIXTURES}/exdailyq_20261007.json"))
    _day(REDUCTION, "2026/10/07", load_fixture(f"{FIXTURES}/revivt_20261006_07.json"))
    _day(PAR_VALUE, "2026/10/07", load_fixture(f"{FIXTURES}/pvchgrslt_20261006_07.json"))


@responses.activate
@pytest.mark.django_db
def test_dividend_events_of_the_session_from_ex_rights_and_restart_tables(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    _answer_the_session(load_fixture)
    events = adapter.normalize("dividend_event", adapter.fetch("dividend_event", OCT_7, OCT_7))
    assert _sent(0) == {"startDate": "2026/10/07", "endDate": "2026/10/07", "response": "json"}
    by_code = events.set_index("stock_id")
    assert sorted(by_code.index) == ["2947", "4527", "5301", "6129"]
    assert by_code.loc["2947", "cash_dividend"] == Decimal("1")
    assert by_code.loc["6129", "adj_factor"] == pytest.approx(14.11 / 14.40)
    assert by_code.loc["4527", "ex_date"] == OCT_7
    assert by_code.loc["4527", "cash_dividend"] == Decimal("0")
    assert by_code.loc["4527", "adj_factor"] == pytest.approx(44.33 / 40.90)
    assert Counter(RawTpex.objects.values_list("dataset", flat=True)) == {
        "ex_rights": 1,
        "capital_reduction": 1,
        "par_value_change": 1,
    }


@responses.activate
@pytest.mark.django_db
def test_dividend_events_replay_from_each_page_kept(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    _answer_the_session(load_fixture)
    fetched = adapter.normalize("dividend_event", adapter.fetch("dividend_event", OCT_7, OCT_7))
    calls = len(responses.calls)
    replay = replay_adapter("tpex", "dividend_event", adapter.calendar)
    replayed = replay.normalize("dividend_event", replay.fetch("dividend_event", OCT_7, OCT_7))
    assert len(responses.calls) == calls
    assert replayed.equals(fetched)


@responses.activate
@pytest.mark.django_db
def test_a_cut_short_answer_is_refused(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture(f"{FIXTURES}/exdailyq_20261007.json")
    payload["tables"][0]["totalCount"] = 100
    _day(EX_DAILY, "2026/10/07", payload)
    with pytest.raises(SchemaChangedError, match="2 of 100"):
        adapter.fetch("dividend_event", OCT_7, OCT_7)


@responses.activate
@pytest.mark.django_db
def test_a_quiet_session_has_no_events(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    _day(EX_DAILY, "2026/10/06", load_fixture(f"{FIXTURES}/exdailyq_20261006.json"))
    _day(REDUCTION, "2026/10/06", load_fixture(f"{FIXTURES}/revivt_20261008.json"))
    _day(PAR_VALUE, "2026/10/06", load_fixture(f"{FIXTURES}/pvchgrslt_20261006_07.json"))
    events = adapter.normalize("dividend_event", adapter.fetch("dividend_event", OCT_6, OCT_6))
    assert events.empty
