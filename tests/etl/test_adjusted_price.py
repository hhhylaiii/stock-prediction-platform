from collections.abc import Callable
from datetime import date
from decimal import Decimal
from io import StringIO
from typing import Any

import pytest
from django.core.management import call_command

from apps.etl.adapters.finmind_datasets import dividend_events, price_bars
from apps.etl.upsert import upsert
from apps.market.adjust import adjusted, refresh_adj_factor
from apps.market.models import DailyPrice, DividendEvent

PRICE_FIELDS = ("open", "high", "low", "close", "volume", "turnover")
EVENT_FIELDS = ("cash_dividend", "stock_dividend", "adj_factor")


@pytest.fixture
def esun(load_fixture: Callable[[str], Any], db: None) -> None:
    for month in ("201108", "201308"):
        rows = load_fixture(f"finmind/price_2884_{month}.json")["data"]
        upsert(DailyPrice, price_bars(rows, "2884"), ("stock_id", "date"), PRICE_FIELDS)
    events = dividend_events(
        load_fixture("finmind/dividend_2884.json")["data"],
        load_fixture("finmind/dividend_result_2884.json")["data"],
        "2884",
    )
    upsert(DividendEvent, events, ("stock_id", "ex_date"), EVENT_FIELDS)
    refresh_adj_factor("2884")


def _gap(before: date, ex_date: date) -> tuple[float, float]:
    """Raw and adjusted change from the close before an ex-date to the ex-date open."""
    prev = DailyPrice.objects.get(stock_id="2884", date=before)
    ex = DailyPrice.objects.get(stock_id="2884", date=ex_date)
    assert prev.close is not None
    assert ex.open is not None
    raw = float(ex.open) / float(prev.close) - 1
    adj_open, adj_close = adjusted(ex.open, ex.adj_factor), adjusted(prev.close, prev.adj_factor)
    assert adj_open is not None
    assert adj_close is not None
    return raw, adj_open / adj_close - 1


@pytest.mark.spec8
@pytest.mark.usefixtures("esun")
def test_adjusted_return_is_flat_across_a_large_stock_dividend() -> None:
    raw, adjusted_gap = _gap(date(2013, 8, 14), date(2013, 8, 15))
    assert raw == pytest.approx(17.85 / 19.95 - 1)
    assert raw < -0.10
    assert abs(adjusted_gap) < 0.005


@pytest.mark.usefixtures("esun")
def test_stock_and_cash_ex_date_right_after_a_rights_issue_is_flat() -> None:
    raw, adjusted_gap = _gap(date(2011, 8, 24), date(2011, 8, 25))
    assert raw < -0.07
    assert abs(adjusted_gap) < 0.005


@pytest.mark.usefixtures("esun")
def test_the_latest_prices_are_unadjusted_once_no_event_follows() -> None:
    DividendEvent.objects.filter(ex_date__gt=date(2013, 8, 31)).delete()
    refresh_adj_factor("2884")
    assert DailyPrice.objects.get(stock_id="2884", date=date(2013, 8, 30)).adj_factor == 1.0


@pytest.mark.usefixtures("esun")
def test_events_without_a_factor_do_not_adjust() -> None:
    DividendEvent.objects.update(adj_factor=None)
    refresh_adj_factor("2884")
    assert set(DailyPrice.objects.values_list("adj_factor", flat=True)) == {1.0}


@pytest.mark.usefixtures("esun")
def test_recompute_command_refreshes_every_or_chosen_stock() -> None:
    DailyPrice.objects.update(adj_factor=1.0)
    out = StringIO()
    call_command("recompute_adj_factor", "--stock", "2884", stdout=out)
    assert "2884" in out.getvalue()
    assert DailyPrice.objects.filter(adj_factor__lt=1.0).exists()
    DailyPrice.objects.update(adj_factor=1.0)
    call_command("recompute_adj_factor", stdout=StringIO())
    assert DailyPrice.objects.filter(adj_factor__lt=1.0).exists()


@pytest.mark.usefixtures("esun")
def test_refreshing_one_stock_leaves_others_alone() -> None:
    DailyPrice.objects.create(
        stock_id="2330", date=date(2013, 8, 14), close=Decimal("100"), volume=1, turnover=100
    )
    refresh_adj_factor("2884")
    assert DailyPrice.objects.get(stock_id="2330").adj_factor == 1.0


@pytest.mark.usefixtures("esun")
def test_refreshing_from_a_date_leaves_earlier_rows_alone() -> None:
    DailyPrice.objects.update(adj_factor=1.0)
    refresh_adj_factor("2884", since=date(2013, 8, 15))
    assert DailyPrice.objects.get(stock_id="2884", date=date(2013, 8, 14)).adj_factor == 1.0
    assert DailyPrice.objects.get(stock_id="2884", date=date(2013, 8, 15)).adj_factor < 1.0
