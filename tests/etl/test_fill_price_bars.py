from collections.abc import Callable
from datetime import date
from decimal import Decimal
from io import StringIO
from typing import Any

import pytest
import responses
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.etl.adapters.tpex import TpexAdapter
from apps.etl.models import EtlRun, RawTpex
from apps.etl.ratelimit import RateLimiter
from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice, DividendEvent, Market, StockMaster

BASE = "https://www.tpex.org.tw/www/zh-tw"
DAY = date(2026, 10, 28)
OTC = f"{BASE}/afterTrading/otc"
QUOTES = f"{BASE}/afterTrading/dailyQuotes"
FILL = "daily_price_fill"

pytestmark = pytest.mark.usefixtures("october_2026_in_db")


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)


@pytest.fixture
def quotes(load_fixture: Callable[[str], Any]) -> None:
    responses.post(OTC, json=load_fixture("tpex/daily_price/otc_20261028.json"))
    responses.post(QUOTES, json=load_fixture("tpex/daily_price/daily_quotes_20261028.json"))


def _listed(code: str, **fields: Any) -> StockMaster:
    return StockMaster.objects.create(stock_id=code, name=code, market=Market.TPEX, **fields)


def _fill(*stocks: str, day: str = "2026-10-28", dry_run: bool = False) -> str:
    out = StringIO()
    chosen = [part for code in stocks for part in ("--stock", code)]
    extra = ["--dry-run"] if dry_run else []
    call_command("fill_price_bars", "--source", "tpex", "--date", day, *chosen, *extra, stdout=out)
    return out.getvalue()


def _kept_page(day: str) -> RawTpex:
    return RawTpex.objects.create(
        dataset="daily_price",
        request_params={
            "date": f"{day[:4]}/{day[4:6]}/{day[6:]}",
            "type": "EW",
            "response": "json",
        },
        payload={"date": day, "stat": "ok", "tables": []},
        source_hash=f"kept-{day}",
    )


@responses.activate
@pytest.mark.usefixtures("quotes")
def test_only_the_chosen_bar_is_added() -> None:
    _listed("6488")
    _listed("3293")
    printed = _fill("6488")
    bar = DailyPrice.objects.get()
    assert (bar.stock_id, bar.date, bar.open, bar.high, bar.low, bar.close) == (
        "6488",
        DAY,
        Decimal("842.00"),
        Decimal("860.00"),
        Decimal("839.00"),
        Decimal("849.00"),
    )
    assert (bar.volume, bar.turnover, bar.adj_factor) == (6_892_320, 5_852_153_367, 1.0)
    assert "insert 6488 2026-10-28 open 842.00 high 860.00 low 839.00 close 849.00" in printed
    assert "daily_price: 1 bars added for 2026-10-28" in printed


@responses.activate
@pytest.mark.usefixtures("quotes")
def test_the_pages_are_kept_apart_from_the_day_quotes() -> None:
    _listed("6488")
    _fill("6488")
    assert set(RawTpex.objects.values_list("dataset", flat=True)) == {FILL}
    assert RawTpex.objects.count() == 2
    run = EtlRun.objects.get()
    assert (run.source, run.dataset, run.status, run.rows) == ("tpex", FILL, "success", 1)
    assert (run.start_date, run.end_date, run.checkpoint) == (DAY, DAY, {"stocks": ["6488"]})


@responses.activate
@pytest.mark.usefixtures("quotes")
def test_the_first_kept_quote_page_stays_where_it_was(
    october_2026: TradingDays, twse_limiter: RateLimiter
) -> None:
    _listed("6488")
    _kept_page("20261030")
    _fill("6488")
    adapter = TpexAdapter(october_2026, twse_limiter)
    assert adapter._quotes_kept_from() == "20261030"
    assert adapter._quoted_codes(DAY, "day_trade") == {"6488"}


@responses.activate
@pytest.mark.usefixtures("quotes")
def test_later_events_give_the_new_bar_its_factor() -> None:
    _listed("6488")
    DividendEvent.objects.create(
        stock_id="6488", ex_date=date(2026, 10, 29), cash_dividend=Decimal("1"), adj_factor=0.9
    )
    _fill("6488")
    assert DailyPrice.objects.get().adj_factor == pytest.approx(0.9)


@responses.activate
@pytest.mark.usefixtures("quotes")
def test_several_stocks_are_added_together() -> None:
    _listed("6488")
    _listed("3293")
    printed = _fill("3293", "6488", "6488")
    assert sorted(DailyPrice.objects.values_list("stock_id", flat=True)) == ["3293", "6488"]
    assert EtlRun.objects.get().checkpoint == {"stocks": ["3293", "6488"]}
    assert "daily_price: 2 bars added for 2026-10-28" in printed


@responses.activate
@pytest.mark.usefixtures("quotes")
def test_a_bar_already_stored_is_refused_before_any_request() -> None:
    _listed("6488")
    _listed("3293")
    DailyPrice.objects.create(stock_id="6488", date=DAY, close=Decimal("1"), volume=1, turnover=1)
    with pytest.raises(CommandError, match="already have a bar on 2026-10-28: 6488"):
        _fill("3293", "6488")
    assert len(responses.calls) == 0
    assert DailyPrice.objects.count() == 1
    assert not EtlRun.objects.exists()
    assert not RawTpex.objects.exists()


@responses.activate
@pytest.mark.usefixtures("quotes")
@pytest.mark.parametrize("day", ["2026-10-28", "2026-10-29"])
def test_a_day_from_the_first_kept_quote_page_on_is_refused(day: str) -> None:
    _listed("6488")
    _kept_page("20261028")
    with pytest.raises(CommandError, match="quote pages are kept from 20261028"):
        _fill("6488", day=day)
    assert len(responses.calls) == 0
    assert not DailyPrice.objects.exists()
    assert not EtlRun.objects.exists()


@responses.activate
@pytest.mark.usefixtures("quotes")
@pytest.mark.parametrize("day", ["2026-10-25", "2027-03-01"], ids=["sunday", "off-calendar"])
def test_a_day_that_is_not_a_known_session_is_refused(day: str) -> None:
    _listed("6488")
    with pytest.raises(CommandError, match="Error"):
        _fill("6488", day=day)
    assert len(responses.calls) == 0
    assert not DailyPrice.objects.exists()
    assert not EtlRun.objects.exists()


@responses.activate
@pytest.mark.usefixtures("quotes")
@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        (None, "not in stock_master: 6488"),
        ({"list_date": date(2026, 10, 29)}, "not listed on 2026-10-28: 6488"),
        ({"delist_date": date(2026, 10, 28)}, "not listed on 2026-10-28: 6488"),
    ],
    ids=["unknown", "before-listing", "delisted"],
)
def test_a_code_that_was_not_listed_that_day_is_refused(
    fields: dict[str, Any] | None, reason: str
) -> None:
    _listed("3293")
    if fields is not None:
        _listed("6488", **fields)
    with pytest.raises(CommandError, match=reason):
        _fill("3293", "6488")
    assert len(responses.calls) == 0
    assert not DailyPrice.objects.exists()
    assert not EtlRun.objects.exists()


@responses.activate
@pytest.mark.usefixtures("quotes")
def test_a_code_the_quotes_do_not_list_stops_every_bar() -> None:
    _listed("6488")
    _listed("5483")
    with pytest.raises(CommandError, match="quotes of 2026-10-28 do not list: 5483"):
        _fill("6488", "5483")
    assert not DailyPrice.objects.exists()
    run = EtlRun.objects.get()
    assert (run.dataset, run.status, run.rows) == (FILL, "failed", 0)
    assert set(RawTpex.objects.values_list("dataset", flat=True)) == {FILL}


@responses.activate
@pytest.mark.usefixtures("quotes")
def test_a_dry_run_prints_the_bar_and_leaves_nothing() -> None:
    _listed("6488")
    printed = _fill("6488", dry_run=True)
    assert "insert 6488 2026-10-28 open 842.00" in printed
    assert "daily_price: 1 bars to add for 2026-10-28" in printed
    assert printed.rstrip().endswith("nothing was written")
    assert not DailyPrice.objects.exists()
    assert not RawTpex.objects.exists()
    assert not EtlRun.objects.exists()


def test_only_tpex_is_taken() -> None:
    with pytest.raises(CommandError, match="invalid choice"):
        call_command(
            "fill_price_bars", "--source", "twse", "--date", "2026-10-28", "--stock", "2330"
        )


@responses.activate
@pytest.mark.usefixtures("quotes")
def test_the_adapter_never_keeps_these_pages_as_the_day_quotes(
    october_2026: TradingDays, twse_limiter: RateLimiter
) -> None:
    adapter = TpexAdapter(october_2026, twse_limiter)
    with pytest.raises(ValueError, match="daily_price"):
        adapter.day_quotes(DAY, "daily_price")
    assert len(responses.calls) == 0


@responses.activate
@pytest.mark.usefixtures("quotes")
def test_fetching_the_day_the_usual_way_is_unchanged(
    october_2026: TradingDays, twse_limiter: RateLimiter
) -> None:
    adapter = TpexAdapter(october_2026, twse_limiter)
    frame = adapter.normalize("daily_price", adapter.fetch("daily_price", DAY, DAY))
    assert set(frame["stock_id"]) == {"006201", "1240", "2064", "3293", "6488"}
    assert set(RawTpex.objects.values_list("dataset", flat=True)) == {"daily_price"}
    assert RawTpex.objects.count() == 2
