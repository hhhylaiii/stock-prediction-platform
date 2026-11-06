import copy
import logging
from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal
from typing import Any
from urllib.parse import parse_qs

import pytest
import responses
from django.core.management import call_command
from django.utils import timezone

from apps.etl.adapters.tpex import TpexAdapter
from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.models import EtlRun, RawTpex
from apps.etl.ratelimit import RateLimiter
from apps.market.calendar import NotATradingDayError, TradingDays
from apps.market.models import DailyChip, DailyPrice, IndexDaily

BASE = "https://www.tpex.org.tw/www/zh-tw"
DAY = date(2026, 10, 28)
OTC = f"{BASE}/afterTrading/otc"
QUOTES = f"{BASE}/afterTrading/dailyQuotes"
INDEX = f"{BASE}/indexInfo/inx"
DATASETS = {
    "institutional": (
        f"{BASE}/insti/dailyTrade",
        {"date": "2026/10/28", "type": "Daily", "sect": "EW", "response": "json"},
        "tpex/institutional/daily_trade_20261028.json",
        "tpex/institutional/daily_trade_20261101_empty.json",
    ),
    "margin": (
        f"{BASE}/margin/balance",
        {"date": "2026/10/28", "response": "json"},
        "tpex/margin/balance_20261028.json",
        "tpex/margin/balance_20261101_empty.json",
    ),
    "sbl": (
        f"{BASE}/margin/sbl",
        {"date": "2026/10/28", "response": "json"},
        "tpex/sbl/sbl_20261028.json",
        "tpex/sbl/sbl_20261101_empty.json",
    ),
    "foreign_holding": (
        f"{BASE}/insti/qfii",
        {"date": "2026/10/28", "response": "json"},
        "tpex/foreign_holding/qfii_20261028.json",
        "tpex/foreign_holding/qfii_20261101_empty.json",
    ),
    "day_trade": (
        f"{BASE}/intraday/stat",
        {"date": "2026/10/28", "type": "Daily", "response": "json"},
        "tpex/day_trade/daytrade_20261028.json",
        "tpex/day_trade/daytrade_20261101_empty.json",
    ),
}
QUOTE_FIXTURES = (
    "tpex/daily_price/otc_20261028.json",
    "tpex/daily_price/daily_quotes_20261028.json",
)


@pytest.fixture
def adapter(october_2026: TradingDays, twse_limiter: RateLimiter) -> TpexAdapter:
    return TpexAdapter(october_2026, twse_limiter)


@pytest.fixture
def quotes_loaded(load_fixture: Callable[[str], Any]) -> None:
    """The day's EW quotes as daily_price stores them; 1240 is quoted but not in the flows table."""
    RawTpex.objects.create(
        dataset="daily_price",
        source_hash="otc-20261028",
        request_params={"date": "2026/10/28", "type": "EW", "response": "json"},
        payload=load_fixture(QUOTE_FIXTURES[0]),
    )


def _sent(call_index: int) -> dict[str, str]:
    body = responses.calls[call_index].request.body or ""
    text = body.decode() if isinstance(body, bytes) else body
    return {key: values[0] for key, values in parse_qs(text).items()}


def _load(adapter: TpexAdapter, dataset: str) -> dict[str, dict[str, Any]]:
    frame = adapter.normalize(dataset, adapter.fetch(dataset, DAY, DAY))
    assert set(frame["date"]) == {DAY}
    return {
        str(row["stock_id"]): {str(key): value for key, value in row.items()}
        for row in frame.to_dict("records")
    }


def _serve(load_fixture: Callable[[str], Any], dataset: str) -> dict[str, Any]:
    payload: dict[str, Any] = load_fixture(DATASETS[dataset][2])
    responses.post(DATASETS[dataset][0], json=payload)
    return payload


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize("dataset", sorted(DATASETS))
def test_table_fetch_posts_the_day_and_stores_raw(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any], dataset: str
) -> None:
    payload = _serve(load_fixture, dataset)
    adapter.fetch(dataset, DAY, DAY)
    stored = RawTpex.objects.get()
    assert (stored.dataset, stored.payload) == (dataset, payload)
    assert stored.request_params == DATASETS[dataset][1] == _sent(0)


@responses.activate
@pytest.mark.django_db
@pytest.mark.usefixtures("quotes_loaded")
def test_institutional_nets_by_position_and_zero_for_quiet_stocks(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    _serve(load_fixture, "institutional")
    rows = _load(adapter, "institutional")
    assert rows["6488"] == {
        "stock_id": "6488",
        "date": DAY,
        "fi_net": -8_548,
        "it_net": -266_666,
        "dealer_self_net": 27_800,
        "dealer_hedge_net": 78_072,
    }
    assert rows["2064"]["dealer_hedge_net"] == -203
    assert (rows["1240"]["fi_net"], rows["1240"]["dealer_hedge_net"]) == (0, 0)
    assert set(rows) == {"006201", "1240", "2064", "3293", "6488"}


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    "change",
    [
        {4: "1"},
        {13: "1"},
        {19: "1"},
        {23: "1"},
        {5: "1", 6: "0", 7: "1"},
        {14: "84,801", 16: "27,801"},
    ],
    ids=["fi", "it", "hedge", "total", "foreign-subtotal", "dealer-subtotal"],
)
def test_institutional_columns_must_add_up(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any], change: dict[int, str]
) -> None:
    payload = load_fixture(DATASETS["institutional"][2])
    row = next(row for row in payload["tables"][0]["data"] if row[0] == "6488")
    for column, value in change.items():
        row[column] = value
    responses.post(DATASETS["institutional"][0], json=payload)
    with pytest.raises(SchemaChangedError, match="6488"):
        adapter.normalize("institutional", adapter.fetch("institutional", DAY, DAY))
    assert RawTpex.objects.count() == 1


@responses.activate
@pytest.mark.django_db
@pytest.mark.usefixtures("quotes_loaded")
def test_margin_lots_become_shares_and_unquoted_codes_are_skipped(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    _serve(load_fixture, "margin")
    rows = _load(adapter, "margin")
    assert rows["6488"] == {
        "stock_id": "6488",
        "date": DAY,
        "margin_balance": 11_847_000,
        "margin_limit": 132_028_000,
        "short_balance": 160_000,
    }
    assert "4804" not in rows


@responses.activate
@pytest.mark.django_db
def test_margin_without_lot_unit_is_a_schema_change(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture(DATASETS["margin"][2])
    table = payload["tables"][0]
    table["fields"] = [name.replace("(張)", "(股)") for name in table["fields"]]
    responses.post(DATASETS["margin"][0], json=payload)
    with pytest.raises(SchemaChangedError, match="張"):
        adapter.fetch("margin", DAY, DAY)


@responses.activate
@pytest.mark.django_db
@pytest.mark.usefixtures("quotes_loaded")
def test_sbl_reads_the_lending_balance_by_position(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    _serve(load_fixture, "sbl")
    rows = _load(adapter, "sbl")
    assert rows["6488"] == {"stock_id": "6488", "date": DAY, "sbl_short_balance": 10_499_000}
    assert rows["3293"]["sbl_short_balance"] == 4_159_575
    assert "4804" not in rows


@responses.activate
@pytest.mark.django_db
@pytest.mark.usefixtures("quotes_loaded")
def test_margin_short_balance_in_shares_matches_the_sbl_table(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    _serve(load_fixture, "margin")
    margin = _load(adapter, "margin")
    lending = load_fixture(DATASETS["sbl"][2])["tables"][0]["data"]
    in_shares = {row[0]: int(row[6].replace(",", "")) for row in lending}
    assert {code: row["short_balance"] for code, row in margin.items()} == {
        code: in_shares[code] for code in margin
    }


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize("column", [6, 12], ids=["short", "lent"])
def test_sbl_balance_that_does_not_roll_forward_is_a_schema_change(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any], column: int
) -> None:
    payload = load_fixture(DATASETS["sbl"][2])
    next(row for row in payload["tables"][0]["data"] if row[0] == "6488")[column] = "1"
    responses.post(DATASETS["sbl"][0], json=payload)
    with pytest.raises(SchemaChangedError, match="6488"):
        adapter.normalize("sbl", adapter.fetch("sbl", DAY, DAY))


@responses.activate
@pytest.mark.django_db
@pytest.mark.usefixtures("quotes_loaded")
def test_foreign_holding_percent_text_and_unquoted_codes_skipped(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    _serve(load_fixture, "foreign_holding")
    rows = _load(adapter, "foreign_holding")
    assert rows["6488"] == {"stock_id": "6488", "date": DAY, "fi_holding_pct": 30.29}
    assert rows["2064"]["fi_holding_pct"] == 0.18
    assert set(rows) == {"1240", "2064", "3293", "6488"}


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    "text",
    ["30.29", "%", "abc%", "nan%", "inf%"],
    ids=["no-sign", "sign-only", "word", "nan", "inf"],
)
def test_foreign_holding_percent_without_a_number_and_sign_is_a_schema_change(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any], text: str
) -> None:
    payload = load_fixture(DATASETS["foreign_holding"][2])
    next(row for row in payload["tables"][0]["data"] if row[1] == "6488")[7] = text
    responses.post(DATASETS["foreign_holding"][0], json=payload)
    with pytest.raises(SchemaChangedError, match="percent"):
        adapter.normalize("foreign_holding", adapter.fetch("foreign_holding", DAY, DAY))


@responses.activate
@pytest.mark.django_db
@pytest.mark.usefixtures("quotes_loaded")
def test_foreign_holding_out_of_range_is_dropped_with_a_warning(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any], caplog: pytest.LogCaptureFixture
) -> None:
    payload = load_fixture(DATASETS["foreign_holding"][2])
    next(row for row in payload["tables"][0]["data"] if row[1] == "6488")[7] = "168.55%"
    responses.post(DATASETS["foreign_holding"][0], json=payload)
    with caplog.at_level(logging.WARNING, logger="apps.etl.adapters.exchange_tables"):
        rows = _load(adapter, "foreign_holding")
    assert rows["6488"]["fi_holding_pct"] is None
    assert "6488" in caplog.text


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("change", "error"),
    [
        (lambda t: t[0]["data"][0].__setitem__(0, "3,186,000"), SchemaChangedError),
        (lambda t: t[0]["data"][0].__setitem__(0, "0"), SourceNotReadyError),
        (lambda t: t[0].__setitem__("data", []), SourceNotReadyError),
        (lambda t: t[0]["fields"].__setitem__(0, "總成交股數"), SchemaChangedError),
        (lambda t: t[1]["fields"].__setitem__(3, "成交股數"), SchemaChangedError),
    ],
    ids=["sum-differs", "nothing-yet", "summary-empty", "total-renamed", "volume-renamed"],
)
def test_day_trade_rows_must_add_up_to_the_market_total(
    adapter: TpexAdapter,
    load_fixture: Callable[[str], Any],
    change: Callable[[list[dict[str, Any]]], None],
    error: type[Exception],
) -> None:
    payload = load_fixture(DATASETS["day_trade"][2])
    change(payload["tables"])
    responses.post(DATASETS["day_trade"][0], json=payload)
    with pytest.raises(error, match="當日沖銷"):
        adapter.fetch("day_trade", DAY, DAY)
    assert RawTpex.objects.count() == 1


@responses.activate
@pytest.mark.django_db
@pytest.mark.usefixtures("quotes_loaded")
def test_day_trade_reads_the_stock_table_and_keeps_zero_volume(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    _serve(load_fixture, "day_trade")
    rows = _load(adapter, "day_trade")
    assert rows["6488"] == {"stock_id": "6488", "date": DAY, "day_trade_volume": 2_749_000}
    assert rows["1240"]["day_trade_volume"] == 0
    assert set(rows) == {"006201", "1240", "2064", "3293", "6488"}


@responses.activate
@pytest.mark.django_db
def test_day_trade_without_the_stock_table_is_a_schema_change(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture(DATASETS["day_trade"][2])
    payload["tables"][1]["fields"][0] = "代號"
    responses.post(DATASETS["day_trade"][0], json=payload)
    with pytest.raises(SchemaChangedError, match="證券代號"):
        adapter.fetch("day_trade", DAY, DAY)
    assert RawTpex.objects.count() == 1


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize("dataset", sorted(DATASETS))
def test_chip_datasets_wait_for_the_days_quotes(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any], dataset: str
) -> None:
    _serve(load_fixture, dataset)
    raw = adapter.fetch(dataset, DAY, DAY)
    with pytest.raises(SourceNotReadyError, match="daily_price"):
        adapter.normalize(dataset, raw)


@responses.activate
@pytest.mark.django_db
@pytest.mark.usefixtures("quotes_loaded")
def test_each_dataset_fills_its_own_columns_of_one_row(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    for dataset in (
        "margin",
        "institutional",
        "sbl",
        "foreign_holding",
        "day_trade",
        "institutional",
    ):
        _serve(load_fixture, dataset)
        adapter.upsert(dataset, adapter.normalize(dataset, adapter.fetch(dataset, DAY, DAY)))
    row = DailyChip.objects.get(stock_id="6488", date=DAY)
    assert (row.fi_net, row.margin_balance, row.sbl_short_balance) == (
        -8_548,
        11_847_000,
        10_499_000,
    )
    assert (row.fi_holding_pct, row.day_trade_volume) == (30.29, 2_749_000)
    assert DailyChip.objects.filter(date=DAY).count() == 5


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize("dataset", sorted(DATASETS))
def test_unpublished_day_is_not_ready_and_keeps_no_raw(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any], dataset: str
) -> None:
    responses.post(DATASETS[dataset][0], json=load_fixture(DATASETS[dataset][3]))
    with pytest.raises(SourceNotReadyError):
        adapter.fetch(dataset, DAY, DAY)
    assert RawTpex.objects.count() == 0


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize("dataset", sorted(DATASETS))
def test_previous_day_answer_means_not_published_yet(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any], dataset: str
) -> None:
    payload = load_fixture(DATASETS[dataset][2])
    responses.post(DATASETS[dataset][0], json={**payload, "date": "20261027"})
    with pytest.raises(SourceNotReadyError, match="still serves 20261027"):
        adapter.fetch(dataset, DAY, DAY)
    assert RawTpex.objects.count() == 1


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("dataset", "change", "message"),
    [
        ("institutional", lambda t: t["fields"].pop(), "fields"),
        ("sbl", lambda t: t["fields"].__setitem__(12, "餘額"), "fields"),
        ("margin", lambda t: t["fields"].__setitem__(6, "融資餘額"), "資餘額"),
        ("margin", lambda t: t["data"].append(copy.deepcopy(t["data"][0])), "rows"),
    ],
    ids=["insti-short", "sbl-renamed", "margin-renamed", "margin-total"],
)
def test_layout_change_is_a_schema_change_and_keeps_raw(
    adapter: TpexAdapter,
    load_fixture: Callable[[str], Any],
    dataset: str,
    change: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    payload = load_fixture(DATASETS[dataset][2])
    change(payload["tables"][0])
    responses.post(DATASETS[dataset][0], json=payload)
    with pytest.raises(SchemaChangedError, match=message):
        adapter.fetch(dataset, DAY, DAY)
    assert RawTpex.objects.count() == 1


@responses.activate
def test_non_trading_day_is_rejected_before_any_request(adapter: TpexAdapter) -> None:
    with pytest.raises(NotATradingDayError):
        adapter.fetch("margin", date(2026, 10, 25), date(2026, 10, 25))
    assert len(responses.calls) == 0


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_fetch_commands_after_daily_price_fill_tpex_rows(
    load_fixture: Callable[[str], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)
    responses.post(OTC, json=load_fixture(QUOTE_FIXTURES[0]))
    responses.post(QUOTES, json=load_fixture(QUOTE_FIXTURES[1]))
    responses.post(INDEX, json=load_fixture("tpex/index_daily/inx_202610.json"))
    _serve(load_fixture, "institutional")
    for dataset in ("daily_price", "index_daily", "institutional"):
        call_command("fetch", source="tpex", dataset=dataset, date="2026-10-28")
    assert DailyPrice.objects.filter(date=DAY).count() == 5
    assert IndexDaily.objects.get(index_id="TPEX", date=DAY).close == Decimal("394.6100")
    assert DailyChip.objects.get(stock_id="1240", date=DAY).it_net == 0
    runs = EtlRun.objects.order_by("run_id").values_list("source", "dataset", "status", "rows")
    assert list(runs) == [
        ("tpex", "daily_price", "success", 5),
        ("tpex", "index_daily", "success", 1),
        ("tpex", "institutional", "success", 5),
    ]


@responses.activate
@pytest.mark.django_db
def test_zero_fill_uses_the_latest_ew_quotes_stored_for_the_day(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    earlier = load_fixture(QUOTE_FIXTURES[0])
    table = earlier["tables"][0]
    table["data"] = [row for row in table["data"] if row[0] != "1240"]
    table["totalCount"] = len(table["data"])
    RawTpex.objects.create(
        dataset="daily_price",
        source_hash="otc-earlier",
        request_params={"date": "2026/10/28", "type": "EW", "response": "json"},
        payload=earlier,
        fetched_at=timezone.now() - timedelta(hours=2),
    )
    RawTpex.objects.create(
        dataset="daily_price",
        source_hash="otc-20261028",
        request_params={"date": "2026/10/28", "type": "EW", "response": "json"},
        payload=load_fixture(QUOTE_FIXTURES[0]),
    )
    _serve(load_fixture, "institutional")
    assert _load(adapter, "institutional")["1240"]["fi_net"] == 0


@responses.activate
@pytest.mark.django_db
def test_ew_quotes_answered_for_the_day_count_whatever_day_was_asked(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    RawTpex.objects.create(
        dataset="daily_price",
        source_hash="otc-asked-on-1029",
        request_params={"date": "2026/10/29", "type": "EW", "response": "json"},
        payload=load_fixture(QUOTE_FIXTURES[0]),
    )
    _serve(load_fixture, "institutional")
    assert _load(adapter, "institutional")["1240"]["fi_net"] == 0


@responses.activate
@pytest.mark.django_db
@pytest.mark.usefixtures("quotes_loaded")
def test_an_earlier_day_uses_its_own_ew_quotes_not_the_latest(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    later = load_fixture(QUOTE_FIXTURES[0])
    table = later["tables"][0]
    table["data"] = [row for row in table["data"] if row[0] != "1240"]
    table["totalCount"] = len(table["data"])
    RawTpex.objects.create(
        dataset="daily_price",
        source_hash="otc-20261029",
        request_params={"date": "2026/10/29", "type": "EW", "response": "json"},
        payload={**later, "date": "20261029"},
        fetched_at=timezone.now() + timedelta(hours=24),
    )
    _serve(load_fixture, "institutional")
    assert _load(adapter, "institutional")["1240"]["fi_net"] == 0


@responses.activate
@pytest.mark.django_db
def test_a_day_without_kept_quotes_takes_its_day_trades_for_the_stocks_priced(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    _serve(load_fixture, "day_trade")
    DailyPrice.objects.bulk_create(
        DailyPrice(stock_id=code, date=DAY, close=10, volume=1, turnover=10)
        for code in ("6488", "1240")
    )
    rows = _load(adapter, "day_trade")
    assert set(rows) == {"6488", "1240"}
    assert rows["1240"]["day_trade_volume"] == 0


@responses.activate
@pytest.mark.django_db
def test_other_chip_tables_still_wait_for_the_quotes(
    adapter: TpexAdapter, load_fixture: Callable[[str], Any]
) -> None:
    _serve(load_fixture, "institutional")
    DailyPrice.objects.create(stock_id="6488", date=DAY, close=10, volume=1, turnover=10)
    with pytest.raises(SourceNotReadyError, match="chip rows need its quotes"):
        _load(adapter, "institutional")
