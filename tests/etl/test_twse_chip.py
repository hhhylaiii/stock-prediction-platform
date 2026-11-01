import copy
import logging
from collections.abc import Callable
from datetime import date
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
import responses
from django.core.management import call_command

from apps.etl.adapters.twse import TwseAdapter
from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.models import EtlRun, RawTwse
from apps.etl.ratelimit import RateLimiter
from apps.market.calendar import NotATradingDayError, TradingDays
from apps.market.models import DailyChip

BASE = "https://www.twse.com.tw/rwd/zh"
DAY = date(2026, 10, 28)
QUOTES_FIXTURE = "twse/daily_price/mi_index_20261028.json"
QUIET_QUOTE = ["1213", "大飲", "7,001", "6", "51,877", "7.35", "7.44", "7.35", "7.44"]
QUIET_QUOTE_TAIL = ["<p> </p>", "0.00", "7.17", "1", "7.69", "1", "0.00"]
DATASETS = {
    "institutional": (
        f"{BASE}/fund/T86",
        {"date": "20261028", "selectType": "ALLBUT0999", "response": "json"},
        "twse/institutional/t86_20261028.json",
    ),
    "margin": (
        f"{BASE}/marginTrading/MI_MARGN",
        {"date": "20261028", "selectType": "ALL", "response": "json"},
        "twse/margin/mi_margn_20261028.json",
    ),
    "sbl": (
        f"{BASE}/marginTrading/TWT93U",
        {"date": "20261028", "response": "json"},
        "twse/sbl/twt93u_20261028.json",
    ),
    "foreign_holding": (
        f"{BASE}/fund/MI_QFIIS",
        {"date": "20261028", "selectType": "ALLBUT0999", "response": "json"},
        "twse/foreign_holding/mi_qfiis_20261028.json",
    ),
    "day_trade": (
        f"{BASE}/dayTrading/TWTB4U",
        {"date": "20261028", "selectType": "All", "response": "json"},
        "twse/day_trade/twtb4u_20261028.json",
    ),
}
NOT_PUBLISHED = {
    "institutional": "twse/institutional/t86_20261101_no_data.json",
    "margin": "twse/margin/mi_margn_20261101_no_data.json",
    "sbl": "twse/sbl/twt93u_20261101_empty.json",
    "foreign_holding": "twse/foreign_holding/mi_qfiis_20261101_after_max.json",
    "day_trade": "twse/day_trade/twtb4u_20261101_empty.json",
}


@pytest.fixture
def adapter(october_2026: TradingDays, twse_limiter: RateLimiter) -> TwseAdapter:
    return TwseAdapter(october_2026, twse_limiter)


@pytest.fixture
def serve(load_fixture: Callable[[str], Any]) -> Callable[[str], dict[str, Any]]:
    def _serve(dataset: str) -> dict[str, Any]:
        url, _, fixture = DATASETS[dataset]
        payload: dict[str, Any] = load_fixture(fixture)
        responses.get(url, json=payload)
        return payload

    return _serve


@pytest.fixture
def quotes_loaded(load_fixture: Callable[[str], Any]) -> None:
    """The day's MI_INDEX as daily_price stores it, plus 1213, which T86 leaves out."""
    payload = load_fixture(QUOTES_FIXTURE)
    table = next(t for t in payload["tables"] if "每日收盤行情" in (t.get("title") or ""))
    table["data"].append(QUIET_QUOTE + QUIET_QUOTE_TAIL)
    RawTwse.objects.create(
        dataset="daily_price",
        source_hash="quotes-20261028",
        request_params={"date": "20261028", "type": "ALLBUT0999", "response": "json"},
        payload=payload,
    )


def _load(adapter: TwseAdapter, dataset: str) -> dict[str, dict[str, Any]]:
    frame = adapter.normalize(dataset, adapter.fetch(dataset, DAY, DAY))
    assert set(frame["date"]) == {DAY}
    return {
        str(row["stock_id"]): {str(key): value for key, value in row.items()}
        for row in frame.to_dict("records")
    }


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize("dataset", sorted(DATASETS))
def test_fetch_sends_the_day_and_stores_raw_under_the_dataset(
    adapter: TwseAdapter, serve: Callable[[str], dict[str, Any]], dataset: str
) -> None:
    payload = serve(dataset)
    adapter.fetch(dataset, DAY, DAY)
    stored = RawTwse.objects.get()
    assert (stored.dataset, stored.payload) == (dataset, payload)
    assert stored.request_params == DATASETS[dataset][1]
    sent = parse_qs(urlparse(responses.calls[0].request.url or "").query)
    assert {key: values[0] for key, values in sent.items()} == stored.request_params


@responses.activate
@pytest.mark.django_db
@pytest.mark.usefixtures("quotes_loaded")
def test_institutional_nets_add_foreign_dealers_and_split_dealers(
    adapter: TwseAdapter, serve: Callable[[str], dict[str, Any]]
) -> None:
    serve("institutional")
    rows = _load(adapter, "institutional")
    assert rows["2330"] == {
        "stock_id": "2330",
        "date": DAY,
        "fi_net": -3_175_021,
        "it_net": 844_056,
        "dealer_self_net": 22_050,
        "dealer_hedge_net": 240_125,
    }
    assert rows["2317"]["fi_net"] == 11_299_686


@responses.activate
@pytest.mark.django_db
@pytest.mark.usefixtures("quotes_loaded")
def test_institutional_foreign_dealer_shares_count_toward_fi_net(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture(DATASETS["institutional"][2])
    tsmc = next(row for row in payload["data"] if row[0] == "2330")
    tsmc[7] = "1,000"
    responses.get(DATASETS["institutional"][0], json=payload)
    assert _load(adapter, "institutional")["2330"]["fi_net"] == -3_174_021


@responses.activate
@pytest.mark.django_db
@pytest.mark.usefixtures("quotes_loaded")
def test_stock_quoted_but_missing_from_t86_nets_zero(
    adapter: TwseAdapter, serve: Callable[[str], dict[str, Any]]
) -> None:
    serve("institutional")
    rows = _load(adapter, "institutional")
    assert {key: rows["1213"][key] for key in ("fi_net", "it_net")} == {"fi_net": 0, "it_net": 0}
    assert (rows["1213"]["dealer_self_net"], rows["1213"]["dealer_hedge_net"]) == (0, 0)
    assert set(rows) == {"0050", "00647L", "00682U", "1101", "1213", "2317", "2330"}


@responses.activate
@pytest.mark.django_db
def test_institutional_waits_for_the_days_quotes(
    adapter: TwseAdapter, serve: Callable[[str], dict[str, Any]]
) -> None:
    serve("institutional")
    raw = adapter.fetch("institutional", DAY, DAY)
    with pytest.raises(SourceNotReadyError, match="daily_price"):
        adapter.normalize("institutional", raw)


@responses.activate
@pytest.mark.django_db
def test_margin_lots_become_shares(
    adapter: TwseAdapter, serve: Callable[[str], dict[str, Any]]
) -> None:
    serve("margin")
    rows = _load(adapter, "margin")
    assert rows["2330"] == {
        "stock_id": "2330",
        "date": DAY,
        "margin_balance": 35_332_000,
        "margin_limit": 6_483_092_000,
        "short_balance": 5_000,
    }
    assert rows["1213"]["margin_balance"] == 0


@responses.activate
@pytest.mark.django_db
def test_margin_short_balance_in_shares_matches_twt93u(
    adapter: TwseAdapter, serve: Callable[[str], dict[str, Any]], load_fixture: Callable[[str], Any]
) -> None:
    serve("margin")
    margin = _load(adapter, "margin")
    lending = load_fixture(DATASETS["sbl"][2])
    in_shares = {row[0]: int(row[6].replace(",", "")) for row in lending["data"] if row[0]}
    assert {code: row["short_balance"] for code, row in margin.items()} == in_shares


@responses.activate
@pytest.mark.django_db
def test_margin_unit_change_is_refused(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture(DATASETS["margin"][2])
    payload["tables"][0]["data"][0][0] = "融資(股)"
    responses.get(DATASETS["margin"][0], json=payload)
    with pytest.raises(SchemaChangedError, match="交易單位"):
        _load(adapter, "margin")


@responses.activate
@pytest.mark.django_db
def test_sbl_reads_the_lending_balance_and_skips_the_total_row(
    adapter: TwseAdapter, serve: Callable[[str], dict[str, Any]]
) -> None:
    serve("sbl")
    rows = _load(adapter, "sbl")
    assert rows["2330"] == {"stock_id": "2330", "date": DAY, "sbl_short_balance": 15_528_514}
    assert set(rows) == {"0050", "1101", "1213", "2317", "2330"}


@responses.activate
@pytest.mark.django_db
def test_foreign_holding_keeps_percent_and_drops_foreign_currency_etf(
    adapter: TwseAdapter, serve: Callable[[str], dict[str, Any]]
) -> None:
    serve("foreign_holding")
    rows = _load(adapter, "foreign_holding")
    assert rows["2330"] == {"stock_id": "2330", "date": DAY, "fi_holding_pct": 68.55}
    assert rows["1213"]["fi_holding_pct"] == 0.01
    assert "00636K" not in rows


@responses.activate
@pytest.mark.django_db
def test_foreign_holding_out_of_range_is_dropped_with_a_warning(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any], caplog: pytest.LogCaptureFixture
) -> None:
    payload = load_fixture(DATASETS["foreign_holding"][2])
    next(row for row in payload["data"] if row[0] == "2330")[7] = 168.55
    responses.get(DATASETS["foreign_holding"][0], json=payload)
    with caplog.at_level(logging.WARNING, logger="apps.etl.adapters.twse_tables"):
        assert _load(adapter, "foreign_holding")["2330"]["fi_holding_pct"] is None
    assert "2330" in caplog.text


@responses.activate
@pytest.mark.django_db
def test_day_trade_volume_in_shares_and_absent_stocks_stay_unknown(
    adapter: TwseAdapter, serve: Callable[[str], dict[str, Any]]
) -> None:
    serve("day_trade")
    rows = _load(adapter, "day_trade")
    assert rows["2330"] == {"stock_id": "2330", "date": DAY, "day_trade_volume": 4_142_000}
    assert "1213" not in rows


@responses.activate
@pytest.mark.django_db
@pytest.mark.usefixtures("quotes_loaded")
def test_each_dataset_fills_its_own_columns_of_one_row(
    adapter: TwseAdapter, serve: Callable[[str], dict[str, Any]]
) -> None:
    for dataset in ("margin", "institutional", "institutional"):
        serve(dataset)
        adapter.upsert(dataset, adapter.normalize(dataset, adapter.fetch(dataset, DAY, DAY)))
    tsmc = DailyChip.objects.get(stock_id="2330", date=DAY)
    assert (tsmc.fi_net, tsmc.margin_balance, tsmc.sbl_short_balance) == (
        -3_175_021,
        35_332_000,
        None,
    )
    assert DailyChip.objects.filter(date=DAY).count() == 7


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize("dataset", sorted(NOT_PUBLISHED))
def test_unpublished_day_raises_not_ready_and_keeps_no_raw(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any], dataset: str
) -> None:
    responses.get(DATASETS[dataset][0], json=load_fixture(NOT_PUBLISHED[dataset]))
    with pytest.raises(SourceNotReadyError):
        adapter.fetch(dataset, DAY, DAY)
    assert RawTwse.objects.count() == 0


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize("dataset", sorted(DATASETS))
def test_previous_day_answer_means_not_published_yet(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any], dataset: str
) -> None:
    payload = load_fixture(DATASETS[dataset][2])
    responses.get(DATASETS[dataset][0], json={**payload, "date": "20261027"})
    with pytest.raises(SourceNotReadyError, match="still serves 20261027"):
        adapter.fetch(dataset, DAY, DAY)


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("dataset", "column"),
    [
        ("institutional", "投信買賣超股數"),
        ("foreign_holding", "全體外資及陸資持股比率"),
        ("day_trade", "當日沖銷交易成交股數"),
    ],
)
def test_lost_column_is_a_schema_change(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any], dataset: str, column: str
) -> None:
    payload = load_fixture(DATASETS[dataset][2])
    body = payload["tables"][1] if "tables" in payload else payload
    body["fields"] = [f"{name}x" if name == column else name for name in body["fields"]]
    responses.get(DATASETS[dataset][0], json=payload)
    with pytest.raises(SchemaChangedError, match=column):
        adapter.fetch(dataset, DAY, DAY)


@responses.activate
@pytest.mark.django_db
def test_margin_summary_without_stock_rows_is_not_ready(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture(DATASETS["margin"][2])
    payload["tables"][1]["data"] = []
    responses.get(DATASETS["margin"][0], json=payload)
    with pytest.raises(SourceNotReadyError, match="no rows"):
        adapter.fetch("margin", DAY, DAY)


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"groups": [{"title": "股票", "span": 2}, {"title": "融券", "span": 6}]}, "groups"),
        ({"data": [["2330", "台積電", "1"]]}, "malformed"),
    ],
    ids=["groups-short", "row-short"],
)
def test_lending_layout_breaks_are_a_schema_change(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any], change: dict[str, Any], message: str
) -> None:
    payload = load_fixture(DATASETS["sbl"][2])
    responses.get(DATASETS["sbl"][0], json={**payload, **change})
    with pytest.raises(SchemaChangedError, match=message):
        adapter.fetch("sbl", DAY, DAY)


@responses.activate
@pytest.mark.django_db
def test_lending_without_column_groups_is_a_schema_change(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any]
) -> None:
    payload = load_fixture(DATASETS["sbl"][2])
    payload["groups"] = []
    responses.get(DATASETS["sbl"][0], json=payload)
    with pytest.raises(SchemaChangedError):
        adapter.fetch("sbl", DAY, DAY)


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize("hints", ["單位：張", None])
def test_unit_hint_change_is_a_schema_change(
    adapter: TwseAdapter, load_fixture: Callable[[str], Any], hints: str | None
) -> None:
    payload = load_fixture(DATASETS["institutional"][2])
    responses.get(DATASETS["institutional"][0], json={**payload, "hints": hints})
    with pytest.raises(SchemaChangedError, match="units"):
        adapter.fetch("institutional", DAY, DAY)


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.append(copy.deepcopy(rows[0])), "repeated"),
        (lambda rows: rows[0].__setitem__(0, "23 30"), "stock code"),
        (lambda rows: rows[0].__setitem__(4, "1,2x"), "count"),
    ],
    ids=["duplicate", "bad-code", "bad-number"],
)
def test_bad_rows_are_a_schema_change(
    adapter: TwseAdapter,
    load_fixture: Callable[[str], Any],
    mutate: Callable[[list[list[str]]], None],
    message: str,
) -> None:
    payload = load_fixture(DATASETS["institutional"][2])
    mutate(payload["data"])
    responses.get(DATASETS["institutional"][0], json=payload)
    with pytest.raises(SchemaChangedError, match=message):
        adapter.normalize("institutional", adapter.fetch("institutional", DAY, DAY))


@responses.activate
def test_non_trading_day_is_rejected_before_any_request(adapter: TwseAdapter) -> None:
    with pytest.raises(NotATradingDayError):
        adapter.fetch("margin", date(2026, 10, 25), date(2026, 10, 25))
    assert len(responses.calls) == 0


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_fetch_command_after_daily_price_zeroes_quiet_stocks(
    load_fixture: Callable[[str], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)
    responses.get(f"{BASE}/afterTrading/MI_INDEX", json=load_fixture(QUOTES_FIXTURE))
    responses.get(DATASETS["institutional"][0], json=load_fixture(DATASETS["institutional"][2]))
    call_command("fetch", source="twse", dataset="daily_price", date="2026-10-28")
    call_command("fetch", source="twse", dataset="institutional", date="2026-10-28")
    quiet = DailyChip.objects.get(stock_id="00647L", date=DAY)
    assert (quiet.fi_net, quiet.it_net, quiet.dealer_self_net, quiet.dealer_hedge_net) == (
        0,
        0,
        0,
        0,
    )
    assert DailyChip.objects.get(stock_id="2330", date=DAY).fi_net == -3_175_021
    run = EtlRun.objects.get(dataset="institutional")
    assert (run.status, run.rows) == ("success", 6)
