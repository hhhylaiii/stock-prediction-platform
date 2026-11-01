import logging
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any

import pandas as pd
import pytest

from apps.etl.adapters.finmind_datasets import (
    dividend_events,
    price_bars,
    share_counts,
    valuations,
)
from apps.etl.errors import SchemaChangedError
from apps.market.models import ShareChangeReason

Rows = list[dict[str, Any]]


@pytest.fixture
def rows(load_fixture: Callable[[str], Any]) -> Callable[[str], Rows]:
    def _rows(name: str) -> Rows:
        data: Rows = load_fixture(f"finmind/{name}.json")["data"]
        return data

    return _rows


def test_price_bars_match_the_twse_definition(rows: Callable[[str], Rows]) -> None:
    frame = price_bars(rows("price_2330_202610"), "2330")
    assert list(frame.columns) == [
        "stock_id",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
    ]
    bar = frame.set_index("date").loc[date(2026, 10, 28)]
    assert (bar["open"], bar["high"], bar["low"], bar["close"]) == (
        Decimal("2420.0"),
        Decimal("2430.0"),
        Decimal("2410.0"),
        Decimal("2425.0"),
    )
    assert (bar["volume"], bar["turnover"]) == (21_866_835, 52_940_678_675)


def test_zero_prices_mean_no_trade(rows: Callable[[str], Rows]) -> None:
    halted = [{**rows("price_2330_202610")[0], "open": 0, "max": 0, "min": 0, "close": 0}]
    bar = price_bars(halted, "2330").iloc[0]
    assert [bar["open"], bar["high"], bar["low"], bar["close"]] == [None, None, None, None]


@pytest.mark.parametrize(
    "change",
    [
        {"Trading_Volume": None},
        {"stock_id": "2317"},
        {"date": "2026/10/27"},
        {"close": -1.0},
        {"open": "n/a"},
    ],
)
def test_price_rows_that_break_the_layout_are_schema_changes(
    rows: Callable[[str], Rows], change: dict[str, Any]
) -> None:
    broken = [{**rows("price_2330_202610")[0], **change}]
    with pytest.raises(SchemaChangedError):
        price_bars(broken, "2330")


def test_price_rows_without_a_key_are_schema_changes(rows: Callable[[str], Rows]) -> None:
    broken = [{k: v for k, v in rows("price_2330_202610")[0].items() if k != "max"}]
    with pytest.raises(SchemaChangedError, match="max"):
        price_bars(broken, "2330")


def test_no_rows_give_an_empty_frame() -> None:
    assert price_bars([], "2330").empty
    assert dividend_events([], [], "2330").empty
    assert valuations([], "2330").empty
    assert share_counts([], "2330").empty


@pytest.fixture
def esun(rows: Callable[[str], Rows]) -> pd.DataFrame:
    events = dividend_events(rows("dividend_2884"), rows("dividend_result_2884"), "2884")
    return events.set_index("ex_date")


def test_every_ex_date_becomes_one_event(esun: pd.DataFrame) -> None:
    assert len(esun) == 21
    assert list(esun.reset_index().columns) == [
        "ex_date",
        "stock_id",
        "cash_dividend",
        "stock_dividend",
        "adj_factor",
    ]


def test_stock_and_cash_on_one_ex_date_are_merged(esun: pd.DataFrame) -> None:
    event = esun.loc[date(2013, 8, 15)]
    assert (event["cash_dividend"], event["stock_dividend"]) == (Decimal("0.3"), Decimal("1.0"))
    assert event["adj_factor"] == pytest.approx(17.86 / 19.95)


def test_rights_issues_adjust_without_dividends(esun: pd.DataFrame) -> None:
    event = esun.loc[date(2011, 8, 17)]
    assert (event["cash_dividend"], event["stock_dividend"]) == (Decimal("0"), Decimal("0"))
    assert event["adj_factor"] == pytest.approx(18.09 / 18.5)


def test_cash_only_event(esun: pd.DataFrame) -> None:
    event = esun.loc[date(2026, 7, 1)]
    assert (event["cash_dividend"], event["stock_dividend"]) == (Decimal("1.4"), Decimal("0"))
    assert event["adj_factor"] == pytest.approx(33.05 / 34.45)


def test_quarterly_cash_dividends(rows: Callable[[str], Rows]) -> None:
    events = dividend_events(rows("dividend_2330"), rows("dividend_result_2330"), "2330")
    first = events.set_index("ex_date").loc[date(2024, 3, 18)]
    assert len(events) == 11
    assert first["cash_dividend"] == Decimal("3.49978969")
    assert first["adj_factor"] == pytest.approx(749.5 / 753.0)


def test_an_announced_event_without_a_result_has_no_factor(rows: Callable[[str], Rows]) -> None:
    future = {
        **rows("dividend_2330")[-1],
        "date": "2026-12-22",
        "CashExDividendTradingDate": "2026-12-16",
        "AnnouncementDate": "2026-12-01",
    }
    events = dividend_events([future], [], "2330")
    assert "announce_date" not in events.columns
    assert events.iloc[0]["ex_date"] == date(2026, 12, 16)
    assert events.iloc[0]["adj_factor"] is None


@pytest.mark.parametrize("change", [{"before_price": 0.0}, {"after_price": 0.0}])
def test_unusable_reference_prices_leave_the_factor_empty(
    rows: Callable[[str], Rows], change: dict[str, Any]
) -> None:
    result = [{**rows("dividend_result_2330")[0], **change}]
    events = dividend_events([], result, "2330")
    assert events.iloc[0]["adj_factor"] is None


@pytest.mark.parametrize(
    ("name", "missing"),
    [("dividend_2330", "CashExDividendTradingDate"), ("dividend_result_2330", "after_price")],
)
def test_dividend_rows_without_a_key_are_schema_changes(
    rows: Callable[[str], Rows], name: str, missing: str
) -> None:
    broken = [{k: v for k, v in rows(name)[0].items() if k != missing}]
    policy, result = (broken, []) if name == "dividend_2330" else ([], broken)
    with pytest.raises(SchemaChangedError, match=missing):
        dividend_events(policy, result, "2330")


def test_valuations_store_the_yield_as_a_fraction(rows: Callable[[str], Rows]) -> None:
    frame = valuations(rows("per_2330_202610"), "2330").set_index("date")
    day = frame.loc[date(2026, 10, 27)]
    assert (day["pe"], day["pb"], day["dividend_yield"]) == pytest.approx((27.82, 9.68, 0.0092))


@pytest.mark.parametrize(
    ("change", "column"),
    [({"PER": 0.0}, "pe"), ({"PBR": 0.0}, "pb"), ({"dividend_yield": 150.0}, "dividend_yield")],
)
def test_impossible_valuations_become_blank(
    rows: Callable[[str], Rows], change: dict[str, Any], column: str
) -> None:
    frame = valuations([{**rows("per_2330_202610")[0], **change}], "2330")
    assert frame.iloc[0][column] is None


def test_share_counts_are_kept_only_when_they_change(rows: Callable[[str], Rows]) -> None:
    frame = share_counts(rows("shareholding_2884_2013q3"), "2884")
    assert frame.to_dict("records") == [
        {
            "stock_id": "2884",
            "effective_date": date(2013, 8, 1),
            "shares": 5_010_700_000,
            "reason": ShareChangeReason.UNKNOWN,
        },
        {
            "stock_id": "2884",
            "effective_date": date(2013, 9, 6),
            "shares": 5_524_300_000,
            "reason": ShareChangeReason.UNKNOWN,
        },
    ]


def test_zero_share_counts_are_skipped(rows: Callable[[str], Rows]) -> None:
    data = rows("shareholding_2884_2013q3")
    zeroed = [{**data[0], "NumberOfSharesIssued": 0}, *data[1:]]
    frame = share_counts(zeroed, "2884")
    assert frame.iloc[0]["effective_date"] == date(2013, 8, 2)


def test_non_numeric_reference_prices_are_schema_changes(rows: Callable[[str], Rows]) -> None:
    broken = [{**rows("dividend_result_2330")[0], "before_price": "753"}]
    with pytest.raises(SchemaChangedError, match="bad prices"):
        dividend_events([], broken, "2330")


def test_missing_valuations_become_blank(rows: Callable[[str], Rows]) -> None:
    frame = valuations(
        [{**rows("per_2330_202610")[0], "PER": None, "dividend_yield": None}], "2330"
    )
    assert (frame.iloc[0]["pe"], frame.iloc[0]["dividend_yield"]) == (None, None)


def test_events_outside_the_requested_range_are_left_out(rows: Callable[[str], Rows]) -> None:
    events = dividend_events(
        rows("dividend_2884"),
        rows("dividend_result_2884"),
        "2884",
        start=date(2013, 1, 1),
        end=date(2013, 12, 31),
    )
    assert events["ex_date"].tolist() == [date(2013, 8, 15)]


@pytest.mark.parametrize(
    ("name", "change"),
    [
        ("per_2330_202610", {"PER": "27.82"}),
        ("per_2330_202610", {"dividend_yield": True}),
        ("shareholding_2884_2013q3", {"NumberOfSharesIssued": 5010700000.0}),
        ("shareholding_2884_2013q3", {"NumberOfSharesIssued": "5010700000"}),
    ],
)
def test_values_of_the_wrong_type_are_schema_changes(
    rows: Callable[[str], Rows], name: str, change: dict[str, Any]
) -> None:
    broken = [{**rows(name)[0], **change}]
    normalize = valuations if name.startswith("per") else share_counts
    with pytest.raises(SchemaChangedError, match="bad"):
        normalize(broken, "2884" if name.startswith("share") else "2330")


def test_missing_share_counts_are_skipped(rows: Callable[[str], Rows]) -> None:
    data = rows("shareholding_2884_2013q3")
    frame = share_counts([{**data[0], "NumberOfSharesIssued": None}, *data[1:]], "2884")
    assert frame.iloc[0]["effective_date"] == date(2013, 8, 2)


def test_a_capital_reduction_becomes_a_factor_only_event(
    rows: Callable[[str], Rows], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="apps.etl.adapters.finmind_datasets"):
        events = dividend_events([], [], "2412", reductions=rows("capred_2412"))
    assert caplog.text == ""
    event = events.iloc[0]
    assert event["ex_date"] == date(2011, 1, 25)
    assert (event["cash_dividend"], event["stock_dividend"]) == (Decimal("0"), Decimal("0"))
    assert event["adj_factor"] == pytest.approx(88.87 / 73.1)


def test_a_split_becomes_a_factor_only_event(rows: Callable[[str], Rows]) -> None:
    splits = [row for row in rows("split_all") if row["stock_id"] == "0050"]
    event = dividend_events([], [], "0050", splits=splits).iloc[0]
    assert event["ex_date"] == date(2025, 6, 18)
    assert event["adj_factor"] == pytest.approx(47.16 / 188.65)


def test_factors_on_one_day_multiply_and_are_flagged(
    rows: Callable[[str], Rows], caplog: pytest.LogCaptureFixture
) -> None:
    result = {**rows("dividend_result_2330")[0], "stock_id": "2412", "date": "2011-01-25"}
    with caplog.at_level(logging.WARNING, logger="apps.etl.adapters.finmind_datasets"):
        events = dividend_events([], [result], "2412", reductions=rows("capred_2412"))
    expected = (749.5 / 753.0) * (88.87 / 73.1)
    assert events.iloc[0]["adj_factor"] == pytest.approx(expected)
    assert "TaiwanStockCapitalReductionReferencePrice, TaiwanStockDividendResult" in caplog.text


@pytest.mark.parametrize(
    "change", [{"PostReductionReferencePrice": 0.0}, {"ClosingPriceonTheLastTradingDay": -1.0}]
)
def test_unusable_reduction_prices_leave_the_factor_empty(
    rows: Callable[[str], Rows], change: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    reduction = [{**rows("capred_2412")[0], **change}]
    with caplog.at_level(logging.WARNING, logger="apps.etl.adapters.finmind_datasets"):
        events = dividend_events([], [], "2412", reductions=reduction)
    assert events.iloc[0]["adj_factor"] is None
    assert "TaiwanStockCapitalReductionReferencePrice has no usable reference price" in caplog.text


@pytest.mark.parametrize(
    ("kind", "name", "missing"),
    [
        ("reductions", "capred_2412", "PostReductionReferencePrice"),
        ("splits", "split_all", "after_price"),
    ],
)
def test_capital_change_rows_without_a_key_are_schema_changes(
    rows: Callable[[str], Rows], kind: str, name: str, missing: str
) -> None:
    row = {k: v for k, v in rows(name)[0].items() if k != missing}
    stock = str(row["stock_id"])
    reductions, splits = ([row], []) if kind == "reductions" else ([], [row])
    with pytest.raises(SchemaChangedError, match=missing):
        dividend_events([], [], stock, reductions=reductions, splits=splits)


def test_each_capital_reduction_is_its_own_event(rows: Callable[[str], Rows]) -> None:
    events = dividend_events([], [], "5371", reductions=rows("capred_5371"))
    assert events["ex_date"].tolist() == [date(2014, 8, 29), date(2016, 8, 26)]
    assert events["adj_factor"].tolist() == pytest.approx([49.47 / 39.6, 34.81 / 29.85])


def test_a_missing_dividend_result_keeps_the_split_factor_and_warns(
    rows: Callable[[str], Rows], caplog: pytest.LogCaptureFixture
) -> None:
    policy = {
        **rows("dividend_2330")[0],
        "stock_id": "0050",
        "CashExDividendTradingDate": "2025-06-18",
    }
    splits = [row for row in rows("split_all") if row["stock_id"] == "0050"]
    with caplog.at_level(logging.WARNING, logger="apps.etl.adapters.finmind_datasets"):
        event = dividend_events([policy], [], "0050", splits=splits).iloc[0]
    assert event["adj_factor"] == pytest.approx(47.16 / 188.65)
    assert "0050 2025-06-18: no ex-right result" in caplog.text


def test_a_repeated_row_from_one_source_counts_once(
    rows: Callable[[str], Rows], caplog: pytest.LogCaptureFixture
) -> None:
    result = rows("dividend_result_2330")[0]
    with caplog.at_level(logging.WARNING, logger="apps.etl.adapters.finmind_datasets"):
        events = dividend_events([], [result, result], "2330")
    assert events.iloc[0]["adj_factor"] == pytest.approx(749.5 / 753.0)
    assert "repeated TaiwanStockDividendResult" in caplog.text


def test_an_unusable_source_does_not_discard_the_others(rows: Callable[[str], Rows]) -> None:
    result = {**rows("dividend_result_2330")[0], "stock_id": "2412", "date": "2011-01-25"}
    bad = [{**rows("capred_2412")[0], "PostReductionReferencePrice": 0.0}]
    event = dividend_events([], [result], "2412", reductions=bad).iloc[0]
    assert event["adj_factor"] == pytest.approx(749.5 / 753.0)
