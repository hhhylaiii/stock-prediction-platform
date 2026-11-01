import logging
from collections.abc import Callable
from datetime import date
from typing import Any

import pytest

from apps.etl.adapters.finmind_datasets import LOTS, chip_rows
from apps.etl.errors import SchemaChangedError

Rows = list[dict[str, Any]]
LOTS_KEYS = tuple(LOTS)


@pytest.fixture
def rows(load_fixture: Callable[[str], Any]) -> Callable[[str], Rows]:
    return lambda name: list(load_fixture(f"finmind/{name}.json")["data"])


@pytest.fixture
def october(rows: Callable[[str], Rows]) -> dict[str, Rows]:
    return {
        "institutional": rows("inst_2330_202610"),
        "margin": rows("margin_2330_202610"),
        "lending": rows("sbl_2330_202610"),
        "holding": rows("shareholding_2330_202610"),
    }


def _day(frame: Any, day: date) -> dict[str, Any]:
    return dict(frame.loc[frame["date"] == day].iloc[0])


def test_chip_columns_match_twse_for_the_same_day(october: dict[str, Rows]) -> None:
    frame = chip_rows(stock_id="2330", **october)
    assert len(frame) == 19
    assert _day(frame, date(2026, 10, 1)) == {
        "stock_id": "2330",
        "date": date(2026, 10, 1),
        "fi_net": 3_090_730,
        "it_net": 596_450,
        "dealer_self_net": 249_100,
        "dealer_hedge_net": 353_428,
        "margin_balance": 30_658_000,
        "margin_limit": 6_483_092_000,
        "short_balance": 20_000,
        "sbl_short_balance": 14_198_514,
        "fi_holding_pct": 69.19,
    }


def test_foreign_dealers_count_as_foreign_investors(october: dict[str, Rows]) -> None:
    institutional = [
        {**row, "buy": 1000, "sell": 0} if row["name"] == "Foreign_Dealer_Self" else row
        for row in october["institutional"]
    ]
    frame = chip_rows(stock_id="2330", **{**october, "institutional": institutional})
    assert _day(frame, date(2026, 10, 1))["fi_net"] == 3_091_730


def test_dealers_before_the_split_leave_both_dealer_columns_empty(
    rows: Callable[[str], Rows],
) -> None:
    frame = chip_rows(rows("inst_2330_201411"), [], [], [], "2330")
    before = _day(frame, date(2014, 11, 28))
    assert (before["fi_net"], before["it_net"]) == (10_563_856, -370_508)
    assert (before["dealer_self_net"], before["dealer_hedge_net"]) == (None, None)
    after = _day(frame, date(2014, 12, 1))
    assert (after["dealer_self_net"], after["dealer_hedge_net"]) == (924_000, 450_000)


def test_days_missing_from_one_dataset_keep_the_others(october: dict[str, Rows]) -> None:
    lending = [row for row in october["lending"] if row["date"] != "2026-10-01"]
    frame = chip_rows(stock_id="2330", **{**october, "lending": lending})
    day = _day(frame, date(2026, 10, 1))
    assert day["sbl_short_balance"] is None
    assert day["margin_balance"] == 30_658_000


def test_an_unknown_investor_type_is_a_schema_change(october: dict[str, Rows]) -> None:
    institutional = [*october["institutional"], {**october["institutional"][0], "name": "Other"}]
    with pytest.raises(SchemaChangedError, match="Other"):
        chip_rows(stock_id="2330", **{**october, "institutional": institutional})


@pytest.mark.parametrize("dataset", ["institutional", "margin", "lending", "holding"])
def test_a_repeated_day_is_a_schema_change(october: dict[str, Rows], dataset: str) -> None:
    repeated = [*october[dataset], october[dataset][0]]
    with pytest.raises(SchemaChangedError, match="repeated"):
        chip_rows(stock_id="2330", **{**october, dataset: repeated})


@pytest.mark.parametrize(
    ("dataset", "key", "value"),
    [
        ("institutional", "buy", "1,000"),
        ("margin", "MarginPurchaseTodayBalance", -1),
        ("lending", "SBLShortSalesCurrentDayBalance", None),
        ("margin", "MarginPurchaseTodayBalance", 30658.7),
        ("lending", "SBLShortSalesCurrentDayBalance", 14198514.9),
    ],
)
def test_bad_counts_are_schema_changes(
    october: dict[str, Rows], dataset: str, key: str, value: Any
) -> None:
    changed = [{**october[dataset][0], key: value}, *october[dataset][1:]]
    with pytest.raises(SchemaChangedError, match="bad count"):
        chip_rows(stock_id="2330", **{**october, dataset: changed})


@pytest.mark.parametrize("ratio", [-1.0, 100.5, None])
def test_foreign_holding_outside_0_to_100_is_left_empty(
    october: dict[str, Rows], ratio: float | None, caplog: pytest.LogCaptureFixture
) -> None:
    holding = [{**october["holding"][0], "ForeignInvestmentSharesRatio": ratio}]
    with caplog.at_level(logging.WARNING, logger="apps.etl.adapters.finmind_datasets"):
        frame = chip_rows(stock_id="2330", **{**october, "holding": holding})
    assert _day(frame, date(2026, 10, 1))["fi_holding_pct"] is None
    assert "foreign holding" in caplog.text


@pytest.mark.parametrize("ratio", [0.0, 100.0])
def test_foreign_holding_at_the_bounds_is_kept(october: dict[str, Rows], ratio: float) -> None:
    holding = [{**october["holding"][0], "ForeignInvestmentSharesRatio": ratio}]
    frame = chip_rows(stock_id="2330", **{**october, "holding": holding})
    assert _day(frame, date(2026, 10, 1))["fi_holding_pct"] == ratio


@pytest.mark.parametrize(
    ("dataset", "missing"),
    [
        ("institutional", "name"),
        ("margin", "MarginPurchaseLimit"),
        ("lending", "SBLShortSalesCurrentDayBalance"),
        ("lending", "MarginShortSalesCurrentDayBalance"),
        ("holding", "ForeignInvestmentSharesRatio"),
    ],
)
def test_rows_without_a_key_are_schema_changes(
    october: dict[str, Rows], dataset: str, missing: str
) -> None:
    changed = [{k: v for k, v in row.items() if k != missing} for row in october[dataset]]
    with pytest.raises(SchemaChangedError, match=missing):
        chip_rows(stock_id="2330", **{**october, dataset: changed})


def test_days_without_institutional_rows_mean_no_institutional_trades(
    october: dict[str, Rows], caplog: pytest.LogCaptureFixture
) -> None:
    institutional = [row for row in october["institutional"] if row["date"] != "2026-10-02"]
    with caplog.at_level(logging.INFO, logger="apps.etl.adapters.finmind_datasets"):
        frame = chip_rows(stock_id="2330", **{**october, "institutional": institutional})
    day = _day(frame, date(2026, 10, 2))
    assert (day["fi_net"], day["it_net"], day["dealer_self_net"], day["dealer_hedge_net"]) == (
        0,
        0,
        0,
        0,
    )
    assert "2330: 1 day without institutional rows" in caplog.text


def test_days_before_the_first_institutional_row_stay_empty(
    october: dict[str, Rows], rows: Callable[[str], Rows]
) -> None:
    institutional = [row for row in october["institutional"] if row["date"] >= "2026-10-05"]
    frame = chip_rows(stock_id="2330", **{**october, "institutional": institutional})
    assert _day(frame, date(2026, 10, 2))["fi_net"] is None
    assert _day(frame, date(2026, 10, 5))["fi_net"] is not None


def test_dealer_columns_are_filled_only_after_the_dealer_split(
    rows: Callable[[str], Rows],
) -> None:
    institutional = [r for r in rows("inst_2330_201411") if r["date"] != "2014-11-27"]
    margin = [{"date": "2014-11-27", "stock_id": "2330", **dict.fromkeys(LOTS_KEYS, 1)}]
    frame = chip_rows(institutional, margin, [], [], "2330")
    day = _day(frame, date(2014, 11, 27))
    assert (day["fi_net"], day["it_net"]) == (0, 0)
    assert (day["dealer_self_net"], day["dealer_hedge_net"]) == (None, None)


def test_lending_rows_that_disagree_with_the_short_balance_are_dropped(
    rows: Callable[[str], Rows], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="apps.etl.adapters.finmind_datasets"):
        frame = chip_rows([], rows("margin_2330_202603"), rows("sbl_2330_202603"), [], "2330")
    assert _day(frame, date(2026, 3, 10))["sbl_short_balance"] is None
    assert _day(frame, date(2026, 3, 9))["sbl_short_balance"] == 2_372_680
    assert _day(frame, date(2026, 3, 11))["sbl_short_balance"] == 2_381_680
    assert "2330 2026-03-10: lending row misaligned" in caplog.text


def test_lending_rows_are_kept_when_the_margin_row_is_missing(
    rows: Callable[[str], Rows],
) -> None:
    frame = chip_rows([], [], rows("sbl_2330_202603"), [], "2330")
    assert _day(frame, date(2026, 3, 10))["sbl_short_balance"] == 15_495_000
