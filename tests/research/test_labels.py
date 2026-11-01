import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from apps.research.labels import label_panel
from tests.research.conftest import weekday_calendar

CALENDAR = weekday_calendar(date(2014, 1, 1), date(2014, 12, 31))
NAN = math.nan


def d(text: str) -> date:
    return date.fromisoformat(text)


def listing(
    stock_id: str,
    closes: list[float],
    start: str = "2014-03-03",
    segment: int = 1,
    **columns: object,
) -> pd.DataFrame:
    """One listing of len(closes) trading days; opens sit 0.5 under the close by default."""
    days = list(CALENDAR.between(d(start), d("2014-12-31")))[: len(closes)]
    base: dict[str, object] = {
        "stock_id": stock_id,
        "date": days,
        "segment": segment,
        "adj_close": closes,
        "adj_open": [close - 0.5 for close in closes],
        "in_universe": True,
        "limit_up_open": False,
    }
    return pd.DataFrame({**base, **columns})


def rising(n: int) -> list[float]:
    return [100.0 + k for k in range(n)]


def row(labels: pd.DataFrame, stock_id: str, day: date, horizon: int) -> pd.Series:
    picked = labels[
        (labels["stock_id"] == stock_id) & (labels["date"] == day) & (labels["horizon"] == horizon)
    ]
    assert len(picked) == 1
    return picked.iloc[0]


def test_open_entry_and_close_to_close_returns_count_trading_days() -> None:
    panel = listing("2330", rising(30))
    labels = label_panel(panel)
    first = panel["date"][0]

    five = row(labels, "2330", first, 5)
    twenty = row(labels, "2330", first, 20)

    assert five["fwd_ret_open"] == pytest.approx(105.0 / 100.5 - 1)
    assert five["fwd_ret"] == pytest.approx(105.0 / 100.0 - 1)
    assert twenty["fwd_ret_open"] == pytest.approx(120.0 / 100.5 - 1)
    assert panel["date"][5] == d("2014-03-10")


def test_halted_exit_day_uses_the_last_close() -> None:
    closes = rising(30)
    closes[5] = NAN
    panel = listing("2330", closes, adj_open=[NAN if math.isnan(c) else c - 0.5 for c in closes])

    five = row(label_panel(panel), "2330", panel["date"][0], 5)

    assert five["fwd_ret_open"] == pytest.approx(104.0 / 100.5 - 1)


def test_listing_that_ends_inside_the_horizon_settles_at_its_last_close() -> None:
    panel = pd.concat([listing("1101", rising(10)), listing("2330", rising(30))], ignore_index=True)
    labels = label_panel(panel)
    gone = panel[panel["stock_id"] == "1101"].reset_index(drop=True)

    seventh = row(labels, "1101", gone["date"][7], 5)
    last = row(labels, "1101", gone["date"][9], 20)

    assert seventh["fwd_ret_open"] == pytest.approx(109.0 / 107.5 - 1)
    assert seventh["fwd_ret"] == pytest.approx(109.0 / 107.0 - 1)
    assert math.isnan(last["fwd_ret_open"])
    assert last["fwd_ret"] == pytest.approx(0.0)
    assert last["untradable_t1"] is True


def test_rows_whose_horizon_passes_the_loaded_data_are_left_open() -> None:
    panel = listing("2330", rising(30))
    labels = label_panel(panel)

    late = row(labels, "2330", panel["date"][26], 5)
    end = row(labels, "2330", panel["date"][29], 5)

    assert math.isnan(late["fwd_ret_open"])
    assert math.isnan(late["fwd_ret"])
    assert late["untradable_t1"] is False
    assert end["untradable_t1"] is None


def test_untradable_when_the_next_open_is_limit_up_or_missing() -> None:
    limit = [False] * 30
    limit[3] = True
    opens = [c - 0.5 for c in rising(30)]
    opens[6] = NAN
    panel = listing("2330", rising(30), limit_up_open=limit, adj_open=opens)
    labels = label_panel(panel)

    flags = [row(labels, "2330", panel["date"][k], 5)["untradable_t1"] for k in (1, 2, 5, 7)]

    assert flags == [False, True, True, False]


def test_rank_pct_uses_only_universe_rows_that_can_be_entered() -> None:
    parts = [
        listing(f"1{k:03d}", [100.0] + [100.0 + k] * 29, in_universe=k != 3, adj_open=[99.5] * 30)
        for k in range(1, 6)
    ]
    limit = [False] * 30
    limit[1] = True
    parts.append(listing("2000", [100.0] * 30, limit_up_open=limit))
    labels = label_panel(pd.concat(parts, ignore_index=True))
    first = parts[0]["date"][0]

    ranks = {
        stock_id: row(labels, stock_id, first, 5)["fwd_rank_pct"]
        for stock_id in ("1001", "1002", "1003", "1004", "1005", "2000")
    }

    assert ranks["1001"] == pytest.approx(0.25)
    assert ranks["1005"] == pytest.approx(1.0)
    assert math.isnan(ranks["1003"])
    assert math.isnan(ranks["2000"])


def test_forward_volatility_is_the_annualized_std_of_the_next_twenty_returns() -> None:
    closes = [100.0 * (1.02 if k % 2 else 0.99) ** (k % 3) for k in range(40)]
    panel = listing("2330", closes)
    labels = label_panel(panel)
    steps = np.array(closes[1:21]) / np.array(closes[0:20]) - 1

    twenty = row(labels, "2330", panel["date"][0], 20)
    five = row(labels, "2330", panel["date"][0], 5)

    assert twenty["fwd_vol_20d"] == pytest.approx(steps.std(ddof=1) * math.sqrt(252))
    assert math.isnan(five["fwd_vol_20d"])
    assert five["fwd_up_5pct"] is None


def test_volatility_of_a_listing_that_ends_counts_flat_days_after_it() -> None:
    closes = [100.0, 110.0, 99.0, 108.9]
    panel = pd.concat([listing("1101", closes), listing("2330", rising(30))], ignore_index=True)
    steps = np.array([0.1, -0.1, 0.1] + [0.0] * 17)

    vol = row(label_panel(panel), "1101", panel["date"][0], 20)["fwd_vol_20d"]

    assert vol == pytest.approx(steps.std(ddof=1) * math.sqrt(252))


@pytest.mark.parametrize(
    ("entry", "exit_close", "expected"),
    [(100.5, 110.0, True), (100.5, 105.0, False), (100.0, 105.0, False), (100.0, 105.01, True)],
)
def test_up_five_percent_is_strictly_above_five_percent(
    entry: float, exit_close: float, expected: bool
) -> None:
    closes = [100.0] * 30
    closes[20] = exit_close
    panel = listing("2330", closes, adj_open=[entry] * 30)

    assert row(label_panel(panel), "2330", panel["date"][0], 20)["fwd_up_5pct"] is expected


def test_labels_stay_inside_one_listing() -> None:
    first = listing("5450", rising(10), segment=1)
    second = listing("5450", [500.0] * 25, start="2014-09-01", segment=2)
    other = listing("2330", rising(30), start="2014-08-01")
    labels = label_panel(pd.concat([first, second, other], ignore_index=True))

    old = row(labels, "5450", first["date"][8], 5)

    assert old["fwd_ret_open"] == pytest.approx(109.0 / 108.5 - 1)


def test_unsorted_panel_is_refused() -> None:
    with pytest.raises(ValueError, match="sorted"):
        label_panel(listing("2330", rising(30)).iloc[::-1])


def test_rank_waits_until_the_whole_day_is_settled() -> None:
    ended = listing("1101", [100.0 - k for k in range(25)], adj_open=[99.5] * 25)
    panel = pd.concat(
        [ended, listing("2317", rising(30)), listing("2330", rising(30))], ignore_index=True
    )
    labels = label_panel(panel)
    days = list(ended["date"])

    early = row(labels, "1101", days[5], 20)
    late = row(labels, "1101", days[15], 20)

    assert early["fwd_rank_pct"] == pytest.approx(1 / 3)
    assert late["fwd_ret_open"] == pytest.approx(76.0 / 99.5 - 1)
    assert math.isnan(late["fwd_rank_pct"])
    assert not math.isnan(row(labels, "1101", days[15], 5)["fwd_rank_pct"])


def test_volatility_of_a_day_without_a_close_is_left_open() -> None:
    closes = rising(30)
    closes[3] = NAN
    panel = listing("2330", closes, adj_open=[NAN if math.isnan(c) else c - 0.5 for c in closes])
    labels = label_panel(panel)

    assert math.isnan(row(labels, "2330", panel["date"][3], 20)["fwd_vol_20d"])
    assert not math.isnan(row(labels, "2330", panel["date"][2], 20)["fwd_vol_20d"])
