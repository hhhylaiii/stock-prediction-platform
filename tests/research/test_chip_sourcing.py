import math
from datetime import date

import pandas as pd
import pytest

from apps.market.models import DailyChip
from apps.research.features.chip import (
    LENDING,
    SOURCED_CHIPS,
    chip_features,
    chip_sourcing,
    lending_features,
    load_chip_starts,
)
from apps.research.features.fundamental import sourced_name
from tests.research.feature_panels import CALENDAR, listing, rising

NAN = math.nan
N = 12
DAYS = CALENDAR.between(date(2014, 1, 2), CALENDAR.end)[:N]
MARGINS = ("chip_margin_chg_5d", "chip_margin_usage", "chip_short_ratio")


def panel(**columns: float | list[float]) -> pd.DataFrame:
    base: dict[str, float | list[float]] = {
        "margin_balance": [10_000.0 + 100 * k for k in range(N)],
        "margin_limit": 50_000.0,
        "short_balance": 2_000.0,
        "sbl_short_balance": [5_000.0 + 10 * k for k in range(N)],
    }
    return listing("2330", rising(N)).assign(**{**base, **columns})


def starts(margin: date | None = DAYS[0], sbl: date | None = DAYS[0]) -> pd.DataFrame:
    return pd.DataFrame(
        {"margin_balance": [margin], "sbl_short_balance": [sbl]},
        index=pd.Index(["2330"], name="stock_id"),
    )


def gaps(rows: pd.DataFrame, first: pd.DataFrame | None = None) -> dict[str, list[int]]:
    """Rows of each lending feature that are missing and whose flag is 0."""
    flags = chip_sourcing(rows, starts() if first is None else first)
    values = lending_features(rows)
    return {
        name: [
            k
            for k in range(N)
            if math.isnan(values[name].iloc[k]) and flags[sourced_name(name)].iloc[k] == 0.0
        ]
        for name in SOURCED_CHIPS
    }


def column(values: list[float], **changes: float) -> list[float]:
    return [changes.get(f"r{k}", value) for k, value in enumerate(values)]


def test_lending_features_are_the_chip_features_they_name() -> None:
    rows = panel().assign(big_holder_pct_chg=NAN, holder_count_chg=NAN)
    rows = rows.assign(fi_net=1.0, it_net=1.0, dealer_self_net=1.0, dealer_hedge_net=1.0)
    rows = rows.assign(fi_holding_pct=30.0)

    pd.testing.assert_frame_equal(
        lending_features(rows), chip_features(rows)[["stock_id", "date", *SOURCED_CHIPS]]
    )
    assert (*MARGINS, "chip_sbl_short_chg_5d") == SOURCED_CHIPS
    assert dict(LENDING) == {
        "margin_balance": ("margin_balance", "margin_limit", "short_balance"),
        "sbl_short_balance": ("sbl_short_balance",),
    }


def test_a_missing_margin_row_after_the_first_is_a_source_gap_on_each_day_reading_it() -> None:
    rows = panel(
        margin_balance=column([10_000.0 + 100 * k for k in range(N)], r6=NAN),
        margin_limit=column([50_000.0] * N, r6=NAN),
        short_balance=column([2_000.0] * N, r6=NAN),
    )
    before = rows.copy()

    found = gaps(rows)

    assert found["chip_margin_chg_5d"] == [6, 11]
    assert found["chip_margin_usage"] == [6]
    assert found["chip_short_ratio"] == [6]
    assert found["chip_sbl_short_chg_5d"] == []
    flags = chip_sourcing(rows, starts())
    assert flags[sourced_name("chip_margin_chg_5d")].iloc[:5].tolist() == [1.0] * 5
    assert list(flags.columns) == ["stock_id", "date", *map(sourced_name, SOURCED_CHIPS)]
    pd.testing.assert_frame_equal(rows, before)


def test_days_before_the_first_row_are_observable_and_a_stock_without_rows_is_not() -> None:
    late = panel(margin_balance=column([10_000.0] * N, r0=NAN, r1=NAN, r2=NAN, r3=NAN))
    late = late.assign(margin_limit=late["margin_limit"].where(late.index >= 4))
    late = late.assign(short_balance=late["short_balance"].where(late.index >= 4))
    none = panel(margin_balance=NAN, margin_limit=NAN, short_balance=NAN)

    assert gaps(late, starts(margin=DAYS[4]))["chip_margin_usage"] == []
    assert gaps(late, starts(margin=DAYS[4]))["chip_margin_chg_5d"] == []
    found = gaps(none, starts(margin=None))
    assert found["chip_margin_usage"] == found["chip_short_ratio"] == list(range(N))
    assert found["chip_margin_chg_5d"] == list(range(5, N))
    assert gaps(none, starts(margin=None))["chip_sbl_short_chg_5d"] == []


def test_a_missing_row_after_the_last_is_a_source_gap() -> None:
    gone = {"r8": NAN, "r9": NAN, "r10": NAN, "r11": NAN}
    rows = panel(
        margin_balance=column([10_000.0] * N, **gone),
        margin_limit=column([50_000.0] * N, **gone),
        short_balance=column([2_000.0] * N, **gone),
    )

    assert gaps(rows)["chip_margin_usage"] == [8, 9, 10, 11]


def test_formula_gaps_on_rows_on_record_are_observable() -> None:
    rows = panel(
        margin_limit=column([50_000.0] * N, r3=NAN, r4=0.0),
        margin_balance=column([10_000.0] * N, r7=0.0),
        short_balance=column([2_000.0] * N, r9=NAN),
    )

    values = lending_features(rows)

    assert values["chip_margin_usage"].isna().tolist() == [k in (3, 4) for k in range(N)]
    assert values["chip_short_ratio"].isna().tolist() == [k in (7, 9) for k in range(N)]
    assert gaps(rows) == {name: [] for name in SOURCED_CHIPS}


def test_missing_shares_are_a_source_gap() -> None:
    rows = panel(shares=column([1_000_000.0] * N, r8=NAN))

    found = gaps(rows)

    assert found["chip_margin_chg_5d"] == [8]
    assert found["chip_sbl_short_chg_5d"] == [8]
    assert found["chip_margin_usage"] == []


def test_lending_rows_have_their_own_first_day() -> None:
    sbl = column([5_000.0] * N, r0=NAN, r1=NAN, r2=NAN, r3=NAN, r4=NAN, r5=NAN, r11=NAN)
    rows = panel(sbl_short_balance=sbl)

    found = gaps(rows, starts(sbl=DAYS[6]))

    assert found["chip_sbl_short_chg_5d"] == [11]
    assert found["chip_margin_chg_5d"] == []
    assert gaps(rows)["chip_sbl_short_chg_5d"] == list(range(5, N))


def test_a_halted_day_without_a_row_carries_the_balance() -> None:
    closes = rising(N)
    closes[6] = NAN
    missing = {"r6": NAN}
    rows = panel(
        margin_balance=column([10_000.0] * N, **missing),
        margin_limit=column([50_000.0] * N, **missing),
        short_balance=column([2_000.0] * N, **missing),
    ).assign(close=closes, has_bar=pd.Series(closes).notna())

    assert gaps(rows)["chip_margin_usage"] == []


@pytest.mark.django_db
def test_chip_starts_are_each_stocks_first_day_with_a_balance_on_record() -> None:
    DailyChip.objects.bulk_create(
        [
            DailyChip(stock_id="2330", date=DAYS[0], fi_net=1),
            DailyChip(stock_id="2330", date=DAYS[1], margin_balance=10),
            DailyChip(stock_id="2330", date=DAYS[2], margin_balance=10, sbl_short_balance=5),
            DailyChip(stock_id="2330", date=DAYS[3], margin_balance=10, sbl_short_balance=5),
            DailyChip(stock_id="1101", date=DAYS[0], fi_net=1),
        ]
    )

    found = load_chip_starts().reindex(["2330", "1101", "9999"])

    assert list(found.columns) == list(LENDING)
    assert list(found.loc["2330"]) == [DAYS[1], DAYS[2]]
    assert found.loc[["1101", "9999"]].isna().all().all()
