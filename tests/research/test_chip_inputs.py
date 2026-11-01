import math
from datetime import date, timedelta

import pandas as pd
import pytest

from apps.market.models import DailyChip, WeeklyHolding
from apps.research.features.chip import (
    attach_chips,
    attach_holding_changes,
    load_chips,
    load_holding_changes,
)

pytestmark = pytest.mark.django_db


def d(text: str) -> date:
    return date.fromisoformat(text)


def week(stock_id: str, data_date: str, big_pct: float, holders: int) -> None:
    day = d(data_date)
    WeeklyHolding.objects.bulk_create(
        WeeklyHolding(
            stock_id=stock_id,
            data_date=day,
            level=level,
            holders=holders if level == 1 else 0,
            shares=1,
            pct=big_pct if level == 15 else 0.0,
            publish_date=day + timedelta(days=1),
        )
        for level in range(1, 18)
    )


def panel_of(days: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"stock_id": "2330", "date": [d(day) for day in days]})


@pytest.mark.spec8
def test_friday_weekly_features_wait_for_the_saturday_publication() -> None:
    week("2330", "2016-02-19", 40.0, 1_000)
    week("2330", "2016-02-26", 41.0, 1_100)
    week("2330", "2016-03-04", 45.0, 1_210)
    days = ["2016-03-04", "2016-03-07"]

    start, end = d(days[0]), d(days[-1])
    attached = attach_holding_changes(panel_of(days), load_holding_changes(start, end))

    friday, monday = attached.iloc[0], attached.iloc[1]
    assert friday["big_holder_pct_chg"] == pytest.approx(1.0)
    assert friday["holder_count_chg"] == pytest.approx(0.1)
    assert monday["big_holder_pct_chg"] == pytest.approx(4.0)


def test_weekly_changes_go_stale_after_two_weeks() -> None:
    week("2330", "2016-02-19", 40.0, 1_000)
    week("2330", "2016-02-26", 41.0, 1_100)
    days = ["2016-03-11", "2016-03-14"]

    attached = attach_holding_changes(panel_of(days), load_holding_changes(d(days[0]), d(days[-1])))

    assert attached["big_holder_pct_chg"].tolist()[0] == pytest.approx(1.0)
    assert math.isnan(attached["big_holder_pct_chg"].iloc[1])


def test_chip_rows_join_by_stock_and_day() -> None:
    DailyChip.objects.create(stock_id="2330", date=d("2016-03-01"), fi_net=5, margin_balance=7)
    DailyChip.objects.create(stock_id="2330", date=d("2016-02-29"), fi_net=1)
    panel = panel_of(["2016-03-01", "2016-03-02"])

    attached = attach_chips(panel, load_chips(d("2016-03-01"), d("2016-03-02")))

    assert attached["fi_net"].tolist()[0] == 5.0
    assert attached["margin_balance"].tolist()[0] == 7.0
    assert attached[["fi_net", "it_net", "sbl_short_balance"]].iloc[1].isna().all()
    assert attached[["stock_id", "date"]].equals(panel)


def test_no_holdings_give_missing_changes() -> None:
    panel = panel_of(["2016-03-04"])

    attached = attach_holding_changes(panel, load_holding_changes(d("2016-03-04"), d("2016-03-04")))

    assert attached[["big_holder_pct_chg", "holder_count_chg"]].isna().all().all()
