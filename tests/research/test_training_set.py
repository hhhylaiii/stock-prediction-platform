from datetime import date

import pandas as pd
import pytest

from apps.market.models import StockMaster
from apps.research.features.store import read_features
from apps.research.features.transforms import STOCK_LEVEL
from apps.research.models import FeatureDaily, LabelDaily
from apps.research.training import LINEAR_FEATURES, load_training_set

pytestmark = pytest.mark.django_db
DAYS = [date(2016, 3, 30), date(2016, 3, 31), date(2016, 4, 1)]


def d(text: str) -> date:
    return date.fromisoformat(text)


def feature(stock_id: str, day: date, value: float | None, universe: bool = True) -> FeatureDaily:
    return FeatureDaily(
        stock_id=stock_id, date=day, in_universe=universe, tech_ret_1d=value, mkt_taiex_ret_5d=0.1
    )


def label(stock_id: str, day: date, rank: float | None, untradable: bool | None) -> LabelDaily:
    return LabelDaily(
        stock_id=stock_id,
        date=day,
        horizon=20,
        fwd_rank_pct=rank,
        fwd_ret_open=0.01,
        untradable_t1=untradable,
    )


def test_training_rows_are_tradable_universe_rows_with_a_label() -> None:
    FeatureDaily.objects.bulk_create(
        [
            feature("1101", DAYS[0], 1.0),
            feature("1102", DAYS[0], 2.0),
            feature("1103", DAYS[0], 3.0),
            feature("1104", DAYS[0], 4.0),
            feature("1105", DAYS[0], 50.0, universe=False),
            feature("1106", DAYS[0], None),
        ]
    )
    LabelDaily.objects.bulk_create(
        [
            label("1101", DAYS[0], 0.2, False),
            label("1102", DAYS[0], 0.4, True),
            label("1103", DAYS[0], 0.6, None),
            label("1104", DAYS[0], None, False),
            label("1105", DAYS[0], 0.8, False),
            label("1106", DAYS[0], 1.0, False),
            LabelDaily(
                stock_id="1102", date=DAYS[0], horizon=5, fwd_rank_pct=0.9, untradable_t1=False
            ),
        ]
    )

    rows = load_training_set(DAYS[0], DAYS[0], 20, "fwd_rank_pct")

    assert rows["stock_id"].tolist() == ["1101", "1106"]
    clipped = pd.Series([1.03, 2.0, 3.0, 3.97])
    z = (1.03 - clipped.mean()) / clipped.std()
    assert rows["tech_ret_1d"].tolist() == pytest.approx([z, 0.0])
    assert rows["tech_ret_1d_missing"].tolist() == [0, 1]
    assert rows["fwd_rank_pct"].tolist() == [0.2, 1.0]
    assert not rows[LINEAR_FEATURES].isna().any().any()
    assert "mkt_taiex_ret_5d" not in LINEAR_FEATURES
    flags = [f"{name}_missing" for name in STOCK_LEVEL]
    assert [*STOCK_LEVEL, *flags] == LINEAR_FEATURES
    assert "ind_sector_id" not in LINEAR_FEATURES


def test_reading_spans_months_and_keeps_both_ends() -> None:
    FeatureDaily.objects.bulk_create(
        feature(stock, day, 1.0)
        for stock in ("1101", "2330")
        for day in [d("2016-03-29"), *DAYS, d("2016-04-04")]
    )

    frame = read_features(DAYS[0], DAYS[-1])

    assert sorted(set(frame["date"])) == DAYS
    assert len(frame) == 6
    assert frame["in_universe"].dtype == bool


def test_reading_an_empty_range_gives_typed_columns() -> None:
    frame = read_features(DAYS[0], DAYS[-1])

    assert frame.empty
    assert frame["tech_ret_1d"].dtype == "float64"


@pytest.mark.spec8
def test_training_set_keeps_delisted_stocks() -> None:
    StockMaster.objects.create(
        stock_id="1101",
        name="1101",
        market="TWSE",
        industry="水泥工業",
        delist_date=d("2016-06-30"),
    )
    StockMaster.objects.create(stock_id="2330", name="2330", market="TWSE", industry="半導體業")
    FeatureDaily.objects.bulk_create(
        feature(stock, day, value)
        for stock, value in (("1101", 1.0), ("2330", 2.0))
        for day in DAYS
    )
    LabelDaily.objects.bulk_create(
        label(stock, day, rank, False)
        for stock, rank in (("1101", 0.5), ("2330", 1.0))
        for day in DAYS
    )

    rows = load_training_set(DAYS[0], DAYS[-1], 20, "fwd_rank_pct")

    delisted = set(
        StockMaster.objects.filter(delist_date__isnull=False).values_list("stock_id", flat=True)
    )
    assert delisted & set(rows["stock_id"])
    assert len(rows) == 6
