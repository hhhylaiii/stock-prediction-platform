from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from pytest_django import DjangoAssertNumQueries

from apps.market.models import StockMaster
from apps.research.features.store import export_year, read_features, read_partitions
from apps.research.features.transforms import STOCK_LEVEL
from apps.research.models import FeatureDaily, LabelDaily
from apps.research.ranges import months
from apps.research.training import LINEAR_FEATURES, load_labels, load_training_set

pytestmark = pytest.mark.django_db
DAYS = [date(2016, 3, 30), date(2016, 3, 31), date(2016, 4, 1)]


def d(text: str) -> date:
    return date.fromisoformat(text)


def feature(stock_id: str, day: date, value: float | None, universe: bool = True) -> FeatureDaily:
    return FeatureDaily(
        stock_id=stock_id, date=day, in_universe=universe, tech_ret_1d=value, mkt_taiex_ret_5d=0.1
    )


def exported(directory: Path, *years: int) -> Path:
    for year in years:
        export_year(year, directory)
    return directory


def label(stock_id: str, day: date, rank: float | None, untradable: bool | None) -> LabelDaily:
    return LabelDaily(
        stock_id=stock_id,
        date=day,
        horizon=20,
        fwd_rank_pct=rank,
        fwd_ret_open=0.01,
        untradable_t1=untradable,
    )


def test_training_rows_are_tradable_universe_rows_with_a_label(tmp_path: Path) -> None:
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

    rows = load_training_set(DAYS[0], DAYS[0], 20, "fwd_rank_pct", exported(tmp_path, 2016))

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


def test_reading_spans_months_and_keeps_both_ends(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    FeatureDaily.objects.bulk_create(
        feature(stock, day, 1.0)
        for stock in ("1101", "2330")
        for day in [d("2016-03-29"), *DAYS, d("2016-04-04")]
    )

    with django_assert_num_queries(2):
        frame = read_features(DAYS[0], DAYS[-1])

    assert sorted(set(frame["date"])) == DAYS
    assert len(frame) == 6
    assert frame["in_universe"].dtype == bool


def test_reading_an_empty_range_gives_typed_columns() -> None:
    frame = read_features(DAYS[0], DAYS[-1])

    assert frame.empty
    assert frame["tech_ret_1d"].dtype == "float64"


def test_months_cut_a_range_at_month_boundaries() -> None:
    assert list(months(d("2015-12-30"), d("2016-02-02"))) == [
        (d("2015-12-30"), d("2015-12-31")),
        (d("2016-01-01"), d("2016-01-31")),
        (d("2016-02-01"), d("2016-02-02")),
    ]
    assert list(months(d("2016-02-03"), d("2016-02-03"))) == [(d("2016-02-03"), d("2016-02-03"))]


def test_partitions_are_read_across_years_and_keep_both_ends(tmp_path: Path) -> None:
    edges = [d("2015-12-29"), d("2015-12-30"), d("2016-01-04"), d("2016-01-05")]
    FeatureDaily.objects.bulk_create(
        feature(stock, day, 1.0) for stock in ("2330", "1101") for day in edges
    )
    exported(tmp_path, 2015, 2016)

    frame = read_partitions(edges[1], edges[2], tmp_path)

    assert frame["date"].tolist() == [edges[1], edges[1], edges[2], edges[2]]
    assert frame["stock_id"].tolist() == ["1101", "2330", "1101", "2330"]
    assert frame.columns.tolist() == read_features(edges[1], edges[2]).columns.tolist()
    assert frame["in_universe"].dtype == bool
    assert frame["tech_ret_1d"].dtype == "float64"


def test_a_range_without_rows_reads_as_a_typed_empty_frame(tmp_path: Path) -> None:
    FeatureDaily.objects.create(stock_id="1101", date=d("2016-01-04"), in_universe=True)

    frame = read_partitions(DAYS[0], DAYS[-1], exported(tmp_path, 2016))

    assert frame.empty
    assert frame["tech_ret_1d"].dtype == "float64"


def test_a_year_without_a_partition_is_an_error(tmp_path: Path) -> None:
    FeatureDaily.objects.create(stock_id="1101", date=d("2016-01-04"), in_universe=True)
    exported(tmp_path, 2016)

    with pytest.raises(FileNotFoundError, match=r"2015.*build_features"):
        read_partitions(d("2015-12-30"), d("2016-01-04"), tmp_path)


def test_a_missing_partition_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="build_features"):
        read_partitions(DAYS[0], DAYS[-1], tmp_path / "absent")


def test_training_reads_the_exported_partitions(tmp_path: Path) -> None:
    FeatureDaily.objects.bulk_create(feature(stock, DAYS[0], 1.0) for stock in ("1101", "2330"))
    LabelDaily.objects.bulk_create(
        label(stock, DAYS[0], rank, False) for stock, rank in (("1101", 0.5), ("2330", 1.0))
    )
    exported(tmp_path, 2016)
    FeatureDaily.objects.filter(stock_id="2330").delete()

    rows = load_training_set(DAYS[0], DAYS[0], 20, "fwd_rank_pct", tmp_path)

    assert rows["stock_id"].tolist() == ["1101", "2330"]


def test_labels_are_read_across_months_and_keep_both_ends(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    LabelDaily.objects.bulk_create(
        label(stock, day, 0.5, False)
        for stock in ("2330", "1101")
        for day in [d("2016-03-29"), *DAYS, d("2016-04-04")]
    )

    with django_assert_num_queries(2):
        labels = load_labels(DAYS[0], DAYS[-1], 20)

    assert labels["date"].tolist() == [day for day in DAYS for _ in range(2)]
    assert labels["stock_id"].tolist() == ["1101", "2330"] * 3


@pytest.mark.parametrize(
    ("target", "horizon"),
    [
        ("fwd_ret", 20),
        ("untradable_t1", 20),
        ("fwd_rank_pct", 10),
        ("fwd_ret_open", 1),
        ("fwd_vol_20d", 5),
        ("fwd_up_5pct", 5),
    ],
)
def test_only_the_model_targets_and_their_horizons_are_accepted(
    target: str, horizon: int, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match=target):
        load_training_set(DAYS[0], DAYS[-1], horizon, target, tmp_path / "absent")


@pytest.mark.parametrize(
    ("target", "horizon"),
    [("fwd_rank_pct", 5), ("fwd_ret_open", 5), ("fwd_vol_20d", 20), ("fwd_up_5pct", 20)],
)
def test_model_targets_load(target: str, horizon: int, tmp_path: Path) -> None:
    FeatureDaily.objects.create(stock_id="1101", date=d("2016-01-04"), in_universe=True)

    rows = load_training_set(DAYS[0], DAYS[-1], horizon, target, exported(tmp_path, 2016))

    assert rows.empty


@pytest.mark.spec8
def test_training_set_keeps_delisted_stocks(tmp_path: Path) -> None:
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

    rows = load_training_set(DAYS[0], DAYS[-1], 20, "fwd_rank_pct", exported(tmp_path, 2016))

    delisted = set(
        StockMaster.objects.filter(delist_date__isnull=False).values_list("stock_id", flat=True)
    )
    assert delisted & set(rows["stock_id"])
    assert len(rows) == 6
