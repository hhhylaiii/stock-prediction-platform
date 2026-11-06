import math
import warnings
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.market.calendar import TradingDays
from apps.market.models import (
    DailyChip,
    DailyPrice,
    DailyValuation,
    DividendEvent,
    FinancialStatement,
    IndexDaily,
    SharesOutstanding,
    StockMaster,
    TradingCalendar,
    WeeklyHolding,
)
from apps.research.features.build import FEATURE_COLUMNS, FRAME_COLUMNS, build_feature_frame
from apps.research.features.registry import feature_names
from apps.research.features.store import export_year, read_partitions, write_features
from apps.research.models import FeatureDaily
from apps.research.universe import load_universe_panel
from tests.research.conftest import weekday_calendar

pytestmark = pytest.mark.django_db
CLOSED = {date(2016, 2, 8) + timedelta(days=k) for k in range(5)}
CALENDAR = TradingDays(
    [
        day
        for day in weekday_calendar(date(2013, 1, 1), date(2016, 12, 31)).between(
            date(2013, 1, 1), date(2016, 12, 31)
        )
        if day not in CLOSED
    ],
    date(2013, 1, 1),
    date(2016, 12, 31),
)


def d(text: str) -> date:
    return date.fromisoformat(text)


def calendar_rows() -> None:
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=open_day) for day, open_day in CALENDAR.all_days()
    )


def listed(stock_id: str, start: str, end: str, step: float = 0.1) -> None:
    StockMaster.objects.create(
        stock_id=stock_id, name=stock_id, market="TWSE", industry="水泥工業", list_date=d(start)
    )
    DailyPrice.objects.bulk_create(
        DailyPrice(
            stock_id=stock_id,
            date=day,
            open=100 + step * k,
            high=101 + step * k,
            low=99 + step * k,
            close=100 + step * k,
            volume=1_000_000,
            turnover=100_000_000,
        )
        for k, day in enumerate(CALENDAR.between(d(start), d(end)))
    )


def taiex_close(k: int) -> float:
    return 8_000.0 + 50.0 * k


def inputs(stock_id: str) -> None:
    """Shares, daily chips, two adjacent TDCC weeks, one valuation day and TAIEX closes."""
    SharesOutstanding.objects.create(
        stock_id=stock_id, effective_date=d("2015-01-02"), shares=1_000_000
    )
    DailyChip.objects.bulk_create(
        DailyChip(stock_id=stock_id, date=day, fi_net=1_000 * day.day)
        for day in CALENDAR.between(d("2015-12-01"), d("2016-03-31"))
    )
    WeeklyHolding.objects.bulk_create(
        WeeklyHolding(
            stock_id=stock_id,
            data_date=d(data_date),
            level=level,
            holders=holders + level - 1,
            shares=1,
            pct=big if level == 12 else 100.0 - big if level == 1 else 0.0,
            publish_date=d(data_date) + timedelta(days=3),
        )
        for data_date, big, holders in [("2016-02-19", 30.0, 1_000), ("2016-02-26", 32.5, 1_100)]
        for level in (1, 12)
    )
    DailyValuation.objects.create(
        stock_id=stock_id, date=d("2016-03-02"), pe=10.0, pb=2.0, dividend_yield=0.03
    )
    IndexDaily.objects.bulk_create(
        IndexDaily(index_id="TAIEX", date=day, close=taiex_close(k))
        for k, day in enumerate(CALENDAR.between(d("2015-12-01"), d("2016-03-31")))
    )


def stored() -> pd.DataFrame:
    rows = FeatureDaily.objects.order_by("date", "stock_id").values_list(*FRAME_COLUMNS)
    frame = pd.DataFrame(list(rows), columns=FRAME_COLUMNS)
    return frame.astype(dict.fromkeys(FEATURE_COLUMNS, "float64"))


def test_feature_daily_holds_the_raw_values_of_both_batches() -> None:
    names = [field.name for field in FeatureDaily._meta.concrete_fields if not field.primary_key]

    assert names == [
        "stock_id",
        "date",
        "in_universe",
        *feature_names(batch=1),
        *feature_names(batch=2),
    ]


def test_the_frame_has_one_row_per_listed_stock_and_trading_day_in_range() -> None:
    listed("2330", "2015-06-01", "2016-03-31")
    listed("1101", "2015-06-01", "2016-02-26", step=0.2)

    frame = build_feature_frame(CALENDAR, d("2016-02-01"), d("2016-03-04"))

    days = CALENDAR.between(d("2016-02-01"), d("2016-03-04"))
    assert list(frame.columns) == FRAME_COLUMNS
    assert set(frame["date"]) == set(days)
    assert len(frame) == 2 * len(days)
    halted = frame[(frame["stock_id"] == "1101") & (frame["date"] > d("2016-02-26"))]
    assert len(halted) == 5
    assert not halted["in_universe"].any()
    first = frame[(frame["stock_id"] == "2330") & (frame["date"] == d("2016-02-02"))].iloc[0]
    k = len(CALENDAR.between(d("2015-06-01"), d("2016-02-02"))) - 1
    assert first["tech_ret_1d"] == pytest.approx((100 + 0.1 * k) / (100 + 0.1 * (k - 1)) - 1)
    assert frame["in_universe"].dtype == bool
    flags = load_universe_panel(CALENDAR, d("2016-02-01"), d("2016-03-04"))
    expected = flags.sort_values(["date", "stock_id"])["in_universe"].astype(bool).tolist()
    assert frame["in_universe"].tolist() == expected
    assert frame.loc[frame["stock_id"] == "2330", "in_universe"].all()


def test_every_input_group_reaches_the_frame() -> None:
    listed("2330", "2015-06-01", "2016-04-29")
    inputs("2330")
    DividendEvent.objects.create(
        stock_id="2330",
        ex_date=d("2016-03-07"),
        cash_dividend=1,
        announce_date=d("2016-02-24"),
    )

    frame = build_feature_frame(CALENDAR, d("2016-03-01"), d("2016-04-15"))

    row = frame[frame["date"] == d("2016-03-02")].iloc[0]
    assert row["chip_fi_net_1d"] == pytest.approx(2_000 / 1_000_000)
    assert row["chip_fi_net_5d"] == pytest.approx((25 + 26 + 29 + 1 + 2) * 1_000 / 1_000_000)
    assert row["chip_big_holder_pct_chg_1w"] == pytest.approx(2.5)
    assert row["chip_holder_count_chg_1w"] == pytest.approx((1_100 + 1_111) / (1_000 + 1_011) - 1)
    assert row["fund_bp"] == pytest.approx(0.5)
    k = len(CALENDAR.between(d("2015-06-01"), d("2016-03-02"))) - 1
    assert row["tech_gap_1d"] == pytest.approx((100 + 0.1 * k) / (100 + 0.1 * (k - 1)) - 1)
    assert row["fund_days_to_ex_date"] == 3.0
    assert row["ind_sector_id"] == 1.0
    assert row["mkt_days_to_quarter_end"] == len(CALENDAR.between(d("2016-03-03"), d("2016-03-31")))
    k = len(CALENDAR.between(d("2015-12-01"), d("2016-03-02"))) - 1
    assert row["mkt_taiex_ret_5d"] == pytest.approx(taiex_close(k) / taiex_close(k - 5) - 1)


def test_industry_neutral_features_reach_the_frame() -> None:
    listed("2330", "2013-01-02", "2016-03-31")
    listed("1101", "2013-01-02", "2016-03-31", step=-0.05)
    quarters = [
        ("2014-Q4", "2015-03-31"),
        *((f"2015-Q{q}", f"2015-{3 * q + 2:02d}-14") for q in (1, 2, 3)),
    ]
    FinancialStatement.objects.bulk_create(
        FinancialStatement(stock_id=stock_id, period=period, eps=eps, announce_date=d(announce))
        for stock_id, eps in (("2330", Decimal("2")), ("1101", Decimal("1")))
        for period, announce in quarters
    )

    frame = build_feature_frame(CALENDAR, d("2016-03-01"), d("2016-03-04"))

    day = frame[frame["date"] == d("2016-03-02")].set_index("stock_id")

    def value(stock_id: str, name: str) -> float:
        return float(day[name].loc[stock_id])

    mom = (value("2330", "tech_mom_250_20") - value("1101", "tech_mom_250_20")) / 2
    ep = (value("2330", "fund_ep") - value("1101", "fund_ep")) / 2
    k = len(CALENDAR.between(d("2013-01-02"), d("2016-03-02"))) - 1
    assert day["in_universe"].all()
    assert value("2330", "ind_mom_neutral") == pytest.approx(mom)
    assert value("1101", "ind_mom_neutral") == pytest.approx(-mom)
    assert value("2330", "ind_ep_neutral") == pytest.approx(ep)
    assert value("1101", "ind_ep_neutral") == pytest.approx(-ep)
    assert mom > 0
    assert value("2330", "fund_ep") == pytest.approx(8 / (100 + 0.1 * k))


def test_monthly_revenue_growth_falls_back_to_the_quarterly_growth_in_the_frame() -> None:
    listed("2330", "2015-06-01", "2016-03-31")
    FinancialStatement.objects.bulk_create(
        FinancialStatement(stock_id="2330", period=period, revenue=revenue, announce_date=d(day))
        for period, revenue, day in [
            ("2014-Q3", 1_000, "2014-11-14"),
            ("2015-Q3", 1_200, "2015-11-16"),
        ]
    )

    frame = build_feature_frame(CALENDAR, d("2016-03-01"), d("2016-03-04"))

    row = frame[frame["date"] == d("2016-03-02")].iloc[0]
    assert row["fund_rev_yoy_q"] == pytest.approx(0.2)
    assert row["fund_rev_yoy"] == pytest.approx(0.2)
    assert row["fund_rev_3m_yoy"] == pytest.approx(0.2)
    assert math.isnan(row["fund_rev_mom"])


def test_a_later_start_gives_the_same_features() -> None:
    listed("2330", "2013-01-02", "2016-03-31")
    listed("1101", "2014-06-02", "2016-03-31", step=-0.05)
    inputs("2330")

    late = build_feature_frame(CALENDAR, d("2016-03-01"), d("2016-03-31"))
    early = build_feature_frame(CALENDAR, d("2015-06-01"), d("2016-03-31"))

    march = early[early["date"] >= d("2016-03-01")].reset_index(drop=True)
    pd.testing.assert_frame_equal(late, march)
    assert late["tech_high_52w_gap"].notna().all()
    mine = late[late["stock_id"] == "2330"]
    assert mine[["chip_fi_net_20d", "mkt_taiex_ret_20d"]].notna().all().all()


def test_writing_replaces_the_range_and_leaves_other_days() -> None:
    listed("2330", "2015-06-01", "2016-03-31")
    frame = build_feature_frame(CALENDAR, d("2016-03-01"), d("2016-03-31"))
    write_features(frame, d("2016-03-01"), d("2016-03-31"))
    inside = frame["date"].between(d("2016-03-07"), d("2016-03-15"))
    changed = frame[inside].assign(tech_ret_1d=np.inf)

    written = write_features(changed, d("2016-03-07"), d("2016-03-31"))

    rows = stored()
    kept = rows["date"] < d("2016-03-07")
    assert written == len(CALENDAR.between(d("2016-03-07"), d("2016-03-15")))
    assert rows["date"].max() == d("2016-03-15")
    assert rows.loc[kept, "tech_ret_1d"].tolist() == pytest.approx(
        frame.loc[frame["date"] < d("2016-03-07"), "tech_ret_1d"].tolist()
    )
    assert rows.loc[~kept, "tech_ret_1d"].isna().all()


def test_writing_refuses_rows_outside_the_range() -> None:
    listed("2330", "2015-06-01", "2016-03-31")
    frame = build_feature_frame(CALENDAR, d("2016-03-01"), d("2016-03-31"))

    with pytest.raises(ValueError, match="outside"):
        write_features(frame, d("2016-03-02"), d("2016-03-31"))


def test_the_year_partition_matches_the_table(tmp_path: Path) -> None:
    listed("2330", "2015-06-01", "2016-03-31")
    frame = build_feature_frame(CALENDAR, d("2016-03-01"), d("2016-03-31"))
    write_features(frame, d("2016-03-01"), d("2016-03-31"))

    path = export_year(2016, tmp_path)

    assert path == tmp_path / "year=2016" / "part-0.parquet"
    pd.testing.assert_frame_equal(pd.read_parquet(path), stored())


def test_a_month_without_values_of_a_feature_exports_without_warnings(tmp_path: Path) -> None:
    listed("2330", "2015-06-01", "2016-03-31")
    frame = build_feature_frame(CALENDAR, d("2016-02-01"), d("2016-03-31"))
    march = frame["date"] >= d("2016-03-01")
    blank = frame.assign(tech_ret_1d=frame["tech_ret_1d"].where(march))
    write_features(blank, d("2016-02-01"), d("2016-03-31"))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        path = export_year(2016, tmp_path)

    assert path is not None
    exported = pd.read_parquet(path)
    pd.testing.assert_frame_equal(exported, stored())
    assert exported["tech_ret_1d"].notna().sum() == int(march.sum())


def test_a_partition_written_before_a_feature_existed_asks_for_a_rebuild(tmp_path: Path) -> None:
    listed("2330", "2015-06-01", "2016-03-31")
    frame = build_feature_frame(CALENDAR, d("2016-03-01"), d("2016-03-31"))
    write_features(frame, d("2016-03-01"), d("2016-03-31"))
    path = export_year(2016, tmp_path)
    assert path is not None
    old = pd.read_parquet(path).drop(columns=["fund_rev_yoy_q", "tech_ret_1d"])
    old.to_parquet(path, index=False)

    with pytest.raises(
        ValueError,
        match=r"2016.*lacks tech_ret_1d, fund_rev_yoy_q.*--start 2016-01-01 --end 2016-12-31",
    ):
        read_partitions(d("2016-03-01"), d("2016-03-31"), tmp_path)


def test_training_reads_the_first_batch_even_from_a_partition_without_the_second(
    tmp_path: Path,
) -> None:
    listed("2330", "2015-06-01", "2016-03-31")
    frame = build_feature_frame(CALENDAR, d("2016-03-01"), d("2016-03-31"))
    write_features(frame, d("2016-03-01"), d("2016-03-31"))
    path = export_year(2016, tmp_path)
    assert path is not None
    second = feature_names(batch=2)
    pd.read_parquet(path).drop(columns=second).to_parquet(path, index=False)

    first = read_partitions(d("2016-03-01"), d("2016-03-31"), tmp_path)

    assert list(first.columns) == ["stock_id", "date", "in_universe", *feature_names(batch=1)]
    pd.testing.assert_frame_equal(first, stored()[list(first.columns)])
    with pytest.raises(ValueError, match=f"lacks {second[0]}, {second[1]}"):
        read_partitions(d("2016-03-01"), d("2016-03-31"), tmp_path, FRAME_COLUMNS)


def test_an_empty_year_leaves_no_partition_and_the_directory_reads(tmp_path: Path) -> None:
    listed("2330", "2015-06-01", "2016-03-31")
    frame = build_feature_frame(CALENDAR, d("2016-03-01"), d("2016-03-31"))
    write_features(frame, d("2016-03-01"), d("2016-03-31"))
    stale = tmp_path / "year=2014" / "part-0.parquet"
    stale.parent.mkdir()
    stale.write_bytes(b"stale")

    assert export_year(2014, tmp_path) is None
    export_year(2016, tmp_path)

    assert not stale.parent.exists()
    assert len(pd.read_parquet(tmp_path)) == len(frame)


def test_a_failed_export_leaves_the_old_partition_and_no_stray_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listed("2330", "2015-06-01", "2016-03-31")
    frame = build_feature_frame(CALENDAR, d("2016-03-01"), d("2016-03-31"))
    write_features(frame, d("2016-03-01"), d("2016-03-31"))
    path = export_year(2016, tmp_path)
    assert path is not None
    before = path.read_bytes()

    def broken(*args: object, **kwargs: object) -> None:
        Path(str(args[1])).write_bytes(b"half")
        raise OSError("disk full")

    monkeypatch.setattr("apps.research.features.store.pq.write_table", broken)

    with pytest.raises(OSError, match="disk full"):
        export_year(2016, tmp_path)

    assert sorted(item.name for item in path.parent.iterdir()) == ["part-0.parquet"]
    assert path.read_bytes() == before


def test_writing_warns_about_values_that_are_not_finite(caplog: pytest.LogCaptureFixture) -> None:
    listed("2330", "2015-06-01", "2016-03-31")
    frame = build_feature_frame(CALENDAR, d("2016-03-01"), d("2016-03-31"))

    write_features(frame.assign(tech_ret_1d=np.inf), d("2016-03-01"), d("2016-03-31"))

    assert f"tech_ret_1d: {len(frame)}" in caplog.text


def test_the_command_rebuilds_one_day_and_its_year_partition(tmp_path: Path, settings: Any) -> None:
    settings.FEATURE_PARQUET_DIR = tmp_path
    calendar_rows()
    listed("2330", "2015-06-01", "2016-03-31")

    call_command("build_features", date=d("2016-03-02"))

    assert list(FeatureDaily.objects.values_list("date", flat=True)) == [d("2016-03-02")]
    assert len(pd.read_parquet(tmp_path / "year=2016" / "part-0.parquet")) == 1


def test_the_command_refuses_a_day_without_trading(tmp_path: Path, settings: Any) -> None:
    settings.FEATURE_PARQUET_DIR = tmp_path
    calendar_rows()
    listed("2330", "2015-06-01", "2016-03-31")

    with pytest.raises(CommandError, match="trading day"):
        call_command("build_features", date=d("2016-02-09"))


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"date": d("2016-03-02"), "start": d("2016-03-01")}, "cannot be combined"),
        ({"date": d("2020-03-02")}, "outside"),
        ({"date": d("2016-04-01")}, "after the last price day"),
    ],
)
def test_the_command_refuses_a_bad_day(
    tmp_path: Path, settings: Any, options: dict[str, date], message: str
) -> None:
    settings.FEATURE_PARQUET_DIR = tmp_path
    calendar_rows()
    listed("2330", "2015-06-01", "2016-03-31")

    with pytest.raises(CommandError, match=message):
        call_command("build_features", **options)


def test_the_command_rebuilds_the_recent_days_by_default(tmp_path: Path, settings: Any) -> None:
    settings.FEATURE_PARQUET_DIR = tmp_path
    calendar_rows()
    listed("2330", "2015-06-01", "2016-03-31")

    call_command("build_features", recent=3)

    days = set(FeatureDaily.objects.values_list("date", flat=True))
    assert days == set(CALENDAR.between(d("2016-03-29"), d("2016-03-31")))


def test_ranges_without_prices_or_trading_days_build_nothing(tmp_path: Path, settings: Any) -> None:
    settings.FEATURE_PARQUET_DIR = tmp_path
    calendar_rows()
    listed("2330", "2015-06-01", "2016-03-31")

    call_command("build_features", start=d("2014-12-01"), end=d("2015-06-05"))
    call_command("build_features", start=d("2016-02-08"), end=d("2016-02-12"))

    days = set(FeatureDaily.objects.values_list("date", flat=True))
    assert days == set(CALENDAR.between(d("2015-06-01"), d("2015-06-05")))
    assert [path.parent.name for path in tmp_path.glob("*/*.parquet")] == ["year=2015"]
    empty = build_feature_frame(CALENDAR, d("2014-12-01"), d("2014-12-31"))
    assert empty.empty
    assert list(empty.columns) == FRAME_COLUMNS
    assert empty["in_universe"].dtype == bool
    assert (empty[feature_names(batch=1)].dtypes == "float64").all()


def test_the_command_works_year_by_year(
    tmp_path: Path, settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.FEATURE_PARQUET_DIR = tmp_path
    calendar_rows()
    listed("2330", "2015-06-01", "2016-03-31")
    calls: list[tuple[date, date]] = []
    original = build_feature_frame

    def spy(calendar: TradingDays, start: date, end: date) -> pd.DataFrame:
        calls.append((start, end))
        return original(calendar, start, end)

    monkeypatch.setattr("apps.research.management.commands.build_features.build_feature_frame", spy)

    call_command("build_features", start=d("2015-12-28"), end=d("2016-01-05"))

    assert calls == [(d("2015-12-28"), d("2015-12-31")), (d("2016-01-01"), d("2016-01-05"))]
    assert sorted(path.parent.name for path in tmp_path.glob("*/*.parquet")) == [
        "year=2015",
        "year=2016",
    ]
    december = pd.read_parquet(tmp_path / "year=2015" / "part-0.parquet")
    assert set(december["date"]) == set(CALENDAR.between(d("2015-12-28"), d("2015-12-31")))


def test_each_year_is_exported_before_the_next_is_built(
    tmp_path: Path, settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.FEATURE_PARQUET_DIR = tmp_path
    calendar_rows()
    listed("2330", "2015-06-01", "2016-03-31")
    original = build_feature_frame

    def failing(calendar: TradingDays, start: date, end: date) -> pd.DataFrame:
        if start.year == 2016:
            raise RuntimeError("worker lost")
        return original(calendar, start, end)

    monkeypatch.setattr(
        "apps.research.management.commands.build_features.build_feature_frame", failing
    )

    with pytest.raises(RuntimeError, match="worker lost"):
        call_command("build_features", start=d("2015-12-28"), end=d("2016-01-05"))

    assert (tmp_path / "year=2015" / "part-0.parquet").exists()


@pytest.mark.spec8
def test_feature_dates_subset_of_calendar(tmp_path: Path, settings: Any) -> None:
    settings.FEATURE_PARQUET_DIR = tmp_path
    calendar_rows()
    listed("2330", "2015-06-01", "2016-03-31")
    listed("1101", "2015-06-01", "2016-03-31")

    call_command("build_features", start=d("2016-02-01"), end=d("2016-02-29"))

    trading = set(
        TradingCalendar.objects.filter(is_trading_day=True).values_list("date", flat=True)
    )
    days = set(FeatureDaily.objects.values_list("date", flat=True))
    assert days <= trading
    assert days.isdisjoint(CLOSED)
    assert days == set(CALENDAR.between(d("2016-02-01"), d("2016-02-29")))
    assert not math.isnan(stored()["tech_ret_1d"].iloc[-1])
