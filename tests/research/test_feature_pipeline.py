import math
import os
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

from apps.etl.models import EtlRun
from apps.market.calendar import TradingDays
from apps.market.models import (
    DailyChip,
    DailyPrice,
    DailyValuation,
    DividendEvent,
    FinancialStatement,
    IndexDaily,
    MarketHistory,
    MonthlyRevenue,
    SharesOutstanding,
    StockMaster,
    TradingCalendar,
    WeeklyHolding,
)
from apps.research.features import store
from apps.research.features.build import (
    CHECKED,
    FEATURE_COLUMNS,
    FRAME_COLUMNS,
    build_feature_frame,
    current_name,
    observable_frame,
)
from apps.research.features.chip import SOURCED_CHIPS
from apps.research.features.registry import FEATURES, feature_names
from apps.research.features.residual import observable_name, rules_signature
from apps.research.features.store import (
    export_year,
    partition_rules,
    read_columns,
    read_partitions,
    write_features,
)
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
    MarketHistory.objects.create(stock_id=stock_id, start_date=d(start), market="TWSE")
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
        *(spec.name for spec in FEATURES),
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


def test_monthly_revenue_growth_left_missing_on_record_takes_no_fallback() -> None:
    listed("2330", "2015-06-01", "2016-03-31")
    FinancialStatement.objects.bulk_create(
        FinancialStatement(stock_id="2330", period=period, revenue=revenue, announce_date=d(day))
        for period, revenue, day in [
            ("2014-Q3", 1_000, "2014-11-14"),
            ("2015-Q3", 1_200, "2015-11-16"),
        ]
    )
    MonthlyRevenue.objects.bulk_create(
        MonthlyRevenue(
            stock_id="2330",
            period=f"{year}-{month:02d}",
            revenue=0 if (year, month) == (2014, 12) else 100,
            announce_date=date(year, month, 10) + timedelta(days=31),
        )
        for year in (2014, 2015)
        for month in range(1, 13)
    )

    frame = build_feature_frame(CALENDAR, d("2016-03-01"), d("2016-03-04"))

    row = frame[frame["date"] == d("2016-03-02")].iloc[0]
    assert row["fund_rev_yoy_q"] == pytest.approx(0.2)
    assert math.isnan(row["fund_rev_yoy"])
    assert row["fund_rev_3m_yoy"] == pytest.approx(300 / 200 - 1)
    assert row["fund_rev_mom"] == pytest.approx(0.0)


def quarterly_growth(announced: str = "2015-11-16") -> None:
    """2015-Q3 revenue 20% above 2014-Q3, the later quarter announced on the given day."""
    FinancialStatement.objects.bulk_create(
        FinancialStatement(stock_id="2330", period=period, revenue=revenue, announce_date=d(day))
        for period, revenue, day in [
            ("2014-Q3", 1_000, "2014-11-14"),
            ("2015-Q3", 1_200, announced),
        ]
    )


def test_monthly_revenue_takes_the_quarter_only_once_it_goes_stale() -> None:
    listed("2330", "2015-06-01", "2016-04-29")
    quarterly_growth()
    MonthlyRevenue.objects.bulk_create(
        MonthlyRevenue(
            stock_id="2330",
            period=f"{year}-{month:02d}",
            revenue=110 if (year, month) == (2015, 12) else 100,
            announce_date=date(year, month, 10) + timedelta(days=31),
        )
        for year in (2014, 2015)
        for month in range(1, 13)
    )

    frame = build_feature_frame(CALENDAR, d("2016-03-10"), d("2016-03-15")).set_index("date")

    assert frame["fund_rev_yoy"].loc[d("2016-03-11")] == pytest.approx(0.1)
    assert frame["fund_rev_mom"].loc[d("2016-03-11")] == pytest.approx(0.1)
    assert frame["fund_rev_yoy"].loc[d("2016-03-14")] == pytest.approx(0.2)
    assert frame["fund_rev_3m_yoy"].loc[d("2016-03-14")] == pytest.approx(0.2)
    assert math.isnan(frame["fund_rev_mom"].loc[d("2016-03-14")])


def test_the_fallback_waits_for_its_statement_and_ignores_later_delisting() -> None:
    listed("2330", "2015-06-01", "2016-03-31")
    quarterly_growth("2016-03-03")

    frame = build_feature_frame(CALENDAR, d("2016-03-01"), d("2016-03-04")).set_index("date")
    StockMaster.objects.filter(stock_id="2330").update(delist_date=d("2016-06-30"))
    delisted = build_feature_frame(CALENDAR, d("2016-03-01"), d("2016-03-04")).set_index("date")

    assert math.isnan(frame["fund_rev_yoy"].loc[d("2016-03-02")])
    assert frame["fund_rev_yoy"].loc[d("2016-03-03")] == pytest.approx(0.2)
    pd.testing.assert_frame_equal(delisted, frame)


def test_month_masks_reach_any_column_of_the_frame_after_the_fallbacks() -> None:
    listed("2330", "2015-06-01", "2016-04-29")
    FinancialStatement.objects.bulk_create(
        FinancialStatement(stock_id="2330", period=period, revenue=revenue, announce_date=d(day))
        for period, revenue, day in [
            ("2014-Q3", 1_000, "2014-11-14"),
            ("2015-Q3", 1_200, "2015-11-16"),
        ]
    )
    masked = {"fund_rev_yoy_q": frozenset({"2016-03"}), "tech_vol_20d": frozenset({"2016-04"})}

    plain = build_feature_frame(CALENDAR, d("2016-03-28"), d("2016-04-04"))
    frame = build_feature_frame(CALENDAR, d("2016-03-28"), d("2016-04-04"), masked=masked)

    march = frame["date"] < d("2016-04-01")
    assert frame.loc[march, "fund_rev_yoy_q"].isna().all()
    assert frame.loc[~march, "fund_rev_yoy_q"].notna().all()
    assert frame["fund_rev_yoy"].tolist() == pytest.approx([0.2] * len(frame))
    assert frame.loc[~march, "tech_vol_20d"].isna().all()
    assert frame.loc[march, "tech_vol_20d"].tolist() == plain.loc[march, "tech_vol_20d"].tolist()
    others = [name for name in FRAME_COLUMNS if name not in masked]
    pd.testing.assert_frame_equal(frame[others], plain[others])


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
    assert partition_rules(tmp_path, 2016) == rules_signature()


def test_a_partition_written_without_the_rules_stamp_reads_as_unstamped(tmp_path: Path) -> None:
    listed("2330", "2015-06-01", "2016-03-31")
    frame = build_feature_frame(CALENDAR, d("2016-03-01"), d("2016-03-31"))
    write_features(frame, d("2016-03-01"), d("2016-03-31"))
    path = export_year(2016, tmp_path)
    assert path is not None
    pd.read_parquet(path).to_parquet(path, index=False)

    assert partition_rules(tmp_path, 2016) is None
    with pytest.raises(FileNotFoundError, match="no feature partition for 2015"):
        partition_rules(tmp_path, 2015)


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


def swapped_on(
    step: str, path: Path, monkeypatch: pytest.MonkeyPatch, newer_rows: pd.DataFrame
) -> None:
    """Replace the partition with newer_rows the moment pyarrow's step is first called."""
    newer = path.with_name("newer.parquet")
    newer_rows.to_parquet(newer, index=False)
    original = getattr(store.pq, step)

    def replaced(source: Any, *args: Any, **kwargs: Any) -> Any:
        if newer.exists():
            os.replace(newer, path)
        return original(source, *args, **kwargs)

    monkeypatch.setattr(store.pq, step, replaced)


def exported_march(tmp_path: Path) -> Path:
    listed("2330", "2015-06-01", "2016-03-31")
    frame = build_feature_frame(CALENDAR, d("2016-03-01"), d("2016-03-31"))
    write_features(frame, d("2016-03-01"), d("2016-03-31"))
    path = export_year(2016, tmp_path)
    assert path is not None
    return path


def test_a_partition_replaced_before_its_schema_is_read_gives_the_file_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = exported_march(tmp_path)
    expected = read_partitions(d("2016-03-01"), d("2016-03-31"), tmp_path)
    lacking = pd.read_parquet(path).drop(columns=["tech_ret_1d"]).assign(tech_ret_5d=999.0)

    swapped_on("ParquetFile", path, monkeypatch, lacking)
    rows = read_partitions(d("2016-03-01"), d("2016-03-31"), tmp_path)

    pd.testing.assert_frame_equal(rows, expected)
    assert "tech_ret_1d" not in pd.read_parquet(path).columns


def test_a_partition_replaced_before_its_rows_are_read_gives_the_file_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = exported_march(tmp_path)
    expected = stored()
    first = list(read_partitions(d("2016-03-01"), d("2016-03-31"), tmp_path).columns)

    swapped_on("read_table", path, monkeypatch, pd.read_parquet(path).assign(tech_ret_1d=999.0))
    rows = read_partitions(d("2016-03-01"), d("2016-03-31"), tmp_path)

    pd.testing.assert_frame_equal(rows, expected[first])
    assert pd.read_parquet(path)["tech_ret_1d"].eq(999.0).all()

    pd.read_parquet(path).assign(tech_ret_1d=1.5).to_parquet(path, index=False)
    swapped_on("read_table", path, monkeypatch, pd.read_parquet(path).assign(tech_ret_1d=999.0))
    columns = read_columns(d("2016-03-01"), d("2016-03-31"), tmp_path, ["tech_ret_1d"])

    assert columns["tech_ret_1d"].eq(1.5).all()
    assert pd.read_parquet(path)["tech_ret_1d"].eq(999.0).all()


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
    assert partition_rules(tmp_path, 2015) == rules_signature()


def test_a_partial_rebuild_keeps_the_stamp_only_while_the_rules_are_unchanged(
    tmp_path: Path, settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.FEATURE_PARQUET_DIR = tmp_path
    calendar_rows()
    listed("2330", "2015-06-01", "2016-03-31")
    call_command("build_features", start=d("2016-03-01"), end=d("2016-03-31"))
    call_command("build_features", date=d("2016-03-02"))
    assert partition_rules(tmp_path, 2016) == rules_signature()

    monkeypatch.setattr(store, "rules_signature", lambda: "changed")
    call_command("build_features", date=d("2016-03-02"))
    assert partition_rules(tmp_path, 2016) is None
    call_command("build_features", start=d("2016-01-01"), end=d("2016-03-31"))
    assert partition_rules(tmp_path, 2016) == "changed"


def test_an_export_without_a_rebuild_range_stamps_the_current_rules(tmp_path: Path) -> None:
    listed("2330", "2015-06-01", "2016-03-31")
    frame = build_feature_frame(CALENDAR, d("2016-03-01"), d("2016-03-31"))
    write_features(frame, d("2016-03-01"), d("2016-03-31"))

    export_year(2016, tmp_path, rebuilt=(d("2016-03-10"), d("2016-03-31")))
    assert partition_rules(tmp_path, 2016) is None
    export_year(2016, tmp_path)
    assert partition_rules(tmp_path, 2016) == rules_signature()


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


def test_observable_gaps_mark_formula_and_industry_gaps_but_not_missing_inputs() -> None:
    for stock_id in ("2330", "2881", "1101"):
        listed(stock_id, "2015-06-01", "2016-03-31")
    StockMaster.objects.filter(stock_id="2881").update(industry="金融業")
    periods = [(2013, 4), *((2014, q) for q in (1, 2, 3, 4)), *((2015, q) for q in (1, 2, 3))]
    blanks = {"gross_margin": None, "op_margin": None, "debt_ratio": None}
    FinancialStatement.objects.bulk_create(
        FinancialStatement(
            stock_id=stock_id,
            period=f"{year}-Q{quarter}",
            eps=Decimal("-1") if k < 4 else Decimal("1"),
            revenue=1_000,
            net_income=100,
            operating_cf=120,
            roe=0.05,
            **(
                blanks
                if stock_id == "2881"
                else {"gross_margin": 0.4, "op_margin": 0.2, "debt_ratio": 0.3}
            ),
            announce_date=date(year, quarter * 3, 28) + timedelta(days=45),
        )
        for stock_id in ("2330", "2881")
        for k, (year, quarter) in enumerate(periods)
    )

    for stock_id in ("2330", "2881", "1101"):
        SharesOutstanding.objects.create(
            stock_id=stock_id, effective_date=d("2015-01-02"), shares=1_000_000
        )
    DailyChip.objects.bulk_create(
        DailyChip(
            stock_id=stock_id,
            date=day,
            margin_balance=1_000 + 10 * k,
            margin_limit=5_000 if stock_id == "2330" else None,
            short_balance=100,
            sbl_short_balance=500 + 5 * k,
        )
        for stock_id in ("2330", "2881")
        for k, day in enumerate(CALENDAR.between(d("2015-06-01"), d("2016-03-31")))
        if (stock_id, day) != ("2330", d("2016-03-02"))
    )

    listed("2882", "2015-06-01", "2016-03-31")
    SharesOutstanding.objects.create(
        stock_id="2882", effective_date=d("2015-01-02"), shares=1_000_000
    )
    DailyChip.objects.bulk_create(
        DailyChip(stock_id="2882", date=day, margin_balance=1_000, margin_limit=5_000)
        for day in CALENDAR.between(d("2016-03-03"), d("2016-03-31"))
    )
    EtlRun.objects.bulk_create(
        EtlRun(
            source=source,
            dataset="fill:margin",
            start_date=d("2016-03-03"),
            end_date=d("2016-03-03"),
            status=EtlRun.Status.SUCCESS,
        )
        for source in ("twse", "tpex")
    )

    frame = build_feature_frame(CALENDAR, d("2016-03-01"), d("2016-03-04"))
    gaps = observable_frame(CALENDAR, d("2016-03-01"), d("2016-03-04"))

    keys = ["stock_id", "date"]
    pd.testing.assert_frame_equal(
        gaps[keys].sort_values(keys, ignore_index=True),
        frame[keys].sort_values(keys, ignore_index=True),
    )
    row = frame[frame["date"] == d("2016-03-02")].set_index("stock_id")
    seen = gaps[gaps["date"] == d("2016-03-02")].set_index("stock_id")

    def gap(stock_id: str, name: str) -> tuple[bool, bool]:
        value: float = row[name].loc[stock_id]
        return math.isnan(value), bool(seen[observable_name(name)].loc[stock_id])

    assert gap("2330", "fund_eps_yoy") == (True, True)
    assert gap("2330", "fund_ep") == (False, True)
    assert gap("2881", "fund_gross_margin") == (True, True)
    assert gap("2881", "fund_gm_yoy_chg") == (True, True)
    assert gap("2881", "fund_debt_ratio") == (True, False)
    assert gap("1101", "fund_ep") == (True, False)
    assert gap("1101", "fund_rev_yoy") == (True, False)
    assert gap("2330", "chip_margin_usage") == (True, False)
    assert gap("2881", "chip_margin_usage") == (True, True)
    assert gap("2881", "chip_short_ratio") == (False, True)
    assert gap("1101", "chip_short_ratio") == (True, False)
    assert gap("2882", "chip_margin_usage") == (True, True)
    shown = gaps[(gaps["date"] == d("2016-03-03")) & (gaps["stock_id"] == "1101")]
    assert shown[observable_name("chip_short_ratio")].tolist() == [True]
    assert set(SOURCED_CHIPS) <= set(CHECKED)
    now = observable_frame(CALENDAR, d("2016-03-01"), d("2016-03-04"), current=True)
    both = frame.merge(now, on=keys)
    assert len(both) == len(frame)
    for name in CHECKED:
        pd.testing.assert_series_equal(both[name], both[current_name(name)], check_names=False)
    assert all(both[name].notna().any() for name in SOURCED_CHIPS)
    hidden = {name: frozenset({"2016-03"}) for name in SOURCED_CHIPS}
    masked = build_feature_frame(CALENDAR, d("2016-03-01"), d("2016-03-04"), masked=hidden)
    later = observable_frame(CALENDAR, d("2016-03-01"), d("2016-03-04"), True, hidden)
    paired = masked.merge(later, on=keys)
    hidden_columns = [*SOURCED_CHIPS, *map(current_name, SOURCED_CHIPS)]
    assert paired[hidden_columns].isna().all().all()
