import math
from datetime import date

import pandas as pd
import pytest

from apps.market.calendar import TradingDays
from apps.research.phase3_report import (
    accepted,
    fill_counts,
    fill_table,
    missing_counts,
    missing_table,
    month_ranges,
    monthly_ic,
    render,
    statement_coverage,
    strongest,
)
from tests.research.conftest import weekday_calendar

NAN = math.nan


def d(text: str) -> date:
    return date.fromisoformat(text)


def rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_id": ["9999", "9999", "9999", "2330", "2330", "2330", "1101"],
            "date": [d("2016-01-04"), d("2016-01-05"), d("2016-01-06")] * 2 + [d("2016-01-04")],
            "in_universe": [True, True, True, True, True, True, False],
            "fund_ep": [NAN, NAN, 0.1, 0.1, 0.1, 0.1, NAN],
            "fund_rev_yoy": [0.3, 0.3, NAN, 0.2, 0.3, 0.2, 0.3],
            "fund_rev_yoy_q": [0.3, 0.3, NAN, 0.1, 0.3, 0.1, 0.3],
        }
    )


DELISTED = pd.Series({"9999": True, "2330": False, "1101": False})


def test_missing_counts_add_up_over_chunks_and_split_by_later_delisting() -> None:
    frame = rows()
    whole = missing_counts(frame, DELISTED, ["fund_ep", "fund_rev_yoy"])
    halves = missing_counts(frame.iloc[:4], DELISTED, ["fund_ep", "fund_rev_yoy"]).add(
        missing_counts(frame.iloc[4:], DELISTED, ["fund_ep", "fund_rev_yoy"]), fill_value=0
    )

    pd.testing.assert_frame_equal(whole, halves, check_dtype=False)
    assert whole.loc["fund_ep"].to_dict() == {
        "missing_current": 0,
        "missing_delisted": 2,
        "rows_current": 3,
        "rows_delisted": 3,
    }


def test_missing_table_matches_the_saved_before_table_and_flags_gaps() -> None:
    counts = pd.DataFrame(
        {
            "missing_current": [1, 0, 30, 0],
            "missing_delisted": [5, 0, 60, 1],
            "rows_current": [100, 100, 100, 100],
            "rows_delisted": [100, 100, 100, 100],
        },
        index=["fund_ep", "tech_ret_1d", "chip_dealer_self_net_5d", "size_log_mcap"],
    )

    table = missing_table(counts)

    assert list(table.columns) == [
        "current",
        "delisted",
        "rows_current",
        "rows_delisted",
        "ratio",
        "gap_pp",
        "flag",
    ]
    assert table.loc["fund_ep", "ratio"] == pytest.approx(5.0)
    assert table.loc["fund_ep", "flag"]
    assert math.isnan(float(table["ratio"]["tech_ret_1d"]))
    assert not table.loc["tech_ret_1d", "flag"]
    assert table.loc["chip_dealer_self_net_5d", "gap_pp"] == pytest.approx(30.0)
    assert table.loc["chip_dealer_self_net_5d", "flag"]
    assert not table.loc["size_log_mcap", "flag"]


def test_fill_counts_take_rows_whose_value_is_the_fallback_value() -> None:
    counts = fill_counts(rows(), DELISTED, {"fund_rev_yoy": "fund_rev_yoy_q"})

    assert counts.loc["fund_rev_yoy"].to_dict() == {
        "filled_current": 1,
        "filled_delisted": 2,
        "rows_current": 3,
        "rows_delisted": 3,
    }
    table = fill_table(counts)
    assert table.loc["fund_rev_yoy", "current"] == pytest.approx(1 / 3)
    assert table.loc["fund_rev_yoy", "delisted"] == pytest.approx(2 / 3)
    assert table.loc["fund_rev_yoy", "ratio"] == pytest.approx(2.0)
    assert not table.loc["fund_rev_yoy", "flag"]


def test_fill_table_flags_fills_that_only_the_delisted_group_has() -> None:
    counts = pd.DataFrame(
        {"filled_current": [0, 0], "filled_delisted": [5, 0]}
        | {"rows_current": [100, 100], "rows_delisted": [10, 10]},
        index=["fund_rev_yoy", "fund_rev_3m_yoy"],
    )

    table = fill_table(counts)

    assert table["flag"].to_dict() == {"fund_rev_yoy": True, "fund_rev_3m_yoy": False}


def test_counts_of_a_chunk_without_universe_rows_are_zero() -> None:
    outside = rows().iloc[6:]

    missing = missing_counts(outside, DELISTED, ["fund_ep", "fund_rev_yoy"])
    filled = fill_counts(outside.iloc[:0], DELISTED, {"fund_rev_yoy": "fund_rev_yoy_q"})

    assert missing.loc["fund_ep"].to_dict() == {
        "missing_current": 0,
        "missing_delisted": 0,
        "rows_current": 0,
        "rows_delisted": 0,
    }
    assert list(missing.index) == ["fund_ep", "fund_rev_yoy"]
    assert filled.loc["fund_rev_yoy"].sum() == 0


def test_month_ranges_join_consecutive_months() -> None:
    months = ["2016-01", "2016-02", "2016-03", "2016-05", "2016-12", "2017-01"]

    assert month_ranges(months) == "2016-01–2016-03、2016-05、2016-12–2017-01"
    assert month_ranges([]) == "無"


def test_monthly_ic_marks_months_with_fewer_ic_days_than_trading_days() -> None:
    calendar: TradingDays = weekday_calendar(date(2016, 1, 1), date(2016, 2, 29))
    days = calendar.between(date(2016, 1, 1), date(2016, 2, 29))
    stocks = [f"{1000 + k}" for k in range(40)]
    frame = pd.DataFrame(
        [
            {
                "stock_id": stock,
                "date": day,
                "tech_ret_1d": float(k) if day.month == 1 or day.day > 5 else NAN,
                "fwd_rank_pct": (k + 1) / len(stocks),
            }
            for day in days
            for k, stock in enumerate(stocks)
        ]
    )

    table = monthly_ic(frame, ["tech_ret_1d"], "fwd_rank_pct", calendar)

    january = table[(table["feature"] == "tech_ret_1d") & (table["month"] == "2016-01")].iloc[0]
    february = table[(table["feature"] == "tech_ret_1d") & (table["month"] == "2016-02")].iloc[0]
    assert january["ic"] == pytest.approx(1.0)
    assert january["days"] == january["trading_days"] == 21
    assert january["complete"]
    assert february["trading_days"] == 21
    assert february["days"] == 16
    assert not february["complete"]


def test_monthly_ic_counts_whole_months_at_both_ends_of_the_window() -> None:
    calendar: TradingDays = weekday_calendar(date(2016, 1, 1), date(2016, 2, 29))
    days = calendar.between(date(2016, 1, 11), date(2016, 2, 10))
    stocks = [f"{1000 + k}" for k in range(40)]
    frame = pd.DataFrame(
        [
            {"stock_id": stock, "date": day, "tech_ret_1d": float(k)}
            | {"fwd_rank_pct": (k + 1) / len(stocks)}
            for day in days
            for k, stock in enumerate(stocks)
        ]
    )

    table = monthly_ic(frame, ["tech_ret_1d"], "fwd_rank_pct", calendar).set_index("month")

    assert table.loc["2016-01", "days"] == 15
    assert table.loc["2016-02", "days"] == 8
    assert table["trading_days"].to_dict() == {"2016-01": 21, "2016-02": 21}
    assert not table["complete"].any()


def test_statement_coverage_lists_fundamental_features_left_unjudged() -> None:
    params = {
        "window_days": 100,
        "min_coverage": 0.5,
        "days": {
            "fund_ep": {"5": 100, "20": 100},
            "fund_roe_ttm": {"5": 40, "20": 30},
            "fund_debt_ratio": {"5": 50, "20": 60},
            "fund_days_since_fs": {"5": 100, "20": 100},
            "fund_bp": {"5": 100, "20": 100},
            "tech_ret_1d": {"5": 10, "20": 10},
        },
    }

    table = statement_coverage(params)

    assert list(table.index) == ["fund_ep", "fund_roe_ttm", "fund_debt_ratio", "fund_days_since_fs"]
    assert table["judged"].tolist() == [True, False, True, False]
    assert table["conditional"].tolist() == [False, False, False, True]
    assert table.to_dict("index")["fund_ep"] == {
        "days_h5": 100,
        "days_h20": 100,
        "coverage": 1.0,
        "conditional": False,
        "judged": True,
    }
    assert table.loc["fund_roe_ttm", "coverage"] == pytest.approx(0.3)
    assert not table.loc["fund_roe_ttm", "judged"]


def test_render_lists_only_flagged_gaps_and_marks_incomplete_months() -> None:
    before = pd.DataFrame(
        {
            "current": [0.004, 0.0, 0.1, 0.0],
            "delisted": [0.209, 0.0, 0.2, 0.0],
            "ratio": [55.3, NAN, 2.0, NAN],
        },
        index=["size_log_mcap", "tech_ret_1d", "chip_foreign_hold_pct", "fund_cfo_to_ni"],
    ).assign(
        rows_current=100,
        rows_delisted=10,
        gap_pp=[20.5, 0.0, 10.0, 0.0],
        flag=[True, False, False, False],
    )
    after = missing_table(
        pd.DataFrame(
            {
                "missing_current": [0, 0, 10, 1],
                "missing_delisted": [1, 0, 2, 10],
                "rows_current": [100, 100, 100, 1000],
                "rows_delisted": [10, 10, 10, 1000],
            },
            index=["size_log_mcap", "tech_ret_1d", "chip_foreign_hold_pct", "fund_cfo_to_ni"],
        )
    )
    fills = fill_table(
        pd.DataFrame(
            {"filled_current": [10], "filled_delisted": [5], "rows_current": [100]}
            | {"rows_delisted": [10]},
            index=["fund_rev_yoy"],
        )
    )
    monthly = pd.DataFrame(
        {
            "horizon": [5, 5],
            "feature": ["fund_ep", "fund_ep"],
            "month": ["2014-01", "2014-02"],
            "ic": [0.0123, -0.02],
            "days": [20, 15],
            "trading_days": [20, 19],
            "complete": [True, False],
        }
    )

    text = render(
        header={"run_id": "k.screening", "start": "2010-12-06", "end": "2015-08-31"}
        | {"window_days": 1170, "kept": 56, "dropped": 2, "before": "before.csv"},
        coverage=statement_coverage(
            {
                "window_days": 100,
                "min_coverage": 0.5,
                "days": {
                    "fund_roe_ttm": {"5": 30, "20": 30},
                    "fund_days_since_fs": {"5": 100, "20": 100},
                },
            }
        ),
        gaps=(before, after, "2010-01-01", "2020-12-31"),
        fills=(fills, {"fund_rev_yoy": "fund_rev_yoy_q"}),
        masked={"fund_rev_mom": frozenset({"2016-01", "2016-02"})},
        monthly=monthly,
        top=["fund_ep"],
    )

    assert "| size_log_mcap | 20.9%／0.4% | 10.0%／0.0% | — | 10.0 | 否 |" in text
    assert "| chip_foreign_hold_pct | 20.0%／10.0% | 20.0%／10.0% | 2.00 | 10.0 | 否 |" in text
    assert "| tech_ret_1d |" not in text
    assert "| fund_cfo_to_ni | 0.0%／0.0% | 1.0%／0.1% | 10.00 | 0.9 | 是 |" in text
    assert "| fund_days_since_fs | 100／100 | 100.0% | 豁免（條件型） |" in text
    assert "C 全期套用" in text
    assert "補後未達驗收 2 個：size_log_mcap、chip_foreign_hold_pct。" in text
    assert "不進任何一批的 `fund_rev_mom` 不列入驗收。" in text
    assert "| fund_rev_yoy | fund_rev_yoy_q | 50.0%／10.0% | 5.00 | 是 |" in text
    assert "| fund_rev_mom | 2016-01–2016-02 |" in text
    assert "| fund_roe_ttm | 30／30 | 30.0% | 不判定 |" in text
    assert "月營收查不到歷史公告日，研究期用法定截止日的下一個交易日（K2 保守值）" in text
    assert "| 2014-01 | 0.0123 |" in text
    assert "| 2014-02 | -0.0200† |" in text


def test_strongest_ranks_features_by_their_largest_absolute_ic() -> None:
    ic: dict[str, dict[str, float | None]] = {
        "tech_ret_1d": {"5": 0.01, "20": -0.05},
        "fund_ep": {"5": 0.02, "20": None},
        "chip_big_holder_pct_chg_1w": {"5": None, "20": None},
        "size_log_mcap": {"5": 0.03, "20": 0.04},
    }

    assert strongest(ic, 2) == ["tech_ret_1d", "size_log_mcap"]
    assert strongest(ic, 10)[-1] == "chip_big_holder_pct_chg_1w"


def test_accepted_takes_a_small_ratio_or_a_small_gap() -> None:
    table = pd.DataFrame(
        {
            "ratio": [1.5, 1.6, 20.0, NAN, NAN, 1.6],
            "gap_pp": [30.0, 5.0, 1.0, 0.0, 6.0, 5.1],
        },
        index=["ratio_edge", "gap_edge", "small_gap", "none_missing", "only_delisted", "both_over"],
    )

    assert accepted(table).to_dict() == {
        "ratio_edge": True,
        "gap_edge": True,
        "small_gap": True,
        "none_missing": True,
        "only_delisted": False,
        "both_over": False,
    }
