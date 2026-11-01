import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from apps.research.features.registry import feature_names
from apps.research.features.transforms import (
    STOCK_LEVEL,
    fill_with_flags,
    market_bins,
    standardize,
    winsorize,
    zscore,
)
from tests.research.conftest import weekday_calendar

NAN = math.nan
DAY, NEXT = date(2016, 3, 1), date(2016, 3, 2)
CALENDAR = weekday_calendar(date(2016, 1, 1), date(2016, 3, 31))


def day_frame(values: list[float], universe: list[bool], day: date = DAY) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_id": [f"{1000 + k}" for k in range(len(values))],
            "date": day,
            "in_universe": universe,
            "x": values,
        }
    )


def test_stock_level_columns_leave_out_market_features() -> None:
    expected = [name for name in feature_names(batch=1) if not name.startswith("mkt_")]

    assert expected == STOCK_LEVEL


def test_winsorize_clips_at_the_universe_percentiles_of_the_day() -> None:
    values = [float(k) for k in range(101)] + [1_000.0, -1_000.0]
    frame = day_frame(values, [True] * 101 + [False, False])

    clipped = winsorize(frame, ["x"])["x"]

    assert clipped.iloc[0] == pytest.approx(1.0)
    assert clipped.iloc[100] == pytest.approx(99.0)
    assert clipped.iloc[50] == 50.0
    assert clipped.iloc[101:].tolist() == pytest.approx([99.0, 1.0])


def test_values_outside_the_universe_do_not_move_the_parameters() -> None:
    values = [float(k) for k in range(101)] + [5.0]
    base = day_frame(values, [True] * 101 + [False])
    moved = base.assign(x=[*values[:-1], 1e9])

    inside = slice(0, 101)
    for transform in (winsorize, zscore, standardize):
        left, right = transform(base, ["x"])["x"], transform(moved, ["x"])["x"]
        assert left.iloc[inside].tolist() == pytest.approx(right.iloc[inside].tolist())


def test_zscore_uses_the_universe_mean_and_sample_std_of_each_day() -> None:
    today = day_frame([1.0, 2.0, 3.0, 10.0], [True, True, True, False])
    tomorrow = day_frame([5.0, 5.0, 7.0, NAN], [True, True, True, True], NEXT)

    scored = zscore(pd.concat([today, tomorrow], ignore_index=True), ["x"])["x"]

    assert scored.iloc[:4].tolist() == pytest.approx([-1.0, 0.0, 1.0, 8.0])
    std = np.std([5.0, 5.0, 7.0], ddof=1)
    assert scored.iloc[4:7].tolist() == pytest.approx(
        [(5 - 17 / 3) / std] * 2 + [(7 - 17 / 3) / std]
    )
    assert math.isnan(scored.iloc[7])


def test_zscore_is_zero_when_the_day_has_no_spread() -> None:
    flat = day_frame([2.0, 2.0, 3.0, NAN], [True, True, False, True])
    single = day_frame([4.0, 5.0], [True, False], NEXT)
    empty = day_frame([NAN, 6.0], [True, False], date(2016, 3, 3))

    scored = zscore(pd.concat([flat, single, empty], ignore_index=True), ["x"])["x"]

    assert scored.iloc[:3].tolist() == [0.0, 0.0, 0.0]
    assert scored.iloc[4:6].tolist() == [0.0, 0.0]
    assert scored.iloc[[3, 6, 7]].isna().all()


@pytest.mark.parametrize("value", [1.0, 3.0])
def test_a_rare_flag_or_count_survives_winsorize(value: float) -> None:
    values = [0.0] * 995 + [value] * 5
    frame = day_frame(values, [True] * 1_000)

    scored = standardize(frame, ["x"])["x"]

    assert scored.iloc[-5:].tolist() == pytest.approx(
        [(value - value * 0.005) / np.std(values, ddof=1)] * 5
    )
    assert scored.iloc[0] < 0


def test_standardize_scores_the_clipped_values() -> None:
    frame = day_frame([float(k) for k in range(101)], [True] * 101)

    scored = standardize(frame, ["x"])["x"]

    assert scored.iloc[100] == pytest.approx(scored.iloc[99])
    assert scored.iloc[0] == pytest.approx(-scored.iloc[100])


def test_only_the_given_columns_change_and_the_input_stays() -> None:
    frame = day_frame([1.0, NAN, 3.0], [True, True, True]).assign(y=[5.0, NAN, 9.0])

    for transform in (winsorize, zscore, standardize, fill_with_flags):
        result = transform(frame, ["x"])
        assert result["y"].tolist()[::2] == [5.0, 9.0]
    assert frame["x"].tolist()[::2] == [1.0, 3.0]
    assert frame[["x", "y"]].iloc[1].isna().all()
    assert list(frame.columns) == ["stock_id", "date", "in_universe", "x", "y"]


def test_the_default_columns_leave_market_and_sector_ids_alone() -> None:
    names = feature_names(batch=1)
    frame = day_frame([1.0, 2.0, 3.0], [True, True, True]).assign(
        **{name: [1.0, 2.0, 4.0] for name in names}, ind_sector_id=[3.0, 3.0, 7.0]
    )

    result = fill_with_flags(standardize(frame))

    assert result["mkt_taiex_ret_5d"].tolist() == [1.0, 2.0, 4.0]
    assert result["ind_sector_id"].tolist() == [3.0, 3.0, 7.0]
    assert result["tech_ret_1d"].tolist() == pytest.approx(
        [-0.872872, -0.218218, 1.091089], abs=1e-5
    )
    assert "mkt_taiex_ret_5d_missing" not in result
    assert "tech_ret_1d_missing" in result


def test_missing_values_take_the_universe_median_and_get_a_flag() -> None:
    today = day_frame([-1.0, 0.5, 5.0, NAN, 9.0, NAN], [True, True, True, True, False, False])
    tomorrow = day_frame([NAN, NAN, 4.0], [True, True, False], NEXT)

    filled = fill_with_flags(pd.concat([today, tomorrow], ignore_index=True), ["x"])

    assert filled["x"].tolist() == [-1.0, 0.5, 5.0, 0.5, 9.0, 0.5, 0.0, 0.0, 4.0]
    assert filled["x_missing"].tolist() == [0, 0, 0, 1, 0, 1, 1, 1, 0]
    assert filled["x_missing"].dtype == np.int8


def test_flags_after_standardize_mark_only_the_missing_raw_values() -> None:
    frame = day_frame([1.0, NAN, 3.0, 2.0, NAN], [True, True, True, False, False])

    filled = fill_with_flags(standardize(frame, ["x"]), ["x"])

    assert filled["x_missing"].tolist() == [0, 1, 0, 0, 1]
    half = 1 / math.sqrt(2)
    assert filled["x"].tolist() == pytest.approx([-half, 0.0, half, 0.0, 0.0])


def test_market_bins_come_from_the_training_window_one_value_per_day() -> None:
    days = CALENDAR.between(date(2016, 1, 4), date(2016, 1, 15))
    counts = [1] * 5 + [3] * 5
    train = pd.DataFrame(
        {
            "stock_id": [f"{1000 + n}" for count in counts for n in range(count)],
            "date": [day for day, count in zip(days, counts, strict=True) for _ in range(count)],
            "mkt_taiex_ret_5d": [float(k) for k, count in enumerate(counts) for _ in range(count)],
        }
    )
    later = pd.DataFrame(
        {
            "stock_id": ["2330"] * 6,
            "date": CALENDAR.between(date(2016, 2, 1), date(2016, 2, 8)),
            "mkt_taiex_ret_5d": [-5.0, 1.9, 3.6, 4.5, 100.0, NAN],
        }
    )

    edges = market_bins(train, ["mkt_taiex_ret_5d"])
    binned = edges.apply(later)["mkt_taiex_ret_5d"].tolist()

    assert edges.edges["mkt_taiex_ret_5d"] == pytest.approx([1.8, 3.6, 5.4, 7.2])
    assert binned[:5] == [0.0, 1.0, 1.0, 2.0, 4.0]
    assert math.isnan(binned[5])
    assert later["mkt_taiex_ret_5d"].tolist()[:5] == [-5.0, 1.9, 3.6, 4.5, 100.0]
    per_day = edges.apply(train).drop_duplicates("date")["mkt_taiex_ret_5d"]
    assert per_day.tolist() == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
