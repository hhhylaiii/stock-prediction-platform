import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from apps.market.calendar import TradingDays
from apps.research.metrics.forecast import (
    coverage_by_bucket,
    daily_absolute_error,
    daily_auc,
    daily_brier,
    daily_coverage,
    daily_direction_hits,
    daily_pinball,
    daily_qlike,
    daily_squared_error,
    daily_width,
    expected_calibration_error,
    mean_by_fold,
    pinball_loss,
    reliability_table,
    research_rows,
    shared_rows,
)
from apps.research.walkforward import generate_folds

D1 = date(2020, 1, 2)
D2 = date(2020, 1, 3)
NAN = math.nan


def interval_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [D1, D1, D2, D2, D2],
            "y": [0.05, -0.2, 0.3, NAN, 0.0],
            "q10": [-0.1, -0.1, -0.1, -0.1, -0.1],
            "q50": [0.0, 0.0, 0.0, 0.0, NAN],
            "q90": [0.1, 0.1, 0.1, 0.1, 0.1],
        }
    )


def test_pinball_loss_weights_misses_by_alpha() -> None:
    actual = np.array([0.1, -0.1, 0.0])
    predicted = np.zeros(3)

    np.testing.assert_allclose(pinball_loss(actual, predicted, 0.9), [0.09, 0.01, 0.0])
    np.testing.assert_allclose(pinball_loss(actual, predicted, 0.1), [0.01, 0.09, 0.0])


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_pinball_alpha_must_be_inside_zero_and_one(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        pinball_loss(np.zeros(1), np.zeros(1), alpha)


def test_daily_pinball_sums_the_three_quantiles_and_averages_per_day() -> None:
    loss = daily_pinball(interval_frame(), "y")

    assert loss.index.tolist() == [D1, D2]
    np.testing.assert_allclose(loss.to_numpy(), [(0.045 + 0.22) / 2, 0.37])
    assert loss.name == "pinball"


def test_coverage_counts_the_bounds_as_inside() -> None:
    frame = interval_frame().assign(y=[0.05, -0.1, 0.3, NAN, 0.1])

    coverage = daily_coverage(frame, "y", "q10", "q90")

    assert coverage.to_dict() == {D1: 1.0, D2: 0.5}


def test_width_uses_rows_with_both_bounds() -> None:
    frame = interval_frame().assign(q90=[0.1, 0.3, 0.1, 0.1, NAN])

    width = daily_width(frame, "q10", "q90")

    np.testing.assert_allclose(width.to_numpy(), [0.3, 0.2])


def bucket_day(day: date, scale: float) -> pd.DataFrame:
    covered = [True, True, True, False, False, False, True, True, True, False]
    return pd.DataFrame(
        {
            "date": [day] * 10,
            "tech_vol_20d": np.arange(10.0, 0.0, -1.0) * scale,
            "y": [0.0 if hit else 1.0 for hit in covered],
            "q10": [-0.1 * (n + 1) for n in range(10)],
            "q90": [0.1] * 10,
        }
    )


def test_coverage_by_bucket_splits_each_day_by_volatility() -> None:
    frame = pd.concat([bucket_day(D1, 1.0), bucket_day(D2, 100.0)], ignore_index=True)

    table = coverage_by_bucket(frame, "y", "q10", "q90")

    assert list(table.columns) == ["date", "bucket", "coverage", "width"]
    for day in (D1, D2):
        part = table[table["date"] == day]
        assert part["bucket"].tolist() == [1, 2, 3, 4, 5]
        np.testing.assert_allclose(part["coverage"].to_numpy(), [0.5, 1.0, 0.0, 0.5, 1.0])
        np.testing.assert_allclose(part["width"].to_numpy(), [1.05, 0.85, 0.65, 0.45, 0.25])


def test_coverage_by_bucket_ignores_rows_without_the_bucket_value() -> None:
    frame = pd.DataFrame(
        {
            "date": [D1] * 4,
            "vol": [1.0, 2.0, NAN, NAN],
            "y": [0.0, 1.0, 1.0, 1.0],
            "q10": [-0.1] * 4,
            "q90": [0.1] * 4,
        }
    )

    table = coverage_by_bucket(frame, "y", "q10", "q90", by="vol", n_buckets=2)

    assert table["coverage"].tolist() == [1.0, 0.0]


def test_coverage_by_bucket_needs_a_bucket() -> None:
    with pytest.raises(ValueError, match="n_buckets"):
        coverage_by_bucket(interval_frame(), "y", "q10", "q90", by="q50", n_buckets=0)


def test_errors_and_direction_per_day() -> None:
    frame = pd.DataFrame(
        {
            "date": [D1, D1, D2, D2],
            "y": [0.1, 0.2, -0.3, 0.1],
            "pred": [0.05, -0.1, -0.1, 0.0],
        }
    )

    np.testing.assert_allclose(daily_absolute_error(frame, "y", "pred").to_numpy(), [0.175, 0.15])
    np.testing.assert_allclose(daily_squared_error(frame, "y", "pred").to_numpy(), [0.04625, 0.025])


def test_direction_skips_flat_labels_and_zero_predictions_miss() -> None:
    frame = pd.DataFrame(
        {
            "date": [D1, D1, D1, D2, D2],
            "y": [0.1, 0.2, 0.0, -0.3, 0.1],
            "pred": [0.05, -0.1, 0.3, -0.1, 0.0],
        }
    )

    assert daily_direction_hits(frame, "y", "pred").tolist() == [0.5, 0.5]


def test_qlike_is_zero_when_right_and_skips_non_positive_volatility() -> None:
    frame = pd.DataFrame(
        {
            "date": [D1, D2, D2, D2, D2],
            "realized": [0.2, 0.4, 0.0, 0.3, 0.3],
            "predicted": [0.2, 0.2, 0.2, 0.0, -0.2],
        }
    )

    loss = daily_qlike(frame, "realized", "predicted")

    np.testing.assert_allclose(loss.to_numpy(), [0.0, 3.0 - math.log(4.0)])


def test_brier_is_the_mean_squared_probability_error() -> None:
    frame = pd.DataFrame({"date": [D1, D1, D2], "up": [1, 0, 1], "prob": [0.8, 0.3, NAN]})

    np.testing.assert_allclose(daily_brier(frame, "up", "prob").to_numpy(), [0.065])


@pytest.mark.parametrize(
    ("labels", "probs", "message"),
    [([1, 2], [0.5, 0.5], "0 or 1"), ([1, 0], [1.2, 0.5], "between 0 and 1")],
)
def test_probability_metrics_check_their_inputs(
    labels: list[int], probs: list[float], message: str
) -> None:
    frame = pd.DataFrame({"date": [D1, D1], "up": labels, "prob": probs})

    for metric in (daily_brier, reliability_table, daily_auc):
        with pytest.raises(ValueError, match=message):
            metric(frame, "up", "prob")


def test_daily_auc_counts_ordered_pairs_and_half_ties() -> None:
    frame = pd.DataFrame(
        {
            "date": [D1] * 4 + [D2] * 4,
            "up": [1, 0, 1, 0, 1, 0, 1, 0],
            "prob": [0.9, 0.8, 0.3, 0.1, 0.5, 0.5, 0.9, 0.1],
        }
    )

    auc = daily_auc(frame, "up", "prob", min_stocks=4)

    assert auc.to_dict() == {D1: 0.75, D2: 0.875}


def test_daily_auc_skips_one_sided_and_thin_days() -> None:
    one_sided = pd.DataFrame({"date": [D1] * 4, "up": [1] * 4, "prob": [0.1, 0.2, 0.3, 0.4]})
    none_up = one_sided.assign(up=0)
    thin = pd.DataFrame({"date": [D2] * 29, "up": [1, 0] * 14 + [1], "prob": np.linspace(0, 1, 29)})

    assert daily_auc(one_sided, "up", "prob", min_stocks=4).empty
    assert daily_auc(none_up, "up", "prob", min_stocks=4).empty
    assert daily_auc(thin, "up", "prob").empty
    assert len(daily_auc(pd.concat([thin, thin.head(1)]), "up", "prob")) == 1


def test_reliability_table_and_ece() -> None:
    frame = pd.DataFrame(
        {"date": [D1] * 5, "up": [0, 1, 0, 1, 1], "prob": [0.05, 0.15, 0.12, 0.95, 1.0]}
    )

    table = reliability_table(frame, "up", "prob")

    assert table.index.tolist() == list(range(1, 11))
    assert table["n_rows"].tolist() == [1, 2, 0, 0, 0, 0, 0, 0, 0, 2]
    np.testing.assert_allclose(table.loc[[1, 2, 10], "mean_prob"].to_numpy(), [0.05, 0.135, 0.975])
    np.testing.assert_allclose(table.loc[[1, 2, 10], "frac_positive"].to_numpy(), [0.0, 0.5, 1.0])
    assert table["mean_prob"].isna().loc[5]
    assert (table.loc[3, "lower"], table.loc[3, "upper"]) == pytest.approx((0.2, 0.3))
    assert expected_calibration_error(table) == pytest.approx(0.166)


def test_reliability_needs_a_bin() -> None:
    frame = pd.DataFrame({"date": [D1], "up": [1], "prob": [0.5]})

    with pytest.raises(ValueError, match="n_bins"):
        reliability_table(frame, "up", "prob", n_bins=0)


def test_ece_of_no_rows_is_nan() -> None:
    empty = pd.DataFrame({"date": [], "up": [], "prob": []})

    assert math.isnan(expected_calibration_error(reliability_table(empty, "up", "prob")))


def test_mean_by_fold_uses_test_start_through_eval_end(long_calendar: TradingDays) -> None:
    folds = generate_folds(long_calendar, date(2026, 10, 30))
    trimmed = next(fold for fold in folds if fold.eval_end < fold.test_end)
    daily = pd.Series({trimmed.test_start: 1.0, trimmed.eval_end: 3.0, trimmed.test_end: 100.0})

    table = mean_by_fold(daily, [trimmed])

    row = table.loc[trimmed.index]
    assert (row["mean"], row["n_days"], row["tuning"]) == (2.0, 2, trimmed.tuning)
    last = folds[-1]
    flags = mean_by_fold(pd.Series({last.test_start: 1.0}), [last]).loc[last.index]
    assert (bool(flags["holdout"]), bool(flags["tuning"])) == (True, False)
    assert list(table.columns) == ["test_start", "eval_end", "tuning", "holdout", "mean", "n_days"]


def test_metrics_of_no_rows_are_empty() -> None:
    empty = interval_frame().iloc[0:0]

    assert daily_pinball(empty, "y").empty
    assert daily_coverage(empty, "y", "q10", "q90").empty
    assert list(coverage_by_bucket(empty, "y", "q10", "q90", by="q50").columns) == [
        "date",
        "bucket",
        "coverage",
        "width",
    ]


def test_shared_rows_keep_rows_every_compared_column_scores() -> None:
    frame = pd.DataFrame(
        {
            "date": [D1, D1, D1],
            "y": [0.1, 0.2, NAN],
            "model": [0.0, NAN, 0.1],
            "base": [0.0, 0.1, 0.1],
        }
    )

    kept = shared_rows(frame, ["y", "model", "base"])

    assert kept.index.tolist() == [0]


def test_research_rows_keep_research_folds_through_eval_end(long_calendar: TradingDays) -> None:
    folds = generate_folds(long_calendar, date(2026, 10, 30))
    trimmed = next(fold for fold in folds if fold.research and fold.eval_end < fold.test_end)
    tuning = next(fold for fold in folds if fold.tuning)
    days = [trimmed.test_start, trimmed.eval_end, trimmed.test_end, tuning.test_start]
    frame = pd.DataFrame({"date": days, "up": [1, 0, 1, 0], "prob": [0.9, 0.1, 0.5, 0.5]})

    kept = research_rows(frame, folds)

    assert kept["date"].tolist() == [trimmed.test_start, trimmed.eval_end]
