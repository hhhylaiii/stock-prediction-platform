import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from apps.research.metrics.ic import (
    daily_rank_ic,
    ic_by_fold,
    ic_by_year,
    neutralized_rank_ic,
    rank_ic_by_bucket,
    summarize_ic,
)
from apps.research.metrics.stats import newey_west_t
from apps.research.walkforward import Fold

DAY = date(2016, 1, 4)


def day(n: int) -> date:
    return DAY + timedelta(days=n)


def one_day(score: list[float], label: list[float], on: date = DAY) -> pd.DataFrame:
    return pd.DataFrame({"date": on, "stock_id": range(len(score)), "s": score, "y": label})


def test_spearman_on_one_day_matches_hand_count() -> None:
    ic = daily_rank_ic(one_day([1, 2, 3, 4], [0.1, 0.2, 0.4, 0.3]), "s", "y", min_stocks=4)
    assert ic.to_dict() == {DAY: pytest.approx(0.8)}
    assert ic.name == "ic"


def test_ties_use_average_ranks() -> None:
    frame = one_day([1, 1, 2, 3, 3, 4], [0.5, 0.1, 0.2, 0.9, 0.3, 0.4])
    expected = np.corrcoef([1.5, 1.5, 3, 4.5, 4.5, 6], [5, 1, 2, 6, 3, 4])[0, 1]
    assert daily_rank_ic(frame, "s", "y", min_stocks=6)[DAY] == pytest.approx(expected)


def test_each_date_is_ranked_on_its_own() -> None:
    frame = pd.concat(
        [one_day([1, 2, 3], [1, 2, 3], day(0)), one_day([10, 20, 30], [3, 2, 1], day(1))]
    )
    ic = daily_rank_ic(frame, "s", "y", min_stocks=3)
    assert list(ic.index) == [day(0), day(1)]
    assert list(ic) == [pytest.approx(1.0), pytest.approx(-1.0)]


def test_missing_values_thin_days_and_flat_scores_are_left_out() -> None:
    frame = pd.concat(
        [
            one_day([1, 2, 3, math.nan], [1, 2, 3, 4], day(0)),
            one_day([1, 2], [1, 2], day(1)),
            one_day([5, 5, 5], [1, 2, 3], day(2)),
            one_day([3, 2, 1, 0], [1, 2, math.nan, 4], day(3)),
        ]
    )
    ic = daily_rank_ic(frame, "s", "y", min_stocks=3)
    assert ic.to_dict() == {day(0): pytest.approx(1.0), day(3): pytest.approx(-1.0)}


def test_summary_uses_newey_west_with_the_given_lag() -> None:
    ic = pd.Series([0.1, 0.05, -0.02, 0.08, 0.03], index=[day(n) for n in range(5)])
    summary = summarize_ic(ic, lag=2)
    assert summary.mean == pytest.approx(0.048)
    assert summary.std == pytest.approx(ic.std(ddof=1))
    assert summary.icir == pytest.approx(0.048 / ic.std(ddof=1))
    assert summary.nw_t == pytest.approx(newey_west_t(ic.to_numpy(), 2))
    assert summary.positive_share == pytest.approx(0.8)
    assert summary.n_days == 5


def test_summary_of_too_few_days_is_nan() -> None:
    summary = summarize_ic(pd.Series([0.1], index=[DAY]), lag=5)
    assert summary.n_days == 1
    assert math.isnan(summary.std)
    assert math.isnan(summary.icir)
    assert math.isnan(summary.nw_t)


def make_fold(index: int, start: date, end: date, *, tuning: bool = False) -> Fold:
    return Fold(index, start, start, start, start, start, end, tuning, holdout=False)


def test_by_fold_keeps_flags_and_only_test_period_days() -> None:
    ic = pd.Series([0.1, 0.3, -0.2, 0.4], index=[day(0), day(1), day(10), day(11)])
    folds = [
        make_fold(0, day(0), day(5), tuning=True),
        make_fold(1, day(10), day(15)),
        make_fold(2, day(20), day(25)),
    ]
    table = ic_by_fold(ic, folds, lag=1)
    assert list(table.index) == [0, 1, 2]
    assert list(table["tuning"]) == [True, False, False]
    assert table.loc[0, "mean"] == pytest.approx(0.2)
    assert table.loc[1, "mean"] == pytest.approx(0.1)
    assert table.loc[2, "n_days"] == 0
    assert table["mean"].isna().tolist() == [False, False, True]


def test_by_year_groups_calendar_years() -> None:
    ic = pd.Series([0.1, 0.3, -0.2], index=[date(2015, 12, 31), date(2016, 1, 4), date(2016, 6, 1)])
    table = ic_by_year(ic, lag=1)
    assert list(table.index) == [2015, 2016]
    assert table.loc[2016, "mean"] == pytest.approx(0.05)
    assert list(table["n_days"]) == [1, 2]


def exposure_panel(n_days: int = 40, n_stocks: int = 300, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for n in range(n_days):
        industry = rng.integers(0, 6, n_stocks)
        size = rng.standard_normal(n_stocks)
        beta = rng.standard_normal(n_stocks)
        vol = rng.standard_normal(n_stocks)
        alpha = rng.standard_normal(n_stocks)
        noise = rng.standard_normal(n_stocks)
        ret = industry * 0.5 + size + 0.5 * beta + alpha
        rows.append(
            pd.DataFrame(
                {
                    "date": day(n),
                    "stock_id": range(n_stocks),
                    "industry": [f"I{i}" for i in industry],
                    "size_log_mcap": size,
                    "risk_beta_250d": beta,
                    "tech_vol_20d": vol,
                    "exposure_score": industry * 0.5 + size + 0.5 * beta + 0.3 * noise,
                    "alpha_score": alpha + 0.3 * size,
                    "fwd_rank_pct": pd.Series(ret).rank(pct=True).to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_score_made_of_exposures_loses_its_ic_after_neutralizing() -> None:
    panel = exposure_panel()
    raw = daily_rank_ic(panel, "exposure_score", "fwd_rank_pct")
    neutral = neutralized_rank_ic(panel, "exposure_score")
    assert raw.mean() > 0.6
    assert abs(neutral.mean()) < 0.05


def test_score_with_residual_alpha_keeps_ic_after_neutralizing() -> None:
    panel = exposure_panel()
    assert neutralized_rank_ic(panel, "alpha_score").mean() > 0.4


def test_rows_without_industry_or_exposure_are_left_out_of_the_regression() -> None:
    panel = exposure_panel(n_days=1, n_stocks=40)
    panel.loc[0, "industry"] = ""
    panel.loc[1, "industry"] = None
    panel.loc[2, "risk_beta_250d"] = math.nan
    kept = panel.iloc[3:]
    assert neutralized_rank_ic(panel, "alpha_score", min_stocks=37)[DAY] == pytest.approx(
        neutralized_rank_ic(kept, "alpha_score", min_stocks=37)[DAY]
    )
    assert neutralized_rank_ic(panel, "alpha_score", min_stocks=38).empty


def test_score_fully_explained_by_exposures_has_no_residual_ic() -> None:
    panel = exposure_panel(n_days=2)
    panel["exact"] = panel["size_log_mcap"] * 2 - panel["tech_vol_20d"]
    assert neutralized_rank_ic(panel, "exact").empty


def test_bucket_ic_shows_where_the_signal_lives() -> None:
    rng = np.random.default_rng(9)
    rows = []
    for n in range(30):
        liquidity = rng.lognormal(16, 1, 300)
        score = rng.standard_normal(300)
        noise = rng.standard_normal(300)
        top = liquidity >= np.quantile(liquidity, 2 / 3)
        label = np.where(top, score, noise)
        rows.append(
            pd.DataFrame(
                {"date": day(n), "liq": liquidity, "s": score, "y": label, "stock_id": range(300)}
            )
        )
    panel = pd.concat(rows, ignore_index=True)
    table = rank_ic_by_bucket(panel, "s", "y", by="liq")
    assert table.columns.equals(pd.Index([1, 2, 3]))
    assert table[3].mean() > 0.95
    assert abs(table[1].mean()) < 0.05
    assert abs(table[2].mean()) < 0.05


def test_buckets_split_each_day_into_equal_counts() -> None:
    frame = one_day(list(range(9)), list(range(9)))
    frame["liq"] = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    frame.loc[0, "liq"] = math.nan
    table = rank_ic_by_bucket(frame, "s", "y", by="liq", n_buckets=2, min_stocks=4)
    assert table.columns.equals(pd.Index([1, 2]))
    assert list(table.loc[DAY]) == [pytest.approx(1.0), pytest.approx(1.0)]
