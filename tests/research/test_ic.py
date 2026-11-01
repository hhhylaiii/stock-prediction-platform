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
    positive_fold_share,
    rank_ic_by_bucket,
    research_ic,
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
    assert summary.positive_day_share == pytest.approx(0.8)
    assert summary.n_days == 5


def test_summary_of_a_flat_series_has_no_icir() -> None:
    summary = summarize_ic(pd.Series([0.2, 0.2, 0.2], index=[day(n) for n in range(3)]), lag=1)
    assert summary.mean == pytest.approx(0.2)
    assert math.isnan(summary.icir)
    assert math.isnan(summary.nw_t)


def test_summary_of_too_few_days_is_nan() -> None:
    summary = summarize_ic(pd.Series([0.1], index=[DAY]), lag=5)
    assert summary.n_days == 1
    assert math.isnan(summary.std)
    assert math.isnan(summary.icir)
    assert math.isnan(summary.nw_t)


def make_fold(
    index: int, start: date, end: date, *, eval_end: date | None = None, **flags: bool
) -> Fold:
    return Fold(
        index,
        start,
        start,
        start,
        start,
        start,
        end,
        end if eval_end is None else eval_end,
        flags.get("tuning", False),
        flags.get("holdout", False),
    )


def test_by_fold_counts_test_days_through_eval_end_and_keeps_flags() -> None:
    days = [0, 1, 3, 4, 10, 12, 15, 30]
    values = [0.1, 0.3, 0.2, 0.9, -0.2, 0.4, 0.6, 0.5]
    ic = pd.Series(values, index=[day(n) for n in days])
    folds = [
        make_fold(5, day(0), day(5), eval_end=day(3), tuning=True),
        make_fold(6, day(10), day(15), holdout=True),
        make_fold(9, day(20), day(25)),
    ]
    table = ic_by_fold(ic, folds, lag=1)
    assert list(table.index) == [5, 6, 9]
    assert list(table["tuning"]) == [True, False, False]
    assert list(table["holdout"]) == [False, True, False]
    assert table.loc[5, "mean"] == pytest.approx(0.2)
    assert table.loc[6, "mean"] == pytest.approx(0.8 / 3)
    expected_t = newey_west_t([-0.2, 0.4, 0.6], 1)
    assert table.loc[6, "nw_t"] == pytest.approx(expected_t)
    assert expected_t != pytest.approx(newey_west_t([-0.2, 0.4, 0.6], 0))
    assert list(table["n_days"]) == [3, 3, 0]
    assert table["mean"].isna().tolist() == [False, False, True]


def test_positive_fold_share_counts_research_folds_with_days() -> None:
    table = pd.DataFrame(
        {
            "mean": [0.1, -0.2, math.nan, 0.3, 0.5, 0.4],
            "n_days": [5, 5, 0, 5, 5, 5],
            "tuning": [False, False, False, False, True, False],
            "holdout": [False, False, False, False, False, True],
        }
    )
    assert positive_fold_share(table) == pytest.approx(2 / 3)
    assert math.isnan(positive_fold_share(table.iloc[2:3]))
    assert math.isnan(positive_fold_share(table.iloc[4:]))


def test_research_ic_keeps_research_days_through_eval_end() -> None:
    ic = pd.Series(range(12), index=[day(n) for n in range(12)], dtype=float)
    folds = [
        make_fold(0, day(0), day(2), tuning=True),
        make_fold(1, day(3), day(6), eval_end=day(4)),
        make_fold(2, day(7), day(9)),
        make_fold(3, day(10), day(11), holdout=True),
    ]
    assert list(research_ic(ic, folds).index) == [day(3), day(4), day(7), day(8), day(9)]


def test_empty_inputs_give_empty_tables_with_columns() -> None:
    by_fold = ic_by_fold(pd.Series([0.1], index=[DAY]), [], lag=1)
    by_year = ic_by_year(pd.Series([], dtype=float), lag=1)
    assert by_fold.empty
    assert {"tuning", "holdout", "mean", "nw_t"} <= set(by_fold.columns)
    assert by_year.empty
    assert {"mean", "nw_t", "n_days"} <= set(by_year.columns)


def test_by_year_groups_calendar_years() -> None:
    ic = pd.Series([0.1, 0.3, -0.2], index=[date(2015, 12, 31), date(2016, 1, 4), date(2016, 6, 1)])
    table = ic_by_year(ic, lag=1)
    assert list(table.index) == [2015, 2016]
    assert table.loc[2016, "mean"] == pytest.approx(0.05)
    assert table.loc[2016, "nw_t"] == pytest.approx(newey_west_t([0.3, -0.2], 1))
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
        sign = 1 if n % 2 == 0 else -1
        exposure = sign * (industry * 0.5 + size) + 0.5 * beta
        ret = exposure + alpha
        rows.append(
            pd.DataFrame(
                {
                    "date": day(n),
                    "stock_id": range(n_stocks),
                    "industry": [f"I{i}" for i in industry],
                    "size_log_mcap": size,
                    "risk_beta_250d": beta,
                    "tech_vol_20d": vol,
                    "exposure_score": exposure + 0.3 * noise,
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
    panel.loc[3, ["fwd_rank_pct", "size_log_mcap"]] = [math.nan, 50.0]
    kept = panel.iloc[4:]
    assert neutralized_rank_ic(panel, "alpha_score", min_stocks=36)[DAY] == pytest.approx(
        neutralized_rank_ic(kept, "alpha_score", min_stocks=36)[DAY]
    )
    assert neutralized_rank_ic(panel, "alpha_score", min_stocks=37).empty


def test_score_fully_explained_by_exposures_has_no_residual_ic() -> None:
    panel = exposure_panel(n_days=2)
    panel["exact"] = panel["size_log_mcap"] * 2 - panel["tech_vol_20d"]
    assert neutralized_rank_ic(panel, "exact").empty


def test_small_residual_is_still_scored() -> None:
    panel = exposure_panel(n_days=2)
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(len(panel))
    panel["nearly"] = panel["size_log_mcap"] * 2 - panel["tech_vol_20d"] + 1e-3 * noise
    assert len(neutralized_rank_ic(panel, "nearly")) == 2


def test_days_below_the_default_stock_count_are_left_out() -> None:
    rng = np.random.default_rng(4)
    frame = pd.concat(
        [
            one_day(list(rng.standard_normal(n)), list(rng.standard_normal(n)), day(n))
            for n in (29, 30)
        ]
    )
    assert list(daily_rank_ic(frame, "s", "y").index) == [day(30)]


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


def test_buckets_are_cut_within_each_day() -> None:
    first = one_day(list(range(9)), list(range(9)), day(0)).assign(liq=range(1, 10))
    second = one_day(list(range(9)), list(range(9)), day(1)).assign(liq=range(100, 1000, 100))
    table = rank_ic_by_bucket(pd.concat([first, second]), "s", "y", by="liq", min_stocks=3)
    assert table.notna().all().all()
    assert table.shape == (2, 3)


def test_rows_without_a_label_do_not_move_the_bucket_edges() -> None:
    base = one_day(list(range(9)), list(range(9))).assign(liq=range(10, 19))
    unlabeled = one_day([0.0] * 6, [math.nan] * 6).assign(liq=range(1, 7), stock_id=range(9, 15))
    unscored = one_day([math.nan] * 6, [0.5] * 6).assign(liq=range(1, 7), stock_id=range(15, 21))
    noisy = pd.concat([base, unlabeled, unscored]).assign(junk="x")
    expected = rank_ic_by_bucket(base, "s", "y", by="liq", min_stocks=3)
    pd.testing.assert_frame_equal(
        rank_ic_by_bucket(noisy, "s", "y", by="liq", min_stocks=3), expected
    )
    assert expected.notna().all().all()


def test_bucket_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="n_buckets"):
        rank_ic_by_bucket(one_day([1.0], [1.0]).assign(liq=1.0), "s", "y", by="liq", n_buckets=0)


def test_bucketing_by_the_scored_column_itself_works() -> None:
    frame = one_day(list(range(9)), list(range(9)))
    table = rank_ic_by_bucket(frame, "s", "y", by="s", min_stocks=3)
    assert table.columns.equals(pd.Index([1, 2, 3]))
    assert table.notna().all().all()
