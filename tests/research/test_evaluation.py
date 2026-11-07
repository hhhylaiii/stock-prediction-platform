import math
from datetime import date, timedelta
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from apps.market.calendar import TradingDays
from apps.research import evaluation
from apps.research.evaluation import (
    ShuffleCheck,
    band_width,
    consistency_report,
    coverage_report,
    decile_report,
    importance_drift,
    leak_flags,
    rank_report,
)
from apps.research.metrics.deciles import top_liquidity, tranche_summary
from apps.research.metrics.forecast import daily_coverage, daily_width
from apps.research.metrics.ic import (
    daily_rank_ic,
    ic_by_fold,
    neutralized_rank_ic,
    rank_ic_by_bucket,
    research_ic,
    summarize_ic,
)
from apps.research.walkforward import Fold, WalkForwardConfig, generate_folds

FIRST, END = date(2015, 7, 1), date(2017, 6, 30)
TESTED = date(2016, 7, 1)
STOCKS = [f"S{k:02d}" for k in range(100)]
CONFIG = WalkForwardConfig(
    train_months=6,
    test_months=2,
    step_months=2,
    valid_months=2,
    data_start=FIRST,
    first_test=TESTED,
    tuning_end=date(2016, 8, 31),
    holdout_start=date(2017, 5, 1),
)
VOL_SCALE = 1.28 * math.sqrt(20 / 252)


def weekdays(start: date, end: date) -> list[date]:
    days = (start + timedelta(days=n) for n in range((end - start).days + 1))
    return [day for day in days if day.weekday() < 5]


def calendar() -> TradingDays:
    return TradingDays(weekdays(FIRST, END), FIRST, END)


def folds() -> list[Fold]:
    return generate_folds(calendar(), END, CONFIG)


def research() -> list[Fold]:
    return [fold for fold in folds() if fold.research]


def inside(frame: pd.DataFrame, chosen: list[Fold]) -> pd.Series:
    keep = pd.Series(False, index=frame.index)
    for fold in chosen:
        keep |= frame["date"].between(fold.test_start, fold.eval_end)
    return keep


def panel(seed: int = 0) -> pd.DataFrame:
    """Test-period rows of 100 stocks with a score near the label; stock k ranks k by volatility."""
    rng = np.random.default_rng(seed)
    days = weekdays(TESTED, END)
    size = len(days) * len(STOCKS)
    ret = rng.normal(0, 0.05, size)
    frame = pd.DataFrame(
        {
            "date": np.repeat(np.array(days, dtype=object), len(STOCKS)),
            "stock_id": np.tile(STOCKS, len(days)),
            "in_universe": True,
            "untradable_t1": False,
            "fwd_ret_open": ret,
            "score": ret + rng.normal(0, 0.05, size),
            "ridge": ret + rng.normal(0, 0.2, size),
            "liq_dollar_vol_med_20d": np.tile(np.arange(1.0, len(STOCKS) + 1), len(days)) * 1e6,
            "size_log_mcap": rng.normal(size=size),
            "risk_beta_250d": rng.normal(size=size),
            "tech_vol_20d": np.tile(np.linspace(0.1, 0.8, len(STOCKS)), len(days)),
            "industry": np.tile([f"ind{k % 4}" for k in range(len(STOCKS))], len(days)),
            "q10": -0.1,
            "q50": 0.0,
            "q90": 0.1,
            "vol": math.asinh(0.1) / VOL_SCALE,
            "prob": 0.3,
        }
    )
    frame["fwd_rank_pct"] = frame.groupby("date")["fwd_ret_open"].rank(pct=True)
    return frame


def test_folds_cover_tuning_research_and_holdout_periods() -> None:
    flags = [(fold.tuning, fold.holdout) for fold in folds()]

    assert flags[0] == (True, False)
    assert flags[-1] == (False, True)
    assert len(research()) == 4
    assert {fold.test_start.year for fold in research()} == {2016, 2017}


def test_rank_report_counts_tradable_universe_rows_of_research_folds() -> None:
    frame = panel()
    excluded = frame["stock_id"].isin(STOCKS[:5])
    frame.loc[excluded, "in_universe"] = False
    frame.loc[frame["stock_id"].isin(STOCKS[5:8]), "untradable_t1"] = True
    frame.loc[excluded | frame["untradable_t1"], "score"] = -frame["fwd_rank_pct"]
    elsewhere = ~inside(frame, research())
    frame.loc[elsewhere, "score"] = frame.loc[elsewhere, "fwd_rank_pct"]
    third = research()[2]
    flipped = inside(frame, [third])
    frame.loc[flipped, "score"] = -frame.loc[flipped, "fwd_rank_pct"]

    report = rank_report(frame, "score", 20, folds())

    kept = frame[frame["in_universe"] & ~frame["untradable_t1"]]
    ic = daily_rank_ic(kept, "score")
    pd.testing.assert_frame_equal(report.by_fold, ic_by_fold(ic, folds(), 20))
    expected = summarize_ic(research_ic(ic, folds()), 20)
    assert report.total == expected
    assert report.total.mean < 0.9
    assert report.positive_share == 0.75
    buckets = rank_ic_by_bucket(kept, "score")
    terciles = [summarize_ic(research_ic(buckets[n], folds()), 20).mean for n in (1, 2, 3)]
    assert report.by_liquidity["mean"].tolist() == pytest.approx(terciles)
    neutral = summarize_ic(research_ic(neutralized_rank_ic(kept, "score"), folds()), 20)
    assert report.neutralized == neutral
    assert math.isnan(report.beat_share)
    days = research_ic(ic, folds())
    years = pd.Index([day.year for day in days.index])
    assert report.by_year.index.tolist() == [2016, 2017]
    assert report.by_year.loc[2016, "n_days"] == (years == 2016).sum()
    assert report.by_year.loc[2017, "mean"] == pytest.approx(days[years == 2017].mean())
    months = pd.Index([day.replace(day=1) for day in days.index])
    assert report.monthly.index.tolist() == sorted(set(months))
    assert report.monthly.iloc[0] == pytest.approx(days[months == months.min()].mean())
    assert report.monthly_days.index.tolist() == report.monthly.index.tolist()
    assert report.monthly_days.iloc[0] == (months == months.min()).sum()
    assert report.monthly_days.sum() == report.total.n_days


def test_beat_share_compares_model_and_baseline_on_the_same_rows() -> None:
    frame = panel()
    best = inside(frame, [research()[0]])
    frame.loc[best, "ridge"] = frame.loc[best, "fwd_rank_pct"]
    missing = frame["stock_id"].isin(STOCKS[10:20])
    frame.loc[missing, "ridge"] = np.nan
    frame.loc[missing, "score"] = -frame.loc[missing, "fwd_rank_pct"]
    alone = rank_report(frame, "score", 20, folds())

    report = rank_report(frame, "score", 20, folds(), baseline="ridge")

    shared = frame[~missing]
    model = ic_by_fold(daily_rank_ic(shared, "score"), folds(), 20)
    base = ic_by_fold(daily_rank_ic(shared, "ridge"), folds(), 20)
    assert report.by_fold["paired_mean"].tolist() == pytest.approx(model["mean"].tolist())
    assert report.by_fold["baseline_mean"].tolist() == pytest.approx(base["mean"].tolist())
    assert report.beat_share == 0.75
    pd.testing.assert_frame_equal(report.by_fold[alone.by_fold.columns], alone.by_fold)
    assert report.total == alone.total
    assert report.neutralized == alone.neutralized
    pd.testing.assert_frame_equal(report.by_liquidity, alone.by_liquidity)


def test_a_baseline_without_a_fold_leaves_the_model_metrics_whole() -> None:
    frame = panel()
    best = inside(frame, [research()[0]])
    frame.loc[best, "ridge"] = frame.loc[best, "fwd_rank_pct"]
    frame.loc[inside(frame, [research()[3]]), "ridge"] = np.nan

    report = rank_report(frame, "score", 20, folds(), baseline="ridge")

    assert report.total == rank_report(frame, "score", 20, folds()).total
    assert report.positive_share == 1.0
    assert report.beat_share == pytest.approx(2 / 3)
    assert report.compared_folds == 3
    beats = report.by_fold["beats"]
    assert pd.isna(beats[research()[3].index])
    assert not beats[research()[0].index]
    assert beats[research()[1].index]


def test_liquidity_terciles_and_neutralized_ic() -> None:
    frame = panel()
    liquid = frame["stock_id"].isin(STOCKS[-33:])
    noise = np.random.default_rng(5).normal(size=len(frame))
    frame["score"] = np.where(liquid, frame["fwd_rank_pct"], noise)
    elsewhere = ~inside(frame, research())
    frame.loc[elsewhere, "score"] = -frame.loc[elsewhere, "fwd_rank_pct"]

    report = rank_report(frame, "score", 5, folds())

    assert report.by_liquidity.index.tolist() == [1, 2, 3]
    means = report.by_liquidity["mean"]
    assert means[3] > 0.9
    assert abs(means[1]) < 0.1
    expected = summarize_ic(research_ic(neutralized_rank_ic(frame, "score"), folds()), 5)
    assert report.neutralized == expected

    exposure = frame.assign(score=2 * frame["tech_vol_20d"] - frame["size_log_mcap"])
    flat = rank_report(exposure, "score", 5, folds())
    assert flat.total.n_days > 0
    assert flat.neutralized.n_days == 0


def test_leak_flags_need_the_model_ahead_of_the_baseline_by_the_gap() -> None:
    paired = pd.Series([0.20, 0.15, 0.10, math.nan, 0.30, -0.01])
    compared = pd.Series([0.14, 0.10, 0.06, 0.10, math.nan, -0.07])

    flags = leak_flags(paired, compared)

    assert flags.tolist()[:3] == [True, True, False]
    assert pd.isna(flags[3])
    assert pd.isna(flags[4])
    assert flags[5]
    assert str(flags.dtype) == "boolean"


def test_a_research_fold_far_ahead_of_its_baseline_is_flagged_as_a_leak() -> None:
    frame = panel()
    ahead = research()[1]
    chosen = inside(frame, [ahead])
    frame["ridge"] = frame["score"].where(~chosen, frame["ridge"])

    report = rank_report(frame, "score", 20, folds(), baseline="ridge")

    assert report.leak_folds == (ahead.index,)
    assert report.unjudged_folds == ()
    assert report.by_fold.loc[ahead.index, "leak"]


def test_an_equal_baseline_clears_every_research_fold() -> None:
    frame = panel()
    tuning = next(fold for fold in folds() if fold.tuning)
    frame["ridge"] = frame["score"]
    noise = np.random.default_rng(4).normal(size=len(frame))
    early = inside(frame, [tuning])
    frame.loc[early, "ridge"] = noise[early.to_numpy()]

    report = rank_report(frame, "score", 5, folds(), baseline="ridge")

    assert report.leak_folds == ()
    assert report.unjudged_folds == ()
    assert report.by_fold.loc[tuning.index, "leak"]


def test_research_folds_without_a_baseline_are_unjudged_not_cleared() -> None:
    frame = panel()
    copied = frame.assign(score=frame["fwd_rank_pct"])
    missing = research()[2]
    copied.loc[inside(copied, [missing]), "ridge"] = np.nan

    alone = rank_report(copied, "score", 20, folds())
    compared = rank_report(copied, "score", 20, folds(), baseline="ridge")

    assert alone.leak_folds == ()
    assert alone.unjudged_folds == tuple(fold.index for fold in research())
    assert compared.unjudged_folds == (missing.index,)
    assert compared.leak_folds == tuple(
        fold.index for fold in research() if fold.index != missing.index
    )


@pytest.mark.parametrize(
    ("ic", "flagged"),
    [(0.0199, False), (-0.0199, False), (0.02, True), (-0.031, True), (math.nan, None)],
)
def test_a_shuffle_check_flags_an_absolute_ic_at_the_limit(ic: float, flagged: bool | None) -> None:
    assert ShuffleCheck(fold=3, horizon=20, ic=ic, limit=0.02).leak is flagged


def test_importance_drift_ranks_top_features_and_counts_overlap() -> None:
    gains = {
        3: {"a": 5.0, "b": 4.0, "c": 3.0, "d": 1.0},
        1: {"a": 1.0, "b": 9.0, "c": 9.0, "e": 2.0},
        2: {"d": 7.0, "e": 6.0, "f": 5.0},
    }

    drift = importance_drift(gains, top=3)

    assert drift.by_fold.index.tolist() == [1, 2, 3]
    assert drift.by_fold.loc[1, "top"] == ("b", "c", "e")
    assert drift.by_fold.loc[2, "top"] == ("d", "e", "f")
    assert drift.by_fold.loc[3, "top"] == ("a", "b", "c")
    kept = drift.by_fold["kept"]
    assert math.isnan(kept[1])
    assert kept[2] == 1
    assert kept[3] == 0
    assert drift.frequency.to_dict() == {"b": 2, "c": 2, "e": 2, "a": 1, "d": 1, "f": 1}
    assert list(drift.frequency.index) == ["b", "c", "e", "a", "d", "f"]
    gap = importance_drift({1: gains[1], 3: gains[3]}, top=3)
    assert gap.by_fold["kept"][3] == 2


def prices(frame: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = weekdays(TESTED, END)
    steps = rng.normal(0, 0.02, (len(days), len(STOCKS)))
    closes = 50 * np.exp(np.cumsum(steps, axis=0))
    opens = closes * np.exp(rng.normal(0, 0.005, closes.shape))
    return pd.DataFrame(
        {
            "date": np.repeat(np.array(days, dtype=object), len(STOCKS)),
            "stock_id": np.tile(STOCKS, len(days)),
            "adj_open": opens.ravel(),
            "adj_close": closes.ravel(),
        }
    )


@pytest.mark.parametrize("horizon", [5, 20])
def test_decile_report_holds_research_signals_net_of_costs(
    horizon: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = panel()
    bars = prices(frame)
    gone = frame["stock_id"].isin(STOCKS[:2]) & inside(frame, [research()[1]])
    frame.loc[gone, "untradable_t1"] = True
    outside = frame["stock_id"].isin(STOCKS[2:6]) & inside(frame, [research()[2]])
    frame.loc[outside, "in_universe"] = False
    for stock, day in frame.loc[gone, ["stock_id", "date"]].itertuples(index=False):
        following = calendar().shift(day, 1)
        bars.loc[(bars["stock_id"] == stock) & (bars["date"] == following), "adj_open"] = np.nan
    seen = []
    original = evaluation.tranche_returns

    def recorded(signal: pd.DataFrame, used: pd.DataFrame, *args: Any) -> pd.DataFrame:
        seen.append(used["date"].max())
        return original(signal, used, *args)

    monkeypatch.setattr(evaluation, "tranche_returns", recorded)

    report = decile_report(frame, "score", horizon, folds(), bars, calendar())

    rows = frame[inside(frame, research()) & frame["in_universe"] & ~frame["untradable_t1"]]
    signal = rows[["date", "stock_id", "liq_dollar_vol_med_20d"]].assign(score=rows["score"])
    for sample, chosen in (("all", signal), ("liquid", top_liquidity(signal))):
        expected = tranche_summary(original(chosen, bars, calendar(), horizon), horizon)
        pd.testing.assert_frame_equal(cast(pd.DataFrame, report.loc[sample]), expected)
    last = max(fold.eval_end for fold in research())
    assert max(seen) == calendar().shift(last, horizon)
    assert max(seen) < next(fold for fold in folds() if fold.holdout).test_start
    assert report["annual_turnover"][("all", "D10")] > 0


def test_coverage_report_buckets_research_rows_by_volatility() -> None:
    frame = panel()
    calm = frame["stock_id"].isin(STOCKS[:30])
    wild = frame["stock_id"].isin(STOCKS[-20:])
    frame.loc[calm, ["q10", "q90"]] = [-1.0, 1.0]
    frame.loc[wild, ["q10", "q90"]] = [10.0, 10.0]
    outside = ~inside(frame, research())
    frame.loc[outside, ["q10", "q90"]] = [-1.0, 1.0]
    dropped = frame["stock_id"].isin(STOCKS[:8])
    frame.loc[dropped, ["q10", "q90"]] = [10.0, 10.0]
    frame.loc[frame["stock_id"].isin(STOCKS[:4]), "in_universe"] = False
    frame.loc[frame["stock_id"].isin(STOCKS[4:8]), "untradable_t1"] = True

    table = coverage_report(frame, "fwd_ret_open", "q10", "q90", folds())

    assert table.index.tolist() == [1, 2, 3, 4, 5, "all"]
    assert table.loc[1, "coverage"] == 1.0
    assert table.loc[1, "width"] == pytest.approx(2.0)
    assert table.loc[5, "coverage"] == 0.0
    assert table.loc[5, "width"] == 0.0
    rows = frame[inside(frame, research()) & ~dropped]
    overall = daily_coverage(rows, "fwd_ret_open", "q10", "q90").mean()
    assert table.loc["all", "coverage"] == pytest.approx(overall)
    assert table.loc["all", "width"] == pytest.approx(daily_width(rows, "q10", "q90").mean())


def test_consistency_report_counts_wide_bands_and_contradictions_per_day() -> None:
    frame = panel()
    wide = frame["stock_id"].isin(STOCKS[:10])
    frame.loc[wide, "vol"] = math.asinh(0.3) / VOL_SCALE
    frame.loc[frame["stock_id"] == STOCKS[99], ["q10", "q90"]] = [0.0, 0.0]
    frame.loc[frame["stock_id"].isin(STOCKS[10:14]), "prob"] = 0.7
    frame.loc[frame["stock_id"].isin(STOCKS[14:16]), ["prob", "q50"]] = [0.2, 0.06]
    rows = inside(frame, research())
    busy = frame["date"] == frame.loc[rows, "date"].min()
    frame.loc[busy & frame["stock_id"].isin(STOCKS[16:20]), "prob"] = 0.9
    frame.loc[frame["stock_id"].isin(STOCKS[20:45]), "in_universe"] = False
    frame.loc[~rows, "vol"] = 9.0

    report = consistency_report(frame, "q10", "q50", "q90", "vol", "prob", folds())

    days = frame.loc[rows, "date"].nunique()
    assert report.loc["width_ratio", "mean_share"] == pytest.approx(11 / 75)
    assert report.loc["width_ratio", "max_share"] == pytest.approx(11 / 75)
    assert report.loc["width_ratio", "days_over"] == days
    assert report.loc["contradiction", "mean_share"] == pytest.approx((6 * days + 4) / (75 * days))
    assert report.loc["contradiction", "max_share"] == pytest.approx(10 / 75)
    assert report.loc["contradiction", "days_over"] == 1
    assert report.loc["contradiction", "n_days"] == days


@pytest.mark.parametrize(("ratio", "share"), [(0.51, 0.0), (1.99, 0.0), (0.49, 1.0), (2.01, 1.0)])
def test_band_ratios_inside_half_to_double_are_consistent(ratio: float, share: float) -> None:
    frame = panel().assign(vol=math.asinh(0.1 * ratio) / VOL_SCALE)

    report = consistency_report(frame, "q10", "q50", "q90", "vol", "prob", folds())

    assert report.loc["width_ratio", "mean_share"] == share


def test_exact_bounds_are_consistent_and_an_alert_needs_more_than_ten_percent() -> None:
    frame = panel()
    width = band_width(frame["vol"])
    frame["q10"] = 0.0
    frame["q90"] = np.where(frame["stock_id"].isin(STOCKS[:50]), 2 * width, width / 2)
    frame.loc[frame["stock_id"].isin(STOCKS[:10]), "prob"] = 0.5
    frame.loc[frame["stock_id"].isin(STOCKS[10:20]), ["prob", "q50"]] = [0.7, 0.05]
    frame.loc[frame["stock_id"].isin(STOCKS[20:30]), ["prob", "q50"]] = [0.3, 0.05]
    frame.loc[frame["stock_id"].isin(STOCKS[30:40]), ["prob", "q50"]] = [0.5, 0.06]
    first = frame.loc[inside(frame, research()), "date"].min()
    frame.loc[(frame["date"] == first) & frame["stock_id"].isin(STOCKS[40:50]), "prob"] = 0.9

    report = consistency_report(frame, "q10", "q50", "q90", "vol", "prob", folds())

    assert report.loc["width_ratio", "mean_share"] == 0.0
    assert report.loc["contradiction", "max_share"] == pytest.approx(0.1)
    assert report.loc["contradiction", "days_over"] == 0


def test_consistency_counts_untradable_rows_and_skips_rows_missing_an_output() -> None:
    frame = panel()
    stuck = frame["stock_id"].isin(STOCKS[:10])
    frame.loc[stuck, "untradable_t1"] = True
    frame.loc[stuck, "vol"] = math.asinh(0.3) / VOL_SCALE
    frame.loc[stuck, "prob"] = 0.9
    frame.loc[frame["stock_id"].isin(STOCKS[10:30]), "vol"] = np.nan
    frame.loc[frame["stock_id"].isin(STOCKS[30:50]), "prob"] = np.nan

    report = consistency_report(frame, "q10", "q50", "q90", "vol", "prob", folds())

    assert report.loc["width_ratio", "mean_share"] == pytest.approx(10 / 80)
    assert report.loc["contradiction", "mean_share"] == pytest.approx(10 / 80)
