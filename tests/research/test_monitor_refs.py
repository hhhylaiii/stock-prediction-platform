import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from apps.market.models import IndexDaily, TradingCalendar
from apps.research.metrics.forecast import daily_brier
from apps.research.metrics.ic import daily_rank_ic
from apps.research.models import LabelDaily, ModelRun
from apps.research.monitor import BRIER_BASELINE, IC_THRESHOLD
from apps.research.monitor_refs import (
    research_brier,
    research_ic_thresholds,
    rolling_ic_threshold,
    store_references,
)
from apps.research.prediction_cache import save_fold_predictions
from apps.research.relative_up import RELATIVE_LABEL
from apps.research.walkforward import Fold

STOCKS = [str(1000 + n) for n in range(40)]
STUCK = STOCKS[:3]
OUTSIDE = [str(9000 + n) for n in range(5)]


def weekdays(start: date, end: date) -> list[date]:
    days = (start + timedelta(days=n) for n in range((end - start).days + 1))
    return [day for day in days if day.weekday() < 5]


def fold(index: int, start: date, end: date, eval_end: date, **flags: bool) -> Fold:
    return Fold(
        index=index,
        train_start=date(2013, 1, 2),
        fit_end=date(2015, 6, 30),
        valid_start=date(2015, 7, 1),
        train_end=start - timedelta(days=30),
        test_start=start,
        test_end=end,
        eval_end=eval_end,
        tuning=flags.get("tuning", False),
        holdout=flags.get("holdout", False),
    )


FOLDS = [
    fold(0, date(2016, 2, 2), date(2016, 2, 27), date(2016, 2, 27), tuning=True),
    fold(1, date(2016, 3, 2), date(2016, 5, 29), date(2016, 5, 13)),
    fold(2, date(2016, 6, 1), date(2016, 8, 31), date(2016, 8, 3)),
    fold(3, date(2016, 9, 1), date(2016, 9, 30), date(2016, 9, 30), holdout=True),
]


def counted(day: date) -> bool:
    """Whether a day sits inside a research fold's evaluated window."""
    return any(f.research and f.test_start <= day <= f.eval_end for f in FOLDS)


@pytest.fixture
def cache(settings: Any, tmp_path: Path) -> None:
    settings.RESEARCH_CACHE_DIR = tmp_path / "predictions"


def test_threshold_is_the_low_tail_of_sixty_day_means() -> None:
    days = pd.Index(weekdays(date(2016, 1, 1), date(2016, 6, 30))[:100])
    ic = pd.Series(np.arange(100) / 1000, index=days)

    shuffled = ic.iloc[np.random.default_rng(0).permutation(100)]
    assert rolling_ic_threshold(shuffled) == pytest.approx(0.0315)
    assert math.isnan(rolling_ic_threshold(ic.iloc[:59]))


def panel(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached outputs and labels; scores are upside down outside the counted rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in weekdays(date(2016, 2, 2), date(2016, 8, 31)):
        truth = rng.normal(0, 0.1, len(STOCKS) + len(OUTSIDE))
        noise = rng.normal(0, 0.05 + 0.2 * rng.random(), len(truth))
        flip = 1.0 if counted(day) else -1.0
        for n, stock in enumerate([*STOCKS, *OUTSIDE]):
            inside = stock in STOCKS
            sign = flip if inside else -1.0
            rows.append(
                {
                    "date": day,
                    "stock_id": stock,
                    "in_universe": inside,
                    "truth": truth[n],
                    "lightgbm_h5": sign * (truth[n] + noise[n]),
                    "lightgbm_h20": sign * (truth[n] + 2 * noise[n]),
                    "logistic_h20": float(np.clip(0.5 + sign * truth[n], 0.01, 0.99)),
                }
            )
    frame = pd.DataFrame(rows)
    stuck = frame["stock_id"].isin(STUCK)
    frame = frame.assign(
        untradable_t1=stuck,
        truth=frame["truth"].mask(stuck, -0.5),
        logistic_h20=frame["logistic_h20"].mask(stuck, 0.99),
    )
    frame["fwd_rank_pct"] = frame["truth"].mask(stuck).groupby(frame["date"]).rank(pct=True)
    frame["fwd_rank_pct_5"] = (-frame["truth"]).mask(stuck).groupby(frame["date"]).rank(pct=True)
    frame["fwd_up_5pct"] = frame["truth"] > 0.05
    return frame, frame


def store(frame: pd.DataFrame, model: str, columns: list[str]) -> None:
    for chosen in FOLDS[:3]:
        part = frame[frame["date"].between(chosen.test_start, chosen.test_end)]
        save_fold_predictions(
            "wf1", model, chosen, part[["date", "stock_id", "in_universe", *columns]]
        )


def write_labels(frame: pd.DataFrame) -> None:
    names = [
        "stock_id",
        "date",
        "fwd_rank_pct_5",
        "fwd_rank_pct",
        "fwd_up_5pct",
        "untradable_t1",
        "truth",
    ]
    columns = zip(*(frame[name] for name in names), strict=True)
    LabelDaily.objects.bulk_create(
        LabelDaily(
            stock_id=str(stock),
            date=day,
            horizon=horizon,
            fwd_rank_pct=None if math.isnan(pct) else float(pct),
            fwd_ret_open=float(ret),
            fwd_up_5pct=bool(up) if horizon == 20 else None,
            untradable_t1=bool(stuck),
        )
        for stock, day, short, long, up, stuck, ret in columns
        for horizon, pct in ((5, short), (20, long))
    )


def expected_threshold(frame: pd.DataFrame, column: str, label: str) -> float:
    rows = frame[frame["in_universe"] & ~frame["untradable_t1"] & frame["date"].map(counted)]
    ic = daily_rank_ic(rows.assign(fwd_rank_pct=rows[label]), column)
    means = ic.rolling(60).mean().dropna()
    return float(means.quantile(0.05))


@pytest.mark.django_db
@pytest.mark.usefixtures("cache")
def test_thresholds_come_from_research_folds_of_the_cache_and_label_daily() -> None:
    frame, labels = panel(1)
    store(frame, "lightgbm", ["lightgbm_h5", "lightgbm_h20"])
    write_labels(labels)

    thresholds = research_ic_thresholds("wf1", "lightgbm", FOLDS)

    assert set(thresholds) == {5, 20}
    assert thresholds[5] == pytest.approx(
        expected_threshold(frame, "lightgbm_h5", "fwd_rank_pct_5")
    )
    assert thresholds[20] == pytest.approx(
        expected_threshold(frame, "lightgbm_h20", "fwd_rank_pct")
    )
    assert thresholds[5] < 0 < thresholds[20]


@pytest.mark.django_db
@pytest.mark.usefixtures("cache")
def test_baseline_brier_is_the_research_mean_of_the_logistic_baseline() -> None:
    frame, labels = panel(2)
    store(frame, "baseline", ["logistic_h20"])
    write_labels(labels)

    brier = research_brier("wf1", FOLDS)

    rows = frame[frame["in_universe"] & ~frame["untradable_t1"] & frame["date"].map(counted)]
    daily = daily_brier(rows.assign(up=rows["fwd_up_5pct"].astype(float)), "up", "logistic_h20")
    assert brier == pytest.approx(daily.mean())
    assert brier < 0.2


def market_rows() -> dict[date, float]:
    """A 2016 weekday calendar and TAIEX_TR's 20-session return from each day."""
    every = (date(2016, 1, 1) + timedelta(days=n) for n in range(366))
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=day.weekday() < 5) for day in every
    )
    days = weekdays(date(2016, 1, 1), date(2016, 12, 31))
    closes = [1000.0 + n for n in range(len(days))]
    IndexDaily.objects.bulk_create(
        IndexDaily(index_id="TAIEX_TR", date=day, close=close)
        for day, close in zip(days, closes, strict=True)
    )
    return {day: closes[n + 20] / closes[n] - 1 for n, day in enumerate(days[:-20])}


def logistic_run(label: str) -> None:
    ModelRun.objects.create(
        run_id="wf1.logistic.h20.f001",
        run_key="wf1",
        model_type="logistic",
        objective="classification",
        version="2016-08",
        train_start=date(2013, 1, 2),
        train_end=date(2016, 1, 31),
        lib_versions={},
        feature_hash="0" * 64,
        params={"horizon": 20, "label": label},
    )


@pytest.mark.django_db
@pytest.mark.usefixtures("cache")
def test_baseline_brier_on_the_relative_label_nets_out_the_market() -> None:
    frame, labels = panel(2)
    store(frame, "baseline", ["logistic_h20"])
    write_labels(labels)
    moves = market_rows()
    logistic_run(RELATIVE_LABEL)

    brier = research_brier("wf1", FOLDS, relative=True)

    rows = frame[frame["in_universe"] & ~frame["untradable_t1"] & frame["date"].map(counted)]
    beats = rows["truth"] - rows["date"].map(moves) > 0.05
    assert 0 < beats.sum() < rows["fwd_up_5pct"].sum()
    daily = daily_brier(rows.assign(up=beats.astype(float)), "up", "logistic_h20")
    alone = daily_brier(rows.assign(up=rows["fwd_up_5pct"].astype(float)), "up", "logistic_h20")
    assert brier == pytest.approx(daily.mean())
    assert brier != pytest.approx(alone.mean())


@pytest.mark.django_db
@pytest.mark.usefixtures("cache")
def test_baseline_brier_refuses_a_label_the_logistic_was_not_fit_on() -> None:
    frame, labels = panel(2)
    store(frame, "baseline", ["logistic_h20"])
    write_labels(labels)
    logistic_run("fwd_up_5pct")

    with pytest.raises(ValueError, match="fit on fwd_up_5pct, not fwd_up_5pct_vs_taiex_tr"):
        research_brier("wf1", FOLDS, relative=True)


def make_run(**params: Any) -> ModelRun:
    return ModelRun.objects.create(
        run_id="prod.rank",
        run_key="prod",
        model_type="lightgbm",
        objective="rank",
        version="2026-09",
        train_start=date(2013, 1, 2),
        train_end=date(2016, 8, 31),
        lib_versions={},
        feature_hash="0" * 64,
        params=params,
    )


@pytest.mark.django_db
def test_references_replace_old_ones_and_keep_other_params() -> None:
    run = make_run(lr=0.1, **{IC_THRESHOLD: {"5": 0.5}, BRIER_BASELINE: 0.3})

    stored = store_references(run.run_id, {5: -0.01, 20: math.nan}, math.nan)

    assert stored.params == {"lr": 0.1, IC_THRESHOLD: {"5": -0.01}}
    assert run.params[BRIER_BASELINE] == 0.3
    again = store_references(run.run_id, {5: -0.02, 20: 0.001}, 0.24)
    assert (
        ModelRun.objects.get(pk=run.pk).params
        == again.params
        == {
            "lr": 0.1,
            IC_THRESHOLD: {"5": -0.02, "20": 0.001},
            BRIER_BASELINE: 0.24,
        }
    )
