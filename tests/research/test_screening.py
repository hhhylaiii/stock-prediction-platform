import math
import re
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice, TradingCalendar
from apps.research.features.store import export_year
from apps.research.features.transforms import STOCK_LEVEL
from apps.research.holdout import claim_holdout
from apps.research.management.commands import ic_report
from apps.research.metrics.ic import daily_rank_ic, summarize_ic
from apps.research.models import FeatureDaily, LabelDaily, ModelRun
from apps.research.screening import (
    CORRELATION_CEILING,
    IC_FLOOR,
    Screening,
    mean_daily_correlation,
    screen,
    screening_window,
    select,
    single_factor_ic,
)
from apps.research.walkforward import WalkForwardConfig, generate_folds

START, END = date(2016, 1, 1), date(2016, 12, 31)
STOCKS = [f"{1100 + k}" for k in range(40)]
CONFIG = WalkForwardConfig(
    train_months=6,
    test_months=1,
    step_months=1,
    valid_months=2,
    data_start=START,
    first_test=date(2016, 9, 1),
    tuning_end=date(2016, 10, 31),
    holdout_start=date(2016, 12, 1),
)


def weekdays() -> list[date]:
    days = (START + timedelta(days=n) for n in range((END - START).days + 1))
    return [day for day in days if day.weekday() < 5]


def table(values: dict[str, tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(values, orient="index", columns=[5, 20])


def correlations(names: list[str], pairs: dict[tuple[str, str], float]) -> pd.DataFrame:
    matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for (first, second), rho in pairs.items():
        matrix.loc[first, second] = matrix.loc[second, first] = rho
    return matrix


def test_signal_features_below_the_floor_at_every_horizon_are_dropped() -> None:
    ic = table(
        {
            "tech_ret_5d": (0.004, -0.004),
            "tech_ret_20d": (0.004, 0.006),
            "chip_fi_net_5d": (-0.03, 0.001),
            "tech_vol_20d": (0.001, 0.0),
            "chip_fi_net_1d": (math.nan, math.nan),
            "size_log_mcap": (math.nan, math.nan),
            "tech_ma_gap_60d": (IC_FLOOR, -IC_FLOOR),
        }
    )

    reasons = select(ic, correlations(list(ic.index), {}))

    assert set(reasons) == {"tech_ret_5d"}
    assert str(IC_FLOOR) in reasons["tech_ret_5d"]


def test_correlated_pairs_keep_the_stronger_feature() -> None:
    ic = table(
        {
            "tech_ret_5d": (0.02, 0.01),
            "tech_ret_20d": (0.01, 0.03),
            "tech_ma_gap_5d": (0.02, 0.02),
            "tech_ma_gap_20d": (0.05, 0.01),
            "tech_rsi_14": (0.02, 0.02),
            "tech_kd_k": (0.01, 0.01),
        }
    )
    pairs = {
        ("tech_ret_5d", "tech_ret_20d"): 0.97,
        ("tech_ma_gap_5d", "tech_ma_gap_20d"): -0.96,
        ("tech_rsi_14", "tech_kd_k"): CORRELATION_CEILING,
    }

    reasons = select(ic, correlations(list(ic.index), pairs))

    assert set(reasons) == {"tech_ret_5d", "tech_ma_gap_5d"}
    assert "duplicate of tech_ret_20d" in reasons["tech_ret_5d"]
    assert "-0.960" in reasons["tech_ma_gap_5d"]


def test_low_ic_pruning_runs_before_deduplication() -> None:
    ic = table({"tech_ret_5d": (0.004, 0.004), "tech_vol_20d": (0.002, 0.003)})

    reasons = select(ic, correlations(list(ic.index), {("tech_ret_5d", "tech_vol_20d"): 0.99}))

    assert list(reasons) == ["tech_ret_5d"]


def test_deduplication_keeps_the_strongest_and_drops_only_twins_of_kept_features() -> None:
    ic = table(
        {"tech_ret_5d": (0.02, 0.0), "tech_ret_20d": (0.01, 0.0), "tech_ret_60d": (0.03, 0.0)}
    )
    pairs = {
        ("tech_ret_5d", "tech_ret_20d"): 0.99,
        ("tech_ret_5d", "tech_ret_60d"): 0.97,
        ("tech_ret_20d", "tech_ret_60d"): 0.90,
    }

    reasons = select(ic, correlations(list(ic.index), pairs))

    assert list(reasons) == ["tech_ret_5d"]
    assert "duplicate of tech_ret_60d" in reasons["tech_ret_5d"]


def test_a_duplicate_names_its_tightest_kept_twin() -> None:
    ic = table(
        {"tech_ret_1d": (0.05, 0.05), "tech_ret_5d": (0.04, 0.04), "tech_ret_20d": (0.03, 0.03)}
    )
    pairs = {("tech_ret_1d", "tech_ret_20d"): 0.96, ("tech_ret_5d", "tech_ret_20d"): -0.99}

    reasons = select(ic, correlations(list(ic.index), pairs))

    assert list(reasons) == ["tech_ret_20d"]
    assert "duplicate of tech_ret_5d" in reasons["tech_ret_20d"]


def test_features_without_an_ic_rank_after_every_scored_feature() -> None:
    ic = table({"chip_fi_net_1d": (math.nan, math.nan), "chip_fi_net_5d": (0.02, 0.0)})

    reasons = select(ic, correlations(list(ic.index), {("chip_fi_net_1d", "chip_fi_net_5d"): 0.99}))

    assert list(reasons) == ["chip_fi_net_1d"]
    assert "duplicate of chip_fi_net_5d" in reasons["chip_fi_net_1d"]


def test_ties_drop_the_later_feature_in_registry_order() -> None:
    ic = table({"tech_ret_20d": (0.02, 0.02), "tech_ret_5d": (0.02, 0.02)})

    reasons = select(ic, correlations(list(ic.index), {("tech_ret_5d", "tech_ret_20d"): 0.99}))

    assert list(reasons) == ["tech_ret_20d"]


def panel(days: list[date], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    size = len(days) * len(STOCKS)
    frame = pd.DataFrame(
        {
            "date": np.repeat(np.array(days, dtype=object), len(STOCKS)),
            "stock_id": np.tile(STOCKS, len(days)),
            "in_universe": np.tile([stock != "1139" for stock in STOCKS], len(days)),
            **{name: rng.normal(size=size) for name in STOCK_LEVEL},
        }
    )
    signal = rng.normal(size=size)
    return frame.assign(
        tech_ret_20d=signal,
        tech_ret_60d=signal + 0.2 * rng.normal(size=size),
        fwd_rank_pct=pd.Series(signal + rng.normal(size=size))
        .groupby(frame["date"])
        .rank(pct=True),
        untradable_t1=False,
    )


def test_single_factor_ic_matches_the_daily_rank_ic() -> None:
    frame = panel(weekdays()[:30], seed=1)

    ic = single_factor_ic(frame, ["tech_ret_20d", "tech_ret_1d"], horizon=20)

    daily = daily_rank_ic(frame, "tech_ret_20d", "fwd_rank_pct")
    assert ic.loc["tech_ret_20d", "mean"] == pytest.approx(daily.mean())
    assert ic.loc["tech_ret_20d", "n_days"] == 30
    assert ic.loc["tech_ret_20d", "nw_t"] == pytest.approx(summarize_ic(daily, lag=20).nw_t)


def test_mean_daily_correlation_averages_each_day_spearman() -> None:
    frame = panel(weekdays()[:5], seed=2)

    matrix = mean_daily_correlation(frame, ["tech_ret_20d", "tech_ret_60d", "tech_ret_1d"])

    expected = np.mean(
        [
            day["tech_ret_20d"].corr(day["tech_ret_1d"], method="spearman")
            for _, day in frame.groupby("date")
        ]
    )
    assert matrix.loc["tech_ret_20d", "tech_ret_1d"] == pytest.approx(expected)
    assert matrix.to_numpy()[0, 1] > CORRELATION_CEILING
    assert matrix.loc["tech_ret_1d", "tech_ret_1d"] == pytest.approx(1.0)


def test_mean_daily_correlation_pairs_shared_stocks_and_skips_thin_days() -> None:
    frame = panel(weekdays()[:5], seed=5)
    first_day = frame["date"].eq(frame["date"].iloc[0])
    frame.loc[first_day & frame["stock_id"].lt("1115"), "tech_ret_1d"] = np.nan
    frame.loc[~first_day & frame["stock_id"].lt("1105"), "tech_ret_1d"] = np.nan
    frame.loc[frame["stock_id"].gt("1134"), "tech_ret_60d"] = np.nan
    names = ["tech_ret_20d", "tech_ret_1d", "tech_ret_60d"]

    matrix = mean_daily_correlation(frame, names)

    expected = np.mean(
        [
            pair["tech_ret_20d"].corr(pair["tech_ret_1d"], method="spearman")
            for _, day in frame[~first_day].groupby("date")
            for pair in [day[["tech_ret_20d", "tech_ret_1d"]].dropna()]
        ]
    )
    assert matrix.loc["tech_ret_20d", "tech_ret_1d"] == pytest.approx(expected)


def test_screen_counts_only_rows_tradable_at_the_next_open() -> None:
    frame = panel(weekdays()[:30], seed=6).assign(
        untradable_t1=lambda rows: rows["stock_id"].lt("1105")
    )
    labels = frame[["stock_id", "date", "fwd_rank_pct", "untradable_t1"]]
    features = frame.drop(columns=["fwd_rank_pct", "untradable_t1"])

    result = screen(features, {5: labels, 20: labels})

    usable = frame[frame["in_universe"] & ~frame["untradable_t1"]]
    expected = daily_rank_ic(usable, "tech_ret_1d", "fwd_rank_pct").mean()
    assert result.table.loc["tech_ret_1d", "ic_h5"] == pytest.approx(expected)


def test_screen_refuses_labels_missing_on_any_day_of_the_window() -> None:
    frame = panel(weekdays()[:30], seed=7)
    labels = frame[["stock_id", "date", "fwd_rank_pct", "untradable_t1"]]
    features = frame.drop(columns=["fwd_rank_pct", "untradable_t1"])
    partial = labels[labels["date"] > weekdays()[3]]

    with pytest.raises(ValueError, match=r"horizon 5 .* 4 of 30 days"):
        screen(features, {5: partial, 20: labels})
    with pytest.raises(ValueError, match=r"horizon 20 .* 30 of 30 days"):
        screen(features, {5: labels, 20: labels.iloc[:0]})


def test_render_escapes_pipes_inside_cells() -> None:
    names = ["tech_ret_5d", "tech_ret_20d", "chip_big_holder_pct_chg_1w"]
    dropped = {
        "tech_ret_5d": select(
            table({"tech_ret_5d": (0.001, 0.001)}), correlations(["tech_ret_5d"], {})
        )["tech_ret_5d"]
    }
    result = Screening(
        pd.DataFrame(
            {
                "ic_h5": [0.001, 0.03, math.nan],
                "nw_t_h5": [0.1, 2.0, math.nan],
                "ic_h20": [0.001, 0.02, math.nan],
                "nw_t_h20": [0.1, 1.5, math.nan],
            },
            index=names,
        ).assign(conditional=False, reason=[dropped["tech_ret_5d"], "", ""]),
        dropped,
        correlations(names, {}),
    )

    text = ic_report.render(result, START, END, "rank-screen1.screening")

    rows = [line for line in text.splitlines() if line.startswith(("| tech_", "| chip_", "| mkt_"))]
    assert "|" in dropped["tech_ret_5d"]
    assert len(rows) == 3 + len(ic_report.UNSCREENED)
    assert all(len(re.findall(r"(?<!\\)\|", row)) == 8 for row in rows)
    assert "| mkt_taiex_ret_5d | 市場 |" in text
    assert rows[1].endswith("| 保留 |")
    assert "未篩選" in rows[2]


def test_screen_keeps_the_signal_and_drops_its_duplicate() -> None:
    frame = panel(weekdays()[:40], seed=3)
    labels = frame[["stock_id", "date", "fwd_rank_pct", "untradable_t1"]]
    features = frame.drop(columns=["fwd_rank_pct", "untradable_t1"])

    result = screen(features, {5: labels, 20: labels})

    assert "tech_ret_20d" in result.kept
    assert "duplicate of tech_ret_20d" in result.dropped["tech_ret_60d"]
    assert set(result.kept) | set(result.dropped) == set(STOCK_LEVEL)
    inside = frame[frame["in_universe"]]
    expected = daily_rank_ic(inside, "tech_ret_20d", "fwd_rank_pct").mean()
    assert result.table.loc["tech_ret_20d", "ic_h20"] == pytest.approx(expected)
    pd.testing.assert_frame_equal(
        result.correlation,
        mean_daily_correlation(inside.drop(columns=["fwd_rank_pct"]), STOCK_LEVEL),
    )
    assert not any(name.startswith("mkt_") for name in result.table.index)


def test_screening_window_spans_the_tuning_folds_training_windows() -> None:
    folds = generate_folds(TradingDays(weekdays(), START, END), END, CONFIG)
    tuning = [fold for fold in folds if fold.tuning]

    assert screening_window(folds) == (tuning[0].train_start, tuning[-1].train_end)
    assert len(tuning) == 2
    with pytest.raises(ValueError, match="tuning"):
        screening_window([fold for fold in folds if not fold.tuning])


@pytest.fixture
def stored(settings: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    settings.FEATURE_PARQUET_DIR = tmp_path / "features"
    settings.MODEL_DIR = tmp_path / "models"
    settings.RESEARCH_CACHE_DIR = tmp_path / "predictions"
    monkeypatch.setattr(ic_report, "DEFAULT_CONFIG", CONFIG)
    days = weekdays()
    TradingCalendar.objects.bulk_create(
        TradingCalendar(
            date=START + timedelta(days=n), is_trading_day=(START + timedelta(days=n)).weekday() < 5
        )
        for n in range((END - START).days + 1)
    )
    frame = panel(days, seed=4)
    FeatureDaily.objects.bulk_create(
        FeatureDaily(**{str(name): value for name, value in row.items()})
        for row in frame.drop(columns=["fwd_rank_pct", "untradable_t1"]).to_dict("records")
    )
    LabelDaily.objects.bulk_create(
        LabelDaily(
            stock_id=row["stock_id"],
            date=row["date"],
            horizon=horizon,
            fwd_rank_pct=row["fwd_rank_pct"],
            untradable_t1=False,
        )
        for row in frame[["stock_id", "date", "fwd_rank_pct"]].to_dict("records")
        for horizon in (5, 20)
    )
    export_year(2016, settings.FEATURE_PARQUET_DIR)
    DailyPrice.objects.create(
        stock_id="1101", date=date(2016, 12, 27), close=10, volume=1000, turnover=10_000
    )
    return tmp_path


@pytest.mark.django_db
def test_ic_report_records_the_kept_list_and_writes_the_report(stored: Path) -> None:
    output = stored / "report.md"
    out = StringIO()

    call_command("ic_report", run_key="rank-screen1", output=str(output), stdout=out)

    run = ModelRun.objects.get()
    assert run.run_id == "rank-screen1.screening"
    assert run.model_type == "screening"
    assert "tech_ret_20d" in run.feature_list
    assert "tech_ret_60d" not in run.feature_list
    assert "duplicate of tech_ret_20d" in run.params["dropped"]["tech_ret_60d"]
    assert run.params["ic_floor"] == IC_FLOOR
    assert run.params["unscreened"] == list(ic_report.UNSCREENED)
    folds = generate_folds(TradingDays(weekdays(), START, END), END, CONFIG)
    assert (run.train_start, run.train_end) == screening_window(folds)
    text = output.read_text()
    assert "| tech_ret_20d |" in text
    assert "duplicate of tech_ret_20d" in text
    assert "rank-screen1.screening" in out.getvalue()
    for position, stock in enumerate(STOCKS):
        later = {"stock_id": stock, "date__gt": run.train_end}
        LabelDaily.objects.filter(**later).update(fwd_rank_pct=(position + 1) / len(STOCKS))
        FeatureDaily.objects.filter(**later).update(tech_ret_20d=float(len(STOCKS) - position))
    export_year(2016, Path(stored) / "features")

    call_command("ic_report", run_key="rank-screen2", output=str(output), stdout=StringIO())

    rerun = ModelRun.objects.get(run_id="rank-screen2.screening")
    assert rerun.params["ic"] == run.params["ic"]
    assert rerun.params["dropped"] == run.params["dropped"]
    assert rerun.feature_list == run.feature_list


@pytest.mark.django_db
@pytest.mark.parametrize("run_key", ["Rank-screen", "rank-" + "a" * 55])
def test_ic_report_checks_the_run_key_before_screening(
    stored: Path, monkeypatch: pytest.MonkeyPatch, run_key: str
) -> None:
    monkeypatch.setattr(ic_report, "screen", lambda *args: pytest.fail("screened first"))

    with pytest.raises(CommandError, match="lowercase"):
        call_command("ic_report", run_key=run_key, output=str(stored / "report.md"))


@pytest.mark.django_db
def test_ic_report_refuses_a_last_date_inside_the_tuning_period(stored: Path) -> None:
    with pytest.raises(CommandError, match="tuning period"):
        call_command(
            "ic_report",
            run_key="rank-screen1",
            last_date="2016-10-31",
            output=str(stored / "report.md"),
        )

    assert not ModelRun.objects.exists()


@pytest.mark.django_db
def test_ic_report_refuses_missing_labels(stored: Path) -> None:
    LabelDaily.objects.filter(horizon=5).delete()

    with pytest.raises(CommandError, match="horizon 5"):
        call_command("ic_report", run_key="rank-screen1", output=str(stored / "report.md"))

    assert not ModelRun.objects.exists()


@pytest.mark.django_db
def test_ic_report_refuses_partitions_that_change_while_reading(
    stored: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    digests = iter(["before", "after"])
    monkeypatch.setattr(ic_report, "feature_hash", lambda *args: next(digests))

    with pytest.raises(CommandError, match="changed while reading"):
        call_command("ic_report", run_key="rank-screen1", output=str(stored / "report.md"))

    assert not ModelRun.objects.exists()


@pytest.mark.django_db
def test_ic_report_refuses_a_holdout_claimed_while_screening(
    stored: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = ic_report.screen

    def claimed(*args: Any) -> Screening:
        claim_holdout("rank", "rank-final", date(2017, 3, 31))
        return real(*args)

    monkeypatch.setattr(ic_report, "screen", claimed)

    with pytest.raises(CommandError, match="holdout"):
        call_command("ic_report", run_key="rank-screen1", output=str(stored / "report.md"))

    assert not ModelRun.objects.filter(model_type="screening").exists()


@pytest.mark.django_db
def test_ic_report_refuses_to_rescreen_after_a_holdout_run(
    stored: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claim_holdout("rank", "rank-final", date(2017, 3, 31))
    monkeypatch.setattr(ic_report, "screen", lambda *args: pytest.fail("screened first"))

    with pytest.raises(CommandError, match="holdout"):
        call_command("ic_report", run_key="rank-screen1", output=str(stored / "report.md"))

    assert not ModelRun.objects.exists()
