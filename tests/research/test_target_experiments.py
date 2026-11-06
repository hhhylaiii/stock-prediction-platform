import math
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice
from apps.research import experiments, lgbm_runs, target_eval, target_experiments
from apps.research.evaluation import tradable_rows
from apps.research.experiments import Arm, arm_key, compared_folds
from apps.research.management.commands import target_experiment
from apps.research.metrics.forecast import (
    daily_absolute_error,
    daily_auc,
    daily_brier,
    daily_coverage,
    daily_direction_hits,
    daily_pinball,
    daily_qlike,
    mean_by_fold,
)
from apps.research.metrics.stats import paired_nw_t
from apps.research.models import HoldoutRun, HoldoutScope, ModelRun
from apps.research.prediction_cache import fold_notes, load_fold_predictions, save_fold_predictions
from apps.research.rank_model import SPEC_RECIPE
from apps.research.runs import SCREENING_MODEL
from apps.research.target_experiments import NO_MARKET, TargetExperiment
from apps.research.target_models import TARGET_RECIPE
from apps.research.training import KEYS, load_labels
from apps.research.walkforward import Fold, WalkForwardConfig
from tests.research.run_panels import CONFIG, weekdays

QUANTILES = {0.1: "lightgbm_q10_h20", 0.5: "lightgbm_q50_h20", 0.9: "lightgbm_q90_h20"}


def test_the_market_experiments_compare_stock_features_alone_with_market_features_added() -> None:
    chosen = target_experiments.target_experiments(WalkForwardConfig())

    assert set(chosen) == {"market-return", "market-interval"}
    for name, scope in (("market-return", "return"), ("market-interval", "interval")):
        experiment = chosen[name]
        assert experiment.scope == scope
        assert experiment.rule == "better"
        alone, market = experiment.arms
        assert (alone.name, market.name) == ("nomarket", "market")
        assert (alone.target, market.target) == (NO_MARKET, TARGET_RECIPE)
        assert not NO_MARKET.market
        assert alone.variant == market.variant == SPEC_RECIPE
        assert alone.config == market.config
        assert alone.config.step_months == alone.config.test_months == 6
    assert target_experiments.PRIMARY == {"return": "mae", "interval": "pinball"}


@pytest.fixture
def small(run_panel: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The market experiments on the one-year panel with monthly folds."""
    monkeypatch.setattr(experiments, "COARSE_MONTHS", 1)
    monkeypatch.setattr(experiments, "WINDOW_FROM", date(2016, 1, 1))
    monkeypatch.setattr(lgbm_runs, "MAX_ROUNDS", 80)
    return run_panel


def calendar() -> TradingDays:
    days = weekdays()
    return TradingDays(days, days[0], days[-1])


def run(
    name: str, directory: Path, seeds: tuple[int, ...] = (0,)
) -> tuple[TargetExperiment, dict[str, list[Fold]]]:
    chosen = target_experiments.target_experiments(CONFIG)[name]
    folds = compared_folds(chosen, calendar(), weekdays()[-1])
    target_experiments.train_arms(chosen, "interval-exp", folds, directory, seeds)
    return chosen, folds


@pytest.mark.django_db
def test_train_arms_fit_each_arm_and_seed_under_its_own_key_with_its_target(
    small: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str, int, Any]] = []
    original = lgbm_runs.run_folds

    def spy(scope: str, folds: Any, run_key: str, *args: Any, **kwargs: Any) -> list[int]:
        calls.append((scope, run_key, args[1], kwargs["target"]))
        return original(scope, folds, run_key, *args, **kwargs)

    monkeypatch.setattr(target_experiments.lgbm_runs, "run_folds", spy)

    _, folds = run("market-interval", small, (0, 3))

    assert calls == [
        ("interval", "interval-exp-nomarket", 0, NO_MARKET),
        ("interval", "interval-exp-nomarket-s3", 3, NO_MARKET),
        ("interval", "interval-exp-market", 0, TARGET_RECIPE),
        ("interval", "interval-exp-market-s3", 3, TARGET_RECIPE),
    ]
    fold = folds["nomarket"][0]
    alone = ModelRun.objects.filter(run_key="interval-exp-nomarket")
    assert {tuple(fitted.feature_list) for fitted in alone} == {
        ("tech_ret_1d", "tech_ret_60d", "tech_vol_20d")
    }
    added = ModelRun.objects.filter(run_key="interval-exp-market")
    assert {"mkt_taiex_ret_5d" in fitted.feature_list for fitted in added} == {True}
    notes = fold_notes("interval-exp-market", "lightgbm", fold.index)
    assert notes["recipe"] != fold_notes("interval-exp-nomarket", "lightgbm", fold.index)["recipe"]


def shared_frames(
    chosen: TargetExperiment, run_key: str, fold: Fold, seed: int
) -> list[pd.DataFrame]:
    """Both arms' tradable cached rows of a fold with labels, built without the module."""
    scope = chosen.scope
    labels = load_labels(
        fold.test_start, fold.eval_end, target_eval.HORIZON[scope], target_eval.LABEL_FIELDS[scope]
    )
    frames = []
    for arm in chosen.arms:
        cached = load_fold_predictions(arm_key(run_key, arm, seed), "lightgbm", fold)
        frames.append(tradable_rows(cached.merge(labels, on=KEYS, how="left")).set_index(KEYS))
    common = frames[0].index.intersection(frames[1].index)
    return [frame.loc[common].reset_index() for frame in frames]


def picked_daily(daily: pd.DataFrame, arm: str, seed: int, fold: int, metric: str) -> pd.Series:
    rows = daily[
        (daily["arm"] == arm)
        & (daily["seed"] == seed)
        & (daily["fold"] == fold)
        & (daily["metric"] == metric)
    ]
    return rows.set_index("date")["value"]


def interval_daily(metric: str, rows: pd.DataFrame) -> pd.Series:
    """An interval metric's daily series, computed straight from the forecast metrics."""
    up = rows.assign(fwd_up_5pct=rows["fwd_up_5pct"].astype(int))
    if metric == "pinball":
        return daily_pinball(rows, "fwd_ret_open", QUANTILES)
    if metric == "coverage":
        return daily_coverage(rows, "fwd_ret_open", QUANTILES[0.1], QUANTILES[0.9])
    if metric == "auc":
        return daily_auc(up, "fwd_up_5pct", "lightgbm_prob_h20")
    if metric == "brier":
        return daily_brier(up, "fwd_up_5pct", "lightgbm_prob_h20")
    if metric == "vol_mae":
        return daily_absolute_error(rows, "fwd_vol_20d", "lightgbm_vol_h20")
    return daily_qlike(rows, "fwd_vol_20d", "lightgbm_vol_h20")


@pytest.mark.django_db
def test_daily_metrics_score_both_interval_arms_on_their_shared_tradable_rows(
    small: Path,
) -> None:
    chosen, folds = run("market-interval", small, (0, 3))
    fold = folds["market"][0]
    key = arm_key("interval-exp", chosen.arms[1], 0)
    cached = load_fold_predictions(key, "lightgbm", fold)
    gap = cached.index[::7]
    holed = cached.assign(
        lightgbm_q90_h20=cached["lightgbm_q90_h20"].mask(cached.index.isin(gap)),
        lightgbm_prob_h20=cached["lightgbm_prob_h20"].mask(cached.index.isin(gap)),
        lightgbm_vol_h20=cached["lightgbm_vol_h20"].mask(cached.index.isin(gap), 0.0),
    )
    save_fold_predictions(
        key, "lightgbm", fold, holed, notes=fold_notes(key, "lightgbm", fold.index)
    )

    daily = target_experiments.daily_metrics(chosen, "interval-exp", folds, (0, 3))

    assert list(daily.columns) == target_experiments.DAILY_COLUMNS
    assert set(daily["metric"]) == {"pinball", "coverage", "vol_mae", "qlike", "auc", "brier"}
    for seed in (0, 3):
        alone, market = shared_frames(chosen, "interval-exp", fold, seed)
        for column, metric in (
            ("lightgbm_q90_h20", "pinball"),
            ("lightgbm_q90_h20", "coverage"),
            ("lightgbm_prob_h20", "auc"),
            ("lightgbm_prob_h20", "brier"),
            ("lightgbm_vol_h20", "vol_mae"),
            ("lightgbm_vol_h20", "qlike"),
        ):
            kept = market[column].notna() & (market[column] > 0 if metric == "qlike" else True)
            for arm, frame in (("nomarket", alone), ("market", market)):
                np.testing.assert_allclose(
                    picked_daily(daily, arm, seed, fold.index, metric).to_numpy(),
                    interval_daily(metric, frame[kept.to_numpy()]).to_numpy(),
                    err_msg=f"{arm} seed {seed} {metric}",
                )
    first, third = (shared_frames(chosen, "interval-exp", fold, seed)[0] for seed in (0, 3))
    assert not np.allclose(first["lightgbm_q50_h20"], third["lightgbm_q50_h20"])
    alone, market = shared_frames(chosen, "interval-exp", fold, 0)
    assert len(alone) > len(alone[market["lightgbm_q90_h20"].notna().to_numpy()])


@pytest.mark.django_db
def test_daily_metrics_score_the_return_arms_by_mae_and_direction(small: Path) -> None:
    chosen = target_experiments.target_experiments(CONFIG)["market-return"]
    folds = compared_folds(chosen, calendar(), weekdays()[-1])
    target_experiments.train_arms(chosen, "return-exp", folds, small, (0,))

    daily = target_experiments.daily_metrics(chosen, "return-exp", folds, (0,))

    assert set(daily["metric"]) == {"mae", "direction"}
    for fold in folds["nomarket"]:
        alone, market = shared_frames(chosen, "return-exp", fold, 0)
        for arm, frame in (("nomarket", alone), ("market", market)):
            np.testing.assert_allclose(
                picked_daily(daily, arm, 0, fold.index, "mae").to_numpy(),
                daily_absolute_error(frame, "fwd_ret_open", "lightgbm_ret_h5").to_numpy(),
            )
            np.testing.assert_allclose(
                picked_daily(daily, arm, 0, fold.index, "direction").to_numpy(),
                daily_direction_hits(frame, "fwd_ret_open", "lightgbm_ret_h5").to_numpy(),
            )


def interval_experiment() -> TargetExperiment:
    return target_experiments.target_experiments(CONFIG)["market-interval"]


def fake_folds() -> list[Fold]:
    days = weekdays()
    return [
        Fold(
            index=index,
            train_start=days[0],
            fit_end=days[7],
            valid_start=days[8],
            train_end=days[10],
            test_start=days[20 + 10 * index],
            test_end=days[29 + 10 * index],
            eval_end=days[29 + 10 * index],
            tuning=False,
            holdout=False,
        )
        for index in range(4)
    ]


def fabricated(values: dict[tuple[str, str], list[float]], seeds: tuple[int, ...] = (0,)) -> Any:
    """Daily rows: each (arm, metric) gets one value per fold, the same on every fold day."""
    rows = []
    for fold in fake_folds():
        days = [day for day in weekdays() if fold.test_start <= day <= fold.eval_end]
        for (arm, metric), per_fold in values.items():
            for seed in seeds:
                for day in days:
                    rows.append([arm, seed, fold.index, metric, day, per_fold[fold.index] + seed])
    return pd.DataFrame(rows, columns=target_experiments.DAILY_COLUMNS)


def test_comparisons_pair_the_market_arm_with_the_stock_only_arm_over_folds() -> None:
    daily = fabricated(
        {
            ("nomarket", "pinball"): [0.10, 0.12, 0.11, 0.13],
            ("market", "pinball"): [0.09, 0.115, 0.10, 0.125],
        },
        seeds=(0, 1),
    )

    compared = target_experiments.comparisons(interval_experiment(), daily, fake_folds())

    pinball = compared["pinball"]
    model = mean_by_fold(
        daily[(daily["arm"] == "market")].groupby("date")["value"].mean(), fake_folds()
    )["mean"]
    baseline = mean_by_fold(
        daily[(daily["arm"] == "nomarket")].groupby("date")["value"].mean(), fake_folds()
    )["mean"]
    assert pinball.model == pytest.approx(np.mean([0.09, 0.115, 0.10, 0.125]) + 0.5)
    assert pinball.baseline == pytest.approx(np.mean([0.10, 0.12, 0.11, 0.13]) + 0.5)
    assert pinball.nw_t == pytest.approx(paired_nw_t(model, baseline, 1))
    assert list(pinball.by_fold.index) == [0, 1, 2, 3]


NOMARKET_PINBALL = [0.10, 0.12, 0.11, 0.13]
CLEAR_PINBALL = [0.08, 0.101, 0.089, 0.112]
NOISY_PINBALL = [0.09, 0.125, 0.10, 0.137]


@pytest.mark.parametrize(
    ("market", "chosen"),
    [
        (CLEAR_PINBALL, "market"),
        (NOISY_PINBALL, "nomarket"),
        (NOMARKET_PINBALL, "nomarket"),
        ([0.11, 0.135, 0.125, 0.14], "nomarket"),
        ([math.nan] * 4, "nomarket"),
    ],
)
def test_the_market_arm_needs_a_lower_primary_metric_with_a_fold_gap_t_below_minus_two(
    market: list[float], chosen: str
) -> None:
    daily = fabricated(
        {
            ("nomarket", "pinball"): NOMARKET_PINBALL,
            ("market", "pinball"): market,
            ("nomarket", "auc"): [0.5] * 4,
            ("market", "auc"): [0.6] * 4,
        }
    )
    experiment = interval_experiment()

    compared = target_experiments.comparisons(experiment, daily, fake_folds())

    assert target_experiments.verdict(experiment, compared) == chosen


def report(values: dict[tuple[str, str], list[float]], seeds: tuple[int, ...] = (0,)) -> str:
    daily = fabricated(values, seeds)
    return target_experiments.experiment_report(
        interval_experiment(), "interval-exp", daily, fake_folds(), "暫定特徵"
    )


INTERVAL_VALUES: dict[tuple[str, str], list[float]] = {
    ("nomarket", "pinball"): NOMARKET_PINBALL,
    ("market", "pinball"): NOISY_PINBALL,
    ("nomarket", "coverage"): [0.80] * 4,
    ("market", "coverage"): [0.78] * 4,
    ("nomarket", "vol_mae"): [0.05] * 4,
    ("market", "vol_mae"): [0.06] * 4,
    ("nomarket", "qlike"): [0.30] * 4,
    ("market", "qlike"): [0.29] * 4,
    ("nomarket", "auc"): [0.55] * 4,
    ("market", "auc"): [0.56] * 4,
    ("nomarket", "brier"): [0.10] * 4,
    ("market", "brier"): [0.09] * 4,
}


def test_the_report_keeps_the_stock_only_arm_on_a_noisy_gap_and_one_seed() -> None:
    text = report(INTERVAL_VALUES)
    compared = target_experiments.comparisons(
        interval_experiment(), fabricated(INTERVAL_VALUES), fake_folds()
    )

    assert text.startswith("# 實驗：market-interval（interval-exp）\n\n> 暫定特徵\n")
    assert "- nomarket：只用篩選後的個股特徵" in text
    assert "- market：個股特徵加上市場特徵（SPEC 配方）" in text
    assert (
        "| 組 | pinball ↓ | 覆蓋率（目標 80%） | 波動率 MAE ↓ | QLIKE ↓ | AUC ↑ | Brier ↓ |" in text
    )
    assert "| nomarket | 0.1150 | 0.8000 | 0.0500 | 0.3000 | 0.5500 | 0.1000 |" in text
    assert "| market | 0.1130 | 0.7800 | 0.0600 | 0.2900 | 0.5600 | 0.0900 |" in text
    assert (
        "| 差（market − nomarket） | -0.0020 | -0.0200 | +0.0100 | -0.0100 | +0.0100 | -0.0100 |"
        in text
    )
    assert "**判定（暫定）：採用 nomarket**" in text
    nw_t = compared["pinball"].nw_t
    assert -2 < nw_t < 0
    assert (
        f"- market 的 pinball 較低，但逐摺差 NW t {nw_t:+.2f} 沒有低於 −2：照規則留 nomarket"
        in text
    )
    assert "- 只跑了 1 個 seed：要用 3 個 seed（`--seeds 0,3,6`）重跑再定案" in text
    assert "- 參考欄（market 對 nomarket）：較好 QLIKE、AUC、Brier；較差 覆蓋率、波動率 MAE" in text
    assert "| 摺 | nomarket | market |" in text
    assert "| 1 | 0.1200 | 0.1250 |" in text
    assert "## 各 seed" not in text


def test_the_report_settles_on_three_seeds_and_adopts_a_clear_gap() -> None:
    three = report(INTERVAL_VALUES, seeds=(0, 3, 6))
    clear = report({**INTERVAL_VALUES, ("market", "pinball"): CLEAR_PINBALL})
    settled = report({**INTERVAL_VALUES, ("market", "pinball"): CLEAR_PINBALL}, seeds=(0, 3, 6))
    losing = report({**INTERVAL_VALUES, ("market", "pinball"): [0.11, 0.135, 0.125, 0.14]})

    assert "**判定：採用 nomarket**" in three
    assert "沒有低於 −2：照規則留 nomarket" in three
    assert "只跑了" not in three
    assert "## 各 seed（pinball）" in three
    assert "| market | 0.1130 | 3.1130 | 6.1130 |" in three
    assert "**判定（暫定）：採用 market**" in clear
    assert "**判定：採用 market**" in settled
    assert "沒有低於 −2" not in clear + settled + losing
    assert "**判定（暫定）：採用 nomarket**" in losing


def test_the_return_report_marks_the_direction_metric_as_higher_is_better() -> None:
    experiment = target_experiments.target_experiments(CONFIG)["market-return"]
    daily = fabricated(
        {
            ("nomarket", "mae"): NOMARKET_PINBALL,
            ("market", "mae"): CLEAR_PINBALL,
            ("nomarket", "direction"): [0.52] * 4,
            ("market", "direction"): [0.51] * 4,
        }
    )

    text = target_experiments.experiment_report(experiment, "return-exp", daily, fake_folds())

    assert "| 組 | MAE ↓ | 方向準確率 ↑ |" in text
    assert "- 參考欄（market 對 nomarket）：較好 無；較差 方向準確率" in text
    assert "**判定（暫定）：採用 market**" in text


@pytest.fixture
def command(small: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(target_experiment, "DEFAULT_CONFIG", CONFIG)
    DailyPrice.objects.create(
        stock_id="1101", date=weekdays()[-1], close=10, volume=1000, turnover=10_000
    )
    return small


@pytest.mark.django_db
def test_target_experiment_trains_the_arms_and_writes_the_report(
    command: Path, tmp_path: Path
) -> None:
    output = tmp_path / "reports" / "market.md"
    out = StringIO()

    call_command(
        "target_experiment",
        name="market-return",
        run_key="return-exp",
        output=str(output),
        note="暫定",
        seeds="0,1",
        stdout=out,
    )

    chosen = target_experiments.target_experiments(CONFIG)["market-return"]
    folds = compared_folds(chosen, calendar(), weekdays()[-1])
    daily = target_experiments.daily_metrics(chosen, "return-exp", folds, (0, 1))
    expected = target_experiments.experiment_report(
        chosen, "return-exp", daily, folds["nomarket"], "暫定"
    )
    assert output.read_text() == expected
    assert "return-exp-market-s1: ran folds" in out.getvalue()
    assert f"wrote {output}" in out.getvalue()
    assert "shuffled OOS rank IC" in out.getvalue()


@pytest.mark.django_db
def test_target_experiment_refuses_bad_keys_and_seeds_before_training(command: Path) -> None:
    with pytest.raises(CommandError, match="start with return-"):
        call_command("target_experiment", name="market-return", run_key="rank-x", stdout=StringIO())
    with pytest.raises(CommandError, match="at most"):
        call_command(
            "target_experiment",
            name="market-interval",
            run_key="interval-" + "x" * 20,
            seeds="0,3",
            stdout=StringIO(),
        )
    with pytest.raises(CommandError, match="--seeds"):
        call_command(
            "target_experiment",
            name="market-return",
            run_key="return-exp",
            seeds="0,0",
            stdout=StringIO(),
        )
    assert not ModelRun.objects.filter(model_type="lightgbm").exists()


@pytest.mark.django_db
def test_target_experiment_refuses_to_run_once_its_scope_holdout_is_claimed(command: Path) -> None:
    HoldoutRun.objects.create(
        scope=HoldoutScope.INTERVAL, attempt=1, run_key="interval-final", last_date=weekdays()[-1]
    )

    with pytest.raises(CommandError, match=r"interval holdout .* frozen"):
        call_command(
            "target_experiment", name="market-interval", run_key="interval-exp", stdout=StringIO()
        )
    assert not ModelRun.objects.filter(model_type="lightgbm").exists()


@pytest.mark.django_db
def test_target_experiment_stops_when_the_shuffle_check_fails(
    command: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiments, "shuffle_check", lambda *args: 0.5)

    with pytest.raises(CommandError, match="shuffle check on fold"):
        call_command(
            "target_experiment", name="market-return", run_key="return-exp", stdout=StringIO()
        )
    assert not ModelRun.objects.filter(model_type="lightgbm").exists()


@pytest.mark.django_db
def test_target_experiment_needs_a_screening_before_the_shuffle_check(
    command: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shuffled: list[Any] = []
    monkeypatch.setattr(experiments, "shuffle_check", lambda *args: shuffled.append(args))
    ModelRun.objects.filter(model_type=SCREENING_MODEL).delete()

    with pytest.raises(CommandError, match="screened features"):
        call_command(
            "target_experiment", name="market-return", run_key="return-exp", stdout=StringIO()
        )
    assert shuffled == []


def test_arms_default_to_the_spec_target_design() -> None:
    assert Arm("base", CONFIG, SPEC_RECIPE).target == TARGET_RECIPE
