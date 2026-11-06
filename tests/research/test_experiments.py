from dataclasses import replace
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
from apps.research import experiments, lgbm_runs
from apps.research.experiments import (
    MIN_ROUNDS,
    SAMPLE_EVERY,
    Arm,
    Experiment,
    compared_folds,
    daily_ics,
    experiment_report,
    stopping_rounds,
    summary,
    verdict,
)
from apps.research.management.commands import rank_experiment
from apps.research.metrics.ic import daily_rank_ic
from apps.research.models import HoldoutRun, HoldoutScope, LabelDaily, ModelRun
from apps.research.prediction_cache import load_predictions
from apps.research.rank_model import LAMBDARANK, SPEC_RECIPE, Variant
from apps.research.runs import SCREENING_MODEL
from apps.research.walkforward import WalkForwardConfig, generate_folds
from tests.research.conftest import weekday_calendar
from tests.research.run_panels import CONFIG, weekdays


def table(means: dict[str, float]) -> pd.DataFrame:
    rows = {name: {"h5": value, "h20": value, "mean": value} for name, value in means.items()}
    return pd.DataFrame.from_dict(rows, orient="index")


def experiment(rule: str, *names: str) -> Experiment:
    return Experiment("demo", rule, tuple(Arm(name, CONFIG, SPEC_RECIPE) for name in names))


def test_the_spec_experiments_compare_these_arms_on_six_month_folds() -> None:
    chosen = experiments.experiments(WalkForwardConfig())

    window, sampling, objective = chosen["window"], chosen["sampling"], chosen["objective"]
    assert [arm.name for arm in window.arms] == ["base", "w60", "expanding"]
    base, long, expanding = window.arms
    assert (base.config.step_months, base.config.test_months) == (6, 6)
    assert (base.config.window, base.config.train_months) == ("rolling", 36)
    assert (long.config.window, long.config.train_months) == ("rolling", 60)
    assert expanding.config.window == "expanding"
    assert expanding.variant == Variant(half_life=24.0)
    assert base.variant == long.variant == SPEC_RECIPE
    assert [(arm.name, arm.variant) for arm in sampling.arms] == [
        ("base", SPEC_RECIPE),
        ("every5", Variant(every=SAMPLE_EVERY)),
    ]
    assert SAMPLE_EVERY == 5
    assert [(arm.name, arm.variant) for arm in objective.arms] == [
        ("base", SPEC_RECIPE),
        ("lambdarank", Variant(objective="lambdarank")),
    ]
    stopping = chosen["stopping"]
    assert [(arm.name, arm.variant, arm.tuned) for arm in stopping.arms] == [
        ("base", SPEC_RECIPE, False),
        ("floor", Variant(min_rounds=MIN_ROUNDS), False),
        ("valid12", SPEC_RECIPE, False),
        ("tuned", SPEC_RECIPE, True),
    ]
    assert stopping.arms[3].config == base.config
    assert MIN_ROUNDS == 200
    assert stopping.arms[2].config == replace(base.config, valid_months=12)
    assert {name: item.rule for name, item in chosen.items()} == {
        "window": "best",
        "sampling": "not_worse",
        "objective": "better",
        "stopping": "best",
    }
    shared = (sampling.arms[0], objective.arms[0], stopping.arms[0], stopping.arms[1], base)
    assert len({arm.config for arm in shared}) == 1


def test_arm_keys_keep_seed_zero_and_suffix_the_other_seeds() -> None:
    arm = Arm("every5", CONFIG, Variant(every=5))

    assert experiments.arm_key("rank-exp1", arm) == "rank-exp1-every5"
    assert experiments.arm_key("rank-exp1", arm, 0) == "rank-exp1-every5"
    assert experiments.arm_key("rank-exp1", arm, 2) == "rank-exp1-every5-s2"


def test_train_arms_fit_every_seed_of_every_arm_under_its_own_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, Variant]] = []

    def fake(
        scope: str, folds: Any, run_key: str, directory: Path, seed: int, **kwargs: Any
    ) -> Any:
        calls.append((run_key, seed, kwargs["variant"]))
        return [seed]

    monkeypatch.setattr(experiments.lgbm_runs, "run_folds", fake)
    chosen = experiments.experiments(CONFIG)["sampling"]

    ran = experiments.train_arms(chosen, "rank-exp", {"base": [], "every5": []}, Path(), (0, 2))

    assert calls == [
        ("rank-exp-base", 0, SPEC_RECIPE),
        ("rank-exp-base-s2", 2, SPEC_RECIPE),
        ("rank-exp-every5", 0, Variant(every=5)),
        ("rank-exp-every5-s2", 2, Variant(every=5)),
    ]
    assert ran == {key: [seed] for key, seed, _ in calls}


def test_compared_folds_share_research_test_periods_from_2015_on() -> None:
    calendar = weekday_calendar(date(2010, 1, 1), date(2026, 12, 31))

    folds = compared_folds(
        experiments.experiments(WalkForwardConfig())["window"], calendar, date(2026, 11, 6)
    )

    periods = {
        name: [(fold.test_start, fold.eval_end) for fold in chosen]
        for name, chosen in folds.items()
    }
    assert periods["base"] == periods["w60"] == periods["expanding"]
    assert all(fold.research for chosen in folds.values() for fold in chosen)
    assert min(start for start, _ in periods["base"]) >= date(2015, 1, 1)
    assert max(end for _, end in periods["base"]) < date(2025, 1, 1)
    assert len(periods["base"]) == 18
    assert (
        folds["expanding"][0].train_start == calendar.between(date(2010, 1, 1), date(2010, 1, 8))[0]
    )


def test_verdict_rules_pick_the_best_the_not_worse_and_the_strictly_better() -> None:
    best = experiment("best", "base", "w60", "expanding")
    assert verdict(best, table({"base": 0.05, "w60": 0.06, "expanding": 0.055})) == "w60"
    assert verdict(best, table({"base": 0.05, "w60": 0.05, "expanding": float("nan")})) == "base"
    sampling = experiment("not_worse", "base", "every5")
    assert verdict(sampling, table({"base": 0.05, "every5": 0.05})) == "every5"
    assert verdict(sampling, table({"base": 0.05, "every5": 0.0499})) == "base"
    better = experiment("better", "base", "lambdarank")
    assert verdict(better, table({"base": 0.05, "lambdarank": 0.05})) == "base"
    assert verdict(better, table({"base": 0.05, "lambdarank": 0.0501})) == "lambdarank"
    assert verdict(better, table({"base": 0.05, "lambdarank": float("nan")})) == "base"


@pytest.fixture
def small(run_panel: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The SPEC experiments on the one-year panel: monthly folds and a 7-month long window."""
    monkeypatch.setattr(experiments, "COARSE_MONTHS", 1)
    monkeypatch.setattr(experiments, "LONG_MONTHS", 7)
    monkeypatch.setattr(experiments, "WINDOW_FROM", date(2016, 1, 1))
    monkeypatch.setattr(lgbm_runs, "MAX_ROUNDS", 80)
    return run_panel


def calendar() -> TradingDays:
    days = weekdays()
    return TradingDays(days, days[0], days[-1])


def run(
    name: str, directory: Path, seeds: tuple[int, ...] = (0,)
) -> tuple[Experiment, dict[str, Any]]:
    chosen = experiments.experiments(CONFIG)[name]
    folds = compared_folds(chosen, calendar(), weekdays()[-1])
    experiments.train_arms(chosen, "rank-exp", folds, directory, seeds)
    return chosen, folds


@pytest.mark.django_db
def test_train_arms_fit_each_arm_under_its_own_key_and_share_the_base(
    small: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Variant, list[int]]] = []
    original = lgbm_runs.run_folds

    def spy(scope: str, folds: Any, run_key: str, *args: Any, **kwargs: Any) -> list[int]:
        ran = original(scope, folds, run_key, *args, **kwargs)
        calls.append((run_key, kwargs["variant"], ran))
        return ran

    monkeypatch.setattr(experiments.lgbm_runs, "run_folds", spy)

    _, folds = run("sampling", small)
    run("objective", small)

    assert [(key, variant) for key, variant, _ in calls] == [
        ("rank-exp-base", SPEC_RECIPE),
        ("rank-exp-every5", Variant(every=5)),
        ("rank-exp-base", SPEC_RECIPE),
        ("rank-exp-lambdarank", Variant(objective="lambdarank")),
    ]
    assert calls[2][2] == []
    assert calls[0][2] == [fold.index for fold in folds["base"]]
    graded = ModelRun.objects.filter(run_key="rank-exp-lambdarank")
    assert {run.params["variant"]["objective"] for run in graded} == {"lambdarank"}
    assert {run.params["lightgbm"]["objective"] for run in graded} == {"lambdarank"}
    for fitted in graded:
        assert fitted.params["lightgbm"].items() >= LAMBDARANK.items()
    assert {run.params["label"] for run in graded} == {"fwd_rank_level"}
    assert {run.params["graded_from"] for run in graded} == {"fwd_rank_pct"}
    plain = ModelRun.objects.filter(run_key="rank-exp-base")
    assert {run.params["lightgbm"]["objective"] for run in plain} == {"regression"}
    assert {run.params["label"] for run in plain} == {"fwd_rank_pct"}


@pytest.mark.django_db
def test_daily_ics_pool_each_arm_over_its_tradable_test_rows(
    small: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chosen, folds = run("sampling", small, (0, 1))
    reads: list[tuple[date, date, int]] = []
    load = experiments.load_labels

    def counted(start: date, end: date, horizon: int, *args: Any) -> pd.DataFrame:
        reads.append((start, end, horizon))
        return load(start, end, horizon, *args)

    monkeypatch.setattr(experiments, "load_labels", counted)

    daily = daily_ics(chosen, "rank-exp", folds, (0, 1))

    periods = [(fold.test_start, fold.eval_end) for fold in folds["base"]]
    assert sorted(reads) == sorted((*period, horizon) for horizon in (5, 20) for period in periods)
    assert set(daily["seed"]) == {0, 1}

    for arm in chosen.arms:
        for seed, horizon in ((0, 5), (0, 20), (1, 5), (1, 20)):
            key = experiments.arm_key("rank-exp", arm, seed)
            cached = load_predictions(key, "lightgbm", folds[arm.name])
            labels = pd.DataFrame(list(LabelDaily.objects.filter(horizon=horizon).values()))
            joined = cached.merge(labels, on=["stock_id", "date"])
            ends = joined["fold"].map({fold.index: fold.eval_end for fold in folds[arm.name]})
            rows = joined[
                (joined["date"] <= ends) & joined["in_universe"] & ~joined["untradable_t1"]
            ]
            expected = daily_rank_ic(rows, f"lightgbm_h{horizon}")
            got = daily[
                (daily["arm"] == arm.name) & (daily["seed"] == seed) & (daily["horizon"] == horizon)
            ]
            np.testing.assert_allclose(
                got.set_index("date")["ic"].sort_index(), expected.sort_index()
            )
    pooled = summary(chosen, daily)
    assert list(pooled.index) == ["base", "every5"]
    base = pooled.loc["base"].astype(float)
    assert base["mean"] == pytest.approx((base["h5"] + base["h20"]) / 2)
    assert base["seeds"] == 2
    rounds = stopping_rounds(chosen, "rank-exp", folds, (0, 1))
    fitted = ModelRun.objects.get(
        run_id=lgbm_runs.run_id("rank-exp-every5-s1", "rank", 20, folds["every5"][0])
    )
    picked = rounds[
        (rounds["arm"] == "every5")
        & (rounds["seed"] == 1)
        & (rounds["horizon"] == 20)
        & (rounds["fold"] == folds["every5"][0].index)
    ]
    assert picked["rounds"].tolist() == [fitted.params["rounds"]]
    assert picked["best"].tolist() == [fitted.params["best_round"]]
    assert len(rounds) == 2 * 2 * 2 * len(periods)


def seeded_daily(values: dict[tuple[str, int], float]) -> pd.DataFrame:
    """One day and fold per arm and seed, with the same IC for both horizons."""
    rows = [
        {"arm": arm, "seed": seed, "fold": 3, "horizon": horizon, "date": date(2016, 10, 3)}
        | {"ic": value}
        for (arm, seed), value in values.items()
        for horizon in (5, 20)
    ]
    return pd.DataFrame(rows)


def test_summary_averages_the_seeds_and_reports_their_spread() -> None:
    chosen = experiment("best", "base", "w60")
    daily = seeded_daily(
        {
            ("base", 0): 0.10,
            ("base", 1): 0.12,
            ("base", 2): 0.14,
            ("w60", 0): 0.11,
            ("w60", 1): 0.11,
            ("w60", 2): 0.11,
        }
    )

    table = summary(chosen, daily)

    assert table.loc["base", "mean"] == pytest.approx(0.12)
    assert table.loc["base", "sd"] == pytest.approx(0.02)
    assert table.loc["w60", "sd"] == pytest.approx(0.0)
    assert table.loc["base", "seeds"] == 3
    assert verdict(chosen, table) == "base"
    one = summary(chosen, seeded_daily({("base", 0): 0.1, ("w60", 0): 0.2}))
    assert np.isnan(float(one["sd"].astype(float)["base"]))


def test_experiment_report_shows_the_verdict_small_gaps_and_folds() -> None:
    chosen = experiment("not_worse", "base", "every5")
    daily = pd.DataFrame(
        {
            "seed": [0] * 8,
            "arm": ["base", "base", "every5", "every5"] * 2,
            "fold": [3, 4] * 4,
            "horizon": [5] * 4 + [20] * 4,
            "date": [date(2016, 10, 3), date(2016, 11, 1)] * 4,
            "ic": [0.05, 0.07, 0.054, 0.07, 0.04, 0.06, 0.04, 0.06],
        }
    )

    text = experiment_report(chosen, "rank-exp", daily, note="暫定特徵")

    assert "暫定特徵" in text
    assert (
        "- every5：rolling 6 個月，驗證段 2 個月，每 1 日一列，半衰期 —，objective regression"
        in text
    )
    assert "| every5 | 0.0620 | 0.0500 | 0.0560 | +0.0010 | — |" in text
    assert "**判定（暫定）：採用 every5**" in text
    assert "差距小於 0.02，只跑了 1 個 seed" in text
    assert "`--seeds 0,3,6`" in text
    assert "## 各 seed" not in text
    assert "## 停止輪數" not in text
    assert "| 3 | 0.0500 | 0.0400 | 0.0540 | 0.0400 |" in text


def test_experiment_report_judges_close_arms_on_three_seeds_and_lists_rounds() -> None:
    chosen = experiment("best", "base", "w60")
    daily = seeded_daily(
        {
            ("base", 0): 0.10,
            ("base", 1): 0.12,
            ("base", 2): 0.14,
            ("w60", 0): 0.11,
            ("w60", 1): 0.11,
            ("w60", 2): 0.11,
        }
    )
    rounds = pd.DataFrame(
        {
            "arm": ["base"] * 3 + ["w60"] * 3,
            "seed": [0, 1, 2] * 2,
            "horizon": [20] * 6,
            "fold": [3] * 6,
            "best": [21, 446, 1681, 30, 310, 320],
            "rounds": [21, 446, 1681, 200, 310, 320],
        }
    )

    text = experiment_report(chosen, "rank-exp", daily, rounds=rounds)

    assert "| base | 0.1200 | 0.1200 | 0.1200 | — | 0.0200 |" in text
    assert "**判定：採用 base**" in text
    assert "差距小於 0.02，判定用 3 個 seed 的平均" in text
    assert "| base | 0.1000 | 0.1200 | 0.1400 |" in text
    assert "| base | 20 日 | 21 | 446 | 1681 | 1 | 446 |" in text
    assert "| w60 | 20 日 | 30 | 310 | 320 | 1 | 310 |" in text
    assert "（摺 × seed）" in text


@pytest.fixture
def command(small: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(rank_experiment, "DEFAULT_CONFIG", CONFIG)
    DailyPrice.objects.create(
        stock_id="1101", date=weekdays()[-1], close=10, volume=1000, turnover=10_000
    )
    return small


@pytest.mark.django_db
def test_rank_experiment_trains_the_arms_and_writes_the_report(
    command: Path, tmp_path: Path
) -> None:
    output = tmp_path / "reports" / "sampling.md"
    out = StringIO()

    call_command(
        "rank_experiment",
        name="sampling",
        run_key="rank-exp",
        output=str(output),
        note="暫定",
        seeds="0,1",
        stdout=out,
    )

    chosen, folds = run("sampling", command, (0, 1))
    daily = daily_ics(chosen, "rank-exp", folds, (0, 1))
    rounds = stopping_rounds(chosen, "rank-exp", folds, (0, 1))
    expected = experiment_report(chosen, "rank-exp", daily, "暫定", rounds)
    assert output.read_text() == expected
    assert f"wrote {output}" in out.getvalue()


@pytest.mark.django_db
def test_rank_experiment_refuses_bad_keys_before_training(command: Path) -> None:
    with pytest.raises(CommandError, match="start with rank-"):
        call_command("rank_experiment", name="window", run_key="interval-x", stdout=StringIO())
    with pytest.raises(CommandError, match="at most"):
        call_command(
            "rank_experiment", name="window", run_key="rank-" + "x" * 30, stdout=StringIO()
        )
    assert not ModelRun.objects.filter(model_type="lightgbm").exists()


def test_compared_folds_drop_periods_an_arm_lacks_and_those_before_2015() -> None:
    calendar = weekday_calendar(date(2005, 1, 1), date(2026, 12, 31))
    late = WalkForwardConfig(data_start=date(2011, 1, 1))
    early = WalkForwardConfig(
        data_start=date(2005, 1, 1), first_test=date(2012, 1, 1), tuning_end=date(2012, 12, 31)
    )

    shortened = compared_folds(experiments.experiments(late)["window"], calendar, date(2026, 11, 6))
    started = compared_folds(experiments.experiments(early)["window"], calendar, date(2026, 11, 6))

    for folds, first in ((shortened, date(2016, 7, 1)), (started, date(2015, 1, 1))):
        periods = {name: [fold.test_start for fold in chosen] for name, chosen in folds.items()}
        assert periods["base"] == periods["w60"] == periods["expanding"]
        assert periods["base"][0] == first


def test_an_arm_missing_a_horizon_has_no_mean_and_never_wins() -> None:
    chosen = experiment("not_worse", "base", "every5")
    daily = pd.DataFrame(
        {
            "arm": ["base", "base", "every5"],
            "seed": [0, 0, 0],
            "fold": [3, 3, 3],
            "horizon": [5, 20, 5],
            "date": [date(2016, 10, 3)] * 3,
            "ic": [0.05, 0.04, 0.09],
        }
    )

    table = summary(chosen, daily)

    assert np.isnan(float(table["mean"]["every5"]))
    assert verdict(chosen, table) == "base"


@pytest.mark.django_db
def test_rank_experiment_stops_when_the_shuffle_check_fails(
    command: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rank_experiment, "shuffle_check", lambda *args: 0.5)

    with pytest.raises(CommandError, match="shuffle check on fold"):
        call_command("rank_experiment", name="sampling", run_key="rank-exp", stdout=StringIO())
    assert not ModelRun.objects.filter(model_type="lightgbm").exists()


@pytest.mark.django_db
def test_rank_experiment_refuses_to_run_once_the_rank_holdout_is_claimed(command: Path) -> None:
    HoldoutRun.objects.create(
        scope=HoldoutScope.RANK, attempt=1, run_key="rank-final", last_date=weekdays()[-1]
    )

    with pytest.raises(CommandError, match=r"rank holdout .* frozen"):
        call_command("rank_experiment", name="sampling", run_key="rank-exp", stdout=StringIO())
    assert not ModelRun.objects.filter(model_type="lightgbm").exists()


@pytest.mark.django_db
def test_rank_experiment_needs_a_screening_before_the_shuffle_check(
    command: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shuffled: list[Any] = []
    monkeypatch.setattr(rank_experiment, "shuffle_check", lambda *args: shuffled.append(args))
    ModelRun.objects.filter(model_type=SCREENING_MODEL).delete()

    with pytest.raises(CommandError, match="screened features"):
        call_command("rank_experiment", name="sampling", run_key="rank-exp", stdout=StringIO())
    assert shuffled == []


def test_the_lambdarank_setting_names_its_truncation_and_gain() -> None:
    graded = experiments.experiments(CONFIG)["objective"].arms[-1]

    line = experiments._setting(graded)

    assert "objective lambdarank" in line
    assert "前 30 名" in line
    assert "label_gain 2^級−1" in line


@pytest.mark.django_db
@pytest.mark.parametrize("seeds", ["x", "0,0", "0, 0", "1,01", "", "-1"])
def test_rank_experiment_refuses_bad_seeds_before_training(command: Path, seeds: str) -> None:
    with pytest.raises(CommandError, match="--seeds"):
        call_command(
            "rank_experiment", name="sampling", run_key="rank-exp", seeds=seeds, stdout=StringIO()
        )
    assert not ModelRun.objects.filter(model_type="lightgbm").exists()


def test_the_stopping_settings_name_the_round_floor_and_the_validation_length() -> None:
    _, floor, longer, tuned = experiments.experiments(WalkForwardConfig())["stopping"].arms

    assert experiments._setting(floor).endswith("，至少 200 輪")
    assert "驗證段 12 個月" in experiments._setting(longer)
    assert "驗證段 6 個月" in experiments._setting(floor)
    assert experiments._setting(tuned).endswith("，輪數由調參摺定")


@pytest.mark.django_db
def test_rank_experiment_checks_every_seed_key_before_training(command: Path) -> None:
    with pytest.raises(CommandError, match="at most"):
        call_command(
            "rank_experiment",
            name="window",
            run_key="rank-" + "x" * 24,
            seeds="0,3",
            stdout=StringIO(),
        )
    assert not ModelRun.objects.filter(model_type="lightgbm").exists()


@pytest.mark.django_db
def test_the_tuned_arm_fixes_each_horizon_to_the_tuning_folds_median_best_round(
    small: Path,
) -> None:
    chosen = experiments.experiments(CONFIG)["stopping"]

    rounds = experiments.tuning_rounds(chosen, "rank-exp", calendar(), weekdays()[-1], small)

    base = chosen.arms[0]
    tuning = [
        fold for fold in generate_folds(calendar(), weekdays()[-1], base.config) if fold.tuning
    ]
    assert tuning
    assert experiments.TUNING_SEEDS == (0, 3, 6)
    for horizon in (5, 20):
        best = [
            ModelRun.objects.get(
                run_id=lgbm_runs.run_id(
                    experiments.arm_key("rank-exp", base, seed), "rank", horizon, fold
                )
            ).params["best_round"]
            for fold in tuning
            for seed in experiments.TUNING_SEEDS
        ]
        assert rounds[horizon] == round(float(np.median(best)))
    resolved = experiments.with_tuned_rounds(chosen, rounds)
    tuned = resolved.arms[3]
    assert tuned.variant == Variant(fixed_rounds=((5, rounds[5]), (20, rounds[20])))
    assert resolved.arms[:3] == chosen.arms[:3]
    line = experiments._setting(tuned)
    assert f"固定 5 日 {rounds[5]} 輪、20 日 {rounds[20]} 輪" in line
    assert "seed 0, 3, 6" in line


def fake_tuning(
    monkeypatch: pytest.MonkeyPatch, best: dict[tuple[int, int], int]
) -> list[tuple[str, int]]:
    """No tuning fits, and stopping rounds read from best keyed by seed and horizon."""
    monkeypatch.setattr(experiments, "COARSE_MONTHS", 1)
    trained: list[tuple[str, int]] = []

    def fit(scope: str, folds: Any, run_key: str, directory: Path, seed: int, **kwargs: Any) -> Any:
        trained.append((run_key, seed))
        return []

    def read(experiment: Experiment, run_key: str, folds: Any, seeds: Any) -> pd.DataFrame:
        (tuning,) = folds.values()
        rows = [
            ["base", seed, horizon, fold.index, best[(seed, horizon)], best[(seed, horizon)]]
            for fold in tuning
            for (seed, horizon) in best
        ]
        return pd.DataFrame(rows, columns=experiments.ROUND_COLUMNS)

    monkeypatch.setattr(experiments.lgbm_runs, "run_folds", fit)
    monkeypatch.setattr(experiments, "stopping_rounds", read)
    return trained


def test_tuning_rounds_take_the_median_over_the_tuning_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    best = {(0, 5): 1, (3, 5): 11, (6, 5): 401, (0, 20): 5, (3, 20): 300, (6, 20): 7}
    trained = fake_tuning(monkeypatch, best)
    chosen = experiments.experiments(CONFIG)["stopping"]

    rounds = experiments.tuning_rounds(chosen, "rank-exp", calendar(), weekdays()[-1], Path())

    assert rounds == {5: 11, 20: 7}
    assert trained == [("rank-exp-base", 0), ("rank-exp-base-s3", 3), ("rank-exp-base-s6", 6)]


def test_tuning_rounds_refuse_tuning_folds_without_a_recorded_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_tuning(monkeypatch, {(0, 5): 1, (3, 5): 11, (6, 5): 401, (0, 20): 5})
    chosen = experiments.experiments(CONFIG)["stopping"]

    with pytest.raises(ValueError, match="tuning folds lack"):
        experiments.tuning_rounds(chosen, "rank-exp", calendar(), weekdays()[-1], Path())


def test_train_arms_refuse_a_tuned_arm_whose_rounds_are_not_fixed() -> None:
    chosen = experiments.experiments(CONFIG)["stopping"]
    half = experiments.with_tuned_rounds(chosen, {5: 7})

    for unresolved in (chosen, half):
        with pytest.raises(ValueError, match="with_tuned_rounds"):
            experiments.train_arms(unresolved, "rank-exp", {}, Path(), (0,))


@pytest.mark.django_db
def test_rank_experiment_fixes_the_tuned_rounds_before_training_the_arms(
    command: Path, tmp_path: Path
) -> None:
    output = tmp_path / "stopping.md"
    out = StringIO()

    call_command(
        "rank_experiment", name="stopping", run_key="rank-exp", output=str(output), stdout=out
    )

    chosen = experiments.experiments(CONFIG)["stopping"]
    rounds = experiments.tuning_rounds(chosen, "rank-exp", calendar(), weekdays()[-1], command)
    assert f"tuned rounds from the tuning folds: {rounds}" in out.getvalue()
    text = output.read_text()
    assert (
        f"- tuned：{experiments._setting(experiments.with_tuned_rounds(chosen, rounds).arms[3])}"
        in text
    )
    fitted = ModelRun.objects.filter(run_key="rank-exp-tuned")
    refits = {(run.run_id.split(".")[2], run.params["rounds"]) for run in fitted}
    assert refits == {("h5", rounds[5]), ("h20", rounds[20])}
