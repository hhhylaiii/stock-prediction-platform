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
    SAMPLE_EVERY,
    Arm,
    Experiment,
    compared_folds,
    daily_ics,
    experiment_report,
    summary,
    verdict,
)
from apps.research.management.commands import rank_experiment
from apps.research.metrics.ic import daily_rank_ic
from apps.research.models import LabelDaily, ModelRun
from apps.research.prediction_cache import load_predictions
from apps.research.rank_model import SPEC_RECIPE, Variant
from apps.research.walkforward import WalkForwardConfig
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
    assert {name: item.rule for name, item in chosen.items()} == {
        "window": "best",
        "sampling": "not_worse",
        "objective": "better",
    }
    assert len({arm.config for arm in (sampling.arms[0], objective.arms[0], base)}) == 1


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


def run(name: str, directory: Path) -> tuple[Experiment, dict[str, Any]]:
    chosen = experiments.experiments(CONFIG)[name]
    folds = compared_folds(chosen, calendar(), weekdays()[-1])
    experiments.train_arms(chosen, "rank-exp", folds, directory, seed=0)
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
    grades = ModelRun.objects.filter(run_key="rank-exp-lambdarank").values_list("params", flat=True)
    assert {params["variant"]["objective"] for params in grades} == {"lambdarank"}


@pytest.mark.django_db
def test_daily_ics_pool_each_arm_over_its_tradable_test_rows(small: Path) -> None:
    chosen, folds = run("sampling", small)

    daily = daily_ics(chosen, "rank-exp", folds)

    for arm in chosen.arms:
        cached = load_predictions(f"rank-exp-{arm.name}", "lightgbm", folds[arm.name])
        for horizon in (5, 20):
            labels = pd.DataFrame(list(LabelDaily.objects.filter(horizon=horizon).values()))
            joined = cached.merge(labels, on=["stock_id", "date"])
            ends = joined["fold"].map({fold.index: fold.eval_end for fold in folds[arm.name]})
            rows = joined[
                (joined["date"] <= ends) & joined["in_universe"] & ~joined["untradable_t1"]
            ]
            expected = daily_rank_ic(rows, f"lightgbm_h{horizon}")
            got = daily[(daily["arm"] == arm.name) & (daily["horizon"] == horizon)]
            np.testing.assert_allclose(
                got.set_index("date")["ic"].sort_index(), expected.sort_index()
            )
    pooled = summary(chosen, daily)
    assert list(pooled.index) == ["base", "every5"]
    base = pooled.loc["base"].astype(float)
    assert base["mean"] == pytest.approx((base["h5"] + base["h20"]) / 2)


def test_experiment_report_shows_the_verdict_small_gaps_and_folds() -> None:
    chosen = experiment("not_worse", "base", "every5")
    daily = pd.DataFrame(
        {
            "arm": ["base", "base", "every5", "every5"] * 2,
            "fold": [3, 4] * 4,
            "horizon": [5] * 4 + [20] * 4,
            "date": [date(2016, 10, 3), date(2016, 11, 1)] * 4,
            "ic": [0.05, 0.07, 0.054, 0.07, 0.04, 0.06, 0.04, 0.06],
        }
    )

    text = experiment_report(chosen, "rank-exp", daily, note="暫定特徵")

    assert "暫定特徵" in text
    assert "| every5 | 0.0620 | 0.0500 | 0.0560 | +0.0010 |" in text
    assert "採用 every5" in text
    assert "小於 0.01" in text
    assert "| 3 | 0.0500 | 0.0400 | 0.0540 | 0.0400 |" in text


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
        stdout=out,
    )

    chosen, folds = run("sampling", command)
    expected = experiment_report(chosen, "rank-exp", daily_ics(chosen, "rank-exp", folds), "暫定")
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
