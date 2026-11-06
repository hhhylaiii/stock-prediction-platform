"""Research-period rank experiments on coarse folds: window, row sampling, objective (SPEC §7)."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import pandas as pd

from apps.market.calendar import TradingDays
from apps.research import lgbm_runs
from apps.research.labels import HORIZONS
from apps.research.metrics.ic import daily_rank_ic
from apps.research.models import HoldoutScope
from apps.research.prediction_cache import load_predictions
from apps.research.rank_model import LAMBDARANK, RANK_LABEL, SPEC_RECIPE, Variant
from apps.research.training import load_labels, training_rows
from apps.research.walkforward import Fold, WalkForwardConfig, generate_folds

COARSE_MONTHS = 6
LONG_MONTHS = 60
HALF_LIFE = 24.0
SAMPLE_EVERY = 5
WINDOW_FROM = date(2015, 1, 1)
SMALL_GAP = 0.01
RULES = {
    "best": "總體 rank IC（5 日與 20 日平均）最高的一組；同分留 base",
    "not_worse": "總體 rank IC 不低於 base 就採用",
    "better": "總體 rank IC 嚴格高於 base 才採用",
}
DAILY_COLUMNS = ["arm", "fold", "horizon", "date", "ic"]
MODEL = lgbm_runs.MODEL


@dataclass(frozen=True)
class Arm:
    """One side of an experiment: its fold layout and rank variant."""

    name: str
    config: WalkForwardConfig
    variant: Variant


@dataclass(frozen=True)
class Experiment:
    """Arms compared under a rule; the first arm is the SPEC recipe."""

    name: str
    rule: str
    arms: tuple[Arm, ...]


def experiments(base: WalkForwardConfig) -> dict[str, Experiment]:
    """The SPEC §7 rank experiments, each on base's layout with a fold every six months."""
    coarse = replace(base, step_months=COARSE_MONTHS, test_months=COARSE_MONTHS)
    plain = Arm("base", coarse, SPEC_RECIPE)
    long = Arm("w60", replace(coarse, train_months=LONG_MONTHS), SPEC_RECIPE)
    decayed = Arm("expanding", replace(coarse, window="expanding"), Variant(half_life=HALF_LIFE))
    return {
        "window": Experiment("window", "best", (plain, long, decayed)),
        "sampling": Experiment(
            "sampling", "not_worse", (plain, Arm("every5", coarse, Variant(every=SAMPLE_EVERY)))
        ),
        "objective": Experiment(
            "objective",
            "better",
            (plain, Arm("lambdarank", coarse, Variant(objective="lambdarank"))),
        ),
    }


def arm_key(run_key: str, arm: Arm) -> str:
    return f"{run_key}-{arm.name}"


def compared_folds(
    experiment: Experiment, calendar: TradingDays, last: date
) -> dict[str, list[Fold]]:
    """Each arm's research folds over the test periods every arm has, from WINDOW_FROM on."""
    by_arm = {
        arm.name: [
            fold
            for fold in generate_folds(calendar, last, arm.config)
            if fold.research and fold.test_start >= WINDOW_FROM
        ]
        for arm in experiment.arms
    }
    shared = set.intersection(
        *({(fold.test_start, fold.eval_end) for fold in folds} for folds in by_arm.values())
    )
    if not shared:
        raise ValueError(f"no research test period is shared by every {experiment.name} arm")
    return {
        name: [fold for fold in folds if (fold.test_start, fold.eval_end) in shared]
        for name, folds in by_arm.items()
    }


def train_arms(
    experiment: Experiment,
    run_key: str,
    folds: Mapping[str, list[Fold]],
    directory: Path,
    seed: int,
) -> dict[str, list[int]]:
    """Fit each arm's folds under its own run_key; current caches are kept."""
    return {
        arm.name: lgbm_runs.run_folds(
            HoldoutScope.RANK,
            folds[arm.name],
            arm_key(run_key, arm),
            directory,
            seed,
            variant=arm.variant,
        )
        for arm in experiment.arms
    }


def daily_ics(
    experiment: Experiment, run_key: str, folds: Mapping[str, list[Fold]]
) -> pd.DataFrame:
    """Every arm's daily OOS rank IC on tradable test rows; each period's labels are read once."""
    cached = {
        arm.name: load_predictions(arm_key(run_key, arm), MODEL, folds[arm.name])
        for arm in experiment.arms
    }
    parts = []
    for horizon in HORIZONS:
        for shared in zip(*(folds[arm.name] for arm in experiment.arms), strict=True):
            first = shared[0]
            labels = load_labels(first.test_start, first.eval_end, horizon, lgbm_runs.RANK_FIELDS)
            for arm, fold in zip(experiment.arms, shared, strict=True):
                frame = cached[arm.name]
                inside = frame[(frame["fold"] == fold.index) & (frame["date"] <= fold.eval_end)]
                rows = training_rows(inside, labels, RANK_LABEL)
                ic = daily_rank_ic(rows, f"{MODEL}_h{horizon}").rename("ic").reset_index()
                parts.append(ic.assign(arm=arm.name, fold=fold.index, horizon=horizon))
    return pd.concat(parts, ignore_index=True)[DAILY_COLUMNS]


def summary(experiment: Experiment, daily: pd.DataFrame) -> pd.DataFrame:
    """Pooled daily IC per arm and horizon, and their mean, in arm order."""
    pooled = daily.groupby(["arm", "horizon"])["ic"].mean().unstack("horizon")
    table = pooled.rename(columns=lambda horizon: f"h{horizon}")
    table = table.reindex([arm.name for arm in experiment.arms])
    return table.assign(mean=table.mean(axis=1, skipna=False))


def verdict(experiment: Experiment, table: pd.DataFrame) -> str:
    """The arm the experiment's rule adopts; the SPEC recipe whenever the rule is not met."""
    default, *others = [arm.name for arm in experiment.arms]
    means = table["mean"]
    chosen = default
    for name in others:
        value, held = means[name], means[chosen]
        if math.isnan(value):
            continue
        ties_adopt = experiment.rule == "not_worse"
        if value > held or (ties_adopt and value == held):
            chosen = name
    return chosen


def _setting(arm: Arm) -> str:
    """One line naming an arm's window and rank variant."""
    config, variant = arm.config, arm.variant
    window = "expanding" if config.window == "expanding" else f"rolling {config.train_months} 個月"
    decay = "—" if variant.half_life is None else f"{variant.half_life:g} 個月"
    line = f"{window}，每 {variant.every} 日一列，半衰期 {decay}，objective {variant.objective}"
    if variant.objective != "lambdarank":
        return line
    top = LAMBDARANK["lambdarank_truncation_level"]
    return f"{line}（只排每天前 {top} 名的配對，label_gain 2^級−1）"


def _cell(value: float) -> str:
    return "—" if math.isnan(value) else f"{value:.4f}"


def experiment_report(
    experiment: Experiment, run_key: str, daily: pd.DataFrame, note: str | None = None
) -> str:
    """Markdown of the pooled IC per arm, the verdict, a small-gap warning and each fold."""
    table = summary(experiment, daily)
    chosen = verdict(experiment, table)
    base = table["mean"].iloc[0]
    lines = [f"# 實驗：{experiment.name}（{run_key}）", ""]
    if note:
        lines += [f"> {note}", ""]
    lines += [f"- {arm.name}：{_setting(arm)}" for arm in experiment.arms]
    folds = sorted(daily["fold"].unique())
    lines += [
        f"- 比較 {len(folds)} 個研究摺（摺 {folds[0]}–{folds[-1]}），每組各自的快取 run_key "
        f"為 `{run_key}-<組名>`",
        f"- 判定規則：{RULES[experiment.rule]}",
        "",
        "| 組 | 5 日 IC | 20 日 IC | 平均 | 與 base 差 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, row in table.iterrows():
        gap = "—" if name == table.index[0] else f"{row['mean'] - base:+.4f}"
        lines.append(
            f"| {name} | {_cell(row['h5'])} | {_cell(row['h20'])} | {_cell(row['mean'])} | {gap} |"
        )
    lines += ["", f"**判定：採用 {chosen}**"]
    close = [
        name
        for name, value in table["mean"].items()
        if name != chosen and abs(value - table["mean"][chosen]) < SMALL_GAP
    ]
    if close:
        lines.append(
            f"- ⚠️ {', '.join(map(str, close))} 與 {chosen} 差距小於 {SMALL_GAP}："
            "定案重建後要重跑一次再定案"
        )
    by_fold = daily.groupby(["fold", "arm", "horizon"])["ic"].mean()
    names = [arm.name for arm in experiment.arms]
    header = " | ".join(f"{name} {horizon} 日" for name in names for horizon in HORIZONS)
    lines += ["", "## 每摺", "", f"| 摺 | {header} |"]
    lines.append("| --- |" + " --- |" * (len(names) * len(HORIZONS)))
    for fold in folds:
        cells = [
            _cell(float(by_fold.get((fold, name, horizon), math.nan)))
            for name in names
            for horizon in HORIZONS
        ]
        lines.append(f"| {fold} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
