"""Rank experiments on coarse research folds: window, sampling, objective, stopping (SPEC §7)."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import pandas as pd
from django.conf import settings

from apps.market.calendar import TradingDays
from apps.research import lgbm_runs
from apps.research.baseline_runs import shuffle_check
from apps.research.gbdt import EARLY_STOPPING_ROUNDS, bag_seeds
from apps.research.labels import HORIZONS
from apps.research.metrics.ic import daily_rank_ic
from apps.research.models import HoldoutScope, ModelRun
from apps.research.prediction_cache import load_predictions
from apps.research.rank_model import LAMBDARANK, RANK_LABEL, SPEC_RECIPE, Variant
from apps.research.target_models import TARGET_RECIPE, TargetVariant
from apps.research.training import load_labels, training_rows
from apps.research.walkforward import Fold, WalkForwardConfig, generate_folds

COARSE_MONTHS = 6
LONG_MONTHS = 60
HALF_LIFE = 24.0
SAMPLE_EVERY = 5
MIN_ROUNDS = 200
VALID_FACTOR = 2
WINDOW_FROM = date(2015, 1, 1)
SMALL_GAP = 0.02
MIN_SEEDS = 3
TUNING_SEEDS = tuple(bag_seeds(0, MIN_SEEDS))
SUGGESTED_SEEDS = ",".join(map(str, TUNING_SEEDS))
RULES = {
    "best": "總體 rank IC（5 日與 20 日平均）最高的一組；同分留 base",
    "not_worse": "總體 rank IC 不低於 base 就採用",
    "better": "總體 rank IC 嚴格高於 base 才採用",
}
DAILY_COLUMNS = ["arm", "seed", "fold", "horizon", "date", "ic"]
ROUND_COLUMNS = ["arm", "seed", "horizon", "fold", "best", "rounds"]
MODEL = lgbm_runs.MODEL


@dataclass(frozen=True)
class Arm:
    """One side of an experiment: its fold layout, rank variant and target variant."""

    name: str
    config: WalkForwardConfig
    variant: Variant
    tuned: bool = False
    target: TargetVariant = TARGET_RECIPE


@dataclass(frozen=True)
class Experiment:
    """Arms compared under a rule; the first arm stays when the rule is not met."""

    name: str
    rule: str
    arms: tuple[Arm, ...]


def experiments(base: WalkForwardConfig) -> dict[str, Experiment]:
    """The SPEC §7 rank experiments, each on base's layout with a fold every six months."""
    coarse = replace(base, step_months=COARSE_MONTHS, test_months=COARSE_MONTHS)
    plain = Arm("base", coarse, SPEC_RECIPE)
    long = Arm("w60", replace(coarse, train_months=LONG_MONTHS), SPEC_RECIPE)
    decayed = Arm("expanding", replace(coarse, window="expanding"), Variant(half_life=HALF_LIFE))
    floor = Arm("floor", coarse, Variant(min_rounds=MIN_ROUNDS))
    months = VALID_FACTOR * coarse.valid_months
    longer = Arm(f"valid{months}", replace(coarse, valid_months=months), SPEC_RECIPE)
    tuned = Arm("tuned", coarse, SPEC_RECIPE, tuned=True)
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
        "stopping": Experiment("stopping", "best", (plain, floor, longer, tuned)),
    }


def arm_key(run_key: str, arm: Arm, seed: int = 0) -> str:
    """The arm's cache run_key; seed 0 keeps the key experiments used before seeds existed."""
    return f"{run_key}-{arm.name}" if seed == 0 else f"{run_key}-{arm.name}-s{seed}"


def parse_seeds(text: str) -> tuple[int, ...]:
    """Distinct non-negative seeds from a comma list."""
    parts = [part.strip() for part in text.split(",")]
    values = tuple(int(part) for part in parts if part.isdigit())
    if len(values) < len(parts) or len(set(values)) < len(values):
        raise ValueError(f"--seeds takes distinct non-negative integers like 0,3,6, not {text!r}")
    return values


def checked_shuffle(fold: Fold, directory: Path, seed: int) -> float:
    """SPEC §8 shuffled-label IC of a fold; one at SHUFFLE_IC_LIMIT or past it stops the run."""
    ic = shuffle_check(fold, directory, seed)
    if math.isnan(ic) or abs(ic) >= settings.SHUFFLE_IC_LIMIT:
        raise ValueError(
            f"shuffle check on fold {fold.index} gave IC {ic:.4f}, not below "
            f"{settings.SHUFFLE_IC_LIMIT}; a label column may be among the features"
        )
    return ic


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
    seeds: Sequence[int],
) -> dict[str, list[int]]:
    """Fit each arm's folds once per seed under its own run_key; current caches are kept."""
    unresolved = [
        arm.name
        for arm in experiment.arms
        if arm.tuned and set(dict(arm.variant.fixed_rounds)) != set(HORIZONS)
    ]
    if unresolved:
        raise ValueError(f"fix the rounds of {unresolved} with with_tuned_rounds before training")
    return {
        arm_key(run_key, arm, seed): lgbm_runs.run_folds(
            HoldoutScope.RANK,
            folds[arm.name],
            arm_key(run_key, arm, seed),
            directory,
            seed,
            variant=arm.variant,
        )
        for arm in experiment.arms
        for seed in seeds
    }


def _fold_ics(
    run_key: str, arm: Arm, fold: Fold, seeds: Sequence[int], labels: pd.DataFrame, horizon: int
) -> list[pd.DataFrame]:
    """One fold's daily IC under each seed's cache of the arm."""
    parts = []
    for seed in seeds:
        cached = load_predictions(arm_key(run_key, arm, seed), MODEL, [fold])
        rows = training_rows(cached[cached["date"] <= fold.eval_end], labels, RANK_LABEL)
        ic = daily_rank_ic(rows, f"{MODEL}_h{horizon}").rename("ic").reset_index()
        parts.append(ic.assign(arm=arm.name, seed=seed, fold=fold.index, horizon=horizon))
    return parts


def daily_ics(
    experiment: Experiment,
    run_key: str,
    folds: Mapping[str, list[Fold]],
    seeds: Sequence[int] = (0,),
) -> pd.DataFrame:
    """Daily OOS rank IC of every arm and seed on tradable test rows; labels read once a period."""
    parts = []
    for horizon in HORIZONS:
        for shared in zip(*(folds[arm.name] for arm in experiment.arms), strict=True):
            first = shared[0]
            labels = load_labels(first.test_start, first.eval_end, horizon, lgbm_runs.RANK_FIELDS)
            for arm, fold in zip(experiment.arms, shared, strict=True):
                parts += _fold_ics(run_key, arm, fold, seeds, labels, horizon)
    return pd.concat(parts, ignore_index=True)[DAILY_COLUMNS]


def per_seed(daily: pd.DataFrame) -> pd.DataFrame:
    """Pooled daily IC per arm and seed for each horizon, and their mean."""
    pooled = daily.groupby(["arm", "seed", "horizon"])["ic"].mean().unstack("horizon")
    table = pooled.reindex(columns=list(HORIZONS)).rename(columns=lambda horizon: f"h{horizon}")
    return table.assign(mean=table.mean(axis=1, skipna=False))


def _strict_mean(values: pd.Series) -> float:
    return float(values.mean(skipna=False))


def summary(experiment: Experiment, daily: pd.DataFrame) -> pd.DataFrame:
    """Seed-averaged pooled IC per arm, the spread of its seeds' means and their count."""
    seeds = per_seed(daily).groupby(level="arm")
    table = seeds.agg(_strict_mean)
    table = table.assign(sd=seeds["mean"].std(), seeds=seeds.size())
    return table.reindex([arm.name for arm in experiment.arms])


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


def stopping_rounds(
    experiment: Experiment, run_key: str, folds: Mapping[str, list[Fold]], seeds: Sequence[int]
) -> pd.DataFrame:
    """Refit rounds each arm, seed, horizon and fold recorded in model_run."""
    rows = []
    for arm in experiment.arms:
        for seed in seeds:
            key = arm_key(run_key, arm, seed)
            ids = {
                lgbm_runs.run_id(key, HoldoutScope.RANK, horizon, fold): (horizon, fold.index)
                for horizon in HORIZONS
                for fold in folds[arm.name]
            }
            fitted = ModelRun.objects.filter(run_id__in=ids).values_list("run_id", "params")
            for run_id, params in fitted:
                horizon, index = ids[run_id]
                best = params.get("best_round", params["rounds"])
                rows.append([arm.name, seed, horizon, index, best, params["rounds"]])
    return pd.DataFrame(rows, columns=ROUND_COLUMNS)


def tuning_rounds(
    experiment: Experiment,
    run_key: str,
    calendar: TradingDays,
    last: date,
    directory: Path,
) -> dict[int, int]:
    """Median best round per horizon of the base arm's tuning folds under TUNING_SEEDS."""
    base = experiment.arms[0]
    tuning = [fold for fold in generate_folds(calendar, last, base.config) if fold.tuning]
    if not tuning:
        raise ValueError(f"{experiment.name} has no tuning folds to fix its rounds on")
    for seed in TUNING_SEEDS:
        key = arm_key(run_key, base, seed)
        lgbm_runs.run_folds(HoldoutScope.RANK, tuning, key, directory, seed, variant=base.variant)
    alone = replace(experiment, arms=(base,))
    rounds = stopping_rounds(alone, run_key, {base.name: tuning}, TUNING_SEEDS)
    expected = len(tuning) * len(TUNING_SEEDS) * len(HORIZONS)
    if len(rounds) < expected:
        raise ValueError(
            f"tuning folds lack {expected - len(rounds)} of {expected} best rounds in model_run "
            f"under {arm_key(run_key, base)}; drop that run and fit its tuning folds again"
        )
    medians = rounds.groupby("horizon")["best"].median()
    return {horizon: round(float(medians[horizon])) for horizon in HORIZONS if horizon in medians}


def with_tuned_rounds(experiment: Experiment, rounds: Mapping[int, int]) -> Experiment:
    """The experiment with each tuned arm refitting every fold for the given rounds."""
    fixed = tuple(sorted(rounds.items()))
    arms = tuple(
        replace(arm, variant=replace(arm.variant, fixed_rounds=fixed)) if arm.tuned else arm
        for arm in experiment.arms
    )
    return replace(experiment, arms=arms)


def _setting(arm: Arm) -> str:
    """One line naming an arm's window, validation tail and rank variant."""
    config, variant = arm.config, arm.variant
    window = "expanding" if config.window == "expanding" else f"rolling {config.train_months} 個月"
    decay = "—" if variant.half_life is None else f"{variant.half_life:g} 個月"
    line = (
        f"{window}，驗證段 {config.valid_months} 個月，每 {variant.every} 日一列，"
        f"半衰期 {decay}，objective {variant.objective}"
    )
    if variant.objective == "lambdarank":
        top = LAMBDARANK["lambdarank_truncation_level"]
        line += f"（只排每天前 {top} 名的配對，label_gain 2^級−1）"
    if variant.min_rounds:
        line += f"，至少 {variant.min_rounds} 輪"
    if variant.fixed_rounds:
        counts = "、".join(f"{horizon} 日 {count} 輪" for horizon, count in variant.fixed_rounds)
        seeds = ", ".join(map(str, TUNING_SEEDS))
        line += f"，固定 {counts}（調參摺 seed {seeds} 最佳輪數的中位數）"
    elif arm.tuned:
        line += "，輪數由調參摺定"
    return line


def _cell(value: float) -> str:
    return "—" if math.isnan(value) else f"{value:.4f}"


def _table_lines(table: pd.DataFrame) -> list[str]:
    base = table["mean"].iloc[0]
    lines = [
        "| 組 | 5 日 IC | 20 日 IC | 平均 | 與 base 差 | seed 間標準差 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, row in table.iterrows():
        gap = "—" if name == table.index[0] else f"{row['mean'] - base:+.4f}"
        cells = [_cell(row["h5"]), _cell(row["h20"]), _cell(row["mean"]), gap, _cell(row["sd"])]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def _verdict_lines(table: pd.DataFrame, chosen: str, seeds: int) -> list[str]:
    """The verdict, provisional when a close arm was judged on fewer than MIN_SEEDS seeds."""
    held = table["mean"][chosen]
    close = [
        str(name)
        for name, value in table["mean"].items()
        if name != chosen and abs(value - held) < SMALL_GAP
    ]
    if not close:
        return [f"**判定：採用 {chosen}**"]
    names = ", ".join(close)
    if seeds >= MIN_SEEDS:
        return [
            f"**判定：採用 {chosen}**",
            f"- {names} 與 {chosen} 差距小於 {SMALL_GAP}，判定用 {seeds} 個 seed 的平均",
        ]
    return [
        f"**判定（暫定）：採用 {chosen}**",
        f"- ⚠️ {names} 與 {chosen} 差距小於 {SMALL_GAP}，只跑了 {seeds} 個 seed："
        f"要用 {MIN_SEEDS} 個 seed（`--seeds {SUGGESTED_SEEDS}`）重跑再判定",
    ]


def _seed_lines(experiment: Experiment, daily: pd.DataFrame) -> list[str]:
    means = per_seed(daily)["mean"].unstack("seed")
    seeds = list(means.columns)
    lines = ["", "## 各 seed", "", "| 組 | " + " | ".join(f"seed {seed}" for seed in seeds) + " |"]
    lines.append("| --- |" + " --- |" * len(seeds))
    for arm in experiment.arms:
        cells = [_cell(float(value)) for value in means.loc[arm.name].to_numpy(dtype=float)]
        lines.append(f"| {arm.name} | " + " | ".join(cells) + " |")
    return lines


def _round_lines(experiment: Experiment, rounds: pd.DataFrame) -> list[str]:
    lines = [
        "",
        "## 停止輪數",
        "",
        "| 組 | horizon | 最佳輪數最少 | 中位數 | 最多 | "
        f"少於 {EARLY_STOPPING_ROUNDS} 輪（摺 × seed） | 重訓輪數中位數 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in experiment.arms:
        for horizon in HORIZONS:
            picked = rounds[(rounds["arm"] == arm.name) & (rounds["horizon"] == horizon)]
            if picked.empty:
                continue
            best = picked["best"]
            short = int((best < EARLY_STOPPING_ROUNDS).sum())
            lines.append(
                f"| {arm.name} | {horizon} 日 | {best.min()} | {best.median():g} | "
                f"{best.max()} | {short} | {picked['rounds'].median():g} |"
            )
    return lines


def _fold_lines(experiment: Experiment, daily: pd.DataFrame) -> list[str]:
    by_fold = daily.groupby(["fold", "arm", "horizon"])["ic"].mean()
    names = [arm.name for arm in experiment.arms]
    header = " | ".join(f"{name} {horizon} 日" for name in names for horizon in HORIZONS)
    lines = ["", "## 每摺", "", f"| 摺 | {header} |"]
    lines.append("| --- |" + " --- |" * (len(names) * len(HORIZONS)))
    for fold in sorted(daily["fold"].unique()):
        cells = [
            _cell(float(by_fold.get((fold, name, horizon), math.nan)))
            for name in names
            for horizon in HORIZONS
        ]
        lines.append(f"| {fold} | " + " | ".join(cells) + " |")
    return lines


def experiment_report(
    experiment: Experiment,
    run_key: str,
    daily: pd.DataFrame,
    note: str | None = None,
    rounds: pd.DataFrame | None = None,
) -> str:
    """Markdown of the seed-averaged IC per arm, the verdict, seeds, stopping rounds and folds."""
    table = summary(experiment, daily)
    seeds = sorted(daily["seed"].unique())
    lines = [f"# 實驗：{experiment.name}（{run_key}）", ""]
    if note:
        lines += [f"> {note}", ""]
    lines += [f"- {arm.name}：{_setting(arm)}" for arm in experiment.arms]
    folds = sorted(daily["fold"].unique())
    lines += [
        f"- 比較 {len(folds)} 個研究摺（摺 {folds[0]}–{folds[-1]}），seed "
        f"{', '.join(map(str, seeds))}；每組的快取 run_key 為 `{run_key}-<組名>`，"
        "seed 不是 0 時再加 `-s<seed>`",
        f"- 判定規則：{RULES[experiment.rule]}；多個 seed 時取各 seed 的平均",
        "",
        *_table_lines(table),
        "",
        *_verdict_lines(table, verdict(experiment, table), len(seeds)),
    ]
    if len(seeds) > 1:
        lines += _seed_lines(experiment, daily)
    if rounds is not None and not rounds.empty:
        lines += _round_lines(experiment, rounds)
    lines += _fold_lines(experiment, daily)
    return "\n".join(lines) + "\n"
