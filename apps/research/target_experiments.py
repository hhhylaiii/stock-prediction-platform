"""Return and interval experiments on coarse research folds: market features, up label (SPEC §7)."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from apps.market.calendar import TradingDays
from apps.research import experiments, lgbm_runs
from apps.research.evaluation import tradable_rows
from apps.research.experiments import MIN_SEEDS, SUGGESTED_SEEDS, Arm, Experiment, arm_key
from apps.research.labels import VOL_DAYS
from apps.research.metrics.forecast import (
    daily_absolute_error,
    daily_auc,
    daily_brier,
    daily_coverage,
    daily_direction_hits,
    daily_pinball,
    daily_qlike,
)
from apps.research.models import HoldoutScope
from apps.research.prediction_cache import load_predictions
from apps.research.rank_model import SPEC_RECIPE
from apps.research.relative_up import relative_up
from apps.research.target_eval import (
    COVERAGE_TOTAL,
    GATE_T,
    HORIZON,
    LABEL,
    LABEL_FIELDS,
    RETURN_COLUMN,
    UP_LABEL,
    VOL_LABEL,
    Comparison,
    beats,
    paired,
    quantile_columns,
)
from apps.research.target_models import TARGET_RECIPE, TargetVariant
from apps.research.training import KEYS, load_labels
from apps.research.walkforward import Fold, WalkForwardConfig

RETURN, INTERVAL = HoldoutScope.RETURN, HoldoutScope.INTERVAL
MODEL = lgbm_runs.MODEL
NO_MARKET = TargetVariant(market=False)
QUANTILE_COLUMNS = quantile_columns(MODEL)
VOL_COLUMN = f"{MODEL}_vol_h{VOL_DAYS}"
PROB_COLUMN = f"{MODEL}_prob_h{VOL_DAYS}"
SCOPE_METRICS: dict[str, tuple[str, ...]] = {
    RETURN: ("mae", "direction"),
    INTERVAL: ("pinball", "coverage", "vol_mae", "qlike", "auc", "brier"),
}
PROBABILITY_METRICS = ("auc",)
SCOPE_PRIMARY: dict[str, str] = {RETURN: "mae", INTERVAL: "pinball"}
LOWER_BETTER = {"mae", "pinball", "vol_mae", "qlike", "brier"}
COVERAGE_AIM = sum(COVERAGE_TOTAL) / 2
NEEDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "mae": (LABEL, (RETURN_COLUMN,)),
    "direction": (LABEL, (RETURN_COLUMN,)),
    "pinball": (LABEL, tuple(QUANTILE_COLUMNS.values())),
    "coverage": (LABEL, (QUANTILE_COLUMNS[0.1], QUANTILE_COLUMNS[0.9])),
    "vol_mae": (VOL_LABEL, (VOL_COLUMN,)),
    "qlike": (VOL_LABEL, (VOL_COLUMN,)),
    "auc": (UP_LABEL, (PROB_COLUMN,)),
    "brier": (UP_LABEL, (PROB_COLUMN,)),
}
TITLES = {
    "mae": "MAE",
    "direction": "方向準確率",
    "pinball": "pinball",
    "coverage": "覆蓋率",
    "vol_mae": "波動率 MAE",
    "qlike": "QLIKE",
    "auc": "AUC",
    "brier": "Brier",
}
SETTINGS = {
    "nomarket": "只用篩選後的個股特徵，不加市場特徵",
    "market": "個股特徵加上市場特徵（SPEC 配方）",
    "relative": "機率模型改用 20 日報酬減 TAIEX 同期報酬是否大於 5%；分位數與波動率模型不變",
    "relative-nomarket": "同 relative，但兩組都不加市場特徵",
}
MARKET_REASON = "OOS 指標沒有提高就不用"
RELATIVE_REASON = "日均 AUC 較高就改用"
RELATIVE_CAVEAT = (
    "日均 AUC 兩組各對自己的 label 評估：第一組看 20 日報酬是否大於 5%，第二組看減掉 TAIEX "
    "同期報酬（收盤 t 到收盤 t+20）後是否大於 5%。TAIEX 是價格指數、個股是含息的還原報酬，"
    "除息旺季的 TAIEX 報酬偏低，第二組的正例會偏多"
)
DAILY_COLUMNS = ["arm", "seed", "fold", "metric", "date", "value"]


@dataclass(frozen=True)
class TargetExperiment(Experiment):
    """Arms of one target scope; the verdict reads its primary metric, not the rank rule."""

    scope: str
    primary: str
    metrics: tuple[str, ...]
    reason: str
    caveats: tuple[str, ...] = ()
    window: str = "base"


def _layout(base: WalkForwardConfig, window: str) -> WalkForwardConfig:
    """The coarse fold layout of a rank window the target scopes can train on."""
    windows = experiments.window_arms(base)
    if window not in windows:
        raise ValueError(f"window must be one of {', '.join(windows)}, not {window!r}")
    if windows[window].variant != SPEC_RECIPE:
        raise ValueError(
            f"the {window} window weighs rows by age, and the target scopes have no time-decay "
            "weights yet; use base or w60"
        )
    return windows[window].config


def target_experiments(
    base: WalkForwardConfig, window: str = "base", market: bool = True
) -> dict[str, TargetExperiment]:
    """The SPEC §7 experiments on a window's folds; relative-up builds on the market verdict."""
    layout = _layout(base, window)
    prefix = "" if window == "base" else f"{window}-"

    def arm(name: str, target: TargetVariant = TARGET_RECIPE) -> Arm:
        return Arm(f"{prefix}{name}", layout, SPEC_RECIPE, target=target)

    market_arms = (arm("nomarket", NO_MARKET), arm("market"))
    chosen = {
        f"market-{scope}": TargetExperiment(
            f"market-{scope}",
            "better",
            market_arms,
            scope,
            SCOPE_PRIMARY[scope],
            SCOPE_METRICS[scope],
            MARKET_REASON,
            window=window,
        )
        for scope in (RETURN, INTERVAL)
    }
    first = market_arms[1 if market else 0]
    relative = arm(
        "relative" if market else "relative-nomarket",
        TargetVariant(market=market, relative_up=True),
    )
    follow = "不加市場特徵時，改帶 --no-market" if market else "採用市場特徵時，拿掉 --no-market"
    chosen["relative-up"] = TargetExperiment(
        "relative-up",
        "better",
        (first, relative),
        INTERVAL,
        "auc",
        PROBABILITY_METRICS,
        RELATIVE_REASON,
        (
            RELATIVE_CAVEAT,
            f"第一組沿用 market-interval 的 {first.name} 組；market-interval 判定{follow} 重跑",
        ),
        window,
    )
    return chosen


def train_arms(
    experiment: TargetExperiment,
    run_key: str,
    folds: Mapping[str, list[Fold]],
    directory: Path,
    seeds: Sequence[int],
) -> dict[str, list[int]]:
    """Fit each arm's folds once per seed under its own run_key; current caches are kept."""
    return {
        arm_key(run_key, arm, seed): lgbm_runs.run_folds(
            experiment.scope,
            folds[arm.name],
            arm_key(run_key, arm, seed),
            directory,
            seed,
            target=arm.target,
        )
        for arm in experiment.arms
        for seed in seeds
    }


def _score(metric: str, rows: pd.DataFrame) -> pd.Series:
    if metric == "mae":
        return daily_absolute_error(rows, LABEL, RETURN_COLUMN)
    if metric == "direction":
        return daily_direction_hits(rows, LABEL, RETURN_COLUMN)
    if metric == "pinball":
        return daily_pinball(rows, LABEL, QUANTILE_COLUMNS)
    if metric == "coverage":
        return daily_coverage(rows, LABEL, QUANTILE_COLUMNS[0.1], QUANTILE_COLUMNS[0.9])
    if metric == "vol_mae":
        return daily_absolute_error(rows, VOL_LABEL, VOL_COLUMN)
    if metric == "qlike":
        return daily_qlike(rows, VOL_LABEL, VOL_COLUMN)
    up = rows.assign(**{UP_LABEL: rows[UP_LABEL].astype(int)})
    if metric == "auc":
        return daily_auc(up, UP_LABEL, PROB_COLUMN)
    return daily_brier(up, UP_LABEL, PROB_COLUMN)


def _shared(metric: str, frames: Sequence[pd.DataFrame]) -> pd.Series:
    """Rows where the label and every arm's columns of a metric are set; QLIKE also needs > 0."""
    label, columns = NEEDS[metric]
    keep = pd.Series(True, index=frames[0].index)
    for frame in frames:
        keep &= frame[label].notna() & frame[list(columns)].notna().all(axis=1)
        if metric == "qlike":
            keep &= (frame[label] > 0) & (frame[VOL_COLUMN] > 0)
    return keep


def _arm_frames(
    experiment: TargetExperiment,
    run_key: str,
    period: Sequence[Fold],
    seed: int,
    labels: Sequence[pd.DataFrame],
) -> list[pd.DataFrame]:
    """Each arm's tradable cached rows of one test period with labels, on the rows all arms hold."""
    frames = []
    for arm, fold, known in zip(experiment.arms, period, labels, strict=True):
        cached = load_predictions(arm_key(run_key, arm, seed), MODEL, [fold])
        joined = cached[cached["date"] <= fold.eval_end].merge(
            known, on=KEYS, how="left", validate="one_to_one"
        )
        frames.append(tradable_rows(joined).set_index(KEYS))
    common = frames[0].index
    for frame in frames[1:]:
        common = common.intersection(frame.index)
    return [frame.loc[common].reset_index() for frame in frames]


def _arm_labels(arm: Arm, labels: pd.DataFrame, calendar: TradingDays | None) -> pd.DataFrame:
    """The labels an arm is scored on: the up label against TAIEX for a relative_up arm."""
    if not arm.target.relative_up or calendar is None:
        return labels
    return labels.assign(**{UP_LABEL: relative_up(labels, calendar)})


def daily_metrics(
    experiment: TargetExperiment,
    run_key: str,
    folds: Mapping[str, list[Fold]],
    seeds: Sequence[int] = (0,),
) -> pd.DataFrame:
    """Daily OOS metrics of each arm and seed on shared tradable rows; labels read once a period."""
    scope, parts = experiment.scope, []
    relative = any(arm.target.relative_up for arm in experiment.arms)
    calendar = TradingDays.from_db() if relative else None
    for period in zip(*(folds[arm.name] for arm in experiment.arms), strict=True):
        first = period[0]
        labels = load_labels(first.test_start, first.eval_end, HORIZON[scope], LABEL_FIELDS[scope])
        own = [_arm_labels(arm, labels, calendar) for arm in experiment.arms]
        for seed in seeds:
            frames = _arm_frames(experiment, run_key, period, seed, own)
            for metric in experiment.metrics:
                keep = _shared(metric, frames).to_numpy()
                for arm, frame, fold in zip(experiment.arms, frames, period, strict=True):
                    daily = _score(metric, frame[keep]).rename("value").reset_index()
                    parts.append(
                        daily.assign(arm=arm.name, seed=seed, fold=fold.index, metric=metric)
                    )
    return pd.concat(parts, ignore_index=True)[DAILY_COLUMNS]


def per_seed(daily: pd.DataFrame, metric: str) -> pd.DataFrame:
    """A metric's research-period mean per arm and seed."""
    picked = daily[daily["metric"] == metric]
    return picked.groupby(["arm", "seed"])["value"].mean().unstack("seed")


def _seed_mean(daily: pd.DataFrame, arm: str, metric: str) -> pd.Series:
    picked = daily[(daily["arm"] == arm) & (daily["metric"] == metric)]
    return picked.groupby("date")["value"].mean()


def comparisons(
    experiment: TargetExperiment, daily: pd.DataFrame, folds: Sequence[Fold]
) -> dict[str, Comparison]:
    """Each metric of the second arm against the first, seed-averaged by day and paired by fold."""
    default, alternative = (arm.name for arm in experiment.arms)
    return {
        metric: _paired_days(
            _seed_mean(daily, alternative, metric), _seed_mean(daily, default, metric), folds
        )
        for metric in experiment.metrics
    }


def _paired_days(model: pd.Series, baseline: pd.Series, folds: Sequence[Fold]) -> Comparison:
    days = model.index.intersection(baseline.index)
    return paired(model[days], baseline[days], folds)


def clears(metric: str, compared: Comparison) -> bool:
    """Whether the second arm is better on a metric with a fold-gap NW t past GATE_T its way."""
    if metric in LOWER_BETTER:
        return beats(compared)
    return compared.model > compared.baseline and compared.nw_t > GATE_T


def verdict(experiment: TargetExperiment, compared: Mapping[str, Comparison]) -> str:
    """The second arm when it clears the primary metric's fold-gap test, else the first."""
    default, alternative = (arm.name for arm in experiment.arms)
    return alternative if clears(experiment.primary, compared[experiment.primary]) else default


def side(metric: str, compared: Comparison) -> int:
    """+1 when the second arm is better on a metric, −1 when worse, 0 on a tie or no value."""
    model, baseline = compared.model, compared.baseline
    if math.isnan(model) or math.isnan(baseline):
        return 0
    if metric == "coverage":
        model, baseline = -abs(model - COVERAGE_AIM), -abs(baseline - COVERAGE_AIM)
    elif metric in LOWER_BETTER:
        model, baseline = -model, -baseline
    return (model > baseline) - (model < baseline)


def _cell(value: float) -> str:
    return "—" if math.isnan(value) else f"{value:.4f}"


def _gap(value: float) -> str:
    return "—" if math.isnan(value) else f"{value:+.4f}"


def _table_lines(experiment: TargetExperiment, compared: Mapping[str, Comparison]) -> list[str]:
    metrics = experiment.metrics
    default, alternative = (arm.name for arm in experiment.arms)
    rows = [
        (default, [_cell(compared[metric].baseline) for metric in metrics]),
        (alternative, [_cell(compared[metric].model) for metric in metrics]),
        (
            f"差（{alternative} − {default}）",
            [_gap(compared[metric].model - compared[metric].baseline) for metric in metrics],
        ),
        ("逐摺差 NW t", [_gap(compared[metric].nw_t) for metric in metrics]),
    ]
    lines = [
        "| 組 | " + " | ".join(_header(metric) for metric in metrics) + " |",
        "| --- |" + " --- |" * len(metrics),
    ]
    return lines + [f"| {name} | " + " | ".join(cells) + " |" for name, cells in rows]


def _header(metric: str) -> str:
    if metric == "coverage":
        return f"{TITLES[metric]}（目標 {COVERAGE_AIM:.0%}）"
    return f"{TITLES[metric]} {'↓' if metric in LOWER_BETTER else '↑'}"


def _verdict_lines(
    experiment: TargetExperiment, compared: Mapping[str, Comparison], seeds: int
) -> list[str]:
    chosen = verdict(experiment, compared)
    default, alternative = (arm.name for arm in experiment.arms)
    primary = experiment.primary
    gap = compared[primary]
    title = "判定" if seeds >= MIN_SEEDS else "判定（暫定）"
    lines = [f"**{title}：採用 {chosen}**"]
    if side(primary, gap) > 0 and chosen == default:
        lower = primary in LOWER_BETTER
        lines.append(
            f"- {alternative} 的 {TITLES[primary]} {'較低' if lower else '較高'}，但逐摺差 NW t "
            f"{gap.nw_t:+.2f} 沒有{'低於 −' if lower else '高於 '}{GATE_T:g}：照規則留 {default}"
        )
    if seeds < MIN_SEEDS:
        lines.append(
            f"- 只跑了 {seeds} 個 seed：要用 {MIN_SEEDS} 個 seed（`--seeds {SUGGESTED_SEEDS}`）"
            "重跑再定案"
        )
    others = [metric for metric in experiment.metrics if metric != primary]
    better = [TITLES[metric] for metric in others if side(metric, compared[metric]) > 0]
    worse = [TITLES[metric] for metric in others if side(metric, compared[metric]) < 0]
    lines.append(
        f"- 參考欄（{alternative} 對 {default}）：較好 {'、'.join(better) or '無'}；"
        f"較差 {'、'.join(worse) or '無'}"
    )
    return lines


def _seed_lines(experiment: TargetExperiment, daily: pd.DataFrame) -> list[str]:
    primary = experiment.primary
    means = per_seed(daily, primary)
    seeds = list(means.columns)
    lines = [
        "",
        f"## 各 seed（{TITLES[primary]}）",
        "",
        "| 組 | " + " | ".join(f"seed {seed}" for seed in seeds) + " |",
        "| --- |" + " --- |" * len(seeds),
    ]
    for arm in experiment.arms:
        cells = [_cell(float(value)) for value in means.loc[arm.name].to_numpy(dtype=float)]
        lines.append(f"| {arm.name} | " + " | ".join(cells) + " |")
    return lines


def _fold_lines(experiment: TargetExperiment, compared: Mapping[str, Comparison]) -> list[str]:
    primary = experiment.primary
    default, alternative = (arm.name for arm in experiment.arms)
    lines = [
        "",
        f"## 每摺（{TITLES[primary]}）",
        "",
        f"| 摺 | {default} | {alternative} |",
        "| --- | --- | --- |",
    ]
    for fold, row in compared[primary].by_fold.iterrows():
        lines.append(f"| {fold} | {_cell(row['baseline'])} | {_cell(row['model'])} |")
    return lines


def experiment_report(
    experiment: TargetExperiment,
    run_key: str,
    daily: pd.DataFrame,
    folds: Sequence[Fold],
    note: str | None = None,
) -> str:
    """Markdown of each arm's seed-averaged metrics, the gaps, the verdict, seeds and folds."""
    compared = comparisons(experiment, daily, folds)
    seeds = sorted(daily["seed"].unique())
    default, alternative = (arm.name for arm in experiment.arms)
    primary = TITLES[experiment.primary]
    lower = experiment.primary in LOWER_BETTER
    indices = [fold.index for fold in folds]
    lines = [f"# 實驗：{experiment.name}（{run_key}）", ""]
    if note:
        lines += [f"> {note}", ""]
    prefix = "" if experiment.window == "base" else f"{experiment.window}-"
    lines += [f"- {arm.name}：{SETTINGS[arm.name.removeprefix(prefix)]}" for arm in experiment.arms]
    layout = experiment.arms[0].config
    span = "expanding" if layout.window == "expanding" else f"rolling {layout.train_months} 個月"
    lines.append(f"- 訓練窗：{span}，驗證段 {layout.valid_months} 個月")
    lines += [
        f"- 比較 {len(indices)} 個研究摺（摺 {indices[0]}–{indices[-1]}），seed "
        f"{', '.join(map(str, seeds))}；每組的快取 run_key 為 `{run_key}-<組名>`，"
        "seed 不是 0 時再加 `-s<seed>`",
        f"- 判定規則：{alternative} 的 {primary} 要{'低' if lower else '高'}於 {default}，"
        f"而且逐摺差的 NW t 要{'低於 −' if lower else '高於 '}{GATE_T:g} 才採用"
        f"（SPEC §7：「{experiment.reason}」；逐摺差的 NW t 門檻是實驗另加的）；其他欄只列參考。"
        "各 seed 先逐日平均，再看研究期與逐摺差",
        *(f"- {caveat}" for caveat in experiment.caveats),
        "",
        *_table_lines(experiment, compared),
        "",
        *_verdict_lines(experiment, compared, len(seeds)),
    ]
    if len(seeds) > 1:
        lines += _seed_lines(experiment, daily)
    lines += _fold_lines(experiment, compared)
    return "\n".join(lines) + "\n"
