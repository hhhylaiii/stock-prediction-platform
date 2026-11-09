"""Evaluation of return and interval runs against their SPEC §7 baselines over research folds."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from apps.market.calendar import TradingDays
from apps.research import baseline_runs, lgbm_runs
from apps.research.baseline_runs import BASELINE, RETURN_HORIZON, check_run_key
from apps.research.baselines import QUANTILES
from apps.research.evaluation import consistency_report, coverage_report, tradable_rows
from apps.research.features.store import read_columns
from apps.research.labels import VOL_DAYS
from apps.research.metrics.forecast import (
    daily_absolute_error,
    daily_auc,
    daily_brier,
    daily_coverage,
    daily_direction_hits,
    daily_pinball,
    daily_qlike,
    daily_squared_error,
    expected_calibration_error,
    mean_by_fold,
    reliability_table,
    research_rows,
    shared_rows,
)
from apps.research.metrics.ic import research_ic
from apps.research.metrics.stats import newey_west_t, paired_nw_t
from apps.research.models import HoldoutScope, ModelObjective, ModelRun
from apps.research.prediction_cache import (
    MissingFoldsError,
    fold_notes,
    load_predictions,
    missing_folds,
)
from apps.research.relative_up import RELATIVE_LABEL, relative_up
from apps.research.target_models import ECE_LIMIT, TargetVariant
from apps.research.training import KEYS, load_labels
from apps.research.walkforward import Fold

MODEL = lgbm_runs.MODEL
RETURN, INTERVAL = HoldoutScope.RETURN, HoldoutScope.INTERVAL
SCOPES = (RETURN, INTERVAL)
LABEL = "fwd_ret_open"
VOL_LABEL = "fwd_vol_20d"
UP_LABEL = "fwd_up_5pct"
RELATIVE_NOTE = (
    "上漲機率的 label 是相對大盤版：20 日報酬減 TAIEX_TR（含息報酬指數）同期報酬後是否大於 5%；"
    "LightGBM 與 logistic baseline 都在這個 label 上訓練與評估；"
    "輸出一致性一節仍拿這個機率對照絕對報酬的 q50 是否大於 5%，矛盾比例會偏高"
)
VOLATILITY = "tech_vol_20d"
LABEL_FIELDS: dict[str, tuple[str, ...]] = {
    RETURN: (LABEL, "untradable_t1"),
    INTERVAL: (LABEL, VOL_LABEL, UP_LABEL, "untradable_t1"),
}
HORIZON: dict[str, int] = {RETURN: RETURN_HORIZON, INTERVAL: VOL_DAYS}
FOLD_LAG = 1
GATE_T = 2.0
COVERAGE_TOTAL = (0.75, 0.85)
COVERAGE_FOLD = (0.70, 0.90)
EVEN = 0.5
RETURN_COLUMN = f"{MODEL}_ret_h{RETURN_HORIZON}"
RETURN_BASELINES = {"zero": f"zero_h{RETURN_HORIZON}", "trailing": f"trailing_h{RETURN_HORIZON}"}
BASELINE_NAMES = {"zero": "零預測", "trailing": "過去 60 日平均"}

DailyMetric = Callable[[pd.DataFrame, str, str], pd.Series]


def quantile_columns(prefix: str) -> dict[float, str]:
    return {alpha: f"{prefix}_{name}_h{VOL_DAYS}" for name, alpha in QUANTILES.items()}


@dataclass(frozen=True)
class Comparison:
    """A metric of the model and a baseline on shared rows: research totals, folds, NW t."""

    model: float
    baseline: float
    nw_t: float
    by_fold: pd.DataFrame
    days: int = 0


@dataclass(frozen=True)
class ReturnEvaluation:
    mae: dict[str, Comparison]
    rmse: dict[str, tuple[float, float]]
    direction: float
    direction_t: float


@dataclass(frozen=True)
class IntervalEvaluation:
    pinball: Comparison
    history: Comparison
    crossed: int
    coverage: float
    coverage_by_fold: pd.DataFrame
    coverage_table: pd.DataFrame
    consistency: pd.DataFrame
    vol_mae: Comparison
    qlike: Comparison
    auc: Comparison
    brier: Comparison
    reliability: pd.DataFrame
    ece: float
    ece_raw: float
    coverage_days: int = 0


def scope_of(run_key: str) -> str:
    """The target scope a run_key names; rank and unknown prefixes are refused."""
    for scope in SCOPES:
        if run_key.startswith(f"{scope}-"):
            check_run_key(scope, run_key)
            return scope
    raise ValueError(f"a target run_key must start with return- or interval-, got {run_key!r}")


def stale_folds(
    scope: str, run_key: str, folds: Sequence[Fold], directory: Path, relative: bool = False
) -> list[int]:
    """Folds whose LightGBM or baseline cache was fit on features or labels other than today's."""
    target = TargetVariant(relative_up=relative)
    stale = []
    for fold in folds:
        for model, current in (
            (MODEL, lgbm_runs.input_notes(scope, fold, directory, target)),
            (BASELINE, baseline_runs.input_notes(scope, fold, directory, relative)),
        ):
            notes = fold_notes(run_key, model, fold.index)
            if {key: notes.get(key) for key in current} != current:
                stale.append(fold.index)
                break
    return stale


def target_frame(
    run_key: str, folds: Sequence[Fold], directory: Path, relative: bool = False
) -> pd.DataFrame:
    """Both caches of the tuning and research folds up to each eval_end, with labels as fit."""
    scope = scope_of(run_key)
    kept = [fold for fold in folds if not fold.holdout]
    keys = [*KEYS, "in_universe", "fold"]
    boosted = load_predictions(run_key, MODEL, kept)
    rules = load_predictions(run_key, BASELINE, kept).drop(columns=["in_universe"])
    frame = boosted.merge(rules, on=[*KEYS, "fold"], how="left", validate="one_to_one")
    frame = frame[frame["date"] <= frame["fold"].map({fold.index: fold.eval_end for fold in kept})]
    start, end = min(fold.test_start for fold in kept), max(fold.eval_end for fold in kept)
    labels = load_labels(start, end, HORIZON[scope], LABEL_FIELDS[scope])
    if relative:
        labels = labels.assign(**{UP_LABEL: relative_up(labels, TradingDays.from_db())})
    frame = frame.merge(labels, on=KEYS, how="left", validate="one_to_one")
    if scope == INTERVAL:
        vols = read_columns(start, end, directory, [VOLATILITY])
        frame = frame.merge(vols, on=KEYS, how="left", validate="one_to_one")
    ordered = [*keys, *(column for column in frame.columns if column not in keys)]
    return frame[ordered].reset_index(drop=True)


def paired(first: pd.Series, second: pd.Series, folds: Sequence[Fold]) -> Comparison:
    """Research means of two daily metrics, every fold's means and the research fold-gap NW t."""
    model, baseline = mean_by_fold(first, folds), mean_by_fold(second, folds)
    by_fold = pd.DataFrame(
        {"model": model["mean"], "baseline": baseline["mean"], "tuning": model["tuning"]}
    )
    research = by_fold[~by_fold["tuning"].astype(bool)]
    nw_t = paired_nw_t(research["model"], research["baseline"], FOLD_LAG)
    return Comparison(
        float(research_ic(first, folds).mean()),
        float(research_ic(second, folds).mean()),
        nw_t,
        by_fold,
        int(research_ic(first, folds).notna().sum()),
    )


def compare(
    rows: pd.DataFrame,
    folds: Sequence[Fold],
    metric: DailyMetric,
    label: str,
    model: str,
    baseline: str,
) -> Comparison:
    """A daily metric of both columns on their shared rows, paired over folds."""
    shared = shared_rows(rows, [label, model, baseline])
    return paired(metric(shared, label, model), metric(shared, label, baseline), folds)


def beats(compared: Comparison) -> bool:
    """A lower-is-better gate: the model is lower and the fold-paired NW t is below −GATE_T."""
    return compared.model < compared.baseline and compared.nw_t < -GATE_T


def no_worse(compared: Comparison) -> bool:
    """A gate the model passes by not being above its baseline; undefined values fail."""
    return compared.model <= compared.baseline


def coverage_outside(by_fold: pd.Series) -> list[int]:
    """Folds whose coverage leaves COVERAGE_FOLD or was never measured."""
    low, high = COVERAGE_FOLD
    outside = by_fold[~by_fold.between(low, high)]
    return [int(fold) for fold in outside.index.to_numpy()]


def coverage_passes(total: float, by_fold: pd.Series) -> bool:
    """Total coverage inside COVERAGE_TOTAL and every fold inside COVERAGE_FOLD."""
    low, high = COVERAGE_TOTAL
    return bool(low <= total <= high) and not coverage_outside(by_fold)


def _rmse(rows: pd.DataFrame, column: str) -> float:
    return math.sqrt(float(daily_squared_error(rows, LABEL, column).mean()))


def evaluate_return(frame: pd.DataFrame, folds: Sequence[Fold]) -> ReturnEvaluation:
    """MAE and RMSE against both baselines, and the direction hit rate against even odds."""
    rows = tradable_rows(frame)
    research = research_rows(rows, folds)
    mae, rmse = {}, {}
    for name, column in RETURN_BASELINES.items():
        mae[name] = compare(rows, folds, daily_absolute_error, LABEL, RETURN_COLUMN, column)
        shared = shared_rows(research, [LABEL, RETURN_COLUMN, column])
        rmse[name] = (_rmse(shared, RETURN_COLUMN), _rmse(shared, column))
    hits = daily_direction_hits(research, LABEL, RETURN_COLUMN)
    return ReturnEvaluation(
        mae=mae,
        rmse=rmse,
        direction=float(hits.mean()),
        direction_t=newey_west_t(hits - EVEN, RETURN_HORIZON),
    )


def _pinball(
    rows: pd.DataFrame, first: dict[float, str], second: dict[float, str], folds: Sequence[Fold]
) -> Comparison:
    shared = shared_rows(rows, [LABEL, *first.values(), *second.values()])
    return paired(daily_pinball(shared, LABEL, first), daily_pinball(shared, LABEL, second), folds)


def _up_rows(rows: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    shared = shared_rows(rows, [UP_LABEL, *columns])
    return shared.assign(**{UP_LABEL: shared[UP_LABEL].astype(int)})


def _ece(rows: pd.DataFrame, column: str) -> float:
    return expected_calibration_error(reliability_table(_up_rows(rows, [column]), UP_LABEL, column))


def evaluate_interval(frame: pd.DataFrame, folds: Sequence[Fold]) -> IntervalEvaluation:
    """Quantile, volatility and probability models against their SPEC §7 baselines."""
    rows = tradable_rows(frame)
    research = research_rows(rows, folds)
    model, band, history = (quantile_columns(prefix) for prefix in (MODEL, "volband", "histq"))
    covered = daily_coverage(rows, LABEL, model[0.1], model[0.9])
    by_fold = mean_by_fold(covered, folds)[["mean", "tuning"]]
    q10, q50, q90 = (research[model[alpha]] for alpha in (0.1, 0.5, 0.9))
    vol, prob = f"{MODEL}_vol_h{VOL_DAYS}", f"{MODEL}_prob_h{VOL_DAYS}"
    logistic = f"logistic_h{VOL_DAYS}"
    positive = rows[(rows[[VOL_LABEL, vol, VOLATILITY]] > 0).all(axis=1)]
    scored = _up_rows(rows, [prob, logistic])
    return IntervalEvaluation(
        pinball=_pinball(rows, model, band, folds),
        history=_pinball(rows, model, history, folds),
        crossed=int(((q10 > q50) | (q50 > q90)).sum()),
        coverage=float(research_ic(covered, folds).mean()),
        coverage_by_fold=by_fold,
        coverage_table=coverage_report(frame, LABEL, model[0.1], model[0.9], folds),
        consistency=consistency_report(frame, model[0.1], model[0.5], model[0.9], vol, prob, folds),
        vol_mae=compare(rows, folds, daily_absolute_error, VOL_LABEL, vol, VOLATILITY),
        qlike=compare(positive, folds, daily_qlike, VOL_LABEL, vol, VOLATILITY),
        auc=compare(scored, folds, daily_auc, UP_LABEL, prob, logistic),
        brier=compare(scored, folds, daily_brier, UP_LABEL, prob, logistic),
        reliability=reliability_table(_up_rows(research, [prob]), UP_LABEL, prob),
        ece=_ece(research, prob),
        ece_raw=_ece(research, f"{MODEL}_prob_raw_h{VOL_DAYS}"),
        coverage_days=int(research_ic(covered, folds).notna().sum()),
    )


Evaluation = ReturnEvaluation | IntervalEvaluation


def check_up_label(run_key: str, relative: bool, folds: Sequence[Fold]) -> None:
    """Refuse folds whose probability models were fit on the other up label."""
    expected = RELATIVE_LABEL if relative else UP_LABEL
    suffixes = tuple(f".f{fold.index:03d}" for fold in folds)
    runs = ModelRun.objects.filter(run_key=run_key, objective=ModelObjective.CLASSIFICATION)
    labels = sorted(
        {run.params.get("label") for run in runs if run.run_id.endswith(suffixes)} - {expected}
    )
    if labels:
        raise ValueError(
            f"{run_key}'s probability models were fit on {', '.join(map(str, labels))}, "
            f"not {expected}; evaluate it with the design it was trained on"
        )


def evaluate_run(
    run_key: str, folds: Sequence[Fold], directory: Path, relative: bool = False
) -> Evaluation:
    """A target run's tuning and research folds; missing caches first, then stale ones refused."""
    scope = scope_of(run_key)
    baseline_runs.relative_flag(scope, TargetVariant(relative_up=relative))
    if scope == INTERVAL:
        check_up_label(run_key, relative, folds)
    kept = [fold for fold in folds if not fold.holdout]
    if not any(fold.research for fold in kept):
        raise ValueError("no research folds to evaluate")
    for model in (BASELINE, MODEL):
        missing = missing_folds(run_key, model, kept)
        if missing:
            raise MissingFoldsError(run_key, model, missing)
    stale = stale_folds(scope, run_key, kept, directory, relative)
    if stale:
        raise ValueError(
            f"folds {stale} of {run_key} cache models fit on features or labels that are stale; "
            "rerun train for the baselines and LightGBM"
        )
    frame = target_frame(run_key, kept, directory, relative)
    if scope == RETURN:
        return evaluate_return(frame, kept)
    return evaluate_interval(frame, kept)


def _number(value: float, digits: int = 4) -> str:
    return "—" if math.isnan(value) else f"{value:.{digits}f}"


def _verdict(passed: bool) -> str:
    return "通過" if passed else "未過"


def _gate(text: str, compared: Comparison) -> str:
    """A line for a gate that needs the model lower with a fold-gap NW t below −GATE_T."""
    return (
        f"- {text}：模型 {_number(compared.model, 5)}、baseline {_number(compared.baseline, 5)}，"
        f"逐摺差 NW t {_number(compared.nw_t, 2)}（需 < −{GATE_T:g}）：{_verdict(beats(compared))}"
    )


def _bound(text: str, compared: Comparison) -> str:
    """A line for a gate the model passes by not being above its baseline."""
    return (
        f"- {text}：模型 {_number(compared.model, 5)}、baseline {_number(compared.baseline, 5)}，"
        f"逐摺差 NW t {_number(compared.nw_t, 2)}（參考）：{_verdict(no_worse(compared))}"
    )


def _reference(text: str, compared: Comparison) -> str:
    """A line for a comparison SPEC §7 reports without a gate."""
    return (
        f"- {text}（參考）：模型 {_number(compared.model, 5)}、"
        f"baseline {_number(compared.baseline, 5)}，逐摺差 NW t {_number(compared.nw_t, 2)}"
    )


def _fold_name(fold: object, tuning: bool) -> str:
    return f"{fold}（調參）" if tuning else str(fold)


def _fold_table(compared: Comparison, name: str) -> list[str]:
    lines = ["", f"| 摺 | 模型 {name} | baseline {name} |", "| --- | --- | --- |"]
    for fold, row in compared.by_fold.iterrows():
        lines.append(
            f"| {_fold_name(fold, bool(row['tuning']))} | {_number(row['model'], 5)} | "
            f"{_number(row['baseline'], 5)} |"
        )
    return lines


def _return_lines(evaluated: ReturnEvaluation) -> list[str]:
    lines = ["## 絕對報酬 5 日", ""]
    for name, compared in evaluated.mae.items():
        lines.append(_gate(f"MAE 低於{BASELINE_NAMES[name]}", compared))
    for name, (model, baseline) in evaluated.rmse.items():
        lines.append(
            f"- RMSE 對{BASELINE_NAMES[name]}（同列）：模型 {_number(model, 5)}、"
            f"baseline {_number(baseline, 5)}"
        )
    lines.append(
        f"- 方向準確率 {_number(evaluated.direction)}，對 50% 的 NW t "
        f"{_number(evaluated.direction_t, 2)}"
    )
    for name, compared in evaluated.mae.items():
        lines += _fold_table(compared, f"MAE（對{BASELINE_NAMES[name]}）")
    return lines


def _coverage_lines(evaluated: IntervalEvaluation) -> list[str]:
    low, high = COVERAGE_TOTAL
    fold_low, fold_high = COVERAGE_FOLD
    research = evaluated.coverage_by_fold[~evaluated.coverage_by_fold["tuning"].astype(bool)]
    outside = coverage_outside(research["mean"])
    passed = coverage_passes(evaluated.coverage, research["mean"])
    lines = [
        f"- 總覆蓋率 {_number(evaluated.coverage)}（需在 {low:.0%} 到 {high:.0%}），"
        f"每摺覆蓋率需在 {fold_low:.0%} 到 {fold_high:.0%}，超出的研究摺："
        f"{'無' if not outside else '、'.join(map(str, outside))}：{_verdict(passed)}",
        "",
        "| 摺 | 覆蓋率 |",
        "| --- | --- |",
    ]
    for fold, row in evaluated.coverage_by_fold.iterrows():
        lines.append(f"| {_fold_name(fold, bool(row['tuning']))} | {_number(row['mean'])} |")
    lines += ["", "| tech_vol_20d 五分位 | 覆蓋率 | 平均寬度 |", "| --- | --- | --- |"]
    for bucket, row in evaluated.coverage_table.iterrows():
        lines.append(f"| {bucket} | {_number(row['coverage'])} | {_number(row['width'])} |")
    return lines


def _probability_lines(evaluated: IntervalEvaluation) -> list[str]:
    lines = [
        "## 上漲機率",
        "",
        _reference("日均 AUC 對 logistic", evaluated.auc),
        _bound("Brier 不高於 logistic", evaluated.brier),
        f"- ECE 低於 {ECE_LIMIT:g}：校準後 {_number(evaluated.ece)}（校準前 "
        f"{_number(evaluated.ece_raw)}）：{_verdict(evaluated.ece < ECE_LIMIT)}",
        *_fold_table(evaluated.auc, "AUC"),
        *_fold_table(evaluated.brier, "Brier"),
        "",
        "| 機率區間 | 平均機率 | 實際比例 | 列數 |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in evaluated.reliability.iterrows():
        lines.append(
            f"| {row['lower']:.1f}–{row['upper']:.1f} | {_number(row['mean_prob'])} | "
            f"{_number(row['frac_positive'])} | {int(row['n_rows'])} |"
        )
    return lines


def _interval_lines(evaluated: IntervalEvaluation) -> list[str]:
    history = evaluated.history
    lines = [
        "## 區間 20 日",
        "",
        _gate("pinball 低於波動率區間", evaluated.pinball),
        f"- pinball 低於歷史分位數（sanity 下限，同列）：模型 {_number(history.model, 5)}、"
        f"歷史分位數 {_number(history.baseline, 5)}：{_verdict(history.model < history.baseline)}",
        f"- 分位數交叉（q10 > q50 或 q50 > q90）的研究列數 {evaluated.crossed}："
        f"{_verdict(evaluated.crossed == 0)}",
        *_coverage_lines(evaluated),
        *_fold_table(evaluated.pinball, "pinball（對波動率區間）"),
        *_fold_table(history, "pinball（對歷史分位數）"),
        "",
        "## 波動率",
        "",
        _reference("MAE 對 tech_vol_20d", evaluated.vol_mae),
        _bound("QLIKE 不高於 tech_vol_20d", evaluated.qlike),
        *_fold_table(evaluated.vol_mae, "MAE"),
        *_fold_table(evaluated.qlike, "QLIKE"),
        "",
        *_probability_lines(evaluated),
        "",
        "## 輸出一致性",
        "",
        "universe 內全部列（含 t+1 不可交易的列），研究摺。",
        "",
        "| 項目 | 平均比例 | 最高比例 | 超過 10% 的天數 | 天數 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, row in evaluated.consistency.iterrows():
        lines.append(
            f"| {name} | {_number(row['mean_share'])} | {_number(row['max_share'])} | "
            f"{int(row['days_over'])} | {int(row['n_days'])} |"
        )
    return lines


INTRO = (
    "指標只算研究摺 test_start..eval_end 內、universe 內 t+1 可交易的列；"
    "調參摺列在每摺表並標註，不計入總體指標與關卡。"
)


def report(run_key: str, evaluated: Evaluation, relative: bool = False) -> str:
    """Markdown of a target run's research-period metrics, baselines and SPEC §7 gates."""
    if isinstance(evaluated, ReturnEvaluation):
        body = _return_lines(evaluated)
    else:
        body = _interval_lines(evaluated)
    label = [RELATIVE_NOTE, ""] if relative else []
    return "\n".join([f"# 目標評估：{run_key}", "", INTRO, "", *label, *body]) + "\n"
