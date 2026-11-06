"""Research-period evaluation of return and interval runs against their SPEC §7 baselines."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

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
from apps.research.metrics.stats import newey_west_t, paired_nw_t
from apps.research.models import HoldoutScope
from apps.research.prediction_cache import (
    MissingFoldsError,
    fold_notes,
    load_predictions,
    missing_folds,
)
from apps.research.training import KEYS, load_labels
from apps.research.walkforward import Fold

MODEL = lgbm_runs.MODEL
RETURN, INTERVAL = HoldoutScope.RETURN, HoldoutScope.INTERVAL
SCOPES = (RETURN, INTERVAL)
LABEL = "fwd_ret_open"
VOL_LABEL = "fwd_vol_20d"
UP_LABEL = "fwd_up_5pct"
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
    """A metric of the model and a baseline on shared rows, with the fold-paired NW t."""

    model: float
    baseline: float
    nw_t: float
    by_fold: pd.DataFrame


@dataclass(frozen=True)
class ReturnEvaluation:
    mae_model: float
    mae: dict[str, Comparison]
    rmse: dict[str, float]
    direction: float
    direction_t: float


@dataclass(frozen=True)
class IntervalEvaluation:
    pinball: Comparison
    history_pinball: float
    coverage: float
    coverage_by_fold: pd.Series
    coverage_table: pd.DataFrame
    consistency: pd.DataFrame
    vol_mae: Comparison
    qlike: Comparison
    auc: float
    logistic_auc: float
    brier: Comparison
    ece: float
    ece_raw: float


def scope_of(run_key: str) -> str:
    """The target scope a run_key names; rank and unknown prefixes are refused."""
    for scope in SCOPES:
        if run_key.startswith(f"{scope}-"):
            check_run_key(scope, run_key)
            return scope
    raise ValueError(f"a target run_key must start with return- or interval-, got {run_key!r}")


def stale_folds(scope: str, run_key: str, folds: Sequence[Fold], directory: Path) -> list[int]:
    """Folds whose LightGBM or baseline cache was fit on features or labels other than today's."""
    stale = []
    for fold in folds:
        for model, current in (
            (MODEL, lgbm_runs.input_notes(scope, fold, directory)),
            (BASELINE, baseline_runs.input_notes(scope, fold, directory)),
        ):
            notes = fold_notes(run_key, model, fold.index)
            if {key: notes.get(key) for key in current} != current:
                stale.append(fold.index)
                break
    return stale


def target_frame(run_key: str, folds: Sequence[Fold], directory: Path) -> pd.DataFrame:
    """Both caches of the research folds up to each eval_end, with labels and tech_vol_20d."""
    scope = scope_of(run_key)
    kept = [fold for fold in folds if fold.research]
    keys = [*KEYS, "in_universe", "fold"]
    boosted = load_predictions(run_key, MODEL, kept)
    rules = load_predictions(run_key, BASELINE, kept).drop(columns=["in_universe"])
    frame = boosted.merge(rules, on=[*KEYS, "fold"], how="left", validate="one_to_one")
    frame = frame[frame["date"] <= frame["fold"].map({fold.index: fold.eval_end for fold in kept})]
    start, end = min(fold.test_start for fold in kept), max(fold.eval_end for fold in kept)
    labels = load_labels(start, end, HORIZON[scope], LABEL_FIELDS[scope])
    frame = frame.merge(labels, on=KEYS, how="left", validate="one_to_one")
    if scope == INTERVAL:
        vols = read_columns(start, end, directory, [VOLATILITY])
        frame = frame.merge(vols, on=KEYS, how="left", validate="one_to_one")
    ordered = [*keys, *(column for column in frame.columns if column not in keys)]
    return frame[ordered].reset_index(drop=True)


def compare(
    rows: pd.DataFrame,
    folds: Sequence[Fold],
    metric: DailyMetric,
    label: str,
    model: str,
    baseline: str,
) -> Comparison:
    """Mean daily metric of both columns on their shared rows and the fold-paired NW t."""
    shared = shared_rows(rows, [label, model, baseline])
    return paired(metric(shared, label, model), metric(shared, label, baseline), folds)


def paired(first: pd.Series, second: pd.Series, folds: Sequence[Fold]) -> Comparison:
    """Two daily metric series, their research fold means and the NW t of the fold gaps."""
    kept = [fold for fold in folds if fold.research]
    by_fold = pd.DataFrame(
        {
            "model": mean_by_fold(first, kept)["mean"],
            "baseline": mean_by_fold(second, kept)["mean"],
        }
    )
    nw_t = paired_nw_t(by_fold["model"], by_fold["baseline"], FOLD_LAG)
    return Comparison(float(first.mean()), float(second.mean()), nw_t, by_fold)


def beats(compared: Comparison) -> bool:
    """A lower-is-better gate: the model is lower and the fold-paired NW t is below −GATE_T."""
    return compared.model < compared.baseline and compared.nw_t < -GATE_T


def coverage_outside(by_fold: pd.Series) -> list[int]:
    """Folds whose coverage leaves COVERAGE_FOLD or was never measured."""
    low, high = COVERAGE_FOLD
    outside = by_fold[~by_fold.between(low, high)]
    return [int(fold) for fold in outside.index.to_numpy()]


def coverage_passes(total: float, by_fold: pd.Series) -> bool:
    """Total coverage inside COVERAGE_TOTAL and every fold inside COVERAGE_FOLD."""
    low, high = COVERAGE_TOTAL
    return bool(low <= total <= high) and not coverage_outside(by_fold)


def evaluate_return(frame: pd.DataFrame, folds: Sequence[Fold]) -> ReturnEvaluation:
    """MAE and RMSE against both baselines, and the direction hit rate against even odds."""
    rows = research_rows(tradable_rows(frame), folds)
    mae = {
        name: compare(rows, folds, daily_absolute_error, LABEL, RETURN_COLUMN, column)
        for name, column in RETURN_BASELINES.items()
    }
    rmse = {
        name: math.sqrt(
            daily_squared_error(shared_rows(rows, [LABEL, column]), LABEL, column).mean()
        )
        for name, column in {MODEL: RETURN_COLUMN, **RETURN_BASELINES}.items()
    }
    hits = daily_direction_hits(rows, LABEL, RETURN_COLUMN)
    return ReturnEvaluation(
        mae_model=float(daily_absolute_error(rows, LABEL, RETURN_COLUMN).mean()),
        mae=mae,
        rmse=rmse,
        direction=float(hits.mean()),
        direction_t=newey_west_t(hits - EVEN, RETURN_HORIZON),
    )


def _ece(rows: pd.DataFrame, column: str) -> float:
    scored = rows.dropna(subset=[column])
    return expected_calibration_error(reliability_table(scored, UP_LABEL, column))


def evaluate_interval(frame: pd.DataFrame, folds: Sequence[Fold]) -> IntervalEvaluation:
    """Quantile, volatility and probability models against their SPEC §7 baselines."""
    rows = research_rows(tradable_rows(frame), folds)
    model, band, history = (quantile_columns(prefix) for prefix in (MODEL, "volband", "histq"))
    quantiled = shared_rows(rows, [LABEL, *model.values(), *band.values()])
    pinball = paired(
        daily_pinball(quantiled, LABEL, model), daily_pinball(quantiled, LABEL, band), folds
    )
    covered = daily_coverage(rows, LABEL, model[0.1], model[0.9])
    kept = [fold for fold in folds if fold.research]
    by_fold = mean_by_fold(covered, kept)["mean"]
    vol = f"{MODEL}_vol_h{VOL_DAYS}"
    prob = f"{MODEL}_prob_h{VOL_DAYS}"
    up = rows.dropna(subset=[UP_LABEL])
    up = up.assign(**{UP_LABEL: up[UP_LABEL].astype(int)})
    return IntervalEvaluation(
        pinball=pinball,
        history_pinball=float(daily_pinball(rows, LABEL, history).mean()),
        coverage=float(covered.mean()),
        coverage_by_fold=by_fold,
        coverage_table=coverage_report(frame, LABEL, model[0.1], model[0.9], folds),
        consistency=consistency_report(frame, model[0.1], model[0.5], model[0.9], vol, prob, folds),
        vol_mae=compare(rows, folds, daily_absolute_error, VOL_LABEL, vol, VOLATILITY),
        qlike=compare(rows, folds, daily_qlike, VOL_LABEL, vol, VOLATILITY),
        auc=float(daily_auc(up, UP_LABEL, prob).mean()),
        logistic_auc=float(daily_auc(up, UP_LABEL, f"logistic_h{VOL_DAYS}").mean()),
        brier=compare(up, folds, daily_brier, UP_LABEL, prob, f"logistic_h{VOL_DAYS}"),
        ece=_ece(up, prob),
        ece_raw=_ece(up, f"{MODEL}_prob_raw_h{VOL_DAYS}"),
    )


Evaluation = ReturnEvaluation | IntervalEvaluation


def evaluate_run(run_key: str, folds: Sequence[Fold], directory: Path) -> Evaluation:
    """A target run's research folds; missing caches are named first, then stale ones refused."""
    scope = scope_of(run_key)
    kept = [fold for fold in folds if fold.research]
    if not kept:
        raise ValueError("no research folds to evaluate")
    for model in (BASELINE, MODEL):
        missing = missing_folds(run_key, model, kept)
        if missing:
            raise MissingFoldsError(run_key, model, missing)
    stale = stale_folds(scope, run_key, kept, directory)
    if stale:
        raise ValueError(
            f"folds {stale} of {run_key} cache models fit on features or labels that are stale; "
            "rerun train for the baselines and LightGBM"
        )
    frame = target_frame(run_key, kept, directory)
    if scope == RETURN:
        return evaluate_return(frame, kept)
    return evaluate_interval(frame, kept)


def _number(value: float, digits: int = 4) -> str:
    return "—" if math.isnan(value) else f"{value:.{digits}f}"


def _verdict(passed: bool) -> str:
    return "通過" if passed else "未過"


def _gate(text: str, compared: Comparison) -> str:
    return (
        f"- {text}：模型 {_number(compared.model, 5)}、baseline {_number(compared.baseline, 5)}，"
        f"逐摺差 NW t {_number(compared.nw_t, 2)}（需 < −{GATE_T:g}）：{_verdict(beats(compared))}"
    )


def _fold_table(compared: Comparison, name: str) -> list[str]:
    lines = ["", f"| 摺 | 模型 {name} | baseline {name} |", "| --- | --- | --- |"]
    for fold, row in compared.by_fold.iterrows():
        lines.append(f"| {fold} | {_number(row['model'], 5)} | {_number(row['baseline'], 5)} |")
    return lines


def _return_lines(evaluated: ReturnEvaluation) -> list[str]:
    lines = ["## 絕對報酬 5 日", ""]
    lines += [
        _gate(f"MAE 低於{BASELINE_NAMES[name]}", compared)
        for name, compared in evaluated.mae.items()
    ]
    rmse = "、".join(f"{name} {_number(value, 5)}" for name, value in evaluated.rmse.items())
    lines += [
        f"- RMSE：{rmse}",
        f"- 方向準確率 {_number(evaluated.direction)}，對 50% 的 NW t "
        f"{_number(evaluated.direction_t, 2)}",
    ]
    return lines + _fold_table(evaluated.mae["zero"], "MAE")


def _coverage_lines(evaluated: IntervalEvaluation) -> list[str]:
    low, high = COVERAGE_TOTAL
    fold_low, fold_high = COVERAGE_FOLD
    outside = coverage_outside(evaluated.coverage_by_fold)
    folds = "無" if not outside else "、".join(map(str, outside))
    lines = [
        f"- 總覆蓋率 {_number(evaluated.coverage)}（需在 {low:.0%} 到 {high:.0%}），"
        f"每摺覆蓋率需在 {fold_low:.0%} 到 {fold_high:.0%}，超出的摺：{folds}："
        f"{_verdict(coverage_passes(evaluated.coverage, evaluated.coverage_by_fold))}",
        "",
        "| tech_vol_20d 五分位 | 覆蓋率 | 平均寬度 |",
        "| --- | --- | --- |",
    ]
    for bucket, row in evaluated.coverage_table.iterrows():
        lines.append(f"| {bucket} | {_number(row['coverage'])} | {_number(row['width'])} |")
    return lines


def _interval_lines(evaluated: IntervalEvaluation) -> list[str]:
    lines = [
        "## 區間 20 日",
        "",
        _gate("pinball 低於波動率區間", evaluated.pinball),
        f"- 歷史分位數（sanity 下限）pinball {_number(evaluated.history_pinball, 5)}",
        *_coverage_lines(evaluated),
        *_fold_table(evaluated.pinball, "pinball"),
        "",
        "## 波動率",
        "",
        _gate("MAE 低於 tech_vol_20d", evaluated.vol_mae),
        _gate("QLIKE 低於 tech_vol_20d", evaluated.qlike),
        "",
        "## 上漲機率",
        "",
        f"- 日均 AUC：模型 {_number(evaluated.auc)}、logistic {_number(evaluated.logistic_auc)}",
        f"- Brier 不高於 logistic：模型 {_number(evaluated.brier.model, 5)}、logistic "
        f"{_number(evaluated.brier.baseline, 5)}，逐摺差 NW t {_number(evaluated.brier.nw_t, 2)}："
        f"{_verdict(evaluated.brier.model <= evaluated.brier.baseline)}",
        f"- ECE：校準後 {_number(evaluated.ece)}、校準前 {_number(evaluated.ece_raw)}",
        "",
        "## 輸出一致性",
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


def report(run_key: str, evaluated: Evaluation) -> str:
    """Markdown of a target run's research-period metrics, baselines and SPEC §7 gates."""
    body = (
        _return_lines(evaluated)
        if isinstance(evaluated, ReturnEvaluation)
        else _interval_lines(evaluated)
    )
    lines = [
        f"# 目標評估：{run_key}",
        "",
        "研究摺的 test_start..eval_end，只算 universe 內 t+1 可交易的列。",
        "",
        *body,
    ]
    return "\n".join(lines) + "\n"
