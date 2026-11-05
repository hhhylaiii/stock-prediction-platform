"""Markdown evaluation report and the backtest_metric rows of a run's rank results (SPEC §7)."""

import calendar as months
import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import cast

import pandas as pd
from django.db import transaction

from apps.etl.upsert import upsert
from apps.research.evaluation import (
    ALERT_SHARE,
    LEAK_LIMITS,
    ImportanceDrift,
    RankReport,
)
from apps.research.models import BacktestMetric, MetricGrain, ModelObjective

METRIC_KEY = [
    "run_key",
    "model_type",
    "objective",
    "horizon",
    "metric",
    "grain",
    "period_start",
    "period_end",
    "segment",
]
RANK_METRICS = (
    "rank_ic",
    "rank_icir",
    "rank_ic_nw_t",
    "positive_fold_share",
    "beat_baseline_share",
    "beat_baseline_folds",
    "neutral_rank_ic",
    "neutral_rank_ic_nw_t",
    "rank_ic_days",
)
SAMPLES = {"all": "全部", "liquid": "成交金額前三分之一"}
LIQUIDITY_NAMES = {"1": "低", "2": "中", "3": "高"}
CHECK_NAMES = {"width_ratio": "寬度比超出 0.5 到 2", "contradiction": "上漲機率與 q50 矛盾"}


def _cell(value: float, digits: int) -> str:
    return "—" if math.isnan(value) else f"{value:.{digits}f}"


def _share(value: float) -> str:
    return "—" if math.isnan(value) else f"{value:.0%}"


def _percent(value: float, digits: int) -> str:
    return "—" if math.isnan(value) else f"{value * 100:.{digits}f}%"


def _summary(label: object, row: pd.Series) -> str:
    return (
        f"| {label} | {_cell(row['mean'], 4)} | {_cell(row['std'], 4)} "
        f"| {_cell(row['icir'], 2)} | {_cell(row['nw_t'], 2)} "
        f"| {_share(row['positive_day_share'])} | {int(row['n_days'])} |"
    )


SUMMARY_HEADER = "| IC 均值 | 標準差 | ICIR | NW t | IC 為正的天數比例 | 天數 |"
SUMMARY_RULE = "| --- | --- | --- | --- | --- | --- | --- |"


def _kind(row: pd.Series) -> str:
    return "調參" if row["tuning"] else "holdout" if row["holdout"] else "研究"


def _compared(report: RankReport) -> bool:
    return "baseline_mean" in report.by_fold


def _verdict(row: pd.Series) -> str:
    """Comparison cells of a fold: model and baseline IC on shared rows, and who won."""
    beats = row["beats"]
    won = "—" if pd.isna(beats) else "是" if beats else "否"
    return f" {_cell(row['paired_mean'], 4)} | {_cell(row['baseline_mean'], 4)} | {won} |"


def _fold_table(report: RankReport) -> list[str]:
    compared = _compared(report)
    extra = " 共同列 IC | baseline IC | 贏 baseline |" if compared else ""
    lines = [
        "### 每摺",
        "",
        f"| 摺 | 類別 | 測試期 {SUMMARY_HEADER}{extra}",
        SUMMARY_RULE + " --- | --- |" + (" --- | --- | --- |" if compared else ""),
    ]
    for fold, row in report.by_fold.iterrows():
        period = f"{row['test_start']}–{row['eval_end']}"
        tail = _verdict(row) if compared else ""
        lines.append(f"| {fold} | {_kind(row)} " + _summary(period, row) + tail)
    return [*lines, ""]


def _rank_section(report: RankReport, deciles: pd.DataFrame | None) -> list[str]:
    horizon, total, neutral = report.horizon, report.total, report.neutralized
    lines = [
        f"## 排名 {horizon} 日",
        "",
        f"- 研究期 IC 均值 {_cell(total.mean, 4)}，標準差 {_cell(total.std, 4)}，"
        f"NW t {_cell(total.nw_t, 2)}，ICIR {_cell(total.icir, 2)}，{total.n_days} 天",
        f"- IC 為正的研究摺比例：{_share(report.positive_share)}",
        f"- 中性化 IC：{_cell(neutral.mean, 4)}，NW t {_cell(neutral.nw_t, 2)}，"
        f"{neutral.n_days} 天",
    ]
    if _compared(report):
        research = sum(fold.research for fold in report.folds)
        lines.append(
            f"- 贏 baseline 的研究摺比例：{_share(report.beat_share)}"
            f"（{report.compared_folds}／{research} 摺可比）"
        )
    if report.leak:
        lines.append(
            f"- 🔴 洩漏警訊：{horizon} 日研究期 IC 均值 {total.mean:.4f} 高於 "
            f"{LEAK_LIMITS[horizon]}，先當成洩漏追查"
        )
    lines += ["", *_fold_table(report), "### 每年", "", f"| 年 {SUMMARY_HEADER}", SUMMARY_RULE]
    lines += [_summary(year, row) for year, row in report.by_year.iterrows()]
    lines += ["", "### 成交金額三分位", "", f"| 組 {SUMMARY_HEADER}", SUMMARY_RULE]
    lines += [
        _summary(LIQUIDITY_NAMES.get(str(bucket), bucket), row)
        for bucket, row in report.by_liquidity.iterrows()
    ]
    return [*lines, "", *([] if deciles is None else _decile_table(deciles))]


def _decile_table(deciles: pd.DataFrame) -> list[str]:
    lines = [
        "### 十分位（扣成本）",
        "",
        "| 樣本 | 組合 | 月報酬 | 月化超額 | 超額最大回撤 | 年化換手率（倍） | 天數 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, row in deciles.iterrows():
        sample, name = cast(tuple[str, str], key)
        lines.append(
            f"| {SAMPLES.get(sample, sample)} | {name} "
            f"| {_percent(row['monthly_return'], 2)} "
            f"| {_percent(row['monthly_excess'], 2)} "
            f"| {_percent(row['excess_drawdown'], 2)} | {_cell(row['annual_turnover'], 2)} "
            f"| {int(row['n_days'])} |"
        )
    return [*lines, ""]


def _importance_section(drift: ImportanceDrift) -> list[str]:
    lines = [
        "## 特徵重要度前 10 名",
        "",
        "| 摺 | 前 10 名（gain） | 和前一摺重疊 |",
        "| --- | --- | --- |",
    ]
    for fold, row in drift.by_fold.iterrows():
        kept = "—" if math.isnan(row["kept"]) else str(int(row["kept"]))
        lines.append(f"| {fold} | {', '.join(row['top'])} | {kept} |")
    counts = "、".join(f"{name} {count}" for name, count in drift.frequency.items())
    return [*lines, "", f"- 進前 10 名的摺數：{counts}", ""]


def _coverage_section(coverage: pd.DataFrame) -> list[str]:
    lines = [
        "## 區間覆蓋率（tech_vol_20d 五分位）",
        "",
        "| 組 | 覆蓋率 | 平均寬度 |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {'全體' if bucket == 'all' else bucket} | {_percent(row['coverage'], 1)} "
        f"| {_cell(row['width'], 4)} |"
        for bucket, row in coverage.iterrows()
    ]
    return [*lines, ""]


def _consistency_section(consistency: pd.DataFrame) -> list[str]:
    lines = [
        "## 輸出一致性",
        "",
        f"| 檢查 | 平均比例 | 最大比例 | 超過 {ALERT_SHARE:.0%} 的天數 | 天數 |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {CHECK_NAMES.get(str(check), check)} | {_percent(row['mean_share'], 1)} "
        f"| {_percent(row['max_share'], 1)} "
        f"| {int(row['days_over'])} | {int(row['n_days'])} |"
        for check, row in consistency.iterrows()
    ]
    return [*lines, ""]


def render_report(
    title: str,
    ranks: Sequence[RankReport],
    deciles: Mapping[int, pd.DataFrame] | None = None,
    importance: ImportanceDrift | None = None,
    coverage: pd.DataFrame | None = None,
    consistency: pd.DataFrame | None = None,
) -> str:
    """Markdown report of the sections given; sections left out are skipped."""
    lines = [f"# 評估報表：{title}", ""]
    for report in ranks:
        lines += _rank_section(report, (deciles or {}).get(report.horizon))
    if importance is not None:
        lines += _importance_section(importance)
    if coverage is not None:
        lines += _coverage_section(coverage)
    if consistency is not None:
        lines += _consistency_section(consistency)
    return "\n".join(lines)


def _month_end(day: date) -> date:
    return day.replace(day=months.monthrange(day.year, day.month)[1])


def rank_metrics(report: RankReport, run_key: str, model_type: str) -> pd.DataFrame:
    """backtest_metric rows of a rank report: research totals by segment, folds, years, months."""
    research = [fold for fold in report.folds if fold.research]
    rows: list[tuple[str, str, date, date, float, str]] = []
    if research:
        span = (research[0].test_start, research[-1].eval_end)
        totals = {
            "rank_ic": report.total.mean,
            "rank_icir": report.total.icir,
            "rank_ic_nw_t": report.total.nw_t,
            "positive_fold_share": report.positive_share,
            "beat_baseline_share": report.beat_share,
            "beat_baseline_folds": report.compared_folds if _compared(report) else math.nan,
            "neutral_rank_ic": report.neutralized.mean,
            "neutral_rank_ic_nw_t": report.neutralized.nw_t,
        }
        rows += [(name, MetricGrain.TOTAL, *span, value, "") for name, value in totals.items()]
        for bucket, row in report.by_liquidity.iterrows():
            segment = f"liq_t{bucket}"
            rows += [
                ("rank_ic", MetricGrain.TOTAL, *span, row["mean"], segment),
                ("rank_ic_nw_t", MetricGrain.TOTAL, *span, row["nw_t"], segment),
            ]
    means, spreads = report.by_fold["mean"], report.by_fold["nw_t"]
    for fold in research:
        period = (fold.test_start, fold.eval_end)
        rows += [
            ("rank_ic", MetricGrain.FOLD, *period, means[fold.index], ""),
            ("rank_ic_nw_t", MetricGrain.FOLD, *period, spreads[fold.index], ""),
        ]
    years = report.by_year
    for year, mean, spread in zip(years.index.tolist(), years["mean"], years["nw_t"], strict=True):
        first, last = date(year, 1, 1), date(year, 12, 31)
        rows += [
            ("rank_ic", MetricGrain.YEAR, first, last, mean, ""),
            ("rank_ic_nw_t", MetricGrain.YEAR, first, last, spread, ""),
        ]
    monthly = zip(report.monthly.index.tolist(), report.monthly, report.monthly_days, strict=True)
    for month, value, days in monthly:
        rows += [
            ("rank_ic", MetricGrain.MONTH, month, _month_end(month), value, ""),
            ("rank_ic_days", MetricGrain.MONTH, month, _month_end(month), float(days), ""),
        ]
    frame = pd.DataFrame(
        rows, columns=["metric", "grain", "period_start", "period_end", "value", "segment"]
    )
    named = frame.assign(
        run_key=run_key,
        model_type=model_type,
        objective=ModelObjective.RANK.value,
        horizon=report.horizon,
        grain=frame["grain"].astype(str),
        value=frame["value"].astype("float64"),
    )
    return named[[*METRIC_KEY, "value"]]


def write_rank_metrics(report: RankReport, run_key: str, model_type: str) -> int:
    """Replace a run model's RANK_METRICS rows at the report's horizon; other metric names stay."""
    rows = rank_metrics(report, run_key, model_type)
    with transaction.atomic():
        BacktestMetric.objects.filter(
            run_key=run_key,
            model_type=model_type,
            objective=ModelObjective.RANK,
            horizon=report.horizon,
            metric__in=RANK_METRICS,
        ).delete()
        return upsert(BacktestMetric, rows, unique_fields=METRIC_KEY, update_fields=["value"])
