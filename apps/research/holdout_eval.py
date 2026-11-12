"""Holdout verdicts on criteria fixed before the run: gate (3) and degradation flags (SPEC §7).

The criteria are the coordinator's of 2026-11-11, written down before any holdout result.
"""

import hashlib
import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path

import pandas as pd

from apps.market.calendar import TradingDays
from apps.research import baseline_runs, lgbm_runs, rank_eval, target_eval
from apps.research.eval_report import write_rank_metrics
from apps.research.evaluation import RankReport, rank_report, tradable_rows
from apps.research.gates import COVERAGE as COVERAGE_GATE
from apps.research.holdout import (
    DEGRADED,
    FAILED,
    PASSED,
    HoldoutError,
    Reported,
    report_marker,
    reported,
)
from apps.research.labels import HORIZONS
from apps.research.metrics.forecast import mean_by_fold
from apps.research.models import HoldoutScope, ModelRun
from apps.research.prediction_cache import (
    FoldMismatchError,
    MissingFoldsError,
    missing_folds,
    other_layout,
)
from apps.research.target_models import ECE_LIMIT
from apps.research.walkforward import Fold

MODEL, RIDGE = lgbm_runs.MODEL, rank_eval.RIDGE
WINDOW = 8
SPEC_CLAUSES: dict[str, tuple[int, str]] = {
    "record": (7, "研究期與最終 holdout"),
    "rank": (7, "排名：日頻 rank IC"),
    "return": (7, "絕對報酬（5 日）"),
    "interval": (7, "區間：主指標是 pinball loss"),
    "probability": (7, "機率：每日 AUC 的平均"),
    "gate": (7, "每月重訓"),
    "bagging": (7, "LightGBM seed bagging"),
    "acceptance": (9, "Phase 4 主模型"),
}
GATE_SHARE = rank_eval.GATE_SHARE
HIGHER, LOWER, INSIDE = "higher", "lower", "inside"
IC_FLAG = "排名 {horizon} 日 IC"
MAE_FLAG = "MAE 差（模型減零預測）"
PINBALL_FLAG = "pinball 差（模型減波動率區間）"
BRIER_FLAG = "Brier 差（模型減 logistic）"
COVERAGE_FLAG = "20 日覆蓋率"
MAE_GATE, COVERAGE_TOTAL = "5 日 MAE", "20 日總覆蓋率"
BRIER_GATE, QLIKE_GATE = "機率 Brier", "波動率 QLIKE"


def cite(name: str) -> str:
    """The SPEC clause a criterion comes from: its section and the words the clause opens with."""
    section, opening = SPEC_CLAUSES[name]
    return f"docs/SPEC.md 第 {section} 節「{opening}」"


@dataclass(frozen=True)
class Check:
    name: str
    value: float
    rule: str
    basis: str
    passed: bool
    digits: int = 5


@dataclass(frozen=True)
class Window:
    mean: float
    first: int
    last: int


Bounds = dict[str, tuple[Window, Window]]


@dataclass(frozen=True)
class Flag:
    """A holdout per-fold mean held to the worst mean of WINDOW consecutive research folds."""

    name: str
    holdout: float
    better: str
    low: Window
    high: Window
    by_fold: tuple[tuple[int, float], ...] = ()

    @property
    def flagged(self) -> bool:
        if math.isnan(self.holdout):
            return True
        below, above = self.holdout < self.low.mean, self.holdout > self.high.mean
        return {HIGHER: below, LOWER: above, INSIDE: below or above}[self.better]

    @property
    def bounds(self) -> list[Window]:
        return {HIGHER: [self.low], LOWER: [self.high], INSIDE: [self.low, self.high]}[self.better]


@dataclass(frozen=True)
class ScopeVerdict:
    scope: str
    gates: tuple[Check, ...]
    flags: tuple[Flag, ...]
    references: tuple[Check, ...]
    waived: tuple[Check, ...]
    holdout_folds: pd.DataFrame

    @property
    def status(self) -> str:
        if not all(gate.passed for gate in self.gates):
            return FAILED
        return DEGRADED if any(flag.flagged for flag in self.flags) else PASSED


def scored(folds: Sequence[Fold]) -> list[Fold]:
    """The folds as the metrics count them: every given fold is a research fold."""
    return [replace(fold, tuning=False, holdout=False) for fold in folds]


def windows(values: pd.Series, size: int) -> tuple[Window, Window]:
    """The lowest and highest mean over size consecutive folds, with the folds each spans."""
    ordered = values.sort_index()
    if len(ordered) < size or ordered.isna().any():
        raise ValueError(
            f"degradation bounds need {size} folds with values, got {ordered.tolist()}"
        )
    means = ordered.rolling(size).mean().to_numpy()[size - 1 :]
    labels = [int(label) for label in ordered.index]

    def window(position: int) -> Window:
        return Window(float(means[position]), labels[position], labels[position + size - 1])

    return window(int(means.argmin())), window(int(means.argmax()))


def flag(name: str, bounds: tuple[Window, Window], holdout: pd.Series, better: str) -> Flag:
    """The degradation flag of one metric: holdout folds weighted equally; a missing fold flags."""
    low, high = bounds
    ordered = holdout.sort_index()
    values = tuple(zip([int(index) for index in ordered.index], ordered.astype(float), strict=True))
    mean = math.nan if holdout.isna().any() else float(holdout.mean())
    return Flag(name, mean, better, low, high, values)


def require_research_caches(run_key: str, folds: Sequence[Fold]) -> None:
    """Refuse before any fit when the research run lacks a research fold the flags read."""
    chosen = [fold for fold in folds if fold.research]
    moved = other_layout(run_key, chosen)
    if moved:
        raise FoldMismatchError(f"{run_key} cached folds {moved} for other fold boundaries")
    for model in (baseline_runs.BASELINE, MODEL):
        missing = missing_folds(run_key, model, chosen)
        if missing:
            raise MissingFoldsError(run_key, model, missing)


def _frame(
    run_key: str, folds: Sequence[Fold], directory: Path, relative: bool, holdout: bool
) -> pd.DataFrame:
    return target_eval.target_frame(run_key, folds, directory, relative, holdout=holdout)


def fold_gap(compared: target_eval.Comparison) -> pd.Series:
    """Per fold, the model's loss less its baseline's."""
    return compared.by_fold["model"] - compared.by_fold["baseline"]


def _refuse_stale(run_key: str, stale: list[int]) -> None:
    if stale:
        raise ValueError(
            f"folds {stale} of {run_key} cache models fit on features or labels that are stale; "
            "rerun train for the baselines and LightGBM before the holdout reads their bounds"
        )


def _rank_series(run_key: str, folds: Sequence[Fold], directory: Path) -> dict[str, pd.Series]:
    _refuse_stale(run_key, rank_eval.stale_folds(run_key, folds, directory))
    series = {}
    for horizon in HORIZONS:
        frame = rank_eval.evaluation_frame(run_key, folds, horizon, directory)
        report = rank_report(frame, f"{MODEL}_h{horizon}", horizon, folds)
        series[IC_FLAG.format(horizon=horizon)] = report.by_fold["mean"]
    return series


def _target_series(
    scope: str, run_key: str, folds: Sequence[Fold], directory: Path, relative: bool
) -> dict[str, pd.Series]:
    if scope == HoldoutScope.INTERVAL:
        target_eval.check_up_label(run_key, relative, folds)
    _refuse_stale(run_key, target_eval.stale_folds(scope, run_key, folds, directory, relative))
    frame = _frame(run_key, folds, directory, relative, False)
    if scope == HoldoutScope.RETURN:
        return {MAE_FLAG: fold_gap(target_eval.evaluate_return(frame, folds).mae["zero"])}
    past = target_eval.evaluate_interval(frame, folds)
    return {
        PINBALL_FLAG: fold_gap(past.pinball),
        BRIER_FLAG: fold_gap(past.brier),
        COVERAGE_FLAG: past.coverage_by_fold["mean"],
    }


def research_bounds(
    scope: str, run_key: str, folds: Sequence[Fold], directory: Path, relative: bool
) -> Bounds:
    """Each flag's worst research windows, read before the claim so a bad research run stops it."""
    chosen = [fold for fold in folds if fold.research]
    require_research_caches(run_key, chosen)
    if scope == HoldoutScope.RANK:
        series = _rank_series(run_key, chosen, directory)
    else:
        series = _target_series(scope, run_key, chosen, directory, relative)
    indexes = [fold.index for fold in chosen]
    return {name: windows(values.loc[indexes], WINDOW) for name, values in series.items()}


def _settled(frame: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """Rows whose label had settled when the holdout was claimed."""
    return frame[frame["date"] <= cutoff]


def rank_verdict(
    holdout_key: str,
    held: Sequence[Fold],
    bounds: Bounds,
    directory: Path,
    cutoffs: dict[int, date],
) -> tuple[ScopeVerdict, dict[int, tuple[RankReport, RankReport]]]:
    counted = scored(held)
    gates, flags, references, days, reports = [], [], [], {}, {}
    for horizon in HORIZONS:
        loaded = rank_eval.evaluation_frame(holdout_key, held, horizon, directory)
        frame = _settled(loaded, cutoffs[horizon])
        boosted, ridge = f"{MODEL}_h{horizon}", f"{RIDGE}_h{horizon}"
        model = rank_report(frame, boosted, horizon, counted, baseline=ridge)
        reports[horizon] = (model, rank_report(frame, ridge, horizon, counted))
        name = IC_FLAG.format(horizon=horizon)
        gates.append(Check(name, model.total.mean, "> 0", cite("gate"), model.total.mean > 0))
        flags.append(flag(name, bounds[name], model.by_fold["mean"], HIGHER))
        significant = model.total.mean > 0 and model.total.nw_t > 2
        references += [
            Check(
                f"{name} NW t", model.total.nw_t, "IC > 0 且 NW t > 2", cite("rank"), significant
            ),
            Check(
                f"排名 {horizon} 日贏 Ridge 的摺比例",
                model.beat_share,
                f">= {GATE_SHARE:.0%}",
                cite("rank"),
                model.beat_share >= GATE_SHARE,
            ),
        ]
        days[f"h{horizon}"] = model.by_fold["n_days"]
    verdict = ScopeVerdict(
        HoldoutScope.RANK, tuple(gates), tuple(flags), tuple(references), (), pd.DataFrame(days)
    )
    return verdict, reports


def _days(frame: pd.DataFrame, label: str, folds: Sequence[Fold]) -> pd.Series:
    """Days of each fold with a known label on a tradable row."""
    rows = tradable_rows(frame)
    known = rows.loc[rows[label].notna(), "date"].drop_duplicates()
    daily = pd.Series(1.0, index=pd.Index(known.to_list()))
    return mean_by_fold(daily, folds)["n_days"]


def return_verdict(
    holdout_key: str, held: Sequence[Fold], bounds: Bounds, directory: Path, cutoff: date
) -> ScopeVerdict:
    frame = _settled(_frame(holdout_key, held, directory, False, True), cutoff)
    counted = scored(held)
    evaluated = target_eval.evaluate_return(frame, counted)
    now, trailing = evaluated.mae["zero"], evaluated.mae["trailing"]
    gate = Check(
        MAE_GATE,
        now.model,
        f"<= 零預測 {now.baseline:.5f}",
        cite("gate"),
        now.model <= now.baseline,
    )
    degraded = flag(MAE_FLAG, bounds[MAE_FLAG], fold_gap(now), LOWER)
    references = (
        Check(
            "MAE 逐摺差 NW t", now.nw_t, "< -2 且 MAE 較低", cite("return"), target_eval.beats(now)
        ),
        Check(
            "MAE 對過去 60 日平均的逐摺差 NW t",
            trailing.nw_t,
            "< -2 且 MAE 較低",
            cite("return"),
            target_eval.beats(trailing),
        ),
    )
    days = pd.DataFrame({"h5": _days(frame, target_eval.LABEL, counted)})
    return ScopeVerdict(HoldoutScope.RETURN, (gate,), (degraded,), references, (), days)


def _interval_gates(now: target_eval.IntervalEvaluation) -> tuple[Check, ...]:
    low, high = COVERAGE_GATE
    return (
        Check(
            COVERAGE_TOTAL,
            now.coverage,
            f"{low:.0%}–{high:.0%}",
            cite("gate"),
            low <= now.coverage <= high,
        ),
        Check(
            BRIER_GATE,
            now.brier.model,
            f"<= logistic {now.brier.baseline:.5f}",
            cite("gate"),
            target_eval.no_worse(now.brier),
        ),
        Check(
            QLIKE_GATE,
            now.qlike.model,
            f"<= tech_vol_20d {now.qlike.baseline:.5f}",
            cite("gate"),
            target_eval.no_worse(now.qlike),
        ),
    )


def _interval_references(now: target_eval.IntervalEvaluation) -> tuple[Check, ...]:
    low, high = target_eval.COVERAGE_TOTAL
    return (
        Check(
            "pinball 逐摺差 NW t",
            now.pinball.nw_t,
            "< -2 且 pinball 較低",
            cite("interval"),
            target_eval.beats(now.pinball),
        ),
        Check(
            COVERAGE_TOTAL,
            now.coverage,
            f"{low:.0%}–{high:.0%}",
            cite("interval"),
            low <= now.coverage <= high,
        ),
        Check(
            "pinball 對歷史分位數",
            now.history.model,
            f"< 歷史分位數 {now.history.baseline:.5f}（sanity 下限）",
            cite("interval"),
            now.history.model < now.history.baseline,
        ),
        Check("分位數交叉列數", float(now.crossed), "= 0", cite("acceptance"), now.crossed == 0, 0),
        Check("機率 ECE", now.ece, f"< {ECE_LIMIT}", cite("probability"), now.ece < ECE_LIMIT),
    )


def _waived(now: target_eval.IntervalEvaluation) -> tuple[Check, ...]:
    low, high = target_eval.COVERAGE_FOLD
    return tuple(
        Check(
            f"摺 {index} 覆蓋率",
            float(value),
            f"{low:.0%}–{high:.0%}",
            cite("interval"),
            low <= value <= high,
        )
        for index, value in now.coverage_by_fold["mean"].items()
    )


def interval_verdict(
    holdout_key: str,
    held: Sequence[Fold],
    bounds: Bounds,
    directory: Path,
    relative: bool,
    cutoff: date,
) -> ScopeVerdict:
    target_eval.check_up_label(holdout_key, relative, held)
    frame = _settled(_frame(holdout_key, held, directory, relative, True), cutoff)
    counted = scored(held)
    now = target_eval.evaluate_interval(frame, counted)
    flags = (
        flag(PINBALL_FLAG, bounds[PINBALL_FLAG], fold_gap(now.pinball), LOWER),
        flag(BRIER_FLAG, bounds[BRIER_FLAG], fold_gap(now.brier), LOWER),
        flag(COVERAGE_FLAG, bounds[COVERAGE_FLAG], now.coverage_by_fold["mean"], INSIDE),
    )
    days = pd.DataFrame({"h20": _days(frame, target_eval.LABEL, counted)})
    return ScopeVerdict(
        HoldoutScope.INTERVAL,
        _interval_gates(now),
        flags,
        _interval_references(now),
        _waived(now),
        days,
    )


def marked_report(verdict: ScopeVerdict, output: str, text: str) -> Reported:
    """A holdout's one report as its marker keeps it: verdict, what fell short, path and text."""
    path = str(Path(output).resolve())
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    failed = [gate.name for gate in verdict.gates if not gate.passed]
    flagged = [raised.name for raised in verdict.flags if raised.flagged]
    return Reported(verdict.scope, verdict.status, path, digest, text, failed, flagged)


def unwritten_report(run_key: str, output: str) -> Reported | None:
    """The marker of a report cut short before its file; refused once written or elsewhere."""
    marked = reported(run_key)
    if marked is None:
        return None
    if Path(marked.path).exists():
        raise HoldoutError(f"holdout {run_key} was reported in {marked.path}; it is reported once")
    if Path(marked.path).resolve() != Path(output).resolve():
        raise HoldoutError(f"holdout {run_key} is marked for {marked.path}; its report goes there")
    return marked


def _link_whole(path: Path, text: str) -> None:
    """Create path with all of text or not at all; an existing path is an error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.staged")
    staged.write_text(text, encoding="utf-8")
    try:
        os.link(staged, path)
    finally:
        staged.unlink()


def write_once(output: str, text: str) -> None:
    """Write a report whole to a path that must not exist yet."""
    _link_whole(Path(output), text)


def mark_reported(run_key: str, marked: Reported) -> None:
    """Record a holdout's one report and verdict before the report file is written."""
    _link_whole(report_marker(run_key), json.dumps(asdict(marked), ensure_ascii=False))


def bags_of(run_key: str) -> list[int]:
    """The bag counts a run's LightGBM models were fit with."""
    runs = ModelRun.objects.filter(run_key=run_key, model_type=MODEL)
    return sorted({int(params.get("bags", 1)) for params in runs.values_list("params", flat=True)})


def write_holdout_metrics(run_key: str, reports: dict[int, tuple[RankReport, RankReport]]) -> int:
    """The holdout's rank metrics under its own run_key, LightGBM and Ridge, every horizon."""
    return sum(
        write_rank_metrics(model, run_key, MODEL) + write_rank_metrics(ridge, run_key, RIDGE)
        for model, ridge in reports.values()
    )


@dataclass(frozen=True)
class Header:
    scope: str
    run_key: str
    research_key: str
    attempt: int | None
    last_date: date
    rehearsal: bool


def _value(value: float, digits: int = 5) -> str:
    return "—" if math.isnan(value) else f"{value:.{digits}f}"


def _mark(passed: bool) -> str:
    return PASSED if passed else FAILED


def _checks(rows: Sequence[Check], judged: bool) -> list[str]:
    head = (
        "| 項目 | 值 | 條件 | 依據 | 結果 |" if judged else "| 項目 | 值 | 條件 | 依據 | 是否滿足 |"
    )
    lines = [head, "| --- | --- | --- | --- | --- |"]
    for row in rows:
        verdict = _mark(row.passed) if judged else ("滿足" if row.passed else "不滿足")
        value = _value(row.value, row.digits)
        lines.append(f"| {row.name} | {value} | {row.rule} | {row.basis} | {verdict} |")
    return lines


def _flags(rows: Sequence[Flag]) -> list[str]:
    lines = [
        "| 指標 | holdout 逐摺平均 | 界線 | 界線來自 | 結果 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        bounds = "、".join(_value(window.mean) for window in row.bounds)
        spans = "、".join(f"研究摺 {window.first}–{window.last}" for window in row.bounds)
        result = DEGRADED if row.flagged else "未觸發"
        lines.append(f"| {row.name} | {_value(row.holdout)} | {bounds} | {spans} | {result} |")
    return lines


def _fold_values(rows: Sequence[Flag]) -> list[str]:
    lines = ["| 摺 | " + " | ".join(row.name for row in rows) + " |"]
    lines.append("| --- | " + " | ".join("---" for _ in rows) + " |")
    folds = sorted({index for row in rows for index, _ in row.by_fold})
    for index in folds:
        cells = [_value(dict(row.by_fold).get(index, math.nan)) for row in rows]
        lines.append(f"| {index} | " + " | ".join(cells) + " |")
    return lines


def _days_table(days: pd.DataFrame) -> list[str]:
    columns = list(days.columns)
    lines = ["| 摺 | " + " | ".join(f"{name} 天數" for name in columns) + " |"]
    lines.append("| --- | " + " | ".join("---" for _ in columns) + " |")
    for index, row in days.iterrows():
        lines.append(f"| {index} | " + " | ".join(str(int(row[name])) for name in columns) + " |")
    return lines


def _shortest(days: pd.DataFrame) -> str:
    """The fold with the fewest labelled days per horizon; the per-fold mean ignores the count."""
    parts = [
        f"{name} 摺 {days[name].idxmin()}（{int(days[name].min())} 天）" for name in days.columns
    ]
    return (
        "只計 claim 的 last_date 當時 label 已到期的日子（各 horizon 往前 h 個交易日）。"
        "逐摺等權平均不依天數加權；天數最少的摺：" + "、".join(parts) + "。"
    )


RELEASE_NOTE = "上線與否由使用者決定，同意後才用 `release_holdout` 記下原因放行。"
HANDLING = {
    PASSED: "門檻 (3) 全部達標、沒有明顯退化旗標：可以進上線流程。",
    FAILED: (
        "有門檻未達標：這個目標預設不上線，報協調；`build_run` 會拒建，"
        f"{RELEASE_NOTE}照 {cite('record')} 只記錄、不改設計重跑。"
    ),
    DEGRADED: (
        "門檻 (3) 全部達標，但有明顯退化旗標：停下，預設不上線；`build_run` 會拒建，"
        f"{RELEASE_NOTE}照 {cite('record')} 只記錄、不改設計重跑。"
    ),
}
REHEARSAL_NOTE = "演練跑在研究摺上，只驗證程式路徑與資料形狀，數字不作任何判定。"


def render(header: Header, verdict: ScopeVerdict, bags: dict[str, list[int]]) -> str:
    """Markdown of one scope's holdout: verdict, gate (3), flags, references, waived, days."""
    kind = "演練（研究摺，不作判定）" if header.rehearsal else f"attempt {header.attempt}"
    status = "演練不作判定" if header.rehearsal else verdict.status
    handling = REHEARSAL_NOTE if header.rehearsal else HANDLING[verdict.status]
    lines = [
        f"# Holdout：{header.scope}（{header.run_key}，{kind}，資料到 {header.last_date}）",
        "",
        f"判讀標準在結果出來前寫死（協調 2026-11-11，{cite('record')}），本報告只套用，不改判；"
        f"{cite('acceptance')} 的 Phase 4 驗收對 holdout 要求只跑一次並記錄。",
        "",
        f"## 結論：{status}",
        "",
        handling,
        "",
        f"## 判定：{cite('gate')} 門檻 (3) 的絕對門檻，holdout 各摺合併",
        "",
        *_checks(verdict.gates, judged=True),
        "",
        f"## 明顯退化旗標（{cite('record')}）："
        f"研究期 `{header.research_key}` 任意連續 {WINDOW} 摺的最差逐摺平均",
        "",
        "研究期與 holdout 都是逐摺等權平均。"
        f"LightGBM 的 bags：研究期 {bags['research']}，holdout {bags['holdout']}"
        f"（{cite('bagging')}：研究階段單 seed、最終驗證 5 seed；"
        "bagging 對各指標的增益沒有量過，界線沒有為此修正）。",
        "",
        *_flags(verdict.flags),
        "",
        "holdout 各摺的數值：",
        "",
        *_fold_values(verdict.flags),
        "",
        f"## 參考（{cite('record')}）：研究期關卡套在 holdout 各摺"
        f"（{len(verdict.holdout_folds)} 摺，檢定力低），"
        "不作判定",
        "",
        *_checks(verdict.references, judged=False),
    ]
    if verdict.waived:
        lines += [
            "",
            f"## 每摺覆蓋率 70%–90%（{cite('record')}）：使用者 2026-11-11 豁免一次，不作判定",
            "",
            *_checks(verdict.waived, judged=False),
        ]
    days = verdict.holdout_folds
    lines += ["", "## 各摺有 label 的天數", "", _shortest(days), "", *_days_table(days), ""]
    return "\n".join(lines)


def judge(
    header: Header,
    held: Sequence[Fold],
    bounds: Bounds,
    directory: Path,
    relative: bool,
    calendar: TradingDays,
) -> tuple[ScopeVerdict, dict[int, tuple[RankReport, RankReport]]]:
    """The scope's verdict on labels settled by the last date, and its rank metric reports."""
    scope, key = header.scope, header.run_key
    cutoffs = {
        horizon: calendar.shift(header.last_date, -horizon)
        for horizon in lgbm_runs.SCOPE_HORIZONS[scope]
    }
    if scope == HoldoutScope.RANK:
        return rank_verdict(key, held, bounds, directory, cutoffs)
    cutoff = next(iter(cutoffs.values()))
    if scope == HoldoutScope.RETURN:
        return return_verdict(key, held, bounds, directory, cutoff), {}
    return interval_verdict(key, held, bounds, directory, relative, cutoff), {}
