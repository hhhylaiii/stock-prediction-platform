"""Holdout verdicts on criteria fixed before the run: gate (3) and degradation flags (SPEC §7).

The criteria are the coordinator's of 2026-11-11, written down before any holdout result.
"""

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import pandas as pd
from django.conf import settings

from apps.research import baseline_runs, lgbm_runs, rank_eval, target_eval
from apps.research.eval_report import write_rank_metrics
from apps.research.evaluation import RankReport, rank_report, tradable_rows
from apps.research.gates import COVERAGE as COVERAGE_GATE
from apps.research.holdout import require_key
from apps.research.labels import HORIZONS
from apps.research.metrics.forecast import mean_by_fold
from apps.research.models import HoldoutScope, ModelRun
from apps.research.prediction_cache import (
    FoldMismatchError,
    MissingFoldsError,
    missing_folds,
    other_layout,
)
from apps.research.walkforward import Fold

MODEL, RIDGE = lgbm_runs.MODEL, rank_eval.RIDGE
WINDOW = 8
SPEC_LINES: dict[str, tuple[int, str]] = {
    "record": (313, "2. 研究期與最終 holdout"),
    "rank": (323, "- 排名："),
    "return": (328, "- 絕對報酬（5 日）"),
    "interval": (329, "- 區間："),
    "probability": (331, "- 機率："),
    "gate": (337, "- 每月重訓"),
    "acceptance": (392, "| Phase 4 主模型"),
}
GATE_SHARE = rank_eval.GATE_SHARE
ECE_LIMIT = 0.05
PASSED, FAILED, DEGRADED = "達標", "未達標", "明顯退化"
HIGHER, LOWER, INSIDE = "higher", "lower", "inside"


def cite(name: str) -> str:
    """The SPEC line a criterion comes from, as docs/SPEC.md:N."""
    return f"docs/SPEC.md:{SPEC_LINES[name][0]}"


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


def flag(name: str, research: pd.Series, holdout: pd.Series, better: str) -> Flag:
    """The degradation flag of one metric: holdout folds weighted equally; a missing fold flags."""
    low, high = windows(research, WINDOW)
    ordered = holdout.sort_index()
    values = tuple(zip([int(index) for index in ordered.index], ordered.astype(float), strict=True))
    mean = math.nan if holdout.isna().any() else float(holdout.mean())
    return Flag(name, mean, better, low, high, values)


def _research_rows(by_fold: pd.DataFrame, folds: Sequence[Fold]) -> pd.DataFrame:
    return by_fold.loc[[fold.index for fold in folds if fold.research]]


def _rank_reports(
    run_key: str, loaded: Sequence[Fold], counted: Sequence[Fold], directory: Path
) -> dict[int, tuple[RankReport, RankReport]]:
    """Per horizon, LightGBM against Ridge and Ridge alone over the counted folds."""
    reports = {}
    for horizon in HORIZONS:
        frame = rank_eval.evaluation_frame(run_key, loaded, horizon, directory)
        boosted, ridge = f"{MODEL}_h{horizon}", f"{RIDGE}_h{horizon}"
        reports[horizon] = (
            rank_report(frame, boosted, horizon, counted, baseline=ridge),
            rank_report(frame, ridge, horizon, counted),
        )
    return reports


def rank_verdict(
    research_key: str,
    research: Sequence[Fold],
    holdout_key: str,
    held: Sequence[Fold],
    directory: Path,
) -> tuple[ScopeVerdict, dict[int, tuple[RankReport, RankReport]]]:
    past = _rank_reports(research_key, research, research, directory)
    now = _rank_reports(holdout_key, held, scored(held), directory)
    gates, flags, references, days = [], [], [], {}
    for horizon in HORIZONS:
        model, earlier = now[horizon][0], past[horizon][0]
        name = f"排名 {horizon} 日 IC"
        gates.append(Check(name, model.total.mean, "> 0", cite("gate"), model.total.mean > 0))
        flags.append(
            flag(
                name,
                _research_rows(earlier.by_fold, research)["mean"],
                model.by_fold["mean"],
                HIGHER,
            )
        )
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
        HoldoutScope.RANK,
        tuple(gates),
        tuple(flags),
        tuple(references),
        (),
        pd.DataFrame(days),
    )
    return verdict, now


def _frame(
    run_key: str, folds: Sequence[Fold], directory: Path, relative: bool, holdout: bool
) -> pd.DataFrame:
    return target_eval.target_frame(run_key, folds, directory, relative, holdout=holdout)


def _days(frame: pd.DataFrame, label: str, folds: Sequence[Fold]) -> pd.Series:
    """Days of each fold with a known label on a tradable row."""
    rows = tradable_rows(frame)
    known = rows.loc[rows[label].notna(), "date"].drop_duplicates()
    daily = pd.Series(1.0, index=pd.Index(known.to_list()))
    return mean_by_fold(daily, folds)["n_days"]


def _diff(compared: target_eval.Comparison) -> pd.Series:
    return compared.by_fold["model"] - compared.by_fold["baseline"]


def return_verdict(
    research_key: str,
    research: Sequence[Fold],
    holdout_key: str,
    held: Sequence[Fold],
    directory: Path,
) -> ScopeVerdict:
    past = target_eval.evaluate_return(
        _frame(research_key, research, directory, False, False), research
    )
    frame = _frame(holdout_key, held, directory, False, True)
    evaluated = target_eval.evaluate_return(frame, scored(held))
    now = evaluated.mae["zero"]
    before = past.mae["zero"]
    gate = Check(
        "5 日 MAE",
        now.model,
        f"<= 零預測 {now.baseline:.5f}",
        cite("gate"),
        now.model <= now.baseline,
    )
    degraded = flag(
        "MAE 差（模型減零預測）",
        _research_rows(before.by_fold.assign(diff=_diff(before)), research)["diff"],
        _diff(now),
        LOWER,
    )
    trailing = evaluated.mae["trailing"]
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
    days = pd.DataFrame({"h5": _days(frame, target_eval.LABEL, scored(held))})
    return ScopeVerdict(HoldoutScope.RETURN, (gate,), (degraded,), references, (), days)


def interval_verdict(
    research_key: str,
    research: Sequence[Fold],
    holdout_key: str,
    held: Sequence[Fold],
    directory: Path,
    relative: bool,
) -> ScopeVerdict:
    target_eval.check_up_label(holdout_key, relative, held)
    target_eval.check_up_label(research_key, relative, research)
    past = target_eval.evaluate_interval(
        _frame(research_key, research, directory, relative, False), research
    )
    frame = _frame(holdout_key, held, directory, relative, True)
    counted = scored(held)
    now = target_eval.evaluate_interval(frame, counted)
    low, high = COVERAGE_GATE
    gates = (
        Check(
            "20 日總覆蓋率",
            now.coverage,
            f"{low:.0%}–{high:.0%}",
            cite("gate"),
            low <= now.coverage <= high,
        ),
        Check(
            "機率 Brier",
            now.brier.model,
            f"<= logistic {now.brier.baseline:.5f}",
            cite("gate"),
            target_eval.no_worse(now.brier),
        ),
        Check(
            "波動率 QLIKE",
            now.qlike.model,
            f"<= tech_vol_20d {now.qlike.baseline:.5f}",
            cite("gate"),
            target_eval.no_worse(now.qlike),
        ),
    )
    covered = _research_rows(past.coverage_by_fold, research)["mean"]
    flags = (
        flag(
            "pinball 差（模型減波動率區間）",
            _research_rows(past.pinball.by_fold.assign(diff=_diff(past.pinball)), research)["diff"],
            _diff(now.pinball),
            LOWER,
        ),
        flag(
            "Brier 差（模型減 logistic）",
            _research_rows(past.brier.by_fold.assign(diff=_diff(past.brier)), research)["diff"],
            _diff(now.brier),
            LOWER,
        ),
        flag("20 日覆蓋率", covered, now.coverage_by_fold["mean"], INSIDE),
    )
    total_low, total_high = target_eval.COVERAGE_TOTAL
    references = (
        Check(
            "pinball 逐摺差 NW t",
            now.pinball.nw_t,
            "< -2 且 pinball 較低",
            cite("interval"),
            target_eval.beats(now.pinball),
        ),
        Check(
            "20 日總覆蓋率",
            now.coverage,
            f"{total_low:.0%}–{total_high:.0%}",
            cite("interval"),
            total_low <= now.coverage <= total_high,
        ),
        Check(
            "pinball 對歷史分位數",
            now.history.model,
            f"< 歷史分位數 {now.history.baseline:.5f}（sanity 下限）",
            cite("interval"),
            now.history.model < now.history.baseline,
        ),
        Check("分位數交叉列數", float(now.crossed), "= 0", cite("interval"), now.crossed == 0, 0),
        Check("機率 ECE", now.ece, f"< {ECE_LIMIT}", cite("probability"), now.ece < ECE_LIMIT),
    )
    fold_low, fold_high = target_eval.COVERAGE_FOLD
    waived = tuple(
        Check(
            f"摺 {index} 覆蓋率",
            float(value),
            f"{fold_low:.0%}–{fold_high:.0%}",
            cite("interval"),
            fold_low <= value <= fold_high,
        )
        for index, value in now.coverage_by_fold["mean"].items()
    )
    days = pd.DataFrame({"h20": _days(frame, target_eval.LABEL, counted)})
    return ScopeVerdict(HoldoutScope.INTERVAL, gates, flags, references, waived, days)


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


def _report_marker(run_key: str) -> Path:
    return Path(settings.RESEARCH_CACHE_DIR) / require_key(run_key) / "holdout.report"


def reported_in(run_key: str) -> str | None:
    """The path a holdout's report was written to, or None before its report."""
    marker = _report_marker(run_key)
    if not marker.exists():
        return None
    return str(json.loads(marker.read_text(encoding="utf-8"))["path"])


def write_once(output: str, text: str) -> None:
    """Write a report to a path that must not exist yet."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def mark_reported(run_key: str, output: str, text: str) -> None:
    """Record that a holdout was reported, so no later path takes a second report."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    marker = _report_marker(run_key)
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("x", encoding="utf-8") as handle:
        json.dump({"path": output, "sha256": digest}, handle)


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


HANDLING = {
    PASSED: "門檻 (3) 全部達標、沒有明顯退化旗標：可以進上線流程。",
    FAILED: f"有門檻未達標：這個目標不上線，報協調；照 {cite('record')} 只記錄、不改設計重跑。",
    DEGRADED: (
        "門檻 (3) 全部達標，但有明顯退化旗標：停下，上線與否由使用者決定；"
        f"照 {cite('record')} 只記錄、不改設計重跑。"
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
        f"LightGBM 的 bags：研究期 {bags['research']}，holdout {bags['holdout']}。",
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
    lines += ["", "## 各摺有 label 的天數", "", *_days_table(verdict.holdout_folds), ""]
    return "\n".join(lines)


def judge(
    scope: str,
    holdout_key: str,
    held: Sequence[Fold],
    research_key: str,
    research: Sequence[Fold],
    directory: Path,
    relative: bool,
) -> tuple[ScopeVerdict, dict[int, tuple[RankReport, RankReport]]]:
    """The scope's verdict, and for rank the reports its backtest_metric rows come from."""
    chosen = [fold for fold in research if fold.research]
    if scope == HoldoutScope.RANK:
        return rank_verdict(research_key, chosen, holdout_key, held, directory)
    if scope == HoldoutScope.RETURN:
        return return_verdict(research_key, chosen, holdout_key, held, directory), {}
    return interval_verdict(research_key, chosen, holdout_key, held, directory, relative), {}
