"""A rehearsal's numbers beside the research run's own fit of the same fold (SPEC §7).

With the same design, data, seed and bags the two agree; a difference is a bug in the path.
"""

from dataclasses import dataclass
from pathlib import Path

from apps.research import rank_eval, target_eval
from apps.research.evaluation import rank_report
from apps.research.holdout_eval import (
    BRIER_FLAG,
    BRIER_GATE,
    COVERAGE_FLAG,
    COVERAGE_TOTAL,
    IC_FLAG,
    MAE_FLAG,
    MAE_GATE,
    MODEL,
    PINBALL_FLAG,
    QLIKE_GATE,
    ScopeVerdict,
    fold_gap,
)
from apps.research.labels import HORIZONS
from apps.research.models import HoldoutScope, ModelRun
from apps.research.prediction_cache import missing_folds
from apps.research.walkforward import Fold

TOLERANCE = 5e-5
Numbers = dict[str, float]


@dataclass(frozen=True)
class Matched:
    name: str
    rehearsed: float
    research: float

    @property
    def agrees(self) -> bool:
        return abs(self.rehearsed - self.research) <= TOLERANCE


@dataclass(frozen=True)
class Reconciled:
    text: str
    differing: list[str]


def _numbers(
    scope: str, run_key: str, fold: Fold, directory: Path, relative: bool
) -> tuple[Numbers, Numbers, Numbers]:
    """A run's gate values, flag values and other metrics over one research fold."""
    folds, index = [fold], fold.index
    if scope == HoldoutScope.RANK:
        reports = {
            IC_FLAG.format(horizon=horizon): rank_report(
                rank_eval.evaluation_frame(run_key, folds, horizon, directory),
                f"{MODEL}_h{horizon}",
                horizon,
                folds,
            )
            for horizon in HORIZONS
        }
        gates = {name: float(report.total.mean) for name, report in reports.items()}
        flags = {name: float(report.by_fold["mean"].loc[index]) for name, report in reports.items()}
        return gates, flags, {}
    frame = target_eval.target_frame(run_key, folds, directory, relative)
    if scope == HoldoutScope.RETURN:
        mae = target_eval.evaluate_return(frame, folds).mae["zero"]
        return {MAE_GATE: mae.model}, {MAE_FLAG: float(fold_gap(mae).loc[index])}, {}
    now = target_eval.evaluate_interval(frame, folds)
    gates = {COVERAGE_TOTAL: now.coverage, BRIER_GATE: now.brier.model, QLIKE_GATE: now.qlike.model}
    flags = {
        PINBALL_FLAG: float(fold_gap(now.pinball).loc[index]),
        BRIER_FLAG: float(fold_gap(now.brier).loc[index]),
        COVERAGE_FLAG: float(now.coverage_by_fold["mean"].loc[index]),
    }
    return gates, flags, {"pinball（模型）": now.pinball.model, "日均 AUC": now.auc.model}


def _fit_as(run_key: str, fold: Fold) -> list[tuple[int, int]]:
    """The (seed, bags) a run's LightGBM models of one fold were fit with."""
    runs = ModelRun.objects.filter(
        run_key=run_key, model_type=MODEL, run_id__endswith=f".f{fold.index:03d}"
    )
    return sorted({(int(run.seed or 0), int(run.params.get("bags", 1))) for run in runs})


def _unreconcilable(rehearsal_key: str, research_key: str, fold: Fold) -> str | None:
    """Why the two fits of the fold are not the same experiment, or None when they are."""
    mine, theirs = _fit_as(rehearsal_key, fold), _fit_as(research_key, fold)
    if mine != theirs:
        return f"演練的 (seed, bags) 是 {mine}，研究 run 這一摺是 {theirs}，數字本來就不會相同。"
    if missing_folds(research_key, MODEL, [fold]):
        return "研究 run 這一摺的摺邊界和演練的不同（`--last-date` 截短了測試期）。"
    return None


def _rows(
    scope: str,
    verdict: ScopeVerdict,
    rehearsal_key: str,
    research_key: str,
    fold: Fold,
    directory: Path,
    relative: bool,
) -> list[Matched]:
    """The rehearsed verdict's gates and flags, and other metrics, beside the research run's."""
    gates, flags, others = _numbers(scope, research_key, fold, directory, relative)
    *_, again = _numbers(scope, rehearsal_key, fold, directory, relative)
    rows = [Matched(f"判定：{gate.name}", gate.value, gates[gate.name]) for gate in verdict.gates]
    rows += [
        Matched(f"旗標：{raised.name}", raised.holdout, flags[raised.name])
        for raised in verdict.flags
    ]
    return rows + [Matched(f"另算：{name}", again[name], value) for name, value in others.items()]


def _table(rows: list[Matched]) -> list[str]:
    lines = ["| 項目 | 演練 | 研究 run | 差 | 結果 |", "| --- | --- | --- | --- | --- |"]
    for row in rows:
        gap = row.rehearsed - row.research
        result = "相符" if row.agrees else "不符"
        lines.append(
            f"| {row.name} | {row.rehearsed:.6f} | {row.research:.6f} | {gap:+.6f} | {result} |"
        )
    return lines


def reconcile(
    scope: str,
    verdict: ScopeVerdict,
    rehearsal_key: str,
    research_key: str,
    fold: Fold,
    directory: Path,
    relative: bool,
) -> Reconciled:
    """The reconciliation section of a rehearsal report, and the rows that differ."""
    reason = _unreconcilable(rehearsal_key, research_key, fold)
    if reason is not None:
        return Reconciled("\n".join(["## 對帳：未對帳", "", reason, ""]), [])
    rows = _rows(scope, verdict, rehearsal_key, research_key, fold, directory, relative)
    differing = [row.name for row in rows if not row.agrees]
    seed, bags = _fit_as(research_key, fold)[0]
    lines = [
        f"## 對帳：演練摺 {fold.index} 對研究 run `{research_key}` 的同一摺",
        "",
        f"設計、資料、seed（{seed}）、bags（{bags}）都相同，數字應該一致；"
        f"容許誤差 {TOLERANCE:.5f}（絕對值）。判定與旗標兩組走 holdout 的評估路徑，"
        "研究 run 那一欄走研究期的評估函式。",
        "",
        *_table(rows),
        "",
        "對帳結論：" + ("不符，不得進正式跑" if differing else "相符"),
        "",
    ]
    return Reconciled("\n".join(lines), differing)
