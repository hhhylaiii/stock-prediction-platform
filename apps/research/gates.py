"""Promotion gates of a production build N against the serving runs (SPEC §7 每月重訓)."""

import math
from dataclasses import asdict, dataclass, field, fields
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from apps.etl.models import DataQualityLog
from apps.research import target_eval
from apps.research.baseline_runs import RETURN_HORIZON
from apps.research.lgbm_runs import MODEL
from apps.research.metrics.ic import daily_rank_ic, research_ic
from apps.research.models import ModelObjective, ModelRun, Prediction
from apps.research.predict import Slot, serving_run
from apps.research.prediction_cache import fold_path, load_fold_predictions
from apps.research.production import SHADOW, shadow_key
from apps.research.runs import PRODUCTION, RunError
from apps.research.training import load_labels
from apps.research.walkforward import Fold

SPEARMAN_LIMIT = 0.7
GAIN_TOP = 10
GAIN_OVERLAP = 6
MIN_STOCKS = 30
MIN_DAYS = 60
COVERAGE = (0.70, 0.90)
GATES = "gates"
FIRST_LAUNCH = "不適用（首次上線）"
FIELDS = {
    ModelObjective.RANK: "score",
    ModelObjective.REGRESSION: "ret_pred",
    ModelObjective.QUANTILE: "q50",
    ModelObjective.VOLATILITY: "vol_pred",
    ModelObjective.CLASSIFICATION: "prob_up_5pct",
}
COLUMNS = {
    ModelObjective.REGRESSION: f"{MODEL}_ret_h{RETURN_HORIZON}",
    ModelObjective.QUANTILE: f"{MODEL}_q50_h20",
    ModelObjective.VOLATILITY: f"{MODEL}_vol_h20",
    ModelObjective.CLASSIFICATION: f"{MODEL}_prob_h20",
}


@dataclass(frozen=True)
class Gate:
    """One gate's verdict; passed is None when it does not apply."""

    name: str
    passed: bool | None
    value: float | None
    threshold: str
    sample: int
    reason: str


@dataclass(frozen=True)
class SlotGates:
    slot: Slot
    run_id: str
    serving: str | None
    gates: list[Gate] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(gate.passed is not False for gate in self.gates)


def _number(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)


def _fold(run: ModelRun) -> Fold:
    """The fold a run recorded in its params, dates parsed back."""
    stored: dict[str, Any] = {
        name: date.fromisoformat(value) if isinstance(value, str) else value
        for name, value in run.params["fold"].items()
    }
    return Fold(**{item.name: stored[item.name] for item in fields(Fold)})


def _column(run: ModelRun) -> str:
    if run.objective == ModelObjective.RANK:
        return f"{MODEL}_h{run.params['horizon']}"
    return COLUMNS[ModelObjective(run.objective)]


def build_runs(run_key: str) -> dict[Slot, ModelRun]:
    """The production model N of run_key by slot; research runs and shadows are refused."""
    runs = ModelRun.objects.filter(run_key=run_key, model_type=MODEL)
    found = {
        Slot(run.objective, int(run.params["horizon"])): run
        for run in runs
        if (run.params.get(PRODUCTION) or {}).get("role") == "model"
    }
    if not found:
        raise RunError(f"{run_key} holds no production model N; gates check build_run's run_key")
    return found


def agreement(run: ModelRun, serving: ModelRun | None, as_of: date) -> Gate:
    """Gate 1: Spearman of N and the serving run over universe stocks on as_of."""
    threshold = f">= {SPEARMAN_LIMIT}"
    if serving is None:
        return Gate("agreement", None, None, threshold, 0, FIRST_LAUNCH)
    cached = load_fold_predictions(run.run_key, MODEL, _fold(run))
    mine = cached[(cached["date"] == as_of) & cached["in_universe"].astype(bool)]
    name = FIELDS[ModelObjective(run.objective)]
    theirs = pd.DataFrame(
        Prediction.objects.filter(
            run=serving, date=as_of, horizon=run.params["horizon"], in_universe=True
        ).values("stock_id", name)
    )
    if theirs.empty:
        return Gate("agreement", False, None, threshold, 0, f"{serving.run_id} 在 {as_of} 沒有預測")
    both = mine[["stock_id", _column(run)]].merge(theirs, on="stock_id").dropna()
    if len(both) < MIN_STOCKS:
        reason = f"兩邊都有值的股票 {len(both)} 檔，少於 {MIN_STOCKS}"
        return Gate("agreement", False, None, threshold, len(both), reason)
    value = _number(both[_column(run)].corr(both[name], method="spearman"))
    passed = value is not None and value >= SPEARMAN_LIMIT
    return Gate("agreement", passed, value, threshold, len(both), "" if passed else "相關太低")


def gain_overlap(run: ModelRun, serving: ModelRun | None) -> Gate:
    """Gate 2: shared features among the GAIN_TOP largest gains of N and the serving run."""
    threshold = f">= {GAIN_OVERLAP} of top {GAIN_TOP}"
    if serving is None:
        return Gate("gain_overlap", None, None, threshold, 0, FIRST_LAUNCH)
    gains = [run.params.get("importance_gain") or {}, serving.params.get("importance_gain") or {}]
    if not all(gains):
        missing = run.run_id if not gains[0] else serving.run_id
        return Gate("gain_overlap", False, None, threshold, 0, f"{missing} 沒有 importance_gain")
    tops = [set(sorted(gain, key=lambda name: (-gain[name], name))[:GAIN_TOP]) for gain in gains]
    shared = len(tops[0] & tops[1])
    sample = min(len(gains[0]), len(gains[1]))
    passed = shared >= GAIN_OVERLAP
    return Gate(
        "gain_overlap", passed, float(shared), threshold, sample, "" if passed else "重疊太少"
    )


def _days_gate(name: str, value: float, days: int, threshold: str, passed: bool) -> Gate:
    if days < MIN_DAYS:
        return Gate(
            name, False, _number(value), threshold, days, f"評估 {days} 天，少於 {MIN_DAYS}"
        )
    if _number(value) is None:
        return Gate(name, False, None, threshold, days, "沒有值")
    return Gate(name, passed, _number(value), threshold, days, "" if passed else "未達門檻")


def _shadow_ic(shadow: ModelRun) -> Gate:
    fold, horizon = _fold(shadow), int(shadow.params["horizon"])
    cached = load_fold_predictions(shadow.run_key, MODEL, fold)
    labels = load_labels(fold.test_start, fold.eval_end, horizon)
    rows = cached.merge(labels, on=["date", "stock_id"])
    rows = rows[rows["in_universe"].astype(bool) & rows["untradable_t1"].eq(False)]
    daily = research_ic(daily_rank_ic(rows, f"{MODEL}_h{horizon}"), [fold])
    value = float(daily.mean()) if len(daily) else math.nan
    return _days_gate("shadow_oos", value, int(daily.notna().sum()), "rank IC > 0", value > 0)


def _shadow_target(shadow: ModelRun, directory: Path) -> Gate:
    fold = _fold(shadow)
    frame = target_eval.target_frame(shadow.run_key, [fold], directory)
    objective = ModelObjective(shadow.objective)
    if objective == ModelObjective.REGRESSION:
        mae = target_eval.evaluate_return(frame, [fold]).mae["zero"]
        threshold = f"MAE <= 零預測 {mae.baseline:.4f}"
        return _days_gate("shadow_oos", mae.model, mae.days, threshold, target_eval.no_worse(mae))
    evaluated = target_eval.evaluate_interval(frame, [fold])
    if objective == ModelObjective.QUANTILE:
        low, high = COVERAGE
        value, days = evaluated.coverage, evaluated.coverage_days
        return _days_gate("shadow_oos", value, days, f"覆蓋率 {low}–{high}", low <= value <= high)
    compared = evaluated.qlike if objective == ModelObjective.VOLATILITY else evaluated.brier
    baseline = "tech_vol_20d" if objective == ModelObjective.VOLATILITY else "logistic"
    threshold = f"<= {baseline} {compared.baseline:.4f}"
    passed = target_eval.no_worse(compared)
    return _days_gate("shadow_oos", compared.model, compared.days, threshold, passed)


def shadow_oos(run: ModelRun, directory: Path) -> Gate:
    """Gate 3: N′ over its last months clears the absolute bar of N's objective."""
    key = shadow_key(run.run_key)
    shadow = ModelRun.objects.filter(
        run_key=key,
        model_type=MODEL,
        objective=run.objective,
        params__horizon=run.params["horizon"],
        params__production__role=SHADOW,
    ).first()
    if shadow is None or not fold_path(key, MODEL, 0).exists():
        return Gate("shadow_oos", False, None, "N′ 的 OOS", 0, f"{key} 沒有對應的 N′")
    if run.objective == ModelObjective.RANK:
        return _shadow_ic(shadow)
    return _shadow_target(shadow, directory)


def data_quality(run: ModelRun) -> Gate:
    """Gate 4: no error in data_quality_log over N's training window, and some checks logged."""
    window = DataQualityLog.objects.filter(date__range=(run.train_start, run.train_end))
    days = window.values("date").distinct().count()
    if not days:
        reason = f"{run.train_start}–{run.train_end} 沒有任何品質檢查紀錄"
        return Gate("data_quality", False, None, "no error", 0, reason)
    errors = sorted(
        set(window.filter(severity=DataQualityLog.Severity.ERROR).values_list("date", "check_name"))
    )
    listed = ", ".join(f"{day} {name}" for day, name in errors)
    passed = not errors
    return Gate("data_quality", passed, float(len(errors)), "no error", days, listed)


def check_gates(run_key: str, directory: Path) -> list[SlotGates]:
    """Every gate of each N slot of run_key, recorded on the N run; nothing changes status."""
    results = []
    for slot, run in build_runs(run_key).items():
        serving = serving_run(slot)
        as_of = date.fromisoformat(run.params[PRODUCTION]["as_of"])
        gates = [
            agreement(run, serving, as_of),
            gain_overlap(run, serving),
            shadow_oos(run, directory),
            data_quality(run),
        ]
        checked = SlotGates(slot, run.run_id, None if serving is None else serving.run_id, gates)
        _record(run, checked)
        results.append(checked)
    return results


def _record(run: ModelRun, checked: SlotGates) -> None:
    note: dict[str, Any] = {
        "serving": checked.serving,
        "passed": checked.passed,
        "gates": [asdict(gate) for gate in checked.gates],
    }
    ModelRun.objects.filter(pk=run.pk).update(params={**run.params, GATES: note})
