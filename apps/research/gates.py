"""Promotion gates of a production build N against the serving runs (SPEC §7 每月重訓)."""

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, fields
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from django.db import transaction

from apps.etl.models import DataQualityLog
from apps.research import target_eval
from apps.research.baseline_runs import BASELINE, RETURN_HORIZON
from apps.research.lgbm_runs import MODEL
from apps.research.metrics.ic import daily_rank_ic, research_ic
from apps.research.models import ModelObjective, ModelRun, ModelStatus, Prediction
from apps.research.predict import Slot, serving_run
from apps.research.prediction_cache import fold_path, load_fold_predictions
from apps.research.production import GATES, SHADOW, shadow_key
from apps.research.relative_up import RELATIVE_LABEL
from apps.research.runs import PRODUCTION, RunError
from apps.research.training import load_labels
from apps.research.walkforward import Fold

SPEARMAN_LIMIT = 0.7
GAIN_TOP = 10
GAIN_OVERLAP = 6
MIN_STOCKS = 30
MIN_DAYS = 60
COVERAGE = (0.70, 0.90)
ERROR_LIMIT = 200
NO_SERVING = "不適用（沒有服務中的 run）"
SAMPLE_UNITS = {
    "agreement": "stocks",
    "gain_overlap": "features",
    "shadow_oos": "days",
    "data_quality": "days",
}
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
    extra: dict[str, Any] = field(default_factory=dict)


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
    """The production models N of run_key still candidate, by slot; a build with none is refused."""
    runs = ModelRun.objects.filter(run_key=run_key, model_type=MODEL).order_by("run_id")
    models = [run for run in runs if (run.params.get(PRODUCTION) or {}).get("role") == "model"]
    if not models:
        raise RunError(f"{run_key} holds no production model N; gates check build_run's run_key")
    found = {
        Slot(run.objective, int(run.params["horizon"])): run
        for run in models
        if run.status == ModelStatus.CANDIDATE
    }
    if not found:
        moved = ", ".join(run.run_id for run in models)
        raise RunError(f"{moved} left candidate; gates judge N before promotion only")
    if not fold_path(run_key, MODEL, 0).exists():
        raise RunError(f"{run_key} has no cached fold 0; its last build_run did not finish")
    return found


def passing_slots(checked: Iterable[SlotGates]) -> set[Slot]:
    """Slots of the horizons where every slot passed; SPEC §7 judges each horizon on its own."""
    judged = list(checked)
    failed = {result.slot.horizon for result in judged if not result.passed}
    return {result.slot for result in judged if result.slot.horizon not in failed}


def agreement(run: ModelRun, serving: ModelRun | None, as_of: date) -> Gate:
    """Gate 1: Spearman of N and the serving run over universe stocks on as_of."""
    threshold = f">= {SPEARMAN_LIMIT}"
    if serving is None:
        return Gate("agreement", None, None, threshold, 0, NO_SERVING)
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
    threshold = f"前 {GAIN_TOP} 名重疊 >= {GAIN_OVERLAP}"
    if serving is None:
        return Gate("gain_overlap", None, None, threshold, 0, NO_SERVING)
    gains = [run.params.get("importance_gain") or {}, serving.params.get("importance_gain") or {}]
    if not all(gains):
        missing = run.run_id if not gains[0] else serving.run_id
        return Gate("gain_overlap", False, None, threshold, 0, f"{missing} 沒有 importance_gain")
    used = [{name: value for name, value in gain.items() if value > 0} for gain in gains]
    tops = [set(sorted(gain, key=lambda name: (-gain[name], name))[:GAIN_TOP]) for gain in used]
    shared = len(tops[0] & tops[1])
    sample = min(len(used[0]), len(used[1]))
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
    objective = ModelObjective(shadow.objective)
    relative = shadow.params.get("label") == RELATIVE_LABEL
    if objective == ModelObjective.CLASSIFICATION:
        target_eval.check_up_label(shadow.run_key, relative, [fold])
    frame = target_eval.target_frame(shadow.run_key, [fold], directory, relative)
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


def paired_shadow(run: ModelRun) -> ModelRun:
    """N′ of the same build as N, with its caches; an unfinished build is refused."""
    key = shadow_key(run.run_key)
    slot = Slot(run.objective, int(run.params["horizon"]))
    shadow = ModelRun.objects.filter(
        run_key=key,
        model_type=MODEL,
        objective=run.objective,
        params__horizon=run.params["horizon"],
        params__production__role=SHADOW,
    ).first()
    unfinished = "its last build_run did not finish"
    if shadow is None:
        raise RunError(f"{key} holds no shadow N′ for {slot}; {unfinished}")
    mine, theirs = run.params[PRODUCTION], shadow.params[PRODUCTION]
    if theirs.get("shadow_of") != run.run_key or theirs.get("as_of") != mine["as_of"]:
        raise RunError(
            f"{shadow.run_id} is the N′ of {theirs.get('shadow_of')} as of {theirs.get('as_of')}, "
            f"not of {run.run_key} as of {mine['as_of']}"
        )
    if not fold_path(key, MODEL, 0).exists():
        raise RunError(f"{key} has no cached fold 0; {unfinished}")
    if run.objective != ModelObjective.RANK and not fold_path(key, BASELINE, 0).exists():
        raise RunError(f"{key} has no baseline cache; {unfinished}")
    return shadow


def shadow_oos(run: ModelRun, shadow: ModelRun, directory: Path) -> Gate:
    """Gate 3: N′ over its last months clears the absolute bar of N's objective."""
    if run.objective == ModelObjective.RANK:
        return _shadow_ic(shadow)
    return _shadow_target(shadow, directory)


def data_quality(run: ModelRun) -> Gate:
    """Gate 4: no error in data_quality_log from train_start to as_of, where N's labels end."""
    as_of = date.fromisoformat(run.params[PRODUCTION]["as_of"])
    window = DataQualityLog.objects.filter(date__range=(run.train_start, as_of))
    days = window.values("date").distinct().count()
    if not days:
        reason = f"{run.train_start}–{as_of} 沒有任何品質檢查紀錄"
        return Gate("data_quality", False, None, "無 error", 0, reason, _errors([]))
    errors = sorted(
        set(
            window.filter(severity=DataQualityLog.Severity.ERROR).values_list("date", "check_name")
        ),
        key=lambda error: (-error[0].toordinal(), error[1]),
    )
    value = float(len(errors))
    return Gate(
        "data_quality", not errors, value, "無 error", days, _summary(errors), _errors(errors)
    )


def _summary(errors: list[tuple[date, str]]) -> str:
    """How many days had an error, and the checks of the latest one."""
    if not errors:
        return ""
    latest = errors[0][0]
    checks = "、".join(name for day, name in errors if day == latest)
    return f"{len({day for day, _ in errors})} 天有 error，最近 {latest} {checks}"


def _errors(errors: list[tuple[date, str]]) -> dict[str, Any]:
    listed = [{"date": day.isoformat(), "check": name} for day, name in errors[:ERROR_LIMIT]]
    return {"errors": listed, "errors_total": len(errors)}


def _stored(gate: Gate) -> dict[str, Any]:
    return {
        **{name: value for name, value in asdict(gate).items() if name != "extra"},
        **gate.extra,
    }


def check_gates(run_key: str, directory: Path) -> list[SlotGates]:
    """Every gate of each candidate N of run_key, recorded on them together; no status changes."""
    runs = build_runs(run_key)
    shadows = {slot: paired_shadow(run) for slot, run in runs.items()}
    checked = []
    for slot, run in runs.items():
        serving = serving_run(slot)
        as_of = date.fromisoformat(run.params[PRODUCTION]["as_of"])
        gates = [
            agreement(run, serving, as_of),
            gain_overlap(run, serving),
            shadow_oos(run, shadows[slot], directory),
            data_quality(run),
        ]
        served = None if serving is None else serving.run_id
        checked.append((run, SlotGates(slot, run.run_id, served, gates)))
    with transaction.atomic():
        for run, result in checked:
            _record(run, result)
    return [result for _, result in checked]


def _without_gates(run: ModelRun) -> dict[str, Any]:
    return {name: value for name, value in run.params.items() if name != GATES}


def _record(run: ModelRun, checked: SlotGates) -> None:
    """Write the verdict on N, locked and rechecked: N must still be the candidate judged."""
    current = ModelRun.objects.select_for_update().get(pk=run.pk)
    if current.status != ModelStatus.CANDIDATE or _without_gates(current) != _without_gates(run):
        raise RunError(f"{run.run_id} changed while its gates were checked; check them again")
    note: dict[str, Any] = {
        "serving": checked.serving,
        "passed": checked.passed,
        "gates": [_stored(gate) for gate in checked.gates],
    }
    ModelRun.objects.filter(pk=run.pk).update(params={**current.params, GATES: note})
