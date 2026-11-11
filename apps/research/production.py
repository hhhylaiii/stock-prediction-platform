"""Production runs (SPEC §7): the frozen design fit through the latest labels, and its shadow."""

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from apps.etl.alerts import dispatch
from apps.etl.models import DataQualityLog
from apps.market.calendar import TradingDays
from apps.research import baseline_runs
from apps.research.baseline_runs import check_run_key
from apps.research.holdout import (
    PRODUCTION_FOLD,
    ProductionPass,
    authorize_production,
    is_holdout,
    latest_ticket,
    production_pass,
    released_verdict,
)
from apps.research.lgbm_runs import (
    MODEL,
    check_design,
    design_features,
    run_fold,
    target_note,
    variant_note,
)
from apps.research.models import (
    OUTPUT_HORIZONS,
    HoldoutScope,
    ModelObjective,
    ModelRun,
    ModelStatus,
)
from apps.research.monitor import BRIER_BASELINE, IC_THRESHOLD, WINDOW
from apps.research.monitor_refs import research_brier, research_ic_thresholds
from apps.research.prediction_cache import fold_written
from apps.research.rank_model import SPEC_RECIPE, Variant
from apps.research.runs import PRODUCTION, RunError
from apps.research.target_models import TARGET_RECIPE, TargetVariant
from apps.research.walkforward import (
    Fold,
    WalkForwardConfig,
    WalkForwardError,
    add_months,
    generate_folds,
    training_window,
)

SHADOW_MONTHS = 6


def _fold(
    calendar: TradingDays,
    config: WalkForwardConfig,
    train_end: date,
    test: tuple[date, date],
    eval_end: date,
) -> Fold:
    window = training_window(calendar, config, train_end, test[0])
    if window is None:
        raise WalkForwardError(
            f"the training window before {test[0]} starts before data_start {config.data_start}"
        )
    train_start, fit_end, valid_start = window
    return Fold(
        index=PRODUCTION_FOLD,
        train_start=train_start,
        fit_end=fit_end,
        valid_start=valid_start,
        train_end=train_end,
        test_start=test[0],
        test_end=test[1],
        eval_end=eval_end,
        tuning=False,
        holdout=False,
    )


def production_fold(calendar: TradingDays, as_of: date, config: WalkForwardConfig) -> Fold:
    """Fold of model N: training ends where as_of's labels end, the test is as_of alone."""
    train_end = calendar.shift(as_of, -config.embargo)
    return _fold(calendar, config, train_end, (as_of, as_of), train_end)


def shadow_fold(calendar: TradingDays, as_of: date, config: WalkForwardConfig) -> Fold:
    """Fold of shadow N′: it tests the SHADOW_MONTHS through as_of after the usual embargo."""
    test_start = calendar.between(add_months(as_of, -SHADOW_MONTHS) + timedelta(days=1), as_of)[0]
    train_end = calendar.shift(test_start, -(config.embargo + 1))
    eval_end = calendar.shift(as_of, -config.embargo)
    return _fold(calendar, config, train_end, (test_start, as_of), eval_end)


@dataclass(frozen=True)
class Design:
    """A scope's frozen design: walk-forward layout, variants, and the research run it froze."""

    config: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    variant: Variant = SPEC_RECIPE
    target: TargetVariant = TARGET_RECIPE
    research_run_key: str | None = None


@dataclass(frozen=True)
class Built:
    model: list[str]
    shadow: list[str]
    references: dict[str, Any]


FROZEN_MONTHS, FROZEN_MIN_ROUNDS = 60, 200
FROZEN_RANK_FEATURES = (
    "tech_ret_1d",
    "tech_ret_5d",
    "tech_ret_20d",
    "tech_ret_60d",
    "tech_mom_250_20",
    "tech_ma_gap_5d",
    "tech_ma_gap_20d",
    "tech_ma_gap_60d",
    "tech_vol_20d",
    "tech_volume_ratio_5_20",
    "tech_turnover_rate_5d",
    "tech_rsi_14",
    "tech_kd_k",
    "tech_kd_d",
    "tech_macd_hist_pct",
    "tech_high_52w_gap",
    "tech_low_52w_gap",
    "tech_limit_up_t",
    "tech_limit_up_cnt_20d",
    "tech_limit_down_cnt_20d",
    "tech_attention_t",
    "size_log_mcap",
    "liq_dollar_vol_med_20d",
    "liq_amihud_20d",
    "chip_fi_net_1d",
    "chip_fi_net_5d",
    "chip_fi_net_20d",
    "chip_it_net_5d",
    "chip_dealer_self_net_5d",
    "chip_dealer_hedge_net_5d",
    "chip_margin_chg_5d",
    "chip_margin_usage",
    "chip_short_ratio",
    "chip_fi_holding_pct",
    "chip_sbl_short_chg_5d",
    "chip_big_holder_pct_chg_1w",
    "chip_holder_count_chg_1w",
    "fund_rev_yoy",
    "fund_rev_3m_yoy",
    "fund_ep",
    "fund_eps_yoy",
    "fund_dividend_yield",
    "fund_roe_ttm",
    "fund_gross_margin",
    "fund_op_margin",
    "fund_debt_ratio",
    "fund_gm_yoy_chg",
    "fund_opm_yoy_chg",
    "fund_cfo_to_ni",
    "fund_days_since_fs",
    "fund_rev_yoy_q",
    "risk_beta_250d",
    "risk_idio_vol_60d",
    "ind_sector_ret_20d",
    "ind_rel_ret_20d",
    "chip_day_trade_ratio_5d",
    "chip_fi_holding_chg_20d",
    "chip_it_net_20d",
    "chip_fi_buy_streak",
    "fund_days_to_ex_date",
)
FROZEN_CONFIG = replace(WalkForwardConfig(), train_months=FROZEN_MONTHS)
FROZEN_RANK = replace(SPEC_RECIPE, min_rounds=FROZEN_MIN_ROUNDS, features=FROZEN_RANK_FEATURES)
SPEC_DESIGNS: dict[str, Design] = {scope: Design() for scope in HoldoutScope.values}
DESIGNS: dict[str, Design] = {
    HoldoutScope.RANK: Design(FROZEN_CONFIG, FROZEN_RANK, research_run_key="rank-cand60"),
    HoldoutScope.RETURN: Design(
        FROZEN_CONFIG, target=TargetVariant(market=False), research_run_key="return-std1109"
    ),
    HoldoutScope.INTERVAL: Design(
        FROZEN_CONFIG,
        target=TargetVariant(market=False, relative_up=True),
        research_run_key="interval-std1109",
    ),
}
FROZEN, SPEC = "frozen", "spec"
DESIGN = "design"
DESIGN_NAMES = (FROZEN, SPEC)


def design_for(scope: str, name: str = FROZEN) -> Design:
    """The scope's frozen design, or the SPEC recipe it was chosen against."""
    if name not in DESIGN_NAMES:
        raise ValueError(f"design must be one of {list(DESIGN_NAMES)}, not {name!r}")
    return (DESIGNS if name == FROZEN else SPEC_DESIGNS)[scope]


BAGS = 5
HOLDOUT_MODELS = (baseline_runs.BASELINE, MODEL)
HOLDOUT_RELEASE = "holdout_release"
RELEASE_CHECK = "model_holdout_release"
ALERTED = ("status", "failed", "flagged", "released_at")
SHADOW = "shadow"
GATES = "gates"
BASELINED = (HoldoutScope.RETURN, HoldoutScope.INTERVAL)
REFERENCED: dict[str, str] = {
    ModelObjective.RANK: IC_THRESHOLD,
    ModelObjective.CLASSIFICATION: BRIER_BASELINE,
}


def shadow_key(run_key: str) -> str:
    """The run_key of a production run's shadow N′."""
    return f"{run_key}-{SHADOW}"


def design_note(design: Design) -> dict[str, Any]:
    """The design as production params record it."""
    config = {
        name: value.isoformat() if isinstance(value, date) else value
        for name, value in asdict(design.config).items()
    }
    return {
        "config": config,
        "variant": variant_note(design.variant),
        "target": target_note(design.target),
    }


def _references(
    scope: str, research_run_key: str, folds: list[Fold], relative: bool
) -> dict[str, Any]:
    """The monitor references N carries, from the research run's folds; none may be missing."""
    if scope == HoldoutScope.RANK:
        thresholds = research_ic_thresholds(research_run_key, MODEL, folds)
        missing = [h for h in OUTPUT_HORIZONS if not math.isfinite(thresholds.get(h, math.nan))]
        if missing:
            raise RunError(
                f"{research_run_key} gives no rolling IC threshold at {missing} days; "
                f"its research folds need {WINDOW} IC days"
            )
        return {IC_THRESHOLD: {str(h): float(thresholds[h]) for h in OUTPUT_HORIZONS}}
    if scope == HoldoutScope.INTERVAL:
        brier = research_brier(research_run_key, folds, relative)
        if not math.isfinite(brier):
            raise RunError(f"{research_run_key} gives no logistic baseline Brier")
        return {BRIER_BASELINE: brier}
    return {}


def _annotation(
    note: dict[str, Any], references: dict[str, Any]
) -> Callable[[ModelRun], dict[str, Any]]:
    def annotate(run: ModelRun) -> dict[str, Any]:
        wanted = REFERENCED.get(str(run.objective))
        return {**note, **{key: value for key, value in references.items() if key == wanted}}

    return annotate


def _canonical(note: Any) -> str:
    return json.dumps(note, sort_keys=True)


def _unwritten(granted: ProductionPass, design: Design, calendar: TradingDays) -> list[int]:
    """Holdout folds of the scope's latest claim that its holdout has not written yet."""
    ticket = latest_ticket(granted.scope)
    if ticket is None:
        return []
    folds = generate_folds(calendar, ticket.last_date, design.config)
    key = granted.frozen_by
    return [
        fold.index
        for fold in folds
        if is_holdout(fold) and not fold_written(key, fold.index, HOLDOUT_MODELS)
    ]


def holdout_mismatch(scope: str, key: str, design: Design, features: list[str]) -> str | None:
    """Why the runs a holdout key recorded are not of this design and features, or None."""
    runs = list(
        ModelRun.objects.filter(run_key=key).values_list("model_type", "params", "feature_list")
    )
    bare = sum(1 for _, params, _ in runs if DESIGN not in params)
    if bare:
        return f"{bare} runs under holdout {key} recorded no design; it is not a holdout run"
    expected = _canonical(json.loads(json.dumps(design_note(design))))
    if {_canonical(params[DESIGN]) for _, params, _ in runs} - {expected}:
        return f"the {scope} design is not the one holdout {key} ran"
    if any(model == MODEL and listed != features for model, _, listed in runs):
        return f"holdout {key} fit other features than the {scope} design reads now"
    return None


def check_holdout_design(
    scope: str, granted: ProductionPass, design: Design, features: list[str], calendar: TradingDays
) -> None:
    """Refuse a build unless the scope's freezing holdout finished on this design and features."""
    key = granted.frozen_by
    if not ModelRun.objects.filter(run_key=key).exists():
        raise RunError(f"holdout {key} recorded no run; run the {scope} holdout before building")
    mismatch = holdout_mismatch(scope, key, design, features)
    if mismatch:
        raise RunError(f"{mismatch}; production builds only the design its holdout ran")
    unwritten = _unwritten(granted, design, calendar)
    if unwritten:
        raise RunError(f"holdout {key} has not written folds {unwritten}; finish it first")


def _check_keys(scope: str, run_key: str, research_run_key: str, granted: ProductionPass) -> None:
    keys = (run_key, shadow_key(run_key))
    for key in (*keys, research_run_key):
        check_run_key(scope, key)
    if run_key.endswith(f"-{SHADOW}"):
        raise RunError(f"{run_key} names a shadow; a production run_key does not end in -{SHADOW}")
    if research_run_key in keys:
        raise RunError(f"{research_run_key} is the research run; production needs its own run_key")
    for key in keys:
        authorize_production(key, granted)
        if ModelRun.objects.filter(run_key=key).exclude(params__has_key=PRODUCTION).exists():
            raise RunError(f"{key} holds runs that are not a production build")
    settled = ModelRun.objects.filter(run_key__in=keys).exclude(status=ModelStatus.CANDIDATE)
    if (run := settled.first()) is not None:
        raise RunError(f"{run.run_id} is {run.status}; a new production build takes a new run_key")
    research = ModelRun.objects.filter(run_key=research_run_key)
    if not research.exists():
        raise RunError(f"no model_run of the research run {research_run_key}")
    if research.filter(params__has_key=PRODUCTION).exists():
        raise RunError(f"{research_run_key} is a production build, not a research run")


def _check_research(scope: str, research_run_key: str, design: Design, features: list[str]) -> None:
    """Refuse rank thresholds from a research run fit on another variant or feature list."""
    if scope != HoldoutScope.RANK:
        return
    variant = json.loads(json.dumps(variant_note(design.variant)))
    fits = ModelRun.objects.filter(run_key=research_run_key, model_type=MODEL)
    if not fits.exists() or any(
        run.params.get("variant") != variant or run.feature_list != features for run in fits
    ):
        raise RunError(f"{research_run_key} was not fit on the frozen rank design")


def _fit(
    scope: str,
    fold: Fold,
    key: str,
    directory: Path,
    features: list[str],
    seed: int,
    design: Design,
    granted: ProductionPass,
    annotate: Callable[[ModelRun], dict[str, Any]],
) -> list[str]:
    run_fold(
        scope,
        fold,
        key,
        directory,
        features,
        seed,
        BAGS,
        design.variant,
        design.target,
        production=granted,
        annotate=annotate,
    )
    return _run_ids(key)


def _clear_gates(run_key: str) -> None:
    """Drop the gates an earlier check recorded, so a rebuild never keeps a stale verdict."""
    for run in ModelRun.objects.filter(run_key=run_key, params__has_key=GATES):
        kept = {name: value for name, value in run.params.items() if name != GATES}
        ModelRun.objects.filter(pk=run.pk).update(params=kept)


def _run_ids(key: str) -> list[str]:
    return list(
        ModelRun.objects.filter(run_key=key).order_by("run_id").values_list("run_id", flat=True)
    )


def _alert_release(
    scope: str, run_key: str, as_of: date, holdout: str, release: dict[str, Any]
) -> None:
    """Warn that a production run was built on a released verdict; the reason stays in params."""
    shown = {name: release[name] for name in ALERTED}
    row = DataQualityLog.objects.create(
        date=as_of,
        check_name=RELEASE_CHECK,
        severity=DataQualityLog.Severity.WARNING,
        detail={"scope": scope, "run_key": run_key, "holdout": holdout, **shown},
    )
    dispatch([row])


def build_run(
    scope: str, run_key: str, research_run_key: str, as_of: date, directory: Path, seed: int = 0
) -> Built:
    """Fit shadow N′ with its return or interval baselines, then N; N carries the references."""
    granted = production_pass(scope)
    frozen = DESIGNS[scope].research_run_key
    if frozen is not None and research_run_key != frozen:
        raise RunError(
            f"{scope}'s frozen design takes its references from {frozen}, not {research_run_key}"
        )
    _check_keys(scope, run_key, research_run_key, granted)
    design = DESIGNS[scope]
    check_design(scope, design.variant, design.target)
    calendar = TradingDays.from_db()
    research = [fold for fold in generate_folds(calendar, as_of, design.config) if fold.research]
    features = design_features(scope, design.variant, design.target)
    check_holdout_design(scope, granted, design, features, calendar)
    release = released_verdict(granted)
    released = {} if release is None else {HOLDOUT_RELEASE: release}
    _check_research(scope, research_run_key, design, features)
    references = _references(scope, research_run_key, research, design.target.relative_up)
    shared = {
        "as_of": as_of.isoformat(),
        "frozen_by": granted.frozen_by,
        DESIGN: design_note(design),
    }
    roles = {
        "shadow": {**shared, "role": SHADOW, "shadow_of": run_key},
        "model": {**shared, **released, "role": "model", "shadow_run_key": shadow_key(run_key)},
    }
    notes = {
        role: {"research_run_key": research_run_key, PRODUCTION: note}
        for role, note in roles.items()
    }
    shadowed = shadow_fold(calendar, as_of, design.config)
    _clear_gates(run_key)
    shadow = _fit(
        scope,
        shadowed,
        shadow_key(run_key),
        directory,
        features,
        seed,
        design,
        granted,
        _annotation(notes["shadow"], {}),
    )
    if scope in BASELINED:
        baseline_runs.run_fold(
            scope,
            shadowed,
            shadow_key(run_key),
            directory,
            baseline_runs.scope_features(scope),
            production=granted,
            annotate=_annotation(notes["shadow"], {}),
            target=design.target,
        )
        shadow = _run_ids(shadow_key(run_key))
    model = _fit(
        scope,
        production_fold(calendar, as_of, design.config),
        run_key,
        directory,
        features,
        seed,
        design,
        granted,
        _annotation(notes["model"], references),
    )
    if release is not None:
        _alert_release(scope, run_key, as_of, granted.frozen_by, release)
    return Built(model, shadow, references)
