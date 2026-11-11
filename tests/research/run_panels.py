"""A year of synthetic features and labels for the walk-forward runner tests."""

import hashlib
import json
import math
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from django.conf import settings

from apps.market.calendar import TradingDays
from apps.market.models import IndexDaily, TradingCalendar
from apps.research import lgbm_runs, production
from apps.research.features.store import export_year
from apps.research.holdout import (
    DEGRADED,
    FAILED,
    PASSED,
    Reported,
    claim_holdout,
    is_holdout,
    report_marker,
)
from apps.research.models import FeatureDaily, HoldoutScope, LabelDaily, ModelRun
from apps.research.prediction_cache import fold_path
from apps.research.production import Design
from apps.research.runs import SCREENING_MODEL
from apps.research.training import linear_features
from apps.research.walkforward import Fold, WalkForwardConfig, generate_folds

START, END = date(2016, 1, 1), date(2016, 12, 31)
STOCKS = [f"{1100 + k}" for k in range(40)]
KEPT = ["tech_ret_1d", "tech_ret_60d", "tech_vol_20d"]
UNSCREENED = ["mkt_taiex_ret_5d"]
SCREENED = linear_features(KEPT)
CONFIG = WalkForwardConfig(
    train_months=6,
    test_months=1,
    step_months=1,
    valid_months=2,
    data_start=START,
    first_test=date(2016, 9, 1),
    tuning_end=date(2016, 9, 30),
    holdout_start=date(2016, 12, 1),
)


def record_screening(kept: list[str], unscreened: list[str] | None = None) -> ModelRun:
    return ModelRun.objects.create(
        run_id="screen-1.screening",
        run_key="screen-1",
        model_type=SCREENING_MODEL,
        objective="rank",
        version="1",
        train_start=START,
        train_end=date(2016, 6, 30),
        params={"unscreened": list(unscreened or [])},
        feature_list=kept,
        lib_versions={},
        feature_hash="0" * 64,
    )


def fitted_runs() -> Any:
    return ModelRun.objects.exclude(model_type=SCREENING_MODEL)


def weekdays() -> list[date]:
    days = (START + timedelta(days=n) for n in range((END - START).days + 1))
    return [day for day in days if day.weekday() < 5]


def build_panel(settings: Any, tmp_path: Path) -> Path:
    """A year of 40 stocks whose tech_ret_1d drives returns and tech_vol_20d their volatility."""
    settings.RESEARCH_CACHE_DIR = tmp_path / "predictions"
    settings.MODEL_DIR = tmp_path / "models"
    settings.FEATURE_PARQUET_DIR = tmp_path / "features"
    rng, extra = np.random.default_rng(3), np.random.default_rng(4)
    days = weekdays()
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=day.weekday() < 5)
        for day in (START + timedelta(days=n) for n in range((END - START).days + 1))
    )
    features, labels = [], []
    for day in days:
        signal = rng.normal(size=len(STOCKS))
        noise = rng.normal(size=len(STOCKS))
        market = float(extra.normal(0, 0.02))
        wobble = np.exp(0.1 * extra.normal(size=len(STOCKS)))
        ranks = pd.Series(signal + noise).rank(pct=True).to_numpy()
        for k, stock in enumerate(STOCKS):
            features.append(
                FeatureDaily(
                    stock_id=stock,
                    date=day,
                    in_universe=stock != "1139",
                    tech_ret_1d=float(signal[k]),
                    tech_ret_60d=0.01 * k,
                    tech_vol_20d=0.2 + 0.01 * k,
                    mkt_taiex_ret_5d=market,
                )
            )
            ret = 0.01 * float(signal[k] + noise[k])
            for horizon in (5, 20):
                labels.append(
                    LabelDaily(
                        stock_id=stock,
                        date=day,
                        horizon=horizon,
                        fwd_ret=ret,
                        fwd_ret_open=ret,
                        fwd_rank_pct=float(ranks[k]),
                        fwd_vol_20d=(0.2 + 0.01 * k) * float(wobble[k]) if horizon == 20 else None,
                        fwd_up_5pct=bool(signal[k] + noise[k] > 0.5) if horizon == 20 else None,
                        untradable_t1=False,
                    )
                )
    FeatureDaily.objects.bulk_create(features)
    LabelDaily.objects.bulk_create(labels)
    export_year(2016, settings.FEATURE_PARQUET_DIR)
    record_screening(KEPT, UNSCREENED)
    return Path(settings.FEATURE_PARQUET_DIR)


def folds() -> list[Fold]:
    calendar = TradingDays(weekdays(), START, END)
    return generate_folds(calendar, END, CONFIG)


def research() -> list[Fold]:
    return [fold for fold in folds() if not fold.holdout]


def store_taiex() -> None:
    """TAIEX_TR closes that swing, so the relative label differs from the absolute one."""
    IndexDaily.objects.bulk_create(
        IndexDaily(index_id="TAIEX_TR", date=day, close=round(100 * (1 + 0.1 * math.sin(k / 6)), 4))
        for k, day in enumerate(weekdays())
    )


def panel_calendar() -> TradingDays:
    days = weekdays()
    return TradingDays(days, days[0], days[-1])


HOLDOUT_OBJECTIVES = {"rank": "rank", "return": "regression", "interval": "quantile"}


def write_holdout_markers(
    scope: str, key: str, models: tuple[str, ...] = production.HOLDOUT_MODELS
) -> None:
    """Mark every holdout fold of the scope's frozen design written under key, files included."""
    calendar = TradingDays.from_db()
    for fold in generate_folds(calendar, weekdays()[-1], production.DESIGNS[scope].config):
        if is_holdout(fold):
            marker = Path(settings.RESEARCH_CACHE_DIR) / key / f"fold-{fold.index:03d}.holdout"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
            for model in models:
                written = fold_path(key, model, fold.index)
                written.parent.mkdir(parents=True, exist_ok=True)
                written.touch()


SHORT = {FAILED: (["5 日 MAE"], ["MAE 差"]), DEGRADED: ([], ["MAE 差"])}


def write_holdout_report(
    scope: str, key: str, status: str = PASSED, path: str = "r.md", text: str = ""
) -> None:
    """Mark the holdout under key reported with a verdict and what fell short in it."""
    marker = report_marker(key)
    marker.parent.mkdir(parents=True, exist_ok=True)
    failed, flagged = SHORT.get(status, ([], []))
    digest = hashlib.sha256(text.encode()).hexdigest()
    stored = Reported(scope, status, path, digest, text, failed, flagged)
    marker.write_text(json.dumps(asdict(stored), ensure_ascii=False), encoding="utf-8")


def finish_holdout(scope: str, key: str, reported: str | None = PASSED) -> None:
    """A claimed holdout as its runner leaves it: folds written, a run of its design, a report."""
    write_holdout_markers(scope, key)
    if reported is not None:
        write_holdout_report(scope, key, reported)
    design = production.DESIGNS[scope]
    note = json.loads(json.dumps(production.design_note(design)))
    features = lgbm_runs.design_features(scope, design.variant, design.target)
    ModelRun.objects.create(
        run_id=f"{key}.lightgbm.h20.f099",
        run_key=key,
        model_type="lightgbm",
        objective=HOLDOUT_OBJECTIVES[scope],
        version="4",
        train_start=START,
        train_end=END,
        params={production.DESIGN: note},
        feature_list=features,
        lib_versions={},
        feature_hash="0" * 64,
    )


def claim_all(finished: bool = True, reported: str | None = PASSED) -> None:
    """Claim every scope's holdout; by default finished on its design and reported as passed."""
    for scope in HOLDOUT_OBJECTIVES:
        key = f"{scope}-final"
        claim_holdout(scope, key, weekdays()[-1])
        if finished:
            finish_holdout(scope, key, reported)


def use_designs(
    monkeypatch: pytest.MonkeyPatch, config: WalkForwardConfig = CONFIG, **frozen: Any
) -> None:
    """Point every scope's frozen and SPEC designs at config; frozen ones may be given instead."""
    spec = {scope: Design(config) for scope in HoldoutScope.values}
    monkeypatch.setattr(production, "SPEC_DESIGNS", spec)
    monkeypatch.setattr(production, "DESIGNS", {**spec, **frozen})
