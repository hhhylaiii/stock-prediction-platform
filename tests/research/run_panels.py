"""A year of synthetic features and labels for the walk-forward runner tests."""

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from apps.market.calendar import TradingDays
from apps.market.models import TradingCalendar
from apps.research.features.store import export_year
from apps.research.models import FeatureDaily, LabelDaily, ModelRun
from apps.research.runs import SCREENING_MODEL
from apps.research.training import linear_features
from apps.research.walkforward import Fold, WalkForwardConfig, generate_folds

START, END = date(2016, 1, 1), date(2016, 12, 31)
STOCKS = [f"{1100 + k}" for k in range(40)]
KEPT = ["tech_ret_1d", "tech_ret_60d", "tech_vol_20d"]
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


def record_screening(kept: list[str]) -> ModelRun:
    return ModelRun.objects.create(
        run_id="screen-1.screening",
        run_key="screen-1",
        model_type=SCREENING_MODEL,
        objective="rank",
        version="1",
        train_start=START,
        train_end=date(2016, 6, 30),
        params={"unscreened": []},
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
    """A year of 40 stocks whose tech_ret_1d drives the labels; 1139 is outside the universe."""
    settings.RESEARCH_CACHE_DIR = tmp_path / "predictions"
    settings.MODEL_DIR = tmp_path / "models"
    settings.FEATURE_PARQUET_DIR = tmp_path / "features"
    rng = np.random.default_rng(3)
    days = weekdays()
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=day.weekday() < 5)
        for day in (START + timedelta(days=n) for n in range((END - START).days + 1))
    )
    features, labels = [], []
    for day in days:
        signal = rng.normal(size=len(STOCKS))
        noise = rng.normal(size=len(STOCKS))
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
                        fwd_vol_20d=0.3 if horizon == 20 else None,
                        fwd_up_5pct=bool(signal[k] + noise[k] > 0.5) if horizon == 20 else None,
                        untradable_t1=False,
                    )
                )
    FeatureDaily.objects.bulk_create(features)
    LabelDaily.objects.bulk_create(labels)
    export_year(2016, settings.FEATURE_PARQUET_DIR)
    record_screening(KEPT)
    return Path(settings.FEATURE_PARQUET_DIR)


def folds() -> list[Fold]:
    calendar = TradingDays(weekdays(), START, END)
    return generate_folds(calendar, END, CONFIG)


def research() -> list[Fold]:
    return [fold for fold in folds() if not fold.holdout]
