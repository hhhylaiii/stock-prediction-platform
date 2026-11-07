from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from apps.market.calendar import TradingDays
from apps.research import baseline_runs, lgbm_runs
from tests.research.run_panels import build_panel, research


def weekday_calendar(start: date, end: date) -> TradingDays:
    """Synthetic calendar trading every Monday to Friday; long fold layouts need 2010 to 2026."""
    days = [
        start + timedelta(days=offset)
        for offset in range((end - start).days + 1)
        if (start + timedelta(days=offset)).weekday() < 5
    ]
    return TradingDays(days, start, end)


@pytest.fixture(scope="session")
def long_calendar() -> TradingDays:
    return weekday_calendar(date(2010, 1, 1), date(2026, 12, 31))


@pytest.fixture
def run_panel(settings: Any, tmp_path: Path, db: None) -> Path:
    """Feature partitions, labels and a screening for the walk-forward runner tests."""
    return build_panel(settings, tmp_path)


@pytest.fixture
def rank_caches(run_panel: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Ridge and LightGBM rank caches of the panel's research folds under rank-wf1."""
    monkeypatch.setattr(lgbm_runs, "MAX_ROUNDS", 120)
    folds = [fold for fold in research() if fold.research]
    baseline_runs.run_folds("rank", folds, "rank-wf1", run_panel)
    lgbm_runs.run_folds("rank", folds, "rank-wf1", run_panel)
    return run_panel


@pytest.fixture
def served_runs(run_panel: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Two-seed rank, return and interval runs of the panel's last research fold, as candidates."""
    monkeypatch.setattr(lgbm_runs, "MAX_ROUNDS", 60)
    last = [fold for fold in research() if fold.research][-1]
    for scope in ("rank", "return", "interval"):
        lgbm_runs.run_folds(scope, [last], f"{scope}-live", run_panel, bags=2)
    return run_panel
