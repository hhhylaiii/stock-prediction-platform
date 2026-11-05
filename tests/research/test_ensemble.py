import subprocess
import sys
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.market.models import DailyPrice, StockMaster
from apps.research import baseline_runs
from apps.research.ensemble import (
    adopt_ensemble,
    ensemble_ics,
    ensemble_report,
    rank_average,
    research_folds,
)
from apps.research.features.store import export_year
from apps.research.management.commands import ensemble_check
from apps.research.metrics.ic import daily_rank_ic
from apps.research.models import FeatureDaily, LabelDaily
from apps.research.prediction_cache import load_fold_predictions
from apps.research.training import training_rows
from tests.research.run_panels import CONFIG, research, weekdays

pytestmark = pytest.mark.django_db


def chosen() -> list[Any]:
    return [fold for fold in research() if fold.research]


def labels(fold: Any, horizon: int) -> pd.DataFrame:
    rows = LabelDaily.objects.filter(date__range=(fold.test_start, fold.eval_end), horizon=horizon)
    return pd.DataFrame(list(rows.values()))


def merged_caches(fold: Any) -> pd.DataFrame:
    ridge = load_fold_predictions("rank-wf1", "baseline", fold)
    lgbm = load_fold_predictions("rank-wf1", "lightgbm", fold)
    return ridge.merge(lgbm, on=["date", "stock_id", "in_universe"])


def test_rank_average_means_each_score_rank_within_the_day_universe() -> None:
    frame = pd.DataFrame(
        {
            "date": [date(2016, 1, 4)] * 4 + [date(2016, 1, 5)] * 2,
            "stock_id": ["a", "b", "c", "d", "a", "b"],
            "in_universe": [True, True, True, False, True, True],
            "ridge": [0.1, 0.3, 0.2, 9.0, 1.0, 2.0],
            "lightgbm": [3.0, 1.0, 2.0, 9.0, 5.0, 4.0],
        }
    )

    averaged = rank_average(frame, ["ridge", "lightgbm"])

    expected = [(1 / 3 + 1) / 2, (1 + 1 / 3) / 2, (2 / 3 + 2 / 3) / 2, np.nan, 0.75, 0.75]
    np.testing.assert_allclose(averaged.to_numpy(), expected)


def test_the_ensemble_is_adopted_only_when_it_beats_lightgbm_alone() -> None:
    assert adopt_ensemble({"lightgbm": 0.05, "ensemble": 0.051})
    assert not adopt_ensemble({"lightgbm": 0.05, "ensemble": 0.05})
    assert not adopt_ensemble({"lightgbm": 0.05, "ensemble": float("nan")})


def test_ensemble_ics_pool_daily_oos_ic_of_universe_ranks(rank_caches: Path) -> None:
    folds = chosen()

    ics = ensemble_ics("rank-wf1", folds, 20)

    parts = []
    for fold in folds:
        frame = merged_caches(fold)
        frame = frame.assign(ensemble=rank_average(frame, ["ridge_h20", "lightgbm_h20"]))
        parts.append(
            training_rows(frame[frame["date"] <= fold.eval_end], labels(fold, 20), "fwd_rank_pct")
        )
    rows = pd.concat(parts, ignore_index=True)
    for name, column in [
        ("ridge", "ridge_h20"),
        ("lightgbm", "lightgbm_h20"),
        ("ensemble", "ensemble"),
    ]:
        assert ics[name] == pytest.approx(daily_rank_ic(rows, column).mean()), name
    assert ics["lightgbm"] > 0.3


def test_ensemble_ranks_the_whole_universe_before_dropping_untradable_rows(
    rank_caches: Path,
) -> None:
    folds = chosen()
    fold = folds[0]
    held = [f"{1100 + k}" for k in range(0, 40, 8)]
    LabelDaily.objects.filter(
        date__range=(fold.test_start, fold.eval_end), stock_id__in=held
    ).update(untradable_t1=True)

    ics = ensemble_ics("rank-wf1", [fold], 5)

    frame = merged_caches(fold)
    frame = frame[frame["date"] <= fold.eval_end]
    whole = frame.assign(ensemble=rank_average(frame, ["ridge_h5", "lightgbm_h5"]))
    kept = training_rows(whole, labels(fold, 5), "fwd_rank_pct")
    late = training_rows(frame, labels(fold, 5), "fwd_rank_pct")
    late = late.assign(ensemble=rank_average(late, ["ridge_h5", "lightgbm_h5"]))
    assert ics["ensemble"] == pytest.approx(daily_rank_ic(kept, "ensemble").mean())
    assert ics["ensemble"] != pytest.approx(daily_rank_ic(late, "ensemble").mean(), abs=1e-9)


def test_ensemble_refuses_caches_built_on_other_feature_partitions(rank_caches: Path) -> None:
    folds = chosen()
    FeatureDaily.objects.filter(date=folds[0].test_start).update(tech_ret_60d=0.5)
    export_year(2016, rank_caches)
    baseline_runs.run_folds("rank", folds, "rank-wf1", rank_caches)

    with pytest.raises(ValueError, match=rf"folds \[{folds[0].index}"):
        ensemble_ics("rank-wf1", folds, 5)


def test_ensemble_report_states_each_horizon_and_its_verdict() -> None:
    text = ensemble_report(
        "rank-wf1",
        {
            5: {"ridge": 0.01, "lightgbm": 0.02, "ensemble": 0.025},
            20: {"ridge": 0.03, "lightgbm": 0.04, "ensemble": 0.035},
        },
    )

    assert text.startswith("# Rank-average ensemble: rank-wf1")
    assert "| 5 | 0.0100 | 0.0200 | 0.0250 | adopt |" in text
    assert "| 20 | 0.0300 | 0.0400 | 0.0350 | keep LightGBM alone |" in text


def test_the_ensemble_module_runs_without_xgboost() -> None:
    script = (
        "import sys, django, os;"
        "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.test');"
        "django.setup();"
        "import apps.research.ensemble, apps.research.management.commands.ensemble_check;"
        "print('xgboost' in sys.modules)"
    )
    shown = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )

    assert shown.stdout.strip() == "False"


@pytest.fixture
def command(rank_caches: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(ensemble_check, "DEFAULT_CONFIG", CONFIG)
    StockMaster.objects.create(stock_id="1101", name="1101", market="TWSE", industry="水泥工業")
    DailyPrice.objects.create(
        stock_id="1101", date=weekdays()[-1], close=10, volume=1000, turnover=10_000
    )
    return rank_caches


def test_ensemble_check_writes_the_report(command: Path, tmp_path: Path) -> None:
    report = tmp_path / "reports" / "ensemble.md"
    out = StringIO()

    call_command("ensemble_check", run_key="rank-wf1", output=str(report), stdout=out)

    ics = {horizon: ensemble_ics("rank-wf1", chosen(), horizon) for horizon in (5, 20)}
    assert report.read_text() == ensemble_report("rank-wf1", ics)
    assert f"wrote {report}" in out.getvalue()


def test_ensemble_check_refuses_other_scopes_missing_caches_and_no_research_folds(
    command: Path,
) -> None:
    with pytest.raises(CommandError, match="start with rank-"):
        call_command("ensemble_check", run_key="interval-wf1", stdout=StringIO())
    with pytest.raises(CommandError, match=r"rank-none/baseline .* folds \[1, 2\]"):
        call_command("ensemble_check", run_key="rank-none", stdout=StringIO())
    with pytest.raises(CommandError, match="no research folds"):
        call_command(
            "ensemble_check", run_key="rank-wf1", last_date="2016-09-20", stdout=StringIO()
        )


def test_research_folds_leave_tuning_and_holdout_out(command: Path) -> None:
    folds = research_folds(CONFIG, None)

    assert [fold.index for fold in folds] == [fold.index for fold in chosen()]
    with pytest.raises(ValueError, match="no research folds"):
        research_folds(CONFIG, date(2016, 9, 20))
