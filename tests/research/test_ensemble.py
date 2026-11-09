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
    DAILY_COLUMNS,
    adopt_ensemble,
    ensemble_daily_ics,
    ensemble_report,
    rank_average,
    research_folds,
)
from apps.research.features.store import export_year
from apps.research.metrics.ic import daily_rank_ic
from apps.research.models import FeatureDaily, LabelDaily
from apps.research.prediction_cache import (
    MissingFoldsError,
    fold_notes,
    fold_path,
    load_fold_predictions,
    save_fold_predictions,
)
from apps.research.training import training_rows
from tests.research.run_panels import CONFIG, research, use_designs, weekdays

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


def test_ensemble_daily_ics_score_each_fold_on_universe_ranks(rank_caches: Path) -> None:
    folds = chosen()

    daily = ensemble_daily_ics("rank-wf1", folds, 20)

    assert daily.columns.tolist() == DAILY_COLUMNS
    assert set(daily["horizon"]) == {20}
    columns = {"ridge": "ridge_h20", "lightgbm": "lightgbm_h20", "ensemble": "ensemble"}
    for fold in folds:
        frame = merged_caches(fold)
        frame = frame.assign(ensemble=rank_average(frame, ["ridge_h20", "lightgbm_h20"]))
        inside = frame[frame["date"] <= fold.eval_end]
        rows = training_rows(inside, labels(fold, 20), "fwd_rank_pct")
        mine = daily[daily["fold"] == fold.index].set_index("date")
        for name, column in columns.items():
            expected = daily_rank_ic(rows, column).to_dict()
            assert mine[name].dropna().to_dict() == pytest.approx(expected), (fold.index, name)
    assert daily["lightgbm"].mean() > 0.3


def test_ensemble_ranks_the_whole_universe_before_dropping_untradable_rows(
    rank_caches: Path,
) -> None:
    folds = chosen()
    fold = folds[0]
    held = [f"{1100 + k}" for k in range(0, 40, 8)]
    LabelDaily.objects.filter(
        date__range=(fold.test_start, fold.eval_end), stock_id__in=held
    ).update(untradable_t1=True)

    daily = ensemble_daily_ics("rank-wf1", [fold], 5)

    frame = merged_caches(fold)
    frame = frame[frame["date"] <= fold.eval_end]
    whole = frame.assign(ensemble=rank_average(frame, ["ridge_h5", "lightgbm_h5"]))
    kept = training_rows(whole, labels(fold, 5), "fwd_rank_pct")
    late = training_rows(frame, labels(fold, 5), "fwd_rank_pct")
    late = late.assign(ensemble=rank_average(late, ["ridge_h5", "lightgbm_h5"]))
    assert daily["ensemble"].mean() == pytest.approx(daily_rank_ic(kept, "ensemble").mean())
    assert daily["ensemble"].mean() != pytest.approx(
        daily_rank_ic(late, "ensemble").mean(), abs=1e-9
    )


def test_ensemble_refuses_caches_built_on_other_feature_partitions(rank_caches: Path) -> None:
    folds = chosen()
    FeatureDaily.objects.filter(date=folds[0].test_start).update(tech_ret_60d=0.5)
    export_year(2016, rank_caches)
    baseline_runs.run_folds("rank", folds, "rank-wf1", rank_caches)

    with pytest.raises(ValueError, match=rf"folds \[{folds[0].index}.* features or labels"):
        ensemble_daily_ics("rank-wf1", folds, 5)


def test_ensemble_refuses_caches_fit_on_other_labels(rank_caches: Path) -> None:
    folds = chosen()
    LabelDaily.objects.filter(date=folds[0].train_start, horizon=5).update(fwd_rank_pct=0.5)
    baseline_runs.run_folds("rank", folds, "rank-wf1", rank_caches)

    with pytest.raises(ValueError, match=rf"folds \[{folds[0].index}.* features or labels"):
        ensemble_daily_ics("rank-wf1", folds, 5)


def test_ensemble_names_the_model_whose_cache_is_missing(rank_caches: Path) -> None:
    folds = chosen()
    fold_path("rank-wf1", "lightgbm", folds[-1].index).unlink()

    with pytest.raises(MissingFoldsError, match=rf"rank-wf1/lightgbm .* \[{folds[-1].index}\]"):
        ensemble_daily_ics("rank-wf1", folds, 5)


def test_ensemble_refuses_caches_holding_different_rows(rank_caches: Path) -> None:
    fold = chosen()[0]
    kept = load_fold_predictions("rank-wf1", "lightgbm", fold).iloc[5:]
    notes = fold_notes("rank-wf1", "lightgbm", fold.index)
    save_fold_predictions("rank-wf1", "lightgbm", fold, kept, notes)

    with pytest.raises(ValueError, match="hold different rows"):
        ensemble_daily_ics("rank-wf1", [fold], 5)


def test_ensemble_report_pools_days_and_lists_each_fold() -> None:
    day, later = date(2016, 10, 3), date(2016, 10, 4)
    rows = [
        [1, 5, day, 0.01, 0.02, 0.04],
        [1, 5, later, 0.01, 0.02, 0.04],
        [2, 5, day, 0.01, 0.10, 0.07],
        [1, 20, day, 0.03, 0.04, 0.035],
    ]

    text = ensemble_report("rank-wf1", pd.DataFrame(rows, columns=DAILY_COLUMNS))

    assert text.startswith("# Rank-average ensemble: rank-wf1")
    assert "| 5 | 0.0100 | 0.0467 | 0.0500 | adopt |" in text
    assert "| 20 | 0.0300 | 0.0400 | 0.0350 | keep LightGBM alone |" in text
    assert "| 1 | 5 | 0.0100 | 0.0200 | 0.0400 |" in text
    assert "| 2 | 5 | 0.0100 | 0.1000 | 0.0700 |" in text


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
    use_designs(monkeypatch)
    StockMaster.objects.create(stock_id="1101", name="1101", market="TWSE", industry="水泥工業")
    DailyPrice.objects.create(
        stock_id="1101", date=weekdays()[-1], close=10, volume=1000, turnover=10_000
    )
    return rank_caches


def test_ensemble_check_writes_the_report(command: Path, tmp_path: Path) -> None:
    report = tmp_path / "reports" / "ensemble.md"
    out = StringIO()

    call_command("ensemble_check", run_key="rank-wf1", output=str(report), stdout=out)

    daily = [ensemble_daily_ics("rank-wf1", chosen(), horizon) for horizon in (5, 20)]
    expected = ensemble_report("rank-wf1", pd.concat(daily, ignore_index=True))
    assert report.read_text() == expected
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
