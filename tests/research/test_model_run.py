import importlib.metadata
import math
import platform
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from django.core.exceptions import ValidationError
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import IntegrityError, transaction

from apps.research.features.store import export_year
from apps.research.holdout import HoldoutError, claim_holdout
from apps.research.models import BacktestMetric, FeatureDaily, ModelRun
from apps.research.prediction_cache import fold_path, fold_record
from apps.research.runs import (
    LIBRARIES,
    SCREENING_MODEL,
    RunError,
    drop_run,
    feature_hash,
    lib_versions,
    load_model,
    model_path,
    record_run,
)
from apps.research.walkforward import Fold

pytestmark = pytest.mark.django_db
SPEC_COLUMNS = {
    "run_id",
    "model_type",
    "objective",
    "version",
    "train_start",
    "train_end",
    "params",
    "feature_list",
    "seed",
    "lib_versions",
    "feature_hash",
    "status",
    "ic_mean",
    "icir",
}


def d(text: str) -> date:
    return date.fromisoformat(text)


@pytest.fixture
def artifacts(settings: Any, tmp_path: Path) -> Path:
    settings.MODEL_DIR = tmp_path / "models"
    settings.RESEARCH_CACHE_DIR = tmp_path / "predictions"
    return tmp_path


def run(run_id: str = "wf1.ridge.h20.f000", run_key: str = "wf1", **changes: Any) -> ModelRun:
    fields: dict[str, Any] = {
        "run_id": run_id,
        "run_key": run_key,
        "model_type": "ridge",
        "objective": "rank",
        "version": "1",
        "train_start": d("2011-01-03"),
        "train_end": d("2013-12-02"),
        "params": {"penalty": 1.0, "horizon": 20},
        "feature_list": ["tech_ret_1d", "tech_ret_1d_missing"],
        "seed": None,
        "lib_versions": {"python": "3.12.0"},
        "feature_hash": "0" * 64,
        "ic_mean": 0.02,
        "icir": 0.2,
    }
    return ModelRun(**{**fields, **changes})


def test_a_run_keeps_every_spec_column_and_starts_as_a_candidate(artifacts: Path) -> None:
    record_run(run())

    stored = ModelRun.objects.get(run_id="wf1.ridge.h20.f000")
    assert {field.name for field in ModelRun._meta.get_fields()} >= SPEC_COLUMNS
    assert stored.status == "candidate"
    assert stored.params == {"penalty": 1.0, "horizon": 20}
    assert stored.feature_list == ["tech_ret_1d", "tech_ret_1d_missing"]
    assert stored.train_end == d("2013-12-02")
    assert stored.ic_mean == pytest.approx(0.02)
    assert stored.created_at is not None


@pytest.mark.parametrize(
    "changes",
    [
        {"objective": "ranking"},
        {"status": "live"},
        {"train_start": d("2014-01-02")},
        {"ic_mean": math.inf},
        {"icir": -math.inf},
    ],
)
def test_the_database_refuses_invalid_runs(changes: dict[str, Any]) -> None:
    with pytest.raises(IntegrityError), transaction.atomic():
        run(**changes).save()


@pytest.mark.parametrize(
    "changes",
    [
        {"objective": "ranking"},
        {"train_start": d("2014-01-02")},
        {"ic_mean": math.inf},
        {"lib_versions": {}},
    ],
)
def test_invalid_runs_are_refused_before_any_write(
    artifacts: Path, changes: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        record_run(run(**changes), {"weights": [1.0]})

    assert not ModelRun.objects.exists()
    assert not model_path("wf1", "wf1.ridge.h20.f000").exists()


@pytest.mark.parametrize("status", ["active", "degraded", "retired", "live"])
def test_runs_are_recorded_only_as_candidates(artifacts: Path, status: str) -> None:
    with pytest.raises(RunError, match="candidate"):
        record_run(run(status=status))

    assert not ModelRun.objects.exists()


def test_rerunning_a_candidate_replaces_its_row_and_model(artifacts: Path) -> None:
    record_run(run(ic_mean=0.01), {"weights": [1.0]})
    record_run(run(ic_mean=0.03, params={"penalty": 3.0}), {"weights": [3.0]})

    stored = ModelRun.objects.get()
    assert stored.ic_mean == pytest.approx(0.03)
    assert stored.params == {"penalty": 3.0}
    assert load_model("wf1", "wf1.ridge.h20.f000") == {"weights": [3.0]}


def test_a_rerun_without_a_model_removes_the_old_file(artifacts: Path) -> None:
    record_run(run(), {"weights": [1.0]})
    record_run(run())

    assert not model_path("wf1", "wf1.ridge.h20.f000").exists()


@pytest.mark.parametrize("status", ["active", "degraded", "retired"])
def test_runs_past_candidate_keep_their_row_and_model(artifacts: Path, status: str) -> None:
    record_run(run(), {"weights": [1.0]})
    ModelRun.objects.update(status=status)

    with pytest.raises(RunError, match=status):
        record_run(run(ic_mean=0.5), {"weights": [9.0]})

    assert ModelRun.objects.get().ic_mean == pytest.approx(0.02)
    assert load_model("wf1", "wf1.ridge.h20.f000") == {"weights": [1.0]}


def fold(index: int, test_start: str, test_end: str) -> Fold:
    start, end = d(test_start), d(test_end)
    return Fold(
        index=index,
        train_start=start - timedelta(days=1100),
        fit_end=start - timedelta(days=250),
        valid_start=start - timedelta(days=200),
        train_end=start - timedelta(days=30),
        test_start=start,
        test_end=end,
        eval_end=end,
        tuning=False,
        holdout=end >= d("2025-01-01"),
    )


HELD = fold(44, "2025-01-02", "2025-03-31")
STUDIED = fold(43, "2024-10-01", "2024-12-02")


def held_run(**changes: Any) -> ModelRun:
    return run("wf1.ridge.h20.f044", **changes)


def test_runs_remember_their_fold(artifacts: Path) -> None:
    record_run(run("wf1.ridge.h20.f043"), fold=STUDIED)

    stored = ModelRun.objects.get()
    assert stored.params["fold"] == fold_record(STUDIED)
    assert stored.params["penalty"] == 1.0


def test_a_holdout_fold_needs_the_claim_of_its_run(artifacts: Path) -> None:
    with pytest.raises(HoldoutError, match="claim_holdout"):
        record_run(held_run(), {"weights": [1.0]}, fold=HELD)

    assert not ModelRun.objects.exists()
    assert not model_path("wf1", "wf1.ridge.h20.f044").exists()


def test_a_holdout_fold_is_rewritten_until_its_predictions_are_cached(artifacts: Path) -> None:
    ticket = claim_holdout("rank", "wf1", d("2025-03-31"))
    record_run(held_run(ic_mean=0.01), {"weights": [1.0]}, fold=HELD, ticket=ticket)
    record_run(held_run(ic_mean=0.02), {"weights": [2.0]}, fold=HELD, ticket=ticket)
    fold_path("wf1", HELD.index).parent.mkdir(parents=True)
    fold_path("wf1", HELD.index).write_bytes(b"cached")

    with pytest.raises(RunError, match="cached"):
        record_run(held_run(ic_mean=0.9), {"weights": [9.0]}, fold=HELD, ticket=ticket)

    assert ModelRun.objects.get().ic_mean == pytest.approx(0.02)
    assert load_model("wf1", "wf1.ridge.h20.f044") == {"weights": [2.0]}


def test_research_folds_of_a_claimed_run_stay_rewritable(artifacts: Path) -> None:
    claim_holdout("rank", "wf1", d("2025-03-31"))
    record_run(run("wf1.ridge.h20.f043", ic_mean=0.01), fold=STUDIED)
    fold_path("wf1", STUDIED.index).parent.mkdir(parents=True)
    fold_path("wf1", STUDIED.index).write_bytes(b"cached")

    record_run(run("wf1.ridge.h20.f043", ic_mean=0.03), fold=STUDIED)

    assert ModelRun.objects.get().ic_mean == pytest.approx(0.03)


@pytest.mark.django_db(transaction=True)
def test_a_run_inside_an_outer_transaction_is_refused(artifacts: Path) -> None:
    with pytest.raises(RuntimeError, match="durable"), transaction.atomic():
        record_run(run(), {"weights": [1.0]})

    assert not ModelRun.objects.exists()
    assert not model_path("wf1", "wf1.ridge.h20.f000").exists()


def test_a_run_id_stays_with_its_run_key(artifacts: Path) -> None:
    record_run(run())

    with pytest.raises(RunError, match="wf1"):
        record_run(run(run_key="wf2"))


@pytest.mark.parametrize("changes", [{"run_id": "../wf1"}, {"run_key": "WF 1"}])
def test_run_ids_and_keys_must_be_plain_names(artifacts: Path, changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="lowercase"):
        record_run(run(**changes))


def test_models_live_under_their_run_key_and_run_id(artifacts: Path) -> None:
    record_run(run(), {"weights": [1.0, 2.0]})

    path = model_path("wf1", "wf1.ridge.h20.f000")
    assert path == artifacts / "models" / "wf1" / "wf1.ridge.h20.f000.pkl"
    assert load_model("wf1", "wf1.ridge.h20.f000") == {"weights": [1.0, 2.0]}
    assert [item.name for item in path.parent.iterdir()] == [path.name]


def test_model_paths_refuse_unsafe_names(artifacts: Path) -> None:
    with pytest.raises(ValueError, match="lowercase"):
        model_path("wf1", "../evil")
    with pytest.raises(ValueError, match="lowercase"):
        model_path("..", "wf1.ridge.h20.f000")


def test_model_dir_follows_the_settings(settings: Any, tmp_path: Path) -> None:
    settings.MODEL_DIR = tmp_path / "elsewhere"

    assert model_path("wf1", "wf1.ridge.h20.f000").parent == tmp_path / "elsewhere" / "wf1"


def feature(stock_id: str, day: date, value: float) -> FeatureDaily:
    return FeatureDaily(stock_id=stock_id, date=day, in_universe=True, tech_ret_1d=value)


def test_feature_hash_follows_the_partition_files_of_the_range(tmp_path: Path) -> None:
    FeatureDaily.objects.bulk_create(
        [feature("1101", d("2015-12-30"), 1.0), feature("1101", d("2016-01-04"), 2.0)]
    )
    export_year(2015, tmp_path)
    export_year(2016, tmp_path)

    both = feature_hash(d("2015-12-30"), d("2016-01-04"), tmp_path)
    again = feature_hash(d("2015-12-01"), d("2016-06-30"), tmp_path)
    later = feature_hash(d("2016-01-04"), d("2016-01-04"), tmp_path)
    FeatureDaily.objects.filter(date=d("2016-01-04")).update(tech_ret_1d=3.0)
    export_year(2016, tmp_path)
    changed = feature_hash(d("2015-12-30"), d("2016-01-04"), tmp_path)

    assert len(both) == 64
    assert both == again
    assert len({both, later, changed}) == 3


def test_feature_hash_needs_every_partition_of_the_range(tmp_path: Path) -> None:
    FeatureDaily.objects.create(stock_id="1101", date=d("2016-01-04"), in_universe=True)
    export_year(2016, tmp_path)

    with pytest.raises(FileNotFoundError, match=r"2015.*build_features"):
        feature_hash(d("2015-12-30"), d("2016-01-04"), tmp_path)


def test_feature_hash_refuses_a_backward_range(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="after"):
        feature_hash(d("2016-01-04"), d("2015-12-30"), tmp_path)


def test_lib_versions_name_python_and_the_libraries() -> None:
    versions = lib_versions()

    assert versions["python"] == platform.python_version()
    assert {"numpy", "pandas", "pyarrow", "scikit-learn"} <= set(LIBRARIES)
    for name in LIBRARIES:
        assert versions[name] == importlib.metadata.version(name)


def cached(artifacts: Path, run_key: str) -> Path:
    path = artifacts / "predictions" / run_key / "fold-000.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"cached")
    return path


def test_drop_run_removes_its_rows_models_and_cached_predictions(artifacts: Path) -> None:
    for run_id, run_key in [
        ("wf1.ridge.h20.f000", "wf1"),
        ("wf1.ridge.h20.f001", "wf1"),
        ("wf2.ridge.h20.f000", "wf2"),
    ]:
        record_run(run(run_id, run_key), {"run": run_id})
    orphan = model_path("wf1", "wf1.ridge.h20.f009")
    orphan.write_bytes(b"left by an interrupted run")
    dropped, kept = cached(artifacts, "wf1"), cached(artifacts, "wf2")

    assert drop_run("wf1") == 2

    assert list(ModelRun.objects.values_list("run_id", flat=True)) == ["wf2.ridge.h20.f000"]
    assert not (artifacts / "models" / "wf1").exists()
    assert model_path("wf2", "wf2.ridge.h20.f000").exists()
    assert not dropped.parent.exists()
    assert kept.exists()


def backtest(run_key: str) -> BacktestMetric:
    return BacktestMetric.objects.create(
        run_key=run_key,
        model_type="ridge",
        objective="rank",
        horizon=20,
        metric="rank_ic",
        grain="total",
        period_start=d("2014-01-02"),
        period_end=d("2024-12-31"),
        value=0.02,
    )


def test_drop_run_removes_its_backtest_metrics(artifacts: Path) -> None:
    record_run(run(), {"run": 0})
    backtest("wf1")
    backtest("wf2")
    backtest("wf3")

    assert drop_run("wf1") == 1
    assert drop_run("wf3") == 0

    assert list(BacktestMetric.objects.values_list("run_key", flat=True)) == ["wf2"]


def test_drop_run_keeps_a_run_with_a_holdout_claim(artifacts: Path) -> None:
    record_run(run(), {"run": 0})
    backtest("wf1")
    path = cached(artifacts, "wf1")
    claim_holdout("rank", "wf1", d("2025-03-31"))

    with pytest.raises(RunError, match="holdout"):
        drop_run("wf1")

    assert ModelRun.objects.exists()
    assert BacktestMetric.objects.filter(run_key="wf1").exists()
    assert model_path("wf1", "wf1.ridge.h20.f000").exists()
    assert path.exists()


def test_drop_run_keeps_the_screening_once_any_holdout_run_exists(artifacts: Path) -> None:
    for run_key in ["wf1", "wf3"]:
        record_run(run(f"{run_key}.screening", run_key, model_type=SCREENING_MODEL))

    assert drop_run("wf3") == 1

    claim_holdout("rank", "wf2", d("2025-03-31"))

    with pytest.raises(RunError, match="screening"):
        drop_run("wf1")

    assert ModelRun.objects.filter(run_id="wf1.screening").exists()


@pytest.mark.parametrize("status", ["active", "degraded", "retired"])
def test_drop_run_keeps_a_run_past_candidate(artifacts: Path, status: str) -> None:
    record_run(run(), {"run": 0})
    record_run(run("wf1.ridge.h20.f001"))
    ModelRun.objects.filter(run_id="wf1.ridge.h20.f001").update(status=status)
    backtest("wf1")
    path = cached(artifacts, "wf1")

    with pytest.raises(RunError, match=f"{status} models, which drop_run keeps"):
        drop_run("wf1")

    assert ModelRun.objects.count() == 2
    assert BacktestMetric.objects.filter(run_key="wf1").exists()
    assert model_path("wf1", "wf1.ridge.h20.f000").exists()
    assert path.exists()


@pytest.mark.django_db(transaction=True)
def test_drop_run_inside_an_outer_transaction_is_refused(artifacts: Path) -> None:
    record_run(run(), {"run": 0})

    with pytest.raises(RuntimeError, match="durable"), transaction.atomic():
        drop_run("wf1")

    assert ModelRun.objects.exists()
    assert model_path("wf1", "wf1.ridge.h20.f000").exists()


def test_drop_run_command_reports_and_refuses(artifacts: Path) -> None:
    record_run(run())
    record_run(run("wf3.ridge.h20.f000", "wf3"))
    claim_holdout("rank", "wf3", d("2025-03-31"))
    out = StringIO()

    call_command("drop_run", "wf1", stdout=out)

    assert "wf1: dropped 1 model_run rows" in out.getvalue()
    with pytest.raises(CommandError, match="holdout"):
        call_command("drop_run", "wf3")
    with pytest.raises(CommandError, match="lowercase"):
        call_command("drop_run", "../wf1")
