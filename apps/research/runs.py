"""model_run rows, model files named by run_id and run cleanup (SPEC §7 模型管理)."""

import hashlib
import importlib.metadata
import os
import pickle
import platform
import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

from django.conf import settings
from django.db import transaction

from apps.research.features.store import existing_partition
from apps.research.holdout import HoldoutTicket, authorize_folds, is_holdout, require_key
from apps.research.models import BacktestMetric, HoldoutRun, ModelRun, ModelStatus
from apps.research.prediction_cache import fold_cached, fold_record
from apps.research.ranges import years
from apps.research.walkforward import Fold

LIBRARIES = ("django", "numpy", "pandas", "pyarrow", "scikit-learn")
MODEL_SUFFIX = ".pkl"
CHUNK = 1 << 20
SCREENING_MODEL = "screening"


class RunError(RuntimeError):
    """A run cannot be written or dropped without losing a record that must stay."""


def lib_versions() -> dict[str, str]:
    """Python and library versions of this process."""
    found = {name: importlib.metadata.version(name) for name in LIBRARIES}
    return {"python": platform.python_version(), **found}


def feature_hash(start: date, end: date, directory: Path) -> str:
    """SHA-256 of the yearly feature partitions that [start, end] reads, in year order."""
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    digest = hashlib.sha256()
    for low, _ in years(start, end):
        path = existing_partition(directory, low.year)
        digest.update(f"year={low.year}\n".encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(CHUNK), b""):
                digest.update(chunk)
    return digest.hexdigest()


def run_dir(run_key: str) -> Path:
    """Directory of one run's model files under MODEL_DIR."""
    return Path(settings.MODEL_DIR) / require_key(run_key)


def model_path(run_key: str, run_id: str) -> Path:
    """Model file named by run_id in its run's directory."""
    return run_dir(run_key) / f"{require_key(run_id)}{MODEL_SUFFIX}"


def load_model(run_key: str, run_id: str) -> Any:
    """The model pickled for run_id; only load files this platform wrote."""
    with model_path(run_key, run_id).open("rb") as handle:
        return pickle.load(handle)  # noqa: S301


def _pickled(path: Path, model: Any) -> str:
    """Temporary file beside path holding the pickled model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as written:
            pickle.dump(model, written, protocol=pickle.HIGHEST_PROTOCOL)
            written.flush()
            os.fsync(written.fileno())
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return temporary


def _replaceable(run: ModelRun) -> None:
    stored = ModelRun.objects.select_for_update().filter(run_id=run.run_id).first()
    if stored is None:
        return
    if stored.run_key != run.run_key:
        raise RunError(f"{run.run_id} belongs to run {stored.run_key}, not {run.run_key}")
    if stored.status != ModelStatus.CANDIDATE:
        raise RunError(f"{run.run_id} is {stored.status}; only candidates are rerun")


def _writable(run_key: str, fold: Fold, ticket: HoldoutTicket | None) -> None:
    """A holdout fold needs its run's ticket and is written until any model caches it."""
    authorize_folds([fold], run_key, ticket)
    if is_holdout(fold) and fold_cached(run_key, fold.index):
        raise RunError(f"holdout fold {fold.index} of {run_key} is cached; it is recorded once")


def record_run(
    run: ModelRun,
    model: Any | None = None,
    *,
    fold: Fold | None = None,
    ticket: HoldoutTicket | None = None,
) -> ModelRun:
    """Write a candidate run of a fold with its model file, replacing only a candidate."""
    path = model_path(run.run_key, run.run_id)
    if run.status != ModelStatus.CANDIDATE:
        raise RunError(f"runs are recorded as candidates, not {run.status}")
    if fold is not None:
        run.params = {**run.params, "fold": fold_record(fold)}
    run.full_clean(validate_unique=False)
    if fold is not None:
        _writable(run.run_key, fold, ticket)
    temporary = None if model is None else _pickled(path, model)
    try:
        with transaction.atomic(durable=True):
            _replaceable(run)
            run.save()
            if temporary is None:
                path.unlink(missing_ok=True)
            else:
                os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return run


def _remove(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)


def drop_run(run_key: str) -> int:
    """Delete a run's model_run rows, backtest metrics, model files and cached predictions.

    Holdout runs stay, and so does the feature screening once any holdout run exists.
    """
    models, cache = run_dir(run_key), Path(settings.RESEARCH_CACHE_DIR) / run_key
    with transaction.atomic(durable=True):
        if HoldoutRun.objects.filter(run_key=run_key).exists():
            raise RunError(f"{run_key} holds a holdout run, which is recorded once and kept")
        rows = list(ModelRun.objects.select_for_update().filter(run_key=run_key))
        if any(row.model_type == SCREENING_MODEL for row in rows) and HoldoutRun.objects.exists():
            raise RunError(f"{run_key} holds the feature screening, which a holdout run freezes")
        kept = sorted({row.status for row in rows if row.status != ModelStatus.CANDIDATE})
        if kept:
            raise RunError(
                f"{run_key} has {', '.join(kept)} models, which drop_run keeps; "
                "change or remove them by hand"
            )
        ModelRun.objects.filter(run_id__in=[row.run_id for row in rows]).delete()
        BacktestMetric.objects.filter(run_key=run_key).delete()
    _remove(models)
    _remove(cache)
    return len(rows)
