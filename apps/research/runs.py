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
from apps.research.holdout import require_key
from apps.research.models import HoldoutRun, ModelRun, ModelStatus
from apps.research.ranges import years

LIBRARIES = ("django", "numpy", "pandas", "pyarrow", "scikit-learn")
MODEL_SUFFIX = ".pkl"
CHUNK = 1 << 20


class RunError(RuntimeError):
    """A run cannot be written or dropped without losing a record that must stay."""


def lib_versions() -> dict[str, str]:
    """Python and library versions of this process."""
    found = {name: importlib.metadata.version(name) for name in LIBRARIES}
    return {"python": platform.python_version(), **found}


def feature_hash(start: date, end: date, directory: Path) -> str:
    """SHA-256 of the yearly feature partitions that [start, end] reads, in year order."""
    digest = hashlib.sha256()
    for low, _ in years(start, end):
        path = existing_partition(directory, low.year)
        digest.update(f"year={low.year}\n".encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(CHUNK), b""):
                digest.update(chunk)
    return digest.hexdigest()


def model_path(run_id: str) -> Path:
    """Model file of run_id under MODEL_DIR."""
    return Path(settings.MODEL_DIR) / f"{require_key(run_id)}{MODEL_SUFFIX}"


def save_model(run_id: str, model: Any) -> Path:
    """Pickle model to its run_id file atomically."""
    path = model_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as written:
            pickle.dump(model, written, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def load_model(run_id: str) -> Any:
    """The model pickled under run_id; only load files this platform wrote."""
    with model_path(run_id).open("rb") as handle:
        return pickle.load(handle)  # noqa: S301


def save_run(run: ModelRun) -> ModelRun:
    """Insert run, or replace a candidate with its run_id; other statuses are never overwritten."""
    require_key(run.run_id)
    require_key(run.run_key)
    run.full_clean(validate_unique=False)
    with transaction.atomic():
        stored = ModelRun.objects.select_for_update().filter(run_id=run.run_id).first()
        if stored is not None and stored.status != ModelStatus.CANDIDATE:
            raise RunError(f"{run.run_id} is {stored.status}; only candidates are rerun")
        run.save()
    return run


def drop_run(run_key: str) -> int:
    """Delete a run's model_run rows, model files and cached predictions; holdout runs stay."""
    require_key(run_key)
    if HoldoutRun.objects.filter(run_key=run_key).exists():
        raise RunError(f"{run_key} holds a holdout run, which is recorded once and kept")
    rows = ModelRun.objects.filter(run_key=run_key)
    kept = sorted(set(rows.exclude(status=ModelStatus.CANDIDATE).values_list("status", flat=True)))
    if kept:
        raise RunError(f"{run_key} has {', '.join(kept)} models; retire them by hand instead")
    run_ids = list(rows.values_list("run_id", flat=True))
    with transaction.atomic():
        dropped, _ = rows.delete()
    for run_id in run_ids:
        model_path(run_id).unlink(missing_ok=True)
    shutil.rmtree(Path(settings.RESEARCH_CACHE_DIR) / run_key, ignore_errors=True)
    return dropped
