"""Per-fold prediction cache in Parquet, so metrics rerun without retraining (SPEC §7)."""

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from django.conf import settings

from apps.research.holdout import (
    SPEC_HOLDOUT_START,
    HoldoutError,
    HoldoutTicket,
    authorize_folds,
    is_holdout,
    require_key,
)
from apps.research.walkforward import Fold

KEYS = ["date", "stock_id"]
FOLD_METADATA = b"fold"
NOTES_METADATA = b"notes"


class MissingFoldsError(LookupError):
    """Some folds have no cached predictions."""

    def __init__(self, run_key: str, folds: list[int]) -> None:
        super().__init__(f"{run_key} has no cached predictions for folds {folds}")
        self.folds = folds


class FoldMismatchError(ValueError):
    """Cached predictions were written for other fold boundaries."""


def fold_path(run_key: str, index: int) -> Path:
    """Cache file of one fold under RESEARCH_CACHE_DIR/run_key."""
    return Path(settings.RESEARCH_CACHE_DIR) / require_key(run_key) / f"fold-{index:03d}.parquet"


def fold_record(fold: Fold) -> dict[str, object]:
    return {
        name: value.isoformat() if isinstance(value, date) else value
        for name, value in asdict(fold).items()
    }


def _check(fold: Fold, predictions: pd.DataFrame) -> None:
    if predictions.empty:
        raise ValueError(f"fold {fold.index} has no predictions")
    absent = [column for column in KEYS if column not in predictions.columns]
    if absent:
        raise ValueError(f"predictions need the columns {absent}")
    days = predictions["date"]
    if not days.map(lambda day: type(day) is date).all():
        raise ValueError("prediction dates must be datetime.date values")
    outside = days[(days < fold.test_start) | (days > fold.test_end)]
    if not outside.empty:
        raise ValueError(
            f"fold {fold.index} has predictions outside its test period on {outside.iloc[0]}"
        )
    if predictions.duplicated(KEYS).any():
        raise ValueError("predictions repeat a (date, stock_id) pair")


def save_fold_predictions(
    run_key: str,
    fold: Fold,
    predictions: pd.DataFrame,
    ticket: HoldoutTicket | None = None,
    notes: Mapping[str, str] | None = None,
) -> Path:
    """Write one fold's predictions atomically, without the index; holdout folds need a ticket."""
    path = fold_path(run_key, fold.index)
    authorize_folds([fold], run_key, ticket)
    once = is_holdout(fold)
    if (once and path.exists()) or _holds_holdout(path):
        raise HoldoutError(f"holdout fold {fold.index} of {run_key} is written once and exists")
    _check(fold, predictions)
    table = pa.Table.from_pandas(predictions, preserve_index=False)
    metadata = {
        **(table.schema.metadata or {}),
        FOLD_METADATA: json.dumps(fold_record(fold)).encode(),
        NOTES_METADATA: json.dumps(dict(notes or {})).encode(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp")
    os.close(handle)
    try:
        pq.write_table(table.replace_schema_metadata(metadata), temporary)
        with open(temporary, "rb") as written:
            os.fsync(written.fileno())
        _publish(temporary, path, once)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def _publish(temporary: str, path: Path, once: bool) -> None:
    if not once:
        os.replace(temporary, path)
        return
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise HoldoutError(f"holdout fold file {path.name} appeared while writing") from error


def _holds_holdout(path: Path) -> bool:
    stored = _stored_record(path) or {}
    return bool(stored.get("holdout")) or str(stored.get("test_end", "")) >= str(SPEC_HOLDOUT_START)


def _stored_record(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    record: dict[str, object] = json.loads(
        (pq.read_schema(path).metadata or {}).get(FOLD_METADATA, b"{}")
    )
    return record


def fold_notes(run_key: str, index: int) -> dict[str, str]:
    """Notes saved with a fold's predictions; empty without a file or notes."""
    path = fold_path(run_key, index)
    if not path.exists():
        return {}
    metadata = pq.read_schema(path).metadata or {}
    notes: dict[str, str] = json.loads(metadata.get(NOTES_METADATA, b"{}"))
    return notes


def load_fold_predictions(run_key: str, fold: Fold) -> pd.DataFrame:
    """One fold's cached predictions, refused when the fold boundaries changed since writing."""
    path = fold_path(run_key, fold.index)
    if not path.exists():
        raise MissingFoldsError(run_key, [fold.index])
    table = pq.read_table(path)
    stored: dict[str, object] = json.loads((table.schema.metadata or {}).get(FOLD_METADATA, b"{}"))
    expected = fold_record(fold)
    changed = sorted(
        name for name in expected.keys() | stored.keys() if stored.get(name) != expected.get(name)
    )
    if changed:
        raise FoldMismatchError(
            f"fold {fold.index} of {run_key} was cached with different {changed}; "
            "rebuild the folds it was written with"
        )
    frame: pd.DataFrame = table.to_pandas()
    return frame


def missing_folds(run_key: str, folds: Iterable[Fold]) -> list[int]:
    """Folds to (re)run: no cache file, or one written for other fold boundaries."""
    return [
        fold.index
        for fold in folds
        if _stored_record(fold_path(run_key, fold.index)) != fold_record(fold)
    ]


def load_predictions(run_key: str, folds: Iterable[Fold]) -> pd.DataFrame:
    """Cached predictions of the folds stacked with a fold column; names every missing fold."""
    chosen = list(folds)
    absent = [fold.index for fold in chosen if not fold_path(run_key, fold.index).exists()]
    if absent:
        raise MissingFoldsError(run_key, absent)
    if not chosen:
        return pd.DataFrame(columns=[*KEYS, "fold"])
    frames = [load_fold_predictions(run_key, fold).assign(fold=fold.index) for fold in chosen]
    return pd.concat(frames, ignore_index=True)
