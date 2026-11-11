"""Per-fold prediction cache in Parquet, a file per model of a run, so metrics skip refits (§7)."""

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
    ProductionPass,
    authorize_folds,
    authorize_production,
    is_holdout,
    require_key,
)
from apps.research.walkforward import Fold

KEYS = ["date", "stock_id"]
FOLD_METADATA = b"fold"
NOTES_METADATA = b"notes"
LAYOUT = ("train_start", "test_start")


class MissingFoldsError(LookupError):
    """Some folds have no cached predictions."""

    def __init__(self, run_key: str, model: str, folds: list[int]) -> None:
        super().__init__(f"{run_key}/{model} has no cached predictions for folds {folds}")
        self.folds = folds


class FoldMismatchError(ValueError):
    """Cached predictions were written for other fold boundaries."""


def _name(index: int) -> str:
    return f"fold-{index:03d}.parquet"


def fold_path(run_key: str, model: str, index: int) -> Path:
    """Cache file of one model's fold under RESEARCH_CACHE_DIR/run_key/model."""
    root = Path(settings.RESEARCH_CACHE_DIR) / require_key(run_key)
    return root / require_key(model) / _name(index)


def _marker(run_key: str, index: int) -> Path:
    """File a holdout write takes before publishing; it stays once the fold is written."""
    return Path(settings.RESEARCH_CACHE_DIR) / require_key(run_key) / f"fold-{index:03d}.holdout"


def fold_cached(run_key: str, index: int) -> bool:
    """Whether any model of the run has cached this fold, or a holdout write has taken it."""
    marker = _marker(run_key, index)
    return marker.exists() or any(marker.parent.glob(f"*/{_name(index)}"))


def fold_written(run_key: str, index: int, models: Iterable[str]) -> bool:
    """Whether a holdout write finished: its marker and every model's file are in place."""
    return _marker(run_key, index).exists() and all(
        fold_path(run_key, model, index).exists() for model in models
    )


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


def _staged(path: Path, fold: Fold, predictions: pd.DataFrame, notes: Mapping[str, str]) -> str:
    """Temporary file beside path holding the predictions, the fold record and the notes."""
    table = pa.Table.from_pandas(predictions, preserve_index=False)
    metadata = {
        **(table.schema.metadata or {}),
        FOLD_METADATA: json.dumps(fold_record(fold)).encode(),
        NOTES_METADATA: json.dumps(dict(notes)).encode(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp")
    os.close(handle)
    try:
        pq.write_table(table.replace_schema_metadata(metadata), temporary)
        with open(temporary, "rb") as written:
            os.fsync(written.fileno())
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return temporary


def save_fold_predictions(
    run_key: str,
    model: str,
    fold: Fold,
    predictions: pd.DataFrame,
    notes: Mapping[str, str] | None = None,
    production: ProductionPass | None = None,
) -> Path:
    """Write one model's research or production fold atomically, without the index."""
    path = fold_path(run_key, model, fold.index)
    if production is not None:
        authorize_production(run_key, production, fold)
    elif is_holdout(fold):
        raise HoldoutError(f"fold {fold.index} is holdout; save_holdout_fold writes its models")
    elif _holds_holdout(path) or _marker(run_key, fold.index).exists():
        raise HoldoutError(f"holdout fold {fold.index} of {run_key} is written once and exists")
    _check(fold, predictions)
    temporary = _staged(path, fold, predictions, notes or {})
    try:
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def save_holdout_fold(
    run_key: str,
    fold: Fold,
    predictions: Mapping[str, pd.DataFrame],
    ticket: HoldoutTicket | None,
    notes: Mapping[str, Mapping[str, str]] | None = None,
) -> list[Path]:
    """Write every model's predictions of a holdout fold in one call; the fold is written once."""
    if not is_holdout(fold):
        raise ValueError(f"fold {fold.index} is not holdout; save_fold_predictions writes it")
    authorize_folds([fold], run_key, ticket)
    paths = {model: fold_path(run_key, model, fold.index) for model in predictions}
    if not paths:
        raise ValueError(f"holdout fold {fold.index} needs the predictions of its models")
    stray = sorted(set(notes or {}) - paths.keys())
    if stray:
        raise ValueError(f"notes name models without predictions: {stray}")
    if fold_cached(run_key, fold.index):
        raise HoldoutError(f"holdout fold {fold.index} of {run_key} is written once and exists")
    for frame in predictions.values():
        _check(fold, frame)
    staged: dict[str, str] = {}
    try:
        for model, frame in predictions.items():
            staged[model] = _staged(paths[model], fold, frame, (notes or {}).get(model, {}))
        _publish(staged, paths, _marker(run_key, fold.index))
    finally:
        for temporary in staged.values():
            Path(temporary).unlink(missing_ok=True)
    return list(paths.values())


def _publish(staged: Mapping[str, str], paths: Mapping[str, Path], marker: Path) -> None:
    """Take the fold's marker, then link every model; a failure removes what this call linked."""
    try:
        os.close(os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError as error:
        raise HoldoutError(f"holdout fold {marker.stem} is being written elsewhere") from error
    linked: list[Path] = []
    try:
        for model, temporary in staged.items():
            _link(temporary, paths[model])
            linked.append(paths[model])
    except BaseException:
        for path in linked:
            path.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
        raise


def _link(temporary: str, path: Path) -> None:
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise HoldoutError(
            f"holdout fold file {path.parent.name}/{path.name} appeared while writing"
        ) from error


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


def fold_notes(run_key: str, model: str, index: int) -> dict[str, str]:
    """Notes saved with a model's fold; empty without a file or notes."""
    path = fold_path(run_key, model, index)
    if not path.exists():
        return {}
    metadata = pq.read_schema(path).metadata or {}
    notes: dict[str, str] = json.loads(metadata.get(NOTES_METADATA, b"{}"))
    return notes


def load_fold_predictions(run_key: str, model: str, fold: Fold) -> pd.DataFrame:
    """A model's cached fold, refused when the fold boundaries changed since writing."""
    path = fold_path(run_key, model, fold.index)
    if not path.exists():
        raise MissingFoldsError(run_key, model, [fold.index])
    table = pq.read_table(path)
    stored: dict[str, object] = json.loads((table.schema.metadata or {}).get(FOLD_METADATA, b"{}"))
    expected = fold_record(fold)
    changed = sorted(
        name for name in expected.keys() | stored.keys() if stored.get(name) != expected.get(name)
    )
    if changed:
        raise FoldMismatchError(
            f"fold {fold.index} of {run_key}/{model} was cached with different {changed}; "
            "rebuild the folds it was written with"
        )
    frame: pd.DataFrame = table.to_pandas()
    return frame


def missing_folds(run_key: str, model: str, folds: Iterable[Fold]) -> list[int]:
    """A model's folds to (re)run: no cache file, or one written for other fold boundaries."""
    return [
        fold.index
        for fold in folds
        if _stored_record(fold_path(run_key, model, fold.index)) != fold_record(fold)
    ]


def other_layout(run_key: str, folds: Iterable[Fold]) -> list[int]:
    """Folds any model of the run cached for another layout: a different train or test start."""
    root = Path(settings.RESEARCH_CACHE_DIR) / require_key(run_key)
    moved = []
    for fold in folds:
        expected = fold_record(fold)
        stored = [_stored_record(path) for path in root.glob(f"*/{_name(fold.index)}")]
        records = [record for record in stored if record is not None]
        if any(record.get(name) != expected[name] for record in records for name in LAYOUT):
            moved.append(fold.index)
    return moved


def load_predictions(run_key: str, model: str, folds: Iterable[Fold]) -> pd.DataFrame:
    """A model's cached folds stacked with a fold column; names every missing fold."""
    chosen = list(folds)
    absent = [fold.index for fold in chosen if not fold_path(run_key, model, fold.index).exists()]
    if absent:
        raise MissingFoldsError(run_key, model, absent)
    if not chosen:
        return pd.DataFrame(columns=[*KEYS, "fold"])
    frames = [
        load_fold_predictions(run_key, model, fold).assign(fold=fold.index) for fold in chosen
    ]
    return pd.concat(frames, ignore_index=True)
