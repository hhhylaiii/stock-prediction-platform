"""feature_daily writes and its yearly Parquet partitions (SPEC §6 處理步驟 6)."""

import logging
import os
import shutil
import tempfile
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from django.db import transaction

from apps.etl.upsert import upsert
from apps.research.features.build import (
    FEATURE_COLUMNS,
    FIRST_BATCH_COLUMNS,
    FRAME_COLUMNS,
    FRAME_TYPES,
)
from apps.research.models import FeatureDaily
from apps.research.ranges import months, years

KEYS = ["stock_id", "date"]
PARTITION_FILE = "part-0.parquet"
logger = logging.getLogger(__name__)


def _finite(frame: pd.DataFrame) -> pd.DataFrame:
    """Frame with ±inf as NaN, logging how many cells each feature lost."""
    counts = np.isinf(frame[FEATURE_COLUMNS]).sum()
    lost = ", ".join(f"{name}: {count}" for name, count in counts[counts > 0].items())
    if lost:
        logger.warning("feature_daily stores non-finite values as NULL, %s", lost)
    return frame[FRAME_COLUMNS].replace([np.inf, -np.inf], np.nan)


def write_features(frame: pd.DataFrame, start: date, end: date) -> int:
    """Replace the feature_daily rows dated in [start, end], a month of objects at a time."""
    if not frame["date"].between(start, end).all():
        raise ValueError(f"features fall outside {start}..{end}")
    values = [column for column in FRAME_COLUMNS if column not in KEYS]
    finite = _finite(frame)
    month = pd.to_datetime(finite["date"]).dt.to_period("M")
    with transaction.atomic():
        FeatureDaily.objects.filter(date__range=(start, end)).delete()
        return sum(
            upsert(FeatureDaily, part, unique_fields=KEYS, update_fields=values)
            for _, part in finite.groupby(month, sort=True)
        )


def _typed(parts: list[pd.DataFrame], columns: Sequence[str] = FRAME_COLUMNS) -> pd.DataFrame:
    types = {name: FRAME_TYPES[name] for name in columns if name in FRAME_TYPES}
    if not parts:
        return pd.DataFrame(columns=list(columns)).astype(types)
    return pd.concat([part.astype(types) for part in parts], ignore_index=True)


def read_features(start: date, end: date) -> pd.DataFrame:
    """feature_daily rows dated in [start, end], read a month at a time."""
    parts = []
    for low, high in months(start, end):
        rows = FeatureDaily.objects.filter(date__range=(low, high))
        listed = list(rows.order_by("date", "stock_id").values_list(*FRAME_COLUMNS))
        if listed:
            parts.append(pd.DataFrame(listed, columns=FRAME_COLUMNS))
    return _typed(parts)


def partition_path(directory: Path, year: int) -> Path:
    """The year's partition file under directory."""
    return directory / f"year={year}" / PARTITION_FILE


def existing_partition(directory: Path, year: int) -> Path:
    """The year's partition file, which must exist before training reads or hashes it."""
    path = partition_path(directory, year)
    if not path.exists():
        raise FileNotFoundError(
            f"no feature partition for {year} in {directory}; run build_features first"
        )
    return path


def _read_partition(directory: Path, low: date, high: date, wanted: list[str]) -> pd.DataFrame:
    """Rows of low..high from one yearly partition, schema and table read through one handle."""
    path = existing_partition(directory, low.year)
    with path.open("rb") as handle:
        stored = set(pq.ParquetFile(handle).schema_arrow.names)
        missing = [name for name in wanted if name not in stored]
        if missing:
            raise ValueError(
                f"feature partition for {low.year} in {directory} lacks {', '.join(missing)}; "
                f"run build_features --start {low.year}-01-01 --end {low.year}-12-31"
            )
        dated = [("date", ">=", low), ("date", "<=", high)]
        frame: pd.DataFrame = pq.read_table(handle, columns=wanted, filters=dated).to_pandas()
    return frame


def read_partitions(
    start: date, end: date, directory: Path, columns: Sequence[str] = FIRST_BATCH_COLUMNS
) -> pd.DataFrame:
    """Rows dated in [start, end] from the yearly partitions; the first batch unless asked."""
    wanted = list(columns)
    parts = [_read_partition(directory, low, high, wanted) for low, high in years(start, end)]
    return _typed(parts, wanted)


def read_columns(start: date, end: date, directory: Path, columns: Sequence[str]) -> pd.DataFrame:
    """stock_id, date and the named feature columns of rows dated in [start, end]."""
    unknown = [name for name in columns if name not in FEATURE_COLUMNS]
    if unknown:
        raise ValueError(f"{unknown} are not feature columns")
    wanted = [*KEYS, *columns]
    types = dict.fromkeys(columns, "float64")
    parts = [
        _read_partition(directory, low, high, wanted).astype(types)
        for low, high in years(start, end)
    ]
    if not parts:
        return pd.DataFrame(columns=wanted).astype(types)
    return pd.concat(parts, ignore_index=True)


def export_year(year: int, directory: Path) -> Path | None:
    """Rewrite one year's partition from feature_daily; a year without rows has no partition."""
    target = partition_path(directory, year)
    frame = read_features(date(year, 1, 1), date(year, 12, 31))
    if frame.empty:
        shutil.rmtree(target.parent, ignore_errors=True)
        return None
    table = pa.Table.from_pandas(frame, preserve_index=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".part-", suffix=".tmp")
    os.close(handle)
    try:
        pq.write_table(table, temporary)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target
