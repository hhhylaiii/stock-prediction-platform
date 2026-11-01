"""feature_daily writes and its yearly Parquet partitions (SPEC §6 處理步驟 6)."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from django.db import transaction

from apps.etl.upsert import upsert
from apps.research.features.build import FEATURE_COLUMNS, FRAME_COLUMNS
from apps.research.models import FeatureDaily

KEYS = ["stock_id", "date"]
PARTITION_FILE = "part-0.parquet"


def write_features(frame: pd.DataFrame, start: date, end: date) -> int:
    """Replace the feature_daily rows dated in [start, end]; ±inf is stored as NULL."""
    if not frame["date"].between(start, end).all():
        raise ValueError(f"features fall outside {start}..{end}")
    values = [column for column in FRAME_COLUMNS if column not in KEYS]
    finite = frame[FRAME_COLUMNS].replace([np.inf, -np.inf], np.nan)
    with transaction.atomic():
        FeatureDaily.objects.filter(date__range=(start, end)).delete()
        return upsert(FeatureDaily, finite, unique_fields=KEYS, update_fields=values)


def export_year(year: int, directory: Path) -> Path:
    """Rewrite one year's partition from feature_daily, so the file always matches the table."""
    rows = FeatureDaily.objects.filter(date__year=year).order_by("date", "stock_id")
    frame = pd.DataFrame(list(rows.values_list(*FRAME_COLUMNS)), columns=FRAME_COLUMNS)
    typed = frame.astype({"in_universe": bool, **dict.fromkeys(FEATURE_COLUMNS, "float64")})
    target = directory / f"year={year}" / PARTITION_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{PARTITION_FILE}.partial")
    typed.to_parquet(partial, index=False)
    partial.replace(target)
    return target
