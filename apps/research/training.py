"""Training rows of the linear baselines: tradable universe rows with a label (SPEC §6, §7)."""

from datetime import date
from pathlib import Path

import pandas as pd
from django.conf import settings

from apps.research.features.store import read_partitions
from apps.research.features.transforms import STOCK_LEVEL, fill_with_flags, standardize
from apps.research.labels import HORIZONS, VOL_DAYS
from apps.research.models import LabelDaily
from apps.research.ranges import months

KEYS = ["stock_id", "date"]
LABEL_FIELDS = [
    "fwd_ret",
    "fwd_ret_open",
    "fwd_rank_pct",
    "fwd_vol_20d",
    "fwd_up_5pct",
    "untradable_t1",
]
LINEAR_FEATURES = [*STOCK_LEVEL, *(f"{name}_missing" for name in STOCK_LEVEL)]
TARGET_HORIZONS = {
    "fwd_rank_pct": HORIZONS,
    "fwd_ret_open": HORIZONS,
    "fwd_vol_20d": (VOL_DAYS,),
    "fwd_up_5pct": (VOL_DAYS,),
}


def load_labels(start: date, end: date, horizon: int) -> pd.DataFrame:
    """label_daily rows of one horizon dated in [start, end], read a month at a time."""
    columns = [*KEYS, *LABEL_FIELDS]
    parts = []
    for low, high in months(start, end):
        rows = LabelDaily.objects.filter(date__range=(low, high), horizon=horizon)
        listed = list(rows.order_by("date", "stock_id").values_list(*columns))
        if listed:
            parts.append(pd.DataFrame(listed, columns=columns))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def linear_inputs(features: pd.DataFrame) -> pd.DataFrame:
    """Stock-level features winsorized, z-scored and filled over each day's universe, with flags."""
    return fill_with_flags(standardize(features))


def training_rows(inputs: pd.DataFrame, labels: pd.DataFrame, label: str) -> pd.DataFrame:
    """Universe rows tradable at the next open whose label is known, in date and stock order."""
    joined = inputs.merge(labels, on=KEYS, how="inner")
    keep = joined["in_universe"] & joined["untradable_t1"].eq(False) & joined[label].notna()
    return joined[keep].sort_values(["date", "stock_id"], ignore_index=True)


def load_training_set(
    start: date, end: date, horizon: int, label: str, directory: Path | None = None
) -> pd.DataFrame:
    """Training rows dated in [start, end]; the transforms see every universe row of each day."""
    if horizon not in TARGET_HORIZONS.get(label, ()):
        raise ValueError(f"{label} at horizon {horizon} is not a model target (SPEC §7)")
    features = read_partitions(start, end, directory or settings.FEATURE_PARQUET_DIR)
    return training_rows(linear_inputs(features), load_labels(start, end, horizon), label)
