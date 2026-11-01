"""Training rows of the linear baselines: tradable universe rows with a label (SPEC §6, §7)."""

from datetime import date

import pandas as pd

from apps.research.features.store import read_features
from apps.research.features.transforms import STOCK_LEVEL, fill_with_flags, standardize
from apps.research.models import LabelDaily

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


def load_labels(start: date, end: date, horizon: int) -> pd.DataFrame:
    """label_daily rows of one horizon dated in [start, end]."""
    rows = LabelDaily.objects.filter(date__range=(start, end), horizon=horizon)
    columns = [*KEYS, *LABEL_FIELDS]
    listed = list(rows.order_by("date", "stock_id").values_list(*columns))
    return pd.DataFrame(listed, columns=columns)


def linear_inputs(features: pd.DataFrame) -> pd.DataFrame:
    """Stock-level features winsorized, z-scored and filled over each day's universe, with flags."""
    return fill_with_flags(standardize(features))


def training_rows(inputs: pd.DataFrame, labels: pd.DataFrame, label: str) -> pd.DataFrame:
    """Universe rows tradable at the next open whose label is known, in date and stock order."""
    joined = inputs.merge(labels, on=KEYS, how="inner")
    keep = joined["in_universe"] & joined["untradable_t1"].eq(False) & joined[label].notna()
    return joined[keep].sort_values(["date", "stock_id"], ignore_index=True)


def load_training_set(start: date, end: date, horizon: int, label: str) -> pd.DataFrame:
    """Training rows dated in [start, end]; the transforms see every universe row of each day."""
    inputs = linear_inputs(read_features(start, end))
    return training_rows(inputs, load_labels(start, end, horizon), label)
