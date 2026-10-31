"""Idempotent bulk writes into standard tables (SPEC §4 rules)."""

from collections.abc import Sequence

import pandas as pd
from django.db import models, transaction

DEFAULT_BATCH_SIZE = 2000


def upsert(
    model: type[models.Model],
    frame: pd.DataFrame,
    unique_fields: Sequence[str],
    update_fields: Sequence[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Insert or update rows by their key; rerunning with the same frame changes nothing."""
    if frame.empty:
        return 0
    duplicated = frame.duplicated(subset=list(unique_fields), keep=False)
    if duplicated.any():
        sample = frame.loc[duplicated, list(unique_fields)].head(3).to_dict("records")
        raise ValueError(f"duplicate keys in upsert frame: {sample}")
    cleaned = frame.astype(object).where(frame.notna(), None)
    records = cleaned.to_dict("records")
    objects = [model(**{str(key): value for key, value in row.items()}) for row in records]
    with transaction.atomic():
        for start in range(0, len(objects), batch_size):
            model._default_manager.bulk_create(
                objects[start : start + batch_size],
                update_conflicts=True,
                unique_fields=list(unique_fields),
                update_fields=list(update_fields),
            )
    return len(objects)
