from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from apps.etl.upsert import insert_missing, upsert
from apps.market.models import DailyPrice

KEY = ("stock_id", "date")
UPDATE = ("open", "high", "low", "close", "volume", "turnover")


def _frame(close: str, volume: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_id": ["2330", "2317"],
            "date": [date(2026, 10, 28)] * 2,
            "open": [Decimal("1"), Decimal("2")],
            "high": [Decimal("1"), Decimal("2")],
            "low": [Decimal("1"), Decimal("2")],
            "close": [Decimal(close), Decimal("2")],
            "volume": [volume, 200],
            "turnover": [1000, 2000],
        }
    )


@pytest.mark.django_db
def test_rerunning_the_same_frame_keeps_one_row_per_key() -> None:
    assert upsert(DailyPrice, _frame("10"), KEY, UPDATE) == 2
    assert upsert(DailyPrice, _frame("10"), KEY, UPDATE) == 2
    assert DailyPrice.objects.count() == 2


@pytest.mark.django_db
def test_conflicting_rows_take_the_new_values() -> None:
    upsert(DailyPrice, _frame("10", volume=100), KEY, UPDATE)
    upsert(DailyPrice, _frame("11", volume=150), KEY, UPDATE)
    row = DailyPrice.objects.get(pk=("2330", date(2026, 10, 28)))
    assert (row.close, row.volume) == (Decimal("11"), 150)


@pytest.mark.django_db
def test_columns_outside_update_fields_are_preserved() -> None:
    upsert(DailyPrice, _frame("10"), KEY, UPDATE)
    DailyPrice.objects.filter(stock_id="2330").update(adj_factor=0.5)
    upsert(DailyPrice, _frame("12"), KEY, UPDATE)
    assert DailyPrice.objects.get(pk=("2330", date(2026, 10, 28))).adj_factor == 0.5


@pytest.mark.django_db
def test_missing_values_are_written_as_null() -> None:
    frame = _frame("10").assign(close=pd.Series([None, np.nan], dtype=object))
    upsert(DailyPrice, frame, KEY, UPDATE)
    assert set(DailyPrice.objects.values_list("close", flat=True)) == {None}


def test_duplicate_keys_are_rejected_before_writing() -> None:
    frame = pd.concat([_frame("10"), _frame("11")], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate keys"):
        upsert(DailyPrice, frame, KEY, UPDATE)


def test_empty_frame_writes_nothing() -> None:
    assert upsert(DailyPrice, pd.DataFrame(), KEY, UPDATE) == 0


@pytest.mark.django_db
def test_batches_cover_every_row() -> None:
    frame = pd.DataFrame(
        {
            "stock_id": [f"{n:04d}" for n in range(5)],
            "date": [date(2026, 10, 28)] * 5,
            "volume": [0] * 5,
            "turnover": [0] * 5,
        }
    )
    assert upsert(DailyPrice, frame, KEY, ("volume", "turnover"), batch_size=2) == 5
    assert DailyPrice.objects.count() == 5


@pytest.mark.django_db
def test_insert_missing_keeps_existing_rows_and_adds_new_ones() -> None:
    upsert(DailyPrice, _frame("10").iloc[:1], KEY, UPDATE)
    assert insert_missing(DailyPrice, _frame("99"), KEY) == 1
    assert DailyPrice.objects.get(stock_id="2330").close == Decimal("10")
    assert DailyPrice.objects.count() == 2


@pytest.mark.django_db
def test_insert_missing_with_nothing_to_insert() -> None:
    assert insert_missing(DailyPrice, _frame("1").iloc[0:0], KEY) == 0


def test_insert_missing_refuses_duplicate_keys() -> None:
    frame = pd.concat([_frame("1"), _frame("2")])
    with pytest.raises(ValueError, match="duplicate keys"):
        insert_missing(DailyPrice, frame, KEY)
