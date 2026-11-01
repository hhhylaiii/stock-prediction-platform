import math
import re
from collections.abc import Callable
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from apps.market.calendar import TradingDays
from apps.research import prediction_cache
from apps.research.holdout import HoldoutError, claim_holdout
from apps.research.prediction_cache import (
    FoldMismatchError,
    MissingFoldsError,
    fold_path,
    load_fold_predictions,
    load_predictions,
    missing_folds,
    save_fold_predictions,
)
from apps.research.walkforward import Fold, WalkForwardConfig, generate_folds

LAST = date(2026, 10, 30)
RUN = "rank-v1"
TAIPEI = ZoneInfo("Asia/Taipei")


@pytest.fixture(autouse=True)
def cache_dir(settings: Any, tmp_path: Path) -> Path:
    root = tmp_path / "cache"
    settings.RESEARCH_CACHE_DIR = root
    return root


@pytest.fixture(scope="module")
def folds(long_calendar: TradingDays) -> list[Fold]:
    return generate_folds(long_calendar, LAST)


@pytest.fixture(scope="module")
def research(folds: list[Fold]) -> list[Fold]:
    return [fold for fold in folds if fold.research]


def predictions(fold: Fold, score: float = 0.1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [fold.test_start, fold.test_start, fold.test_end],
            "stock_id": ["2330", "2317", "2330"],
            "score": [score, math.nan, -score],
        }
    )


def test_saved_predictions_read_back_equal(research: list[Fold], cache_dir: Path) -> None:
    fold = research[0]
    frame = predictions(fold)

    path = save_fold_predictions(RUN, fold, frame)
    loaded = load_fold_predictions(RUN, fold)

    assert path == cache_dir / RUN / f"fold-{fold.index:03d}.parquet"
    assert path == fold_path(RUN, fold.index)
    pd.testing.assert_frame_equal(loaded, frame)
    assert type(loaded["date"].iloc[0]) is date


def test_saving_again_replaces_the_fold(research: list[Fold]) -> None:
    fold = research[0]
    save_fold_predictions(RUN, fold, predictions(fold, 0.1))

    save_fold_predictions(RUN, fold, predictions(fold, 0.7))

    assert load_fold_predictions(RUN, fold)["score"].iloc[0] == 0.7


def test_changed_fold_boundaries_refuse_the_cache(research: list[Fold]) -> None:
    fold = research[0]
    save_fold_predictions(RUN, fold, predictions(fold))

    with pytest.raises(FoldMismatchError, match=r"\['fit_end', 'train_start'\]"):
        load_fold_predictions(
            RUN, replace(fold, train_start=date(2012, 1, 2), fit_end=date(2015, 1, 2))
        )


def test_another_fold_layout_refuses_the_cache(
    long_calendar: TradingDays, research: list[Fold]
) -> None:
    fold = research[0]
    save_fold_predictions(RUN, fold, predictions(fold))
    longer = generate_folds(long_calendar, LAST, WalkForwardConfig(train_months=60))
    same_period = next(other for other in longer if other.index == fold.index)

    with pytest.raises(FoldMismatchError, match="train_start"):
        load_fold_predictions(RUN, same_period)


def test_missing_folds_are_listed_together(research: list[Fold]) -> None:
    chosen = research[:4]
    for fold in (chosen[0], chosen[2]):
        save_fold_predictions(RUN, fold, predictions(fold))
    absent = [chosen[1].index, chosen[3].index]

    assert missing_folds(RUN, chosen) == absent
    with pytest.raises(MissingFoldsError, match=re.escape(str(absent))) as raised:
        load_predictions(RUN, chosen)
    assert raised.value.folds == absent


def test_missing_single_fold_names_it(research: list[Fold]) -> None:
    with pytest.raises(MissingFoldsError) as raised:
        load_fold_predictions(RUN, research[0])

    assert raised.value.folds == [research[0].index]


def test_load_predictions_stacks_folds_in_order_with_their_index(research: list[Fold]) -> None:
    chosen = research[:2]
    for fold in chosen:
        save_fold_predictions(RUN, fold, predictions(fold))

    stacked = load_predictions(RUN, chosen)

    assert stacked["fold"].tolist() == [chosen[0].index] * 3 + [chosen[1].index] * 3
    assert stacked["date"].iloc[3] == chosen[1].test_start
    assert list(stacked.columns) == ["date", "stock_id", "score", "fold"]


def test_load_predictions_of_no_folds_is_empty_with_the_key_columns() -> None:
    assert list(load_predictions(RUN, []).columns) == ["date", "stock_id", "fold"]


def test_dates_outside_the_test_period_are_refused(
    long_calendar: TradingDays, research: list[Fold]
) -> None:
    fold = research[0]
    before = long_calendar.shift(fold.test_start, -1)
    after = long_calendar.shift(fold.test_end, 1)

    for day in (before, after):
        frame = predictions(fold).assign(date=[fold.test_start, day, fold.test_end])
        with pytest.raises(ValueError, match="outside"):
            save_fold_predictions(RUN, fold, frame)
    assert not fold_path(RUN, fold.index).exists()


def test_repeated_stock_days_are_refused(research: list[Fold]) -> None:
    fold = research[0]
    frame = predictions(fold).assign(stock_id=["2330", "2330", "2330"], date=[fold.test_start] * 3)

    with pytest.raises(ValueError, match="repeat"):
        save_fold_predictions(RUN, fold, frame)


def test_key_columns_are_required(research: list[Fold]) -> None:
    fold = research[0]

    with pytest.raises(ValueError, match="stock_id"):
        save_fold_predictions(RUN, fold, predictions(fold).drop(columns="stock_id"))


@pytest.mark.parametrize(
    "wrap", [pd.Timestamp, lambda day: datetime(day.year, day.month, day.day, tzinfo=TAIPEI)]
)
def test_dates_must_be_plain_dates(research: list[Fold], wrap: Callable[[date], datetime]) -> None:
    fold = research[0]
    frame = predictions(fold)
    stamped = frame.assign(date=[wrap(day) for day in frame["date"]])

    with pytest.raises(ValueError, match=r"datetime\.date"):
        save_fold_predictions(RUN, fold, stamped)


def test_holdout_fold_without_a_ticket_is_not_written(folds: list[Fold]) -> None:
    fold = next(fold for fold in folds if fold.holdout)

    with pytest.raises(HoldoutError):
        save_fold_predictions(RUN, fold, predictions(fold))
    assert not fold_path(RUN, fold.index).exists()


@pytest.mark.django_db
def test_holdout_fold_is_written_with_a_recorded_ticket(folds: list[Fold]) -> None:
    fold = next(fold for fold in folds if fold.holdout)
    ticket = claim_holdout("rank", RUN)

    save_fold_predictions(RUN, fold, predictions(fold), ticket)

    assert load_fold_predictions(RUN, fold)["stock_id"].tolist() == ["2330", "2317", "2330"]


@pytest.mark.parametrize("run_key", ["../escape", "a/b", ".."])
def test_run_key_with_path_characters_is_refused(
    research: list[Fold], cache_dir: Path, run_key: str
) -> None:
    fold = research[0]

    with pytest.raises(ValueError, match="letters"):
        save_fold_predictions(run_key, fold, predictions(fold))
    assert not cache_dir.parent.joinpath("escape").exists()
    assert not cache_dir.exists()


def test_failed_write_keeps_the_old_file_and_no_temporary(
    research: list[Fold], cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fold = research[0]
    original = predictions(fold, 0.1)
    save_fold_predictions(RUN, fold, original)

    def broken(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(prediction_cache.pq, "write_table", broken)
    with pytest.raises(OSError, match="disk full"):
        save_fold_predictions(RUN, fold, predictions(fold, 0.9))

    pd.testing.assert_frame_equal(load_fold_predictions(RUN, fold), original)
    assert [path.name for path in (cache_dir / RUN).iterdir()] == [f"fold-{fold.index:03d}.parquet"]
