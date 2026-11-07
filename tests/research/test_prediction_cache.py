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
from apps.research.holdout import HoldoutError, ProductionPass, claim_holdout, production_pass
from apps.research.prediction_cache import (
    FoldMismatchError,
    MissingFoldsError,
    fold_cached,
    fold_notes,
    fold_path,
    load_fold_predictions,
    load_predictions,
    missing_folds,
    save_fold_predictions,
    save_holdout_fold,
)
from apps.research.walkforward import Fold, WalkForwardConfig, generate_folds

LAST = date(2026, 10, 30)
RUN = "rank-v1"
MODEL = "baseline"
OTHER = "lightgbm"
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

    path = save_fold_predictions(RUN, MODEL, fold, frame)
    loaded = load_fold_predictions(RUN, MODEL, fold)

    assert path == cache_dir / RUN / MODEL / f"fold-{fold.index:03d}.parquet"
    assert path == fold_path(RUN, MODEL, fold.index)
    pd.testing.assert_frame_equal(loaded, frame)
    assert type(loaded["date"].iloc[0]) is date


def test_notes_are_kept_with_the_fold(research: list[Fold]) -> None:
    fold = research[0]
    assert fold_notes("rank-v1", MODEL, fold.index) == {}

    save_fold_predictions("rank-v1", MODEL, fold, predictions(fold), notes={"feature_hash": "ab12"})
    assert fold_notes("rank-v1", MODEL, fold.index) == {"feature_hash": "ab12"}

    save_fold_predictions("rank-v1", MODEL, fold, predictions(fold))
    assert fold_notes("rank-v1", MODEL, fold.index) == {}


def test_index_is_not_kept(research: list[Fold]) -> None:
    chosen = research[:2]
    for fold in chosen:
        save_fold_predictions(RUN, MODEL, fold, predictions(fold).set_axis([5, 9, 11]))

    assert isinstance(load_fold_predictions(RUN, MODEL, chosen[0]).index, pd.RangeIndex)
    assert load_predictions(RUN, MODEL, chosen).index.tolist() == list(range(6))


def test_empty_predictions_are_refused(research: list[Fold]) -> None:
    fold = research[0]

    with pytest.raises(ValueError, match="no predictions"):
        save_fold_predictions(RUN, MODEL, fold, predictions(fold).iloc[0:0])
    assert missing_folds(RUN, MODEL, [fold]) == [fold.index]


def test_saving_again_replaces_the_fold(research: list[Fold]) -> None:
    fold = research[0]
    save_fold_predictions(RUN, MODEL, fold, predictions(fold, 0.1))

    save_fold_predictions(RUN, MODEL, fold, predictions(fold, 0.7))

    assert load_fold_predictions(RUN, MODEL, fold)["score"].iloc[0] == 0.7


def test_changed_fold_boundaries_refuse_the_cache(research: list[Fold]) -> None:
    fold = research[0]
    save_fold_predictions(RUN, MODEL, fold, predictions(fold))

    with pytest.raises(FoldMismatchError, match=r"\['fit_end', 'train_start'\]"):
        load_fold_predictions(
            RUN, MODEL, replace(fold, train_start=date(2012, 1, 2), fit_end=date(2015, 1, 2))
        )


def test_another_fold_layout_refuses_the_cache(
    long_calendar: TradingDays, research: list[Fold]
) -> None:
    fold = research[0]
    save_fold_predictions(RUN, MODEL, fold, predictions(fold))
    longer = generate_folds(long_calendar, LAST, WalkForwardConfig(train_months=60))
    same_period = next(other for other in longer if other.index == fold.index)

    with pytest.raises(FoldMismatchError, match="train_start"):
        load_fold_predictions(RUN, MODEL, same_period)


def test_missing_folds_are_listed_together(research: list[Fold]) -> None:
    chosen = research[:4]
    for fold in (chosen[0], chosen[2]):
        save_fold_predictions(RUN, MODEL, fold, predictions(fold))
    absent = [chosen[1].index, chosen[3].index]

    assert missing_folds(RUN, MODEL, chosen) == absent
    with pytest.raises(MissingFoldsError, match=re.escape(str(absent))) as raised:
        load_predictions(RUN, MODEL, chosen)
    assert raised.value.folds == absent


def test_folds_cached_for_other_boundaries_count_as_missing(research: list[Fold]) -> None:
    fold = research[0]
    save_fold_predictions(RUN, MODEL, fold, predictions(fold))
    moved = replace(fold, train_start=date(2012, 1, 2))

    assert missing_folds(RUN, MODEL, [fold]) == []
    assert missing_folds(RUN, MODEL, [moved]) == [fold.index]
    with pytest.raises(FoldMismatchError):
        load_predictions(RUN, MODEL, [moved])


def test_missing_single_fold_names_it(research: list[Fold]) -> None:
    with pytest.raises(MissingFoldsError) as raised:
        load_fold_predictions(RUN, MODEL, research[0])

    assert raised.value.folds == [research[0].index]


def test_load_predictions_stacks_folds_in_order_with_their_index(research: list[Fold]) -> None:
    chosen = research[:2]
    for fold in chosen:
        save_fold_predictions(RUN, MODEL, fold, predictions(fold))

    stacked = load_predictions(RUN, MODEL, chosen)

    assert stacked["fold"].tolist() == [chosen[0].index] * 3 + [chosen[1].index] * 3
    assert stacked["date"].iloc[3] == chosen[1].test_start
    assert list(stacked.columns) == ["date", "stock_id", "score", "fold"]


def test_load_predictions_of_no_folds_is_empty_with_the_key_columns() -> None:
    assert list(load_predictions(RUN, MODEL, []).columns) == ["date", "stock_id", "fold"]


def test_dates_outside_the_test_period_are_refused(
    long_calendar: TradingDays, research: list[Fold]
) -> None:
    fold = research[0]
    before = long_calendar.shift(fold.test_start, -1)
    after = long_calendar.shift(fold.test_end, 1)

    for day in (before, after):
        frame = predictions(fold).assign(date=[fold.test_start, day, fold.test_end])
        with pytest.raises(ValueError, match="outside"):
            save_fold_predictions(RUN, MODEL, fold, frame)
    assert not fold_path(RUN, MODEL, fold.index).exists()


def test_repeated_stock_days_are_refused(research: list[Fold]) -> None:
    fold = research[0]
    frame = predictions(fold).assign(stock_id=["2330", "2330", "2330"], date=[fold.test_start] * 3)

    with pytest.raises(ValueError, match="repeat"):
        save_fold_predictions(RUN, MODEL, fold, frame)


def test_key_columns_are_required(research: list[Fold]) -> None:
    fold = research[0]

    with pytest.raises(ValueError, match="stock_id"):
        save_fold_predictions(RUN, MODEL, fold, predictions(fold).drop(columns="stock_id"))


@pytest.mark.parametrize(
    "wrap", [pd.Timestamp, lambda day: datetime(day.year, day.month, day.day, tzinfo=TAIPEI)]
)
def test_dates_must_be_plain_dates(research: list[Fold], wrap: Callable[[date], datetime]) -> None:
    fold = research[0]
    frame = predictions(fold)
    stamped = frame.assign(date=[wrap(day) for day in frame["date"]])

    with pytest.raises(ValueError, match=r"datetime\.date"):
        save_fold_predictions(RUN, MODEL, fold, stamped)


def holdout_of(folds: list[Fold]) -> Fold:
    return next(fold for fold in folds if fold.holdout)


def both(fold: Fold, score: float = 0.1) -> dict[str, pd.DataFrame]:
    return {MODEL: predictions(fold, score), OTHER: predictions(fold, score + 0.1)}


@pytest.mark.django_db
def test_holdout_fold_without_a_ticket_is_not_written(folds: list[Fold]) -> None:
    fold = holdout_of(folds)

    with pytest.raises(HoldoutError, match="save_holdout_fold"):
        save_fold_predictions(RUN, MODEL, fold, predictions(fold))
    for ticket in (None, claim_holdout("return", "return-v1", LAST)):
        with pytest.raises(HoldoutError):
            save_holdout_fold(RUN, fold, both(fold), ticket)
    assert not fold_cached(RUN, fold.index)


@pytest.mark.django_db
def test_holdout_fold_writes_every_model_with_a_recorded_ticket(folds: list[Fold]) -> None:
    fold = holdout_of(folds)
    ticket = claim_holdout("rank", RUN, LAST)

    paths = save_holdout_fold(RUN, fold, both(fold), ticket, notes={OTHER: {"seed": "7"}})

    assert paths == [fold_path(RUN, MODEL, fold.index), fold_path(RUN, OTHER, fold.index)]
    assert load_fold_predictions(RUN, MODEL, fold)["stock_id"].tolist() == ["2330", "2317", "2330"]
    assert load_fold_predictions(RUN, OTHER, fold)["score"].iloc[0] == pytest.approx(0.2)
    assert fold_notes(RUN, MODEL, fold.index) == {}
    assert fold_notes(RUN, OTHER, fold.index) == {"seed": "7"}


@pytest.mark.django_db
def test_holdout_fold_is_written_once_for_every_model(folds: list[Fold]) -> None:
    fold = holdout_of(folds)
    ticket = claim_holdout("rank", RUN, LAST)
    save_holdout_fold(RUN, fold, both(fold, 0.1), ticket)

    for again in (both(fold, 0.8), {"late": predictions(fold, 0.9)}):
        with pytest.raises(HoldoutError, match="written once"):
            save_holdout_fold(RUN, fold, again, ticket)
    with pytest.raises(HoldoutError):
        save_fold_predictions(RUN, "late", fold, predictions(fold, 0.9))

    assert not fold_path(RUN, "late", fold.index).exists()
    assert load_fold_predictions(RUN, MODEL, fold)["score"].iloc[0] == 0.1


@pytest.mark.django_db
def test_holdout_fold_written_meanwhile_is_kept(
    folds: list[Fold], monkeypatch: pytest.MonkeyPatch
) -> None:
    fold = holdout_of(folds)
    ticket = claim_holdout("rank", RUN, LAST)
    target = fold_path(RUN, MODEL, fold.index)
    write = prediction_cache.pq.write_table

    def racing(table: object, where: str, *args: object, **kwargs: object) -> None:
        write(table, where, *args, **kwargs)
        target.write_bytes(b"other")

    monkeypatch.setattr(prediction_cache.pq, "write_table", racing)
    with pytest.raises(HoldoutError, match="appeared"):
        save_holdout_fold(RUN, fold, {MODEL: predictions(fold)}, ticket)
    assert target.read_bytes() == b"other"
    assert [path.name for path in target.parent.iterdir()] == [target.name]


@pytest.mark.django_db
def test_holdout_fold_with_a_bad_model_writes_none(
    folds: list[Fold], cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fold = holdout_of(folds)
    ticket = claim_holdout("rank", RUN, LAST)
    bad = predictions(fold).drop(columns="stock_id")

    with pytest.raises(ValueError, match="stock_id"):
        save_holdout_fold(RUN, fold, {MODEL: predictions(fold), OTHER: bad}, ticket)
    with pytest.raises(ValueError, match="notes"):
        save_holdout_fold(RUN, fold, {MODEL: predictions(fold)}, ticket, notes={OTHER: {}})
    with pytest.raises(ValueError, match="models"):
        save_holdout_fold(RUN, fold, {}, ticket)
    write = prediction_cache.pq.write_table
    calls = []

    def broken(table: object, where: str, *args: object, **kwargs: object) -> None:
        calls.append(where)
        write(table, where, *args, **kwargs)
        if len(calls) == 2:
            raise OSError("disk full")

    monkeypatch.setattr(prediction_cache.pq, "write_table", broken)
    with pytest.raises(OSError, match="disk full"):
        save_holdout_fold(RUN, fold, both(fold), ticket)

    assert not fold_cached(RUN, fold.index)
    assert [path for path in cache_dir.rglob("*") if path.is_file()] == []


def marker(cache_dir: Path, fold: Fold) -> Path:
    return cache_dir / RUN / f"fold-{fold.index:03d}.holdout"


@pytest.mark.django_db
def test_a_failed_publish_takes_the_holdout_fold_back(
    folds: list[Fold], cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fold = holdout_of(folds)
    ticket = claim_holdout("rank", RUN, LAST)
    link = prediction_cache.os.link
    calls = []

    def broken(source: str, target: Path) -> None:
        calls.append(target)
        if len(calls) == 2:
            raise OSError(28, "No space left on device")
        link(source, target)

    monkeypatch.setattr(prediction_cache.os, "link", broken)
    with pytest.raises(OSError, match="No space"):
        save_holdout_fold(RUN, fold, both(fold), ticket)
    monkeypatch.setattr(prediction_cache.os, "link", link)

    assert not fold_cached(RUN, fold.index)
    assert [path for path in cache_dir.rglob("*") if path.is_file()] == []
    save_holdout_fold(RUN, fold, both(fold), ticket)
    assert load_fold_predictions(RUN, OTHER, fold)["score"].iloc[0] == pytest.approx(0.2)
    assert marker(cache_dir, fold).exists()


@pytest.mark.django_db
def test_a_concurrent_holdout_write_of_other_models_is_refused(
    folds: list[Fold], monkeypatch: pytest.MonkeyPatch
) -> None:
    fold = holdout_of(folds)
    ticket = claim_holdout("rank", RUN, LAST)
    write = prediction_cache.pq.write_table
    nested: list[str] = []

    def meanwhile(table: object, where: str, *args: object, **kwargs: object) -> None:
        write(table, where, *args, **kwargs)
        if not nested:
            nested.append(where)
            save_holdout_fold(RUN, fold, {"xgboost": predictions(fold, 0.5)}, ticket)

    monkeypatch.setattr(prediction_cache.pq, "write_table", meanwhile)
    with pytest.raises(HoldoutError, match="being written"):
        save_holdout_fold(RUN, fold, both(fold), ticket)

    assert load_fold_predictions(RUN, "xgboost", fold)["score"].iloc[0] == 0.5
    assert not fold_path(RUN, MODEL, fold.index).exists()
    assert not fold_path(RUN, OTHER, fold.index).exists()


@pytest.mark.django_db
def test_an_interrupted_holdout_write_keeps_its_fold_closed(
    long_calendar: TradingDays, folds: list[Fold], cache_dir: Path
) -> None:
    shifted = generate_folds(long_calendar, LAST, WalkForwardConfig(first_test=date(2015, 1, 1)))
    fold = holdout_of(shifted)
    ticket = claim_holdout("rank", RUN, LAST)
    marker(cache_dir, fold).parent.mkdir(parents=True)
    marker(cache_dir, fold).touch()
    research = next(other for other in folds if other.index == fold.index)

    assert not research.holdout
    assert fold_cached(RUN, fold.index)
    with pytest.raises(HoldoutError, match="written once"):
        save_holdout_fold(RUN, fold, both(fold), ticket)
    with pytest.raises(HoldoutError, match="written once"):
        save_fold_predictions(RUN, "late", research, predictions(research))


@pytest.mark.django_db
def test_research_folds_are_not_written_as_holdout(research: list[Fold], cache_dir: Path) -> None:
    fold = research[0]
    ticket = claim_holdout("rank", RUN, LAST)

    with pytest.raises(ValueError, match="not holdout"):
        save_holdout_fold(RUN, fold, both(fold), ticket)
    assert not cache_dir.exists()


@pytest.mark.django_db
def test_research_fold_of_another_layout_cannot_replace_a_holdout_file(
    long_calendar: TradingDays, folds: list[Fold]
) -> None:
    shifted = generate_folds(long_calendar, LAST, WalkForwardConfig(first_test=date(2015, 1, 1)))
    holdout = holdout_of(shifted)
    ticket = claim_holdout("rank", RUN, LAST)
    save_holdout_fold(RUN, holdout, {MODEL: predictions(holdout, 0.1)}, ticket)
    research = next(fold for fold in folds if fold.index == holdout.index)

    assert not research.holdout
    for model in (MODEL, "late"):
        with pytest.raises(HoldoutError, match="written once"):
            save_fold_predictions(RUN, model, research, predictions(research, 0.9))
    assert load_fold_predictions(RUN, MODEL, holdout)["score"].iloc[0] == 0.1
    assert not fold_path(RUN, "late", research.index).exists()


def test_holdout_fold_with_a_moved_start_still_needs_a_ticket(
    long_calendar: TradingDays,
) -> None:
    moved = generate_folds(long_calendar, LAST, WalkForwardConfig(holdout_start=date(2099, 1, 1)))
    fold = next(fold for fold in moved if fold.test_end >= date(2025, 1, 1))

    with pytest.raises(HoldoutError):
        save_fold_predictions(RUN, MODEL, fold, predictions(fold))


@pytest.mark.parametrize("run_key", ["../escape", "a/b", ".."])
def test_run_key_with_path_characters_is_refused(
    research: list[Fold], cache_dir: Path, run_key: str
) -> None:
    fold = research[0]

    with pytest.raises(ValueError, match="letters"):
        save_fold_predictions(run_key, MODEL, fold, predictions(fold))
    assert not cache_dir.parent.joinpath("escape").exists()
    assert not cache_dir.exists()


def test_failed_write_keeps_the_old_file_and_no_temporary(
    research: list[Fold], cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fold = research[0]
    original = predictions(fold, 0.1)
    save_fold_predictions(RUN, MODEL, fold, original)

    def broken(table: object, where: str, *args: object, **kwargs: object) -> None:
        Path(where).write_bytes(b"PAR1")
        raise OSError("disk full")

    monkeypatch.setattr(prediction_cache.pq, "write_table", broken)
    with pytest.raises(OSError, match="disk full"):
        save_fold_predictions(RUN, MODEL, fold, predictions(fold, 0.9))

    pd.testing.assert_frame_equal(load_fold_predictions(RUN, MODEL, fold), original)
    assert [path.name for path in (cache_dir / RUN / MODEL).iterdir()] == [
        f"fold-{fold.index:03d}.parquet"
    ]


def test_models_of_a_fold_are_cached_side_by_side(research: list[Fold]) -> None:
    fold = research[0]
    save_fold_predictions(RUN, MODEL, fold, predictions(fold, 0.1), notes={"feature_hash": "aa"})
    save_fold_predictions(RUN, OTHER, fold, predictions(fold, 0.5), notes={"feature_hash": "bb"})

    save_fold_predictions(RUN, MODEL, fold, predictions(fold, 0.7))

    assert load_fold_predictions(RUN, MODEL, fold)["score"].iloc[0] == 0.7
    assert load_fold_predictions(RUN, OTHER, fold)["score"].iloc[0] == 0.5
    assert fold_notes(RUN, MODEL, fold.index) == {}
    assert fold_notes(RUN, OTHER, fold.index) == {"feature_hash": "bb"}


def test_a_fold_another_model_cached_is_still_missing(research: list[Fold]) -> None:
    chosen = research[:2]
    for fold in chosen:
        save_fold_predictions(RUN, OTHER, fold, predictions(fold))
    save_fold_predictions(RUN, MODEL, chosen[0], predictions(chosen[0]))

    assert missing_folds(RUN, MODEL, chosen) == [chosen[1].index]
    assert missing_folds(RUN, OTHER, chosen) == []
    with pytest.raises(MissingFoldsError, match=f"{RUN}/{MODEL} has no") as raised:
        load_predictions(RUN, MODEL, chosen)
    assert raised.value.folds == [chosen[1].index]


def test_a_fold_is_cached_once_any_model_holds_it(research: list[Fold]) -> None:
    fold = research[0]
    assert not fold_cached(RUN, fold.index)

    save_fold_predictions(RUN, OTHER, fold, predictions(fold))

    assert fold_cached(RUN, fold.index)
    assert not fold_cached(RUN, research[1].index)
    assert not fold_cached("rank-v2", fold.index)


@pytest.mark.parametrize("model", ["../escape", "a/b", "..", "Baseline", ""])
def test_model_names_with_path_characters_are_refused(
    research: list[Fold], cache_dir: Path, model: str
) -> None:
    fold = research[0]

    with pytest.raises(ValueError, match="letters"):
        save_fold_predictions(RUN, model, fold, predictions(fold))
    with pytest.raises(ValueError, match="letters"):
        fold_cached("../escape", fold.index)
    assert not cache_dir.exists()


@pytest.mark.django_db
def test_a_production_pass_caches_a_fold_past_the_holdout_start(folds: list[Fold]) -> None:
    fold = holdout_of(folds)
    forged = ProductionPass("rank", "rank-final")
    with pytest.raises(HoldoutError, match="rank design is not frozen"):
        save_fold_predictions("rank-prod", MODEL, fold, predictions(fold), production=forged)
    assert not fold_cached("rank-prod", fold.index)

    claim_holdout("rank", "rank-final", LAST)
    granted = production_pass("rank")
    save_fold_predictions("rank-prod", MODEL, fold, predictions(fold), production=granted)
    save_fold_predictions("rank-prod", MODEL, fold, predictions(fold, 0.5), production=granted)

    assert load_fold_predictions("rank-prod", MODEL, fold)["score"].iloc[0] == pytest.approx(0.5)
    with pytest.raises(HoldoutError, match="rank-final is a holdout run"):
        save_fold_predictions("rank-final", MODEL, fold, predictions(fold), production=granted)
