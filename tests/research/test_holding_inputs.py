import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from apps.market.models import DailyPrice, IndexDaily, StockMaster
from apps.research import holding_inputs
from apps.research.holding import Rule
from apps.research.holding_inputs import (
    SCORE_COLUMNS,
    HoldoutRuleError,
    cached_scores,
    check_holdout_rule,
    index_levels,
    load_market,
    stored_scores,
)
from apps.research.models import LabelDaily, ModelRun, Prediction
from apps.research.prediction_cache import (
    FOLD_METADATA,
    MissingFoldsError,
    fold_path,
    holds_holdout,
)
from tests.research.conftest import weekday_calendar

pytestmark = pytest.mark.django_db
NAN = math.nan
CALENDAR = weekday_calendar(date(2016, 1, 4), date(2016, 3, 31))
DAYS = list(CALENDAR.between(date(2016, 1, 4), date(2016, 3, 31)))
KEY = "rank-test"


@pytest.fixture
def cache(settings: Any, tmp_path: Path) -> Path:
    settings.RESEARCH_CACHE_DIR = tmp_path
    return tmp_path


def write_fold(index: int, days: list[date], record: dict[str, object] | None = None) -> None:
    frame = pd.DataFrame(
        [
            {"date": day, "stock_id": stock, "in_universe": stock != "1001", "lightgbm_h5": n}
            for n, day in enumerate(days)
            for stock in ("1000", "1001")
        ]
    )
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if record is not None:
        table = table.replace_schema_metadata({FOLD_METADATA: json.dumps(record).encode()})
    path = fold_path(KEY, "lightgbm", index)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_cached_scores_stack_folds_through_the_last_signal_day(cache: Path) -> None:
    write_fold(1, DAYS[0:3])
    write_fold(2, DAYS[3:6])

    scores = cached_scores(KEY, [1, 2], DAYS[4])

    assert scores.columns.tolist() == SCORE_COLUMNS
    assert sorted(set(scores["date"])) == DAYS[0:5]
    assert scores.loc[scores["date"] == DAYS[4], "score"].tolist() == [1.0, 1.0]
    assert scores.loc[scores["stock_id"] == "1001", "in_universe"].eq(False).all()


def test_cached_scores_refuse_a_day_two_folds_score(cache: Path) -> None:
    write_fold(1, DAYS[0:3])
    write_fold(2, DAYS[2:5])

    with pytest.raises(ValueError, match="more than one fold"):
        cached_scores(KEY, [1, 2], DAYS[4])


def test_cached_scores_name_the_missing_folds(cache: Path) -> None:
    write_fold(1, DAYS[0:3])

    with pytest.raises(MissingFoldsError) as raised:
        cached_scores(KEY, [1, 2, 3], DAYS[4])

    assert "2" in str(raised.value)
    assert "3" in str(raised.value)


def test_a_fold_written_as_holdout_is_recognised(cache: Path) -> None:
    write_fold(44, DAYS[0:3], {"holdout": True})
    write_fold(8, DAYS[0:3], {"holdout": False, "test_end": "2016-01-06"})

    assert holds_holdout(KEY, "lightgbm", 44) is True
    assert holds_holdout(KEY, "lightgbm", 8) is False


@pytest.mark.parametrize(("name", "slots"), [("H70", 30), ("H90", 30), ("H90", 100)])
def test_the_three_rules_of_the_one_holdout_pass_may_rerun_on_holdout_folds(
    monkeypatch: pytest.MonkeyPatch, name: str, slots: int
) -> None:
    monkeypatch.setattr(holding_inputs, "holds_holdout", lambda key, model, index: index >= 44)

    check_holdout_rule(KEY, [43, 44], Rule.parse(name, slots))


@pytest.mark.parametrize(("name", "slots"), [("H50", 30), ("H70", 100), ("H80", 30)])
def test_any_other_rule_is_refused_on_holdout_folds(
    monkeypatch: pytest.MonkeyPatch, name: str, slots: int
) -> None:
    monkeypatch.setattr(holding_inputs, "holds_holdout", lambda key, model, index: index >= 44)

    with pytest.raises(HoldoutRuleError, match=r"folds \[44\]"):
        check_holdout_rule(KEY, [43, 44], Rule.parse(name, slots))
    check_holdout_rule(KEY, [8, 43], Rule.parse(name, slots))


def make_run(run_id: str, objective: str) -> ModelRun:
    return ModelRun.objects.create(
        run_id=run_id,
        run_key="prod",
        model_type="lightgbm",
        objective=objective,
        version="2026-11",
        train_start=date(2015, 1, 5),
        train_end=date(2015, 12, 31),
        lib_versions={},
        feature_hash="0" * 64,
        status="active",
        params={},
    )


def test_stored_scores_read_only_the_rank_five_day_rows() -> None:
    rank, other = make_run("rank.h5", "rank"), make_run("return.h5", "regression")
    cells = [
        (rank, "1000", DAYS[0], 5, {"score": 0.4}, True),
        (rank, "1001", DAYS[1], 5, {"score": 0.1}, False),
        (rank, "1000", DAYS[0], 20, {"score": 0.9}, True),
        (other, "1002", DAYS[0], 5, {"ret_pred": 0.01}, True),
        (rank, "1003", DAYS[5], 5, {"score": 0.7}, True),
    ]
    rows = [
        Prediction(run=run, stock_id=name, date=day, horizon=h, in_universe=inside, **values)
        for run, name, day, h, values, inside in cells
    ]
    Prediction.objects.bulk_create(rows)

    scores = stored_scores(DAYS[0], DAYS[1]).sort_values("stock_id", ignore_index=True)

    assert scores.columns.tolist() == [*SCORE_COLUMNS, "run_id"]
    assert scores.values.tolist() == [
        [DAYS[0], "1000", True, 0.4, "rank.h5"],
        [DAYS[1], "1001", False, 0.1, "rank.h5"],
    ]


def stock(stock_id: str, **fields: object) -> None:
    defaults: dict[str, object] = {"name": stock_id, "market": "TWSE", "industry": "半導體業"}
    StockMaster.objects.create(stock_id=stock_id, **{**defaults, **fields})


def bars(stock_id: str, days: list[date], opens: list[float], adj: float = 1.0) -> None:
    DailyPrice.objects.bulk_create(
        DailyPrice(
            stock_id=stock_id,
            date=day,
            open=price,
            high=max(price, 10.0),
            low=min(price, 10.0),
            close=10.0,
            volume=1_000,
            turnover=10_000,
            adj_factor=adj,
        )
        for day, price in zip(days, opens, strict=True)
    )


def test_the_market_grids_adjust_prices_and_mark_flags_by_day_and_stock() -> None:
    stock("1000", list_date=DAYS[0], delist_date=DAYS[6])
    stock("1001", list_date=DAYS[0])
    bars("1000", DAYS[0:6], [10.0] * 6)
    bars("1001", DAYS[0:10], [10.0] * 8 + [9.0, 10.0], adj=2.0)
    LabelDaily.objects.bulk_create(
        [
            LabelDaily(stock_id="1001", date=DAYS[3], horizon=5, untradable_t1=True),
            LabelDaily(stock_id="1000", date=DAYS[3], horizon=5, untradable_t1=False),
            LabelDaily(stock_id="1000", date=DAYS[2], horizon=20, untradable_t1=True),
        ]
    )
    scores = pd.DataFrame(
        [
            {"date": day, "stock_id": name, "in_universe": (n, name) != (1, "1000"), "score": n}
            for n, day in enumerate(DAYS[0:5])
            for name in ("1000", "1001")
        ]
    )

    grid = load_market(scores, DAYS[9], CALENDAR)

    assert grid.days == DAYS[0:10]
    assert grid.stocks == ["1000", "1001"]
    assert np.isnan(grid.closes[6:, 0]).all()
    assert grid.closes[:6, 0].tolist() == [10.0] * 6
    assert grid.closes[:, 1].tolist() == [20.0] * 10
    assert grid.opens[8, 1] == 18.0
    assert grid.traded[:, 0].tolist() == [True] * 6 + [False] * 4
    assert np.argwhere(grid.blocked).tolist() == [[3, 1]]
    assert np.argwhere(~grid.in_universe[:5]).tolist() == [[1, 0]]
    assert np.isnan(grid.score[5:]).all()
    assert grid.delisted.tolist() == [7, 10]
    assert np.argwhere(grid.limit_down_open).tolist() == [[8, 1]]


def test_a_market_needs_scores() -> None:
    with pytest.raises(ValueError, match="no scores"):
        load_market(pd.DataFrame(columns=SCORE_COLUMNS), DAYS[3], CALENDAR)


def test_index_levels_read_taiex_tr_by_date() -> None:
    IndexDaily.objects.bulk_create(
        [
            IndexDaily(index_id="TAIEX_TR", date=DAYS[1], close=110),
            IndexDaily(index_id="TAIEX_TR", date=DAYS[0], close=100),
            IndexDaily(index_id="TAIEX", date=DAYS[0], close=9_000),
            IndexDaily(index_id="TAIEX_TR", date=DAYS[5], close=130),
        ]
    )

    levels = index_levels(DAYS[0], DAYS[2])

    assert levels.to_dict() == {DAYS[0]: 100.0, DAYS[1]: 110.0}
