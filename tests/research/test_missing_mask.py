from dataclasses import replace
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice, StockMaster
from apps.research import experiments, lgbm_runs, missing_mask
from apps.research.experiments import arm_key, compared_folds, daily_ics, window_arms
from apps.research.features.registry import FEATURES, Group
from apps.research.features.store import export_year
from apps.research.features.transforms import STOCK_LEVEL
from apps.research.management.commands import missing_mask as command_module
from apps.research.metrics.stats import paired_nw_t
from apps.research.missing_mask import (
    MASKED,
    MASKED_GROUPS,
    SCORE_COLUMNS,
    delisting_scores,
    delisting_summary,
    delisting_windows,
    later_delisted,
    mask_experiment,
    mask_report,
    masked_gain_share,
    masked_values,
    missing_split,
    soon_delisted,
)
from apps.research.models import FeatureDaily, HoldoutRun, HoldoutScope, ModelObjective, ModelRun
from apps.research.prediction_cache import load_predictions
from apps.research.runs import SCREENING_MODEL
from apps.research.target_eval import FOLD_LAG
from apps.research.walkforward import WalkForwardConfig
from tests.research.run_panels import CONFIG, weekdays

SPEC_LAYOUT = WalkForwardConfig()


def first_batch(group: Group) -> set[str]:
    return {
        spec.name
        for spec in FEATURES
        if spec.batch == 1 and spec.group == group and spec.name in STOCK_LEVEL
    }


def test_the_masked_groups_are_the_sources_delisted_stocks_lack() -> None:
    assert list(MASKED_GROUPS) == ["財報", "月營收", "每日估值", "籌碼", "股本", "產業"]
    assert len(MASKED) == len(set(MASKED)) == 33
    assert set(MASKED) <= set(STOCK_LEVEL)
    assert set(MASKED_GROUPS["籌碼"]) == first_batch(Group.CHIP)
    fundamentals = {*MASKED_GROUPS["財報"], *MASKED_GROUPS["月營收"], *MASKED_GROUPS["每日估值"]}
    assert fundamentals == first_batch(Group.FUNDAMENTAL)
    assert MASKED_GROUPS["月營收"] == ("fund_rev_yoy", "fund_rev_3m_yoy", "fund_rev_mom")
    assert MASKED_GROUPS["每日估值"] == ("fund_bp", "fund_dividend_yield")
    assert MASKED_GROUPS["股本"] == ("size_log_mcap", "tech_turnover_rate_5d")
    assert MASKED_GROUPS["產業"] == ("ind_sector_ret_20d", "ind_rel_ret_20d")


@pytest.mark.parametrize(("window", "name"), [("base", "masked"), ("w60", "w60-masked")])
def test_the_mask_experiment_masks_the_chosen_window(window: str, name: str) -> None:
    plain, masked = mask_experiment(SPEC_LAYOUT, window).arms

    assert plain == window_arms(SPEC_LAYOUT)[window]
    assert masked.name == name
    assert masked.config == plain.config
    assert masked.variant == replace(plain.variant, masked=MASKED)


def test_the_mask_experiment_keeps_the_decay_of_the_expanding_window() -> None:
    plain, masked = mask_experiment(SPEC_LAYOUT, "expanding").arms

    assert (plain.name, masked.name) == ("expanding", "expanding-masked")
    assert masked.variant.half_life == plain.variant.half_life == experiments.HALF_LIFE


def test_the_mask_experiment_refuses_an_unknown_window() -> None:
    with pytest.raises(ValueError, match="window must be one of"):
        mask_experiment(SPEC_LAYOUT, "w36")


def calendar() -> TradingDays:
    days = weekdays()
    return TradingDays(days, days[0], days[-1])


def delist(stock_id: str, day: date | None, *quotes: date) -> None:
    StockMaster.objects.create(stock_id=stock_id, name=stock_id, market="TWSE", delist_date=day)
    DailyPrice.objects.bulk_create(
        DailyPrice(stock_id=stock_id, date=quote, close=10, volume=1000, turnover=10_000)
        for quote in quotes
    )


@pytest.mark.django_db
def test_delisting_windows_end_at_the_last_quote_before_delisting() -> None:
    delist("1101", date(2016, 12, 1), date(2016, 11, 10), date(2016, 11, 15))
    delist("1102", date(2016, 10, 3), date(2016, 9, 30), date(2016, 10, 3))
    delist("1103", date(2016, 1, 5), date(2016, 1, 4))
    delist("1104", date(2015, 6, 1), date(2015, 5, 29))
    delist("1105", None, date(2016, 11, 30))
    delist("1106", date(2017, 3, 1), date(2016, 12, 30))
    delist("1107", date(2016, 11, 1))

    windows = delisting_windows(calendar(), sessions=3)

    assert windows == {
        "1101": (date(2016, 11, 11), date(2016, 11, 15)),
        "1102": (date(2016, 9, 28), date(2016, 9, 30)),
        "1103": (date(2016, 1, 1), date(2016, 1, 4)),
        "1106": (date(2016, 12, 28), date(2016, 12, 30)),
    }


def test_soon_delisted_flags_the_rows_inside_their_stock_window() -> None:
    rows = pd.DataFrame(
        {
            "stock_id": ["1101", "1101", "1101", "1102", "1103", "1102"],
            "date": [
                date(2016, 11, 25),
                date(2016, 11, 28),
                date(2016, 11, 30),
                date(2016, 11, 28),
                date(2016, 11, 28),
                date(2016, 9, 1),
            ],
        },
        index=[4, 5, 6, 7, 8, 9],
    )
    windows = {
        "1101": (date(2016, 11, 28), date(2016, 11, 30)),
        "1102": (date(2016, 9, 28), date(2016, 9, 30)),
    }

    assert soon_delisted(rows, windows).to_dict() == {
        4: False,
        5: True,
        6: True,
        7: False,
        8: False,
        9: False,
    }
    assert later_delisted(rows, windows).to_dict() == {
        4: True,
        5: True,
        6: True,
        7: False,
        8: False,
        9: True,
    }


def scored(
    values: dict[tuple[str, int], list[float]], seeds: tuple[int, ...] = (0,), near: bool = True
) -> Any:
    """Score rows: each (arm, fold) gets one stock-day per listed value at both horizons."""
    rows = []
    for (arm, fold), per_row in values.items():
        for seed in seeds:
            for k, pct in enumerate(per_row):
                for horizon in (5, 20):
                    day = date(2016, 1 + fold, 1 + k + (0 if near else 10))
                    stock = f"{1101 + k}"
                    rows.append([arm, seed, fold, horizon, stock, day, near, pct + seed / 100])
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


FIRST = {0: [0.20, 0.30], 1: [0.25, 0.35], 2: [0.10, 0.20], 3: [0.30, 0.20]}
RISEN = {0: [0.40, 0.50], 1: [0.46, 0.54], 2: [0.30, 0.42], 3: [0.47, 0.45]}


def both(first: dict[int, list[float]], second: dict[int, list[float]]) -> Any:
    values = {("base", fold): pcts for fold, pcts in first.items()}
    values.update({("masked", fold): pcts for fold, pcts in second.items()})
    return values


def test_delisting_summary_pairs_the_seed_averaged_fold_means() -> None:
    scores = scored(both(FIRST, RISEN), seeds=(0, 2))

    table = delisting_summary(mask_experiment(SPEC_LAYOUT), scores).to_dict(orient="index")

    first = pd.Series({fold: np.mean(pcts) + 0.01 for fold, pcts in FIRST.items()})
    second = pd.Series({fold: np.mean(pcts) + 0.01 for fold, pcts in RISEN.items()})
    for horizon in (5, 20):
        row = table[("near", horizon)]
        assert table[("all", horizon)] == row
        assert row["first"] == pytest.approx(np.mean([*FIRST.values()]) + 0.01)
        assert row["second"] == pytest.approx(np.mean([*RISEN.values()]) + 0.01)
        assert row["gap"] == pytest.approx(row["second"] - row["first"])
        assert row["nw_t"] == pytest.approx(paired_nw_t(second, first, FOLD_LAG))
        assert (row["folds"], row["stocks"], row["rows"]) == (4, 2, 8)
    assert table[("near", 20)]["nw_t"] > 2


def test_delisting_summary_counts_every_row_before_delisting_in_the_all_group() -> None:
    far = {fold: [0.6, 0.6] for fold in FIRST}
    scores = pd.concat(
        [scored(both(FIRST, RISEN)), scored(both(far, far), near=False)], ignore_index=True
    )

    table = delisting_summary(mask_experiment(SPEC_LAYOUT), scores).to_dict(orient="index")

    near, every = table[("near", 20)], table[("all", 20)]
    assert (near["rows"], every["rows"]) == (8, 16)
    assert near["first"] == pytest.approx(np.mean([*FIRST.values()]))
    assert every["first"] == pytest.approx((np.mean([*FIRST.values()]) + 0.6) / 2)
    assert every["gap"] == pytest.approx(near["gap"] / 2)


def test_delisting_summary_pairs_only_the_folds_both_arms_have() -> None:
    second = {fold: pcts for fold, pcts in RISEN.items() if fold != 3}
    scores = scored(both(FIRST, second))

    table = delisting_summary(mask_experiment(SPEC_LAYOUT), scores).to_dict(orient="index")

    first = pd.Series({fold: np.mean(pcts) for fold, pcts in FIRST.items() if fold != 3})
    risen = pd.Series({fold: np.mean(pcts) for fold, pcts in second.items()})
    assert table[("near", 5)]["nw_t"] == pytest.approx(paired_nw_t(risen, first, FOLD_LAG))
    assert table[("near", 5)]["folds"] == 3


def test_delisting_summary_of_no_rows_is_empty() -> None:
    scores = pd.DataFrame(columns=SCORE_COLUMNS)

    table = delisting_summary(mask_experiment(SPEC_LAYOUT), scores)

    assert list(table.index) == [("near", 5), ("near", 20), ("all", 5), ("all", 20)]
    assert table["rows"].tolist() == [0, 0, 0, 0]
    assert table[["first", "second", "nw_t"]].isna().all().all()


def test_missing_split_groups_the_rows_by_their_masked_gaps() -> None:
    scores = scored(both({0: [0.2, 0.3, 0.4]}, {0: [0.5, 0.3, 0.6]}))
    share = pd.DataFrame(
        {
            "stock_id": ["1101", "1102", "1103"],
            "date": [date(2016, 1, 1), date(2016, 1, 2), date(2016, 1, 3)],
            "missing_share": [1.0, 0.0, 0.5],
        }
    )

    split = missing_split(mask_experiment(SPEC_LAYOUT), scores, share).to_dict(orient="index")

    many, few = split[("near", 20, "缺值多")], split[("near", 20, "缺值少")]
    assert split[("all", 20, "缺值多")] == many
    assert (many["rows"], few["rows"]) == (2, 1)
    assert many["first"] == pytest.approx(0.3)
    assert many["second"] == pytest.approx(0.55)
    assert many["gap"] == pytest.approx(0.25)
    assert (few["first"], few["second"], few["gap"]) == pytest.approx((0.3, 0.3, 0.0))


@pytest.fixture
def small(run_panel: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The check on the one-year panel: monthly folds; fund_ep is masked but never screened."""
    monkeypatch.setattr(experiments, "COARSE_MONTHS", 1)
    monkeypatch.setattr(experiments, "WINDOW_FROM", date(2016, 1, 1))
    monkeypatch.setattr(lgbm_runs, "MAX_ROUNDS", 80)
    monkeypatch.setattr(missing_mask, "MASKED", ("tech_ret_60d", "fund_ep"))
    return run_panel


def trained(directory: Path, seeds: tuple[int, ...] = (0,)) -> tuple[Any, dict[str, Any]]:
    chosen = mask_experiment(CONFIG)
    folds = compared_folds(chosen, calendar(), weekdays()[-1])
    experiments.train_arms(chosen, "rank-mask", folds, directory, seeds)
    return chosen, folds


@pytest.mark.django_db
def test_delisting_scores_rank_each_day_and_keep_the_rows_before_delisting(small: Path) -> None:
    chosen, folds = trained(small, (0, 1))
    delist("1105", date(2016, 12, 1), date(2016, 11, 30))
    windows = delisting_windows(calendar(), sessions=30)

    scores = delisting_scores(chosen, "rank-mask", folds, (0, 1), windows)

    assert set(scores["stock_id"]) == {"1105"}
    first, last = windows["1105"]
    assert (scores["date"] <= last).all()
    assert scores["near"].tolist() == (scores["date"] >= first).tolist()
    assert scores["near"].any()
    assert not scores["near"].all()
    for arm in chosen.arms:
        for seed, horizon in ((0, 5), (1, 20)):
            cached = load_predictions(arm_key("rank-mask", arm, seed), "lightgbm", folds[arm.name])
            ends = cached["fold"].map({fold.index: fold.eval_end for fold in folds[arm.name]})
            tested = cached[(cached["date"] <= ends) & cached["in_universe"].astype(bool)]
            pct = tested.groupby("date")[f"lightgbm_h{horizon}"].rank(pct=True)
            later = (tested["stock_id"] == "1105") & (tested["date"] <= last)
            expected = pd.Series(pct[later].to_numpy(), index=tested.loc[later, "date"].to_numpy())
            got = scores[
                (scores["arm"] == arm.name)
                & (scores["seed"] == seed)
                & (scores["horizon"] == horizon)
            ]
            assert not got.empty
            pd.testing.assert_series_equal(
                got.set_index("date")["pct"].sort_index(),
                expected.sort_index(),
                check_names=False,
                check_index_type=False,
            )


@pytest.mark.django_db
def test_masked_values_give_each_scored_row_its_share_of_missing_masked_features(
    small: Path,
) -> None:
    FeatureDaily.objects.filter(stock_id="1105", date=date(2016, 11, 1)).update(tech_ret_60d=None)
    rows = pd.DataFrame(
        {"stock_id": ["1105", "1105"], "date": [date(2016, 11, 1), date(2016, 11, 2)]}
    )
    export_year(2016, small)

    share = masked_values(rows, small, ["tech_ret_60d", "tech_ret_1d"])

    assert share.set_index("date")["missing_share"].to_dict() == {
        date(2016, 11, 1): 0.5,
        date(2016, 11, 2): 0.0,
    }


@pytest.mark.django_db
def test_masked_gain_share_reads_the_first_arm_gain_on_the_masked_features(small: Path) -> None:
    chosen, folds = trained(small, (0, 1))

    shares = masked_gain_share(chosen, "rank-mask", folds, (0, 1))

    for horizon in (5, 20):
        expected = []
        for seed in (0, 1):
            key = arm_key("rank-mask", chosen.arms[0], seed)
            for fold in folds[chosen.arms[0].name]:
                run_id = lgbm_runs.run_id(key, ModelObjective.RANK, horizon, fold)
                gains = ModelRun.objects.get(run_id=run_id).params["importance_gain"]
                expected.append(gains["tech_ret_60d"] / sum(gains.values()))
        assert shares[horizon] == pytest.approx(np.mean(expected))


def ic_daily() -> pd.DataFrame:
    rows = [
        [arm, 0, fold, horizon, date(2016, 1 + fold, 4), ic]
        for arm, ic in (("base", 0.06), ("masked", 0.05))
        for fold in range(4)
        for horizon in (5, 20)
    ]
    return pd.DataFrame(rows, columns=experiments.DAILY_COLUMNS)


def report(scores: pd.DataFrame, split: pd.DataFrame | None = None) -> str:
    chosen = mask_experiment(SPEC_LAYOUT)
    shares = {5: 0.123, 20: 0.2}
    return mask_report(
        chosen, "rank-mask", ic_daily(), scores, split, shares, ["fund_ep", "chip_short_ratio"]
    )


def test_the_report_flags_a_clear_rise_of_the_soon_delisted_scores() -> None:
    scores = scored(both(FIRST, RISEN))

    text = report(scores)

    table = delisting_summary(mask_experiment(SPEC_LAYOUT), scores)
    assert text.startswith("# 缺值遮蔽檢查（rank-mask）\n")
    assert "- masked：rolling 36 個月" in text
    assert "，遮掉 33 個特徵" in text
    assert "- 遮掉下市股缺值偏高的 6 類特徵，共 33 個，篩選後實際用到 2 個" in text
    assert "  - 月營收：fund_rev_yoy、fund_rev_3m_yoy、fund_rev_mom" in text
    assert "這是診斷，不是採用判定" in text
    assert "倖存者樣本（SPEC §8 第 3 條），不做" in text
    assert "| base | 0.0600 | 0.0600 | 0.0600 | — |" in text
    assert "| masked | 0.0500 | 0.0500 | 0.0500 | -0.0100 |" in text
    assert "靠的是缺值還是特徵值，看「依缺值拆分」" in text
    assert "## 之後會下市的股票的分數" in text
    assert "- 下市前 120 個交易日：從該檔下市前最後一個有報價的交易日往前數" in text
    row = table.to_dict(orient="index")[("near", 20)]
    gap, nw_t, first, second = row["gap"], row["nw_t"], row["first"], row["second"]
    cells = f"{first:.4f} | {second:.4f} | {gap:+.4f} | {nw_t:+.2f} | 4 | 2 | 8 |"
    assert f"| 下市前 120 個交易日 | 20 日 | {cells}" in text
    assert f"| 下市前全部 | 20 日 | {cells}" in text
    assert (
        "**判讀：⚠️ 遮掉後 下市前 120 個交易日 5 日、下市前 120 個交易日 20 日、下市前全部 5 日、"
        "下市前全部 20 日的分數回升，逐摺差 NW t 高於 2"
    ) in text
    assert "| 5 日 | 12.3% |" in text
    assert "| 20 日 | 20.0% |" in text
    assert "## 每摺" in text


def test_the_report_states_when_no_rise_shows_and_when_no_row_is_near_delisting() -> None:
    flat = report(scored(both(FIRST, FIRST)))
    empty = report(pd.DataFrame(columns=SCORE_COLUMNS))

    assert "**判讀：沒有看到遮掉後分數明顯回升" in flat
    assert "- 比較的測試期裡沒有之後會下市的列" in empty
    assert "**判讀：沒有看到遮掉後分數明顯回升" in empty


def test_the_report_lists_the_missing_split() -> None:
    scores = scored(both({0: [0.2, 0.3, 0.4]}, {0: [0.5, 0.3, 0.6]}))
    share = pd.DataFrame(
        {
            "stock_id": ["1101", "1102", "1103"],
            "date": [date(2016, 1, 1), date(2016, 1, 2), date(2016, 1, 3)],
            "missing_share": [1.0, 0.0, 0.5],
        }
    )
    split = missing_split(mask_experiment(SPEC_LAYOUT), scores, share)

    text = report(scores, split)
    skipped = report(scores)

    assert "## 依缺值拆分" in text
    assert "回升集中在缺值多，代表模型靠的是缺值本身；兩群都回升，代表靠的是特徵值" in text
    assert "| 下市前 120 個交易日 | 20 日 | 缺值多 | 2 | 0.3000 | 0.5500 | +0.2500 |" in text
    assert "| 下市前全部 | 20 日 | 缺值少 | 1 | 0.3000 | 0.3000 | +0.0000 |" in text
    assert "- 篩選後沒有被遮掉的特徵，或沒有之後會下市的列：不拆" in skipped


@pytest.fixture
def command(small: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(command_module, "DEFAULT_CONFIG", CONFIG)
    DailyPrice.objects.create(
        stock_id="1101", date=weekdays()[-1], close=10, volume=1000, turnover=10_000
    )
    delist("1105", date(2016, 12, 1), date(2016, 11, 30))
    return small


@pytest.mark.django_db
def test_missing_mask_trains_both_arms_and_writes_the_report(command: Path, tmp_path: Path) -> None:
    output = tmp_path / "reports" / "mask.md"
    out = StringIO()

    call_command(
        "missing_mask",
        run_key="rank-mask",
        seeds="0,1",
        note="暫定",
        output=str(output),
        stdout=out,
    )

    chosen, folds = trained(command, (0, 1))
    daily = daily_ics(chosen, "rank-mask", folds, (0, 1))
    scores = delisting_scores(chosen, "rank-mask", folds, (0, 1), delisting_windows(calendar()))
    shares = masked_gain_share(chosen, "rank-mask", folds, (0, 1))
    split = missing_split(chosen, scores, masked_values(scores, command, ["tech_ret_60d"]))
    expected = mask_report(
        chosen, "rank-mask", daily, scores, split, shares, ["tech_ret_60d"], "暫定"
    )
    assert output.read_text() == expected
    assert "，共 2 個，篩選後實際用到 1 個" in expected
    assert "| 下市前 120 個交易日 | 20 日 | 缺值少 |" in expected
    assert "rank-mask-masked-s1: ran folds" in out.getvalue()
    assert f"wrote {output}" in out.getvalue()


@pytest.mark.django_db
def test_missing_mask_reuses_the_window_arm_cache(command: Path) -> None:
    folds = compared_folds(mask_experiment(CONFIG), calendar(), weekdays()[-1])
    lgbm_runs.run_folds("rank", folds["base"], "rank-mask-base", command)
    out = StringIO()

    call_command("missing_mask", run_key="rank-mask", stdout=out)

    assert "rank-mask-base: ran folds []" in out.getvalue()
    assert "rank-mask-masked: ran folds [" in out.getvalue()


@pytest.mark.django_db
def test_missing_mask_refuses_bad_keys_and_seeds_before_training(command: Path) -> None:
    with pytest.raises(CommandError, match="start with rank-"):
        call_command("missing_mask", run_key="interval-x", stdout=StringIO())
    with pytest.raises(CommandError, match="at most"):
        call_command("missing_mask", run_key="rank-" + "x" * 27, seeds="0,3", stdout=StringIO())
    with pytest.raises(CommandError, match="--seeds"):
        call_command("missing_mask", run_key="rank-mask", seeds="0,0", stdout=StringIO())
    with pytest.raises(CommandError, match="window must be one of"):
        call_command("missing_mask", run_key="rank-mask", window="w36", stdout=StringIO())
    assert not ModelRun.objects.filter(model_type="lightgbm").exists()


@pytest.mark.django_db
def test_missing_mask_refuses_to_run_once_the_rank_holdout_is_claimed(command: Path) -> None:
    HoldoutRun.objects.create(
        scope=HoldoutScope.RANK, attempt=1, run_key="rank-final", last_date=weekdays()[-1]
    )

    with pytest.raises(CommandError, match=r"rank holdout .* frozen"):
        call_command("missing_mask", run_key="rank-mask", stdout=StringIO())
    assert not ModelRun.objects.filter(model_type="lightgbm").exists()


@pytest.mark.django_db
def test_missing_mask_needs_a_screening_and_a_clean_shuffle_check(
    command: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiments, "shuffle_check", lambda *args: 0.5)
    with pytest.raises(CommandError, match="shuffle check on fold"):
        call_command("missing_mask", run_key="rank-mask", stdout=StringIO())
    shuffled: list[Any] = []
    monkeypatch.setattr(experiments, "shuffle_check", lambda *args: shuffled.append(args))
    ModelRun.objects.filter(model_type=SCREENING_MODEL).delete()

    with pytest.raises(CommandError, match="screened features"):
        call_command("missing_mask", run_key="rank-mask", stdout=StringIO())
    assert shuffled == []
    assert not ModelRun.objects.filter(model_type="lightgbm").exists()
