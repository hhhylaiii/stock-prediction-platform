"""Second-batch feature groups on the rank design and the quarter-end feature (PR 4.8b-2)."""

from datetime import date
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice
from apps.research import experiments, feature_groups, lgbm_runs
from apps.research.feature_groups import (
    BLOCKED,
    FEATURE_GROUPS,
    base_arm,
    check_recorded_screening,
    check_same_ics,
    group_experiment,
    group_names,
    group_screening,
    screening_lines,
    screening_span,
)
from apps.research.features.registry import FEATURES
from apps.research.features.store import columns_for, export_year, read_partitions
from apps.research.features.transforms import SECOND_STOCK_LEVEL, STOCK_LEVEL
from apps.research.holdout import claim_holdout
from apps.research.inference import day_rows, model_inputs
from apps.research.management.commands import feature_group, target_experiment
from apps.research.models import FeatureDaily, ModelRun
from apps.research.rank_model import SPEC_RECIPE, Variant
from apps.research.runs import SCREENING_MODEL, feature_hash, load_model
from apps.research.screening import Screening, screen, screening_window
from apps.research.target_experiments import QUARTER_END, target_experiments
from apps.research.target_models import TargetVariant
from apps.research.training import load_labels
from apps.research.walkforward import WalkForwardConfig, generate_folds
from tests.research.run_panels import (
    CONFIG,
    END,
    KEPT,
    START,
    panel_calendar,
    research,
    store_taiex,
    weekdays,
)

GAP = "tech_gap_1d"


def test_the_groups_cover_the_second_batch_but_the_blocked_industry_code() -> None:
    second = [spec.name for spec in FEATURES if spec.batch == 2]
    grouped = [name for names in FEATURE_GROUPS.values() for name in names]
    assert sorted([*grouped, *QUARTER_END, *BLOCKED]) == sorted(second)
    assert set(grouped) == set(SECOND_STOCK_LEVEL)
    assert group_names(["tech"], "chip") == [
        *STOCK_LEVEL,
        *FEATURE_GROUPS["tech"],
        *FEATURE_GROUPS["chip"],
    ]
    with pytest.raises(ValueError, match="distinct names"):
        group_names(["tech"], "tech")
    with pytest.raises(ValueError, match="distinct names"):
        group_names(["sector"])


def test_fits_read_only_the_second_batch_columns_they_ask_for() -> None:
    first = columns_for(KEPT)
    assert GAP not in first
    assert columns_for([*KEPT, GAP, GAP]) == [*first, GAP]
    assert columns_for(["mkt_days_to_quarter_end"])[-1] == "mkt_days_to_quarter_end"


def test_variants_take_second_batch_stock_features_and_target_extras_take_market_ones() -> None:
    assert Variant(features=(*KEPT, GAP)).features[-1] == GAP
    with pytest.raises(ValueError, match="stock-level features only"):
        Variant(features=(*KEPT, "mkt_days_to_quarter_end"))
    with pytest.raises(ValueError, match="twice"):
        Variant(features=(GAP, GAP))
    assert TargetVariant(extra=QUARTER_END).extra == QUARTER_END
    with pytest.raises(ValueError, match="market-level features only"):
        TargetVariant(extra=(GAP,))


@pytest.fixture
def gapped(run_panel: Path) -> Path:
    """The panel with tech_gap_1d and mkt_days_to_quarter_end filled in."""
    rng = np.random.default_rng(7)
    rows = list(FeatureDaily.objects.all())
    for row in rows:
        row.tech_gap_1d = float(rng.normal())
        row.mkt_days_to_quarter_end = float(91 - row.date.timetuple().tm_yday % 91)
    FeatureDaily.objects.bulk_update(
        rows, ["tech_gap_1d", "mkt_days_to_quarter_end"], batch_size=500
    )
    export_year(2016, run_panel)
    return run_panel


@pytest.mark.django_db
def test_a_rank_fit_reads_a_second_batch_feature_it_lists(
    gapped: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lgbm_runs, "MAX_ROUNDS", 60)
    fold = research()[0]
    listed = Variant(features=(*KEPT, GAP))

    lgbm_runs.run_folds("rank", [fold], "rank-g1", gapped, variant=listed)
    lgbm_runs.run_folds("rank", [fold], "rank-g0", gapped)

    run = ModelRun.objects.get(run_id=f"rank-g1.lightgbm.h20.f{fold.index:03d}")
    assert run.feature_list == [*KEPT, GAP]
    assert run.params["variant"]["features"] == [*KEPT, GAP]
    plain = ModelRun.objects.get(run_id=f"rank-g0.lightgbm.h20.f{fold.index:03d}")
    assert "features" not in plain.params["variant"]
    assert lgbm_runs.run_folds("rank", [fold], "rank-g1", gapped, variant=listed) == []
    rows = day_rows(fold.test_start, gapped, run.feature_list)
    served = model_inputs(run, load_model("rank-g1", run.run_id), rows)
    assert list(served.columns) == [*KEPT, GAP]
    assert served[GAP].notna().any()


@pytest.mark.django_db
def test_a_target_fit_bins_the_extra_market_feature(
    gapped: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lgbm_runs, "MAX_ROUNDS", 60)
    fold = research()[0]
    extra = TargetVariant(market=False, extra=QUARTER_END)

    lgbm_runs.run_folds("return", [fold], "return-q", gapped, target=extra)

    run = ModelRun.objects.get(run_id=f"return-q.lightgbm.h5.f{fold.index:03d}")
    assert run.feature_list == [*KEPT, *QUARTER_END]
    assert set(run.params["market_bins"]) == set(QUARTER_END)
    rows = read_partitions(fold.test_start, fold.test_end, gapped, columns_for(QUARTER_END))
    served = model_inputs(run, load_model("return-q", run.run_id), rows)
    assert list(served.columns) == [*KEPT, *QUARTER_END]
    assert served[QUARTER_END[0]].dropna().isin(range(5)).all()


@pytest.mark.django_db
def test_screening_takes_a_feature_list_with_second_batch_names(gapped: Path) -> None:
    features = read_partitions(START, END, gapped, columns_for([*KEPT, GAP]))
    labels = {horizon: load_labels(START, END, horizon) for horizon in (5, 20)}

    result = screen(features, labels, [*KEPT, GAP])

    assert list(result.table.index) == [*KEPT, GAP]
    assert GAP in result.table.index


def test_group_arms_build_on_the_chosen_base_and_name_their_groups() -> None:
    plain = base_arm(CONFIG, "base", "plain")
    floor = base_arm(CONFIG, "w60", "floor")
    every5 = base_arm(CONFIG, "w60", "every5")
    assert (plain.name, floor.name, every5.name) == ("base", "w60-floor", "w60-every5")
    assert floor.variant.min_rounds == experiments.MIN_ROUNDS
    with pytest.raises(ValueError, match="base must be one of"):
        base_arm(CONFIG, "w60", "tuned")

    first = group_experiment(floor, [], None, "tech", [*KEPT, GAP])
    assert [arm.name for arm in first.arms] == ["w60-floor", "w60-floor-tech"]
    assert first.arms[0] is floor
    assert first.arms[1].variant == Variant(
        min_rounds=experiments.MIN_ROUNDS, features=(*KEPT, GAP)
    )
    later = group_experiment(floor, ["tech"], [*KEPT, GAP], "chip", KEPT)
    assert [arm.name for arm in later.arms] == ["w60-floor-tech", "w60-floor-tech-chip"]
    assert later.arms[0].variant.features == (*KEPT, GAP)
    assert first.rule == later.rule == "group"


def test_the_screening_lines_name_the_group_verdicts_and_displaced_features() -> None:
    twin = "duplicate of tech_parkinson_vol_20d (mean daily Spearman +0.970)"
    names = ["tech_vol_20d", "tech_vol_60d", *FEATURE_GROUPS["tech"]]
    table = pd.DataFrame(
        {
            "ic_h5": [0.02, 0.01, 0.01, np.nan, 0.0, 0.0, 0.0],
            "ic_h20": [0.03, 0.01, 0.001, np.nan, 0.0, 0.0, 0.0],
        },
        index=names,
    ).assign(
        reason=[twin, "", "", "", "", "", ""],
        covered=[True, True, True, False, True, True, True],
    )
    result = Screening(table, {"tech_vol_20d": twin}, table, 10)

    lines = "\n".join(screening_lines(result, "tech", ["tech_vol_20d", "tech_ret_1d"]))

    assert "| `tech_parkinson_vol_20d` | +0.0100 | +0.0010 | 保留 |" in lines
    assert (
        "| `tech_gap_1d` | — | — | 保留：IC 有值的天數不到篩選窗的 50%，未做低 IC 篩選 |" in lines
    )
    assert (
        "加入這組後被剔除的原有特徵：`tech_vol_20d`：duplicate of tech_parkinson_vol_20d" in lines
    )
    assert "加入這組後重新保留的原有特徵：`tech_vol_60d`" in lines


def test_the_screening_window_stays_on_the_36_month_tuning_folds(
    long_calendar: TradingDays,
) -> None:
    last = date(2026, 6, 30)
    layout = WalkForwardConfig(train_months=36)

    assert screening_span(long_calendar, last) == screening_window(
        generate_folds(long_calendar, last, layout)
    )
    with pytest.raises(ValueError, match="falls inside the tuning period"):
        screening_span(long_calendar, date(2015, 12, 31))


@pytest.mark.django_db
def test_a_group_screening_needs_the_partitions_the_recorded_screening_read(gapped: Path) -> None:
    with pytest.raises(ValueError, match="features changed since the recorded screening"):
        check_recorded_screening(gapped)
    screening = ModelRun.objects.get(model_type=SCREENING_MODEL)
    screening.feature_hash = feature_hash(screening.train_start, screening.train_end, gapped)
    screening.save()

    check_recorded_screening(gapped)


def test_quarter_end_compares_one_market_feature_on_the_no_market_designs() -> None:
    chosen = target_experiments(CONFIG, "w60", market=False)
    on_return, on_interval = chosen["quarter-end-return"], chosen["quarter-end-interval"]
    assert [arm.name for arm in on_return.arms] == ["w60-nomarket", "w60-nomarket-qend"]
    assert on_return.arms[1].target == TargetVariant(market=False, extra=QUARTER_END)
    assert [arm.name for arm in on_interval.arms] == [
        "w60-relative-nomarket",
        "w60-relative-nomarket-qend",
    ]
    assert on_interval.arms[1].target == TargetVariant(
        market=False, relative_up=True, extra=QUARTER_END
    )
    assert (on_return.primary, on_interval.primary) == ("mae", "pinball")
    assert {arm.variant for arm in (*on_return.arms, *on_interval.arms)} == {SPEC_RECIPE}


@pytest.fixture
def command(gapped: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(feature_group, "DEFAULT_CONFIG", CONFIG)
    monkeypatch.setattr(experiments, "COARSE_MONTHS", 1)
    monkeypatch.setattr(experiments, "LONG_MONTHS", 7)
    monkeypatch.setattr(experiments, "WINDOW_FROM", date(2016, 1, 1))
    monkeypatch.setattr(lgbm_runs, "MAX_ROUNDS", 60)
    DailyPrice.objects.create(
        stock_id="1101", date=weekdays()[-1], close=10, volume=1000, turnover=10_000
    )
    screening = ModelRun.objects.get(model_type=SCREENING_MODEL)
    screening.feature_hash = feature_hash(screening.train_start, screening.train_end, gapped)
    screening.params = {
        **screening.params,
        "ic": {name: {"5": 0.01, "20": 0.02} for name in STOCK_LEVEL},
    }
    screening.save()
    return gapped


def _screened(kept: list[str], names: list[str] | None = None) -> Screening:
    names = names or [*KEPT, *FEATURE_GROUPS["tech"]]
    table = pd.DataFrame({"ic_h5": 0.01, "ic_h20": 0.02}, index=names).assign(
        reason=[
            "" if name in kept else "|rank IC| below 0.005 at horizons 5 and 20" for name in names
        ],
        covered=True,
    )
    dropped = {name: str(table.loc[name, "reason"]) for name in names if name not in kept}
    return Screening(table, dropped, pd.DataFrame(), 100)


@pytest.mark.django_db
def test_feature_group_fits_the_design_with_the_screened_group_and_reports(
    command: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(feature_group, "group_screening", lambda *args: _screened([*KEPT, GAP]))
    output = tmp_path / "tech.md"
    out = StringIO()

    call_command(
        "feature_group",
        group="tech",
        base="plain",
        window="base",
        run_key="rank-fg",
        seeds="0",
        output=str(output),
        stdout=out,
    )

    text = output.read_text()
    assert "# 實驗：group-tech（rank-fg）" in text
    assert "## 這組的篩選" in text
    assert "| `tech_gap_1d` | +0.0100 | +0.0200 | 保留 |" in text
    assert "## 受阻、沒評估的特徵" in text
    assert "`ind_sector_id`：受阻：要等時點正確（PIT）的產業資料" in text
    assert "特徵 4 個（重新篩選後的清單）" in text
    assert "base-tech: ran folds [" in out.getvalue()
    tech = ModelRun.objects.filter(run_key="rank-fg-base-tech", model_type="lightgbm")
    assert {tuple(run.feature_list) for run in tech} == {(*KEPT, GAP)}


@pytest.mark.django_db
def test_a_group_the_screening_drops_whole_is_not_fit(
    command: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(feature_group, "group_screening", lambda *args: _screened(list(KEPT)))
    output = tmp_path / "tech.md"

    call_command(
        "feature_group",
        group="tech",
        base="plain",
        window="base",
        run_key="rank-fg",
        seeds="0",
        output=str(output),
        stdout=StringIO(),
    )

    assert "這組在篩選全部剔除，不跑 walk-forward" in output.read_text()
    assert "`ind_sector_id`：受阻" in output.read_text()
    assert not ModelRun.objects.filter(run_key__startswith="rank-fg").exists()


@pytest.mark.django_db
def test_a_later_group_compares_against_the_accepted_groups_screened_again(
    command: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ex_date = FEATURE_GROUPS["exdt"][0]
    accepted, both = [*KEPT, GAP], [KEPT[0], KEPT[1], GAP, ex_date]

    def screened(names: list[str], *args: object) -> Screening:
        return _screened(both if ex_date in names else accepted, list(names))

    monkeypatch.setattr(feature_group, "group_screening", screened)
    output = tmp_path / "exdt.md"

    call_command(
        "feature_group",
        group="exdt",
        accepted="tech",
        base="plain",
        window="base",
        run_key="rank-fg",
        seeds="0",
        output=str(output),
        stdout=StringIO(),
    )

    runs = ModelRun.objects.filter(model_type="lightgbm")
    lists = {run.run_key: tuple(run.feature_list) for run in runs}
    assert lists == {"rank-fg-base-tech": tuple(accepted), "rank-fg-base-tech-exdt": tuple(both)}
    assert "加入這組後被剔除的原有特徵：`tech_vol_20d`" in output.read_text()


@pytest.mark.django_db
def test_feature_group_refuses_an_unknown_accepted_group(command: Path) -> None:
    with pytest.raises(CommandError, match="distinct names"):
        call_command("feature_group", group="tech", accepted="sector", run_key="rank-fg")


@pytest.mark.django_db
@pytest.mark.parametrize("name", ["quarter-end-return", "quarter-end-interval"])
def test_the_quarter_end_experiments_write_their_reports(
    gapped: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    monkeypatch.setattr(target_experiment, "DEFAULT_CONFIG", CONFIG)
    monkeypatch.setattr(experiments, "COARSE_MONTHS", 1)
    monkeypatch.setattr(experiments, "WINDOW_FROM", date(2016, 1, 1))
    monkeypatch.setattr(lgbm_runs, "MAX_ROUNDS", 60)
    DailyPrice.objects.create(
        stock_id="1101", date=weekdays()[-1], close=10, volume=1000, turnover=10_000
    )
    store_taiex()
    output = tmp_path / "quarter.md"
    scope = name.removeprefix("quarter-end-")

    call_command(
        "target_experiment", name=name, run_key=f"{scope}-qe", output=str(output), stdout=StringIO()
    )

    text = output.read_text()
    assert f"# 實驗：{name}（{scope}-qe）" in text
    assert "再加 `mkt_days_to_quarter_end`" in text
    assert "SPEC §6：「總體 OOS rank IC 沒有提高就不留」" in text
    assert "沒有 rank IC 可比" in text
    assert "**判定" in text


@pytest.fixture
def screened_panel(gapped: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    span = (weekdays()[0], weekdays()[-1])
    monkeypatch.setattr(feature_groups, "screening_span", lambda calendar, last: span)
    return gapped


@pytest.mark.django_db
def test_a_group_screening_reads_the_group_columns(screened_panel: Path) -> None:
    result = group_screening([*KEPT, GAP], screened_panel, panel_calendar(), END)

    assert result.table["ic_h5"].notna().to_dict()[GAP]


@pytest.mark.django_db
def test_a_group_screening_refuses_missing_days_and_partitions_rewritten_mid_read(
    screened_panel: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    days = sorted([*weekdays(), date(2016, 3, 5)])
    with pytest.raises(ValueError, match="features are missing on 1 of"):
        group_screening(KEPT, screened_panel, TradingDays(days, days[0], days[-1]), END)

    reader = feature_groups.read_partitions

    def rewritten(start: date, end: date, directory: Path, columns: list[str]) -> pd.DataFrame:
        frame = reader(start, end, directory, columns)
        FeatureDaily.objects.filter(stock_id="1101").update(tech_ret_1d=9.0)
        export_year(2016, directory)
        return frame

    monkeypatch.setattr(feature_groups, "read_partitions", rewritten)
    with pytest.raises(RuntimeError, match="changed while reading"):
        group_screening(KEPT, screened_panel, panel_calendar(), END)


def test_a_rescreening_must_reproduce_the_recorded_first_batch_ics() -> None:
    table = pd.DataFrame({"ic_h5": [0.01, np.nan], "ic_h20": [0.02, 0.03]}, index=KEPT[:2])
    result = Screening(table, {}, table, 10)
    recorded = {KEPT[0]: {"5": 0.01, "20": 0.02}, KEPT[1]: {"5": None, "20": 0.03}}

    check_same_ics(result, ModelRun(params={"ic": recorded}))
    with pytest.raises(ValueError, match=r"tech_ret_60d's 20-day IC 0\.030000 differs"):
        check_same_ics(
            result, ModelRun(params={"ic": {**recorded, KEPT[1]: {"5": None, "20": 0.04}}})
        )


@pytest.mark.django_db
def test_feature_group_checks_its_run_keys_before_screening(
    command: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(*args: object) -> Screening:
        raise AssertionError("screened before the run keys were checked")

    monkeypatch.setattr(feature_group, "group_screening", unexpected)

    with pytest.raises(CommandError, match="at most 40 characters"):
        call_command("feature_group", group="tech", run_key=f"rank-{'x' * 30}", stdout=StringIO())


@pytest.mark.django_db
def test_feature_group_refuses_once_the_rank_holdout_is_claimed(command: Path) -> None:
    claim_holdout("rank", "rank-final", weekdays()[-1])

    with pytest.raises(CommandError, match="rank design is frozen"):
        call_command("feature_group", group="tech", run_key="rank-fg", stdout=StringIO())
