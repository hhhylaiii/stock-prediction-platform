"""Missingness check: refit rank without the feature groups delisted stocks lack (K17)."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from django.db.models import OuterRef, Subquery

from apps.market.calendar import OutsideCalendarError, TradingDays
from apps.market.models import DailyPrice, StockMaster
from apps.research.experiments import (
    MODEL,
    Arm,
    Experiment,
    arm_key,
    arm_setting,
    fold_lines,
    ic_table_lines,
    seed_lines,
    summary,
    window_arms,
)
from apps.research.features.store import read_columns
from apps.research.labels import HORIZONS
from apps.research.metrics.stats import paired_nw_t
from apps.research.prediction_cache import load_predictions
from apps.research.ranges import years
from apps.research.rank_eval import fold_gains
from apps.research.target_eval import FOLD_LAG, GATE_T
from apps.research.walkforward import Fold, WalkForwardConfig

MASKED_GROUPS: dict[str, tuple[str, ...]] = {
    "財報": (
        "fund_ep",
        "fund_eps_yoy",
        "fund_roe_ttm",
        "fund_gross_margin",
        "fund_op_margin",
        "fund_debt_ratio",
        "fund_gm_yoy_chg",
        "fund_opm_yoy_chg",
        "fund_cfo_to_ni",
        "fund_days_since_fs",
        "fund_rev_yoy_q",
    ),
    "月營收": ("fund_rev_yoy", "fund_rev_3m_yoy", "fund_rev_mom"),
    "每日估值": ("fund_bp", "fund_dividend_yield"),
    "籌碼": (
        "chip_fi_net_1d",
        "chip_fi_net_5d",
        "chip_fi_net_20d",
        "chip_it_net_5d",
        "chip_dealer_self_net_5d",
        "chip_dealer_hedge_net_5d",
        "chip_margin_chg_5d",
        "chip_margin_usage",
        "chip_short_ratio",
        "chip_fi_holding_pct",
        "chip_sbl_short_chg_5d",
        "chip_big_holder_pct_chg_1w",
        "chip_holder_count_chg_1w",
    ),
    "股本": ("size_log_mcap", "tech_turnover_rate_5d"),
    "產業": ("ind_sector_ret_20d", "ind_rel_ret_20d"),
}
MASKED = tuple(name for names in MASKED_GROUPS.values() for name in names)
DELIST_SESSIONS = 120
MOSTLY_MISSING = 0.5
SCORE_COLUMNS = ["arm", "seed", "fold", "horizon", "stock_id", "date", "near", "pct"]
ROW_KEYS = ["arm", "horizon", "fold", "stock_id", "date", "near"]
POPULATIONS = {"near": f"下市前 {DELIST_SESSIONS} 個交易日", "all": "下市前全部"}
SUMMARY_COLUMNS = ["first", "second", "gap", "nw_t", "folds", "stocks", "rows"]
MANY, FEW = "缺值多", "缺值少"


def mask_experiment(base: WalkForwardConfig, window: str = "base") -> Experiment:
    """A window's arm against the same arm with the MASKED features left out."""
    windows = window_arms(base)
    if window not in windows:
        raise ValueError(f"window must be one of {', '.join(windows)}, not {window!r}")
    plain = windows[window]
    prefix = "" if window == "base" else f"{window}-"
    masked = replace(plain, name=f"{prefix}masked", variant=replace(plain.variant, masked=MASKED))
    return Experiment("missing-mask", "diagnostic", (plain, masked))


def delisting_windows(
    calendar: TradingDays, sessions: int = DELIST_SESSIONS
) -> dict[str, tuple[date, date]]:
    """Each delisted stock's last sessions trading days up to its last quote before delisting."""
    quotes = DailyPrice.objects.filter(
        stock_id=OuterRef("stock_id"), date__lt=OuterRef("delist_date")
    ).order_by("-date")
    rows = (
        StockMaster.objects.filter(delist_date__isnull=False)
        .annotate(quoted=Subquery(quotes.values("date")[:1]))
        .values_list("stock_id", "quoted")
    )
    windows = {}
    for stock_id, quoted in rows:
        if quoted is None:
            continue
        try:
            last = calendar.latest_on_or_before(quoted)
        except OutsideCalendarError:
            continue
        try:
            first = calendar.shift(last, 1 - sessions)
        except OutsideCalendarError:
            first = calendar.start
        windows[stock_id] = (first, last)
    return windows


def _ends(rows: pd.DataFrame, windows: Mapping[str, tuple[date, date]], end: int) -> pd.Series:
    return pd.to_datetime(rows["stock_id"].map({key: span[end] for key, span in windows.items()}))


def soon_delisted(rows: pd.DataFrame, windows: Mapping[str, tuple[date, date]]) -> pd.Series:
    """Whether each row is dated inside its stock's delisting window."""
    days = pd.to_datetime(rows["date"])
    return (days >= _ends(rows, windows, 0)) & (days <= _ends(rows, windows, 1))


def later_delisted(rows: pd.DataFrame, windows: Mapping[str, tuple[date, date]]) -> pd.Series:
    """Whether each row belongs to a stock that delists later, dated up to its last window day."""
    return pd.to_datetime(rows["date"]) <= _ends(rows, windows, 1)


def _arm_scores(
    run_key: str,
    arm: Arm,
    seed: int,
    folds: Sequence[Fold],
    windows: Mapping[str, tuple[date, date]],
) -> list[pd.DataFrame]:
    cached = load_predictions(arm_key(run_key, arm, seed), MODEL, folds)
    ends = cached["fold"].map({fold.index: fold.eval_end for fold in folds})
    tested = cached[(cached["date"] <= ends) & cached["in_universe"].astype(bool)]
    later, near = later_delisted(tested, windows), soon_delisted(tested, windows)
    parts = []
    for horizon in HORIZONS:
        pct = tested.groupby("date")[f"{MODEL}_h{horizon}"].rank(pct=True)
        picked = tested.loc[later, ["fold", "stock_id", "date"]]
        parts.append(
            picked.assign(
                pct=pct[later], near=near[later], arm=arm.name, seed=seed, horizon=horizon
            )
        )
    return parts


def delisting_scores(
    experiment: Experiment,
    run_key: str,
    folds: Mapping[str, list[Fold]],
    seeds: Sequence[int],
    windows: Mapping[str, tuple[date, date]],
) -> pd.DataFrame:
    """Daily universe score percentiles of the test rows of stocks that delist later."""
    parts = [
        part
        for arm in experiment.arms
        for seed in seeds
        for part in _arm_scores(run_key, arm, seed, folds[arm.name], windows)
    ]
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return frame.reindex(columns=SCORE_COLUMNS)


def _by_row(scores: pd.DataFrame) -> pd.DataFrame:
    """Percentiles averaged over seeds, one per arm, horizon and stock-day."""
    typed = scores.astype({"pct": "float64"})
    return typed.groupby(ROW_KEYS)["pct"].mean().reset_index()


def _sides(experiment: Experiment, rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    first, second = (arm.name for arm in experiment.arms)
    return rows[rows["arm"] == first], rows[rows["arm"] == second]


def _populations(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"near": rows[rows["near"].astype(bool)], "all": rows}


def delisting_summary(experiment: Experiment, scores: pd.DataFrame) -> pd.DataFrame:
    """Per population and horizon: both arms' mean percentile, the gap and the fold-gap NW t."""
    table = {}
    for population, rows in _populations(_by_row(scores)).items():
        for horizon in HORIZONS:
            table[(population, horizon)] = _side_summary(
                experiment, rows[rows["horizon"] == horizon]
            )
    return pd.DataFrame.from_dict(table, orient="index").reindex(columns=SUMMARY_COLUMNS)


def _side_summary(experiment: Experiment, rows: pd.DataFrame) -> dict[str, float | int]:
    first, second = _sides(experiment, rows)
    first_folds = first.groupby("fold")["pct"].mean()
    second_folds = second.groupby("fold")["pct"].mean()
    shared = first_folds.index.intersection(second_folds.index)
    low, high = float(first["pct"].mean()), float(second["pct"].mean())
    return {
        "first": low,
        "second": high,
        "gap": high - low,
        "nw_t": paired_nw_t(second_folds, first_folds, FOLD_LAG) if len(shared) else math.nan,
        "folds": len(shared),
        "stocks": first["stock_id"].nunique(),
        "rows": len(first),
    }


def masked_values(rows: pd.DataFrame, directory: Path, features: Sequence[str]) -> pd.DataFrame:
    """Each stock-day's share of the features that are missing, read a year at a time."""
    keys = rows[["stock_id", "date"]].drop_duplicates()
    parts = []
    for low, high in years(min(keys["date"]), max(keys["date"])):
        stored = read_columns(low, high, directory, features)
        wanted = keys[(keys["date"] >= low) & (keys["date"] <= high)]
        joined = wanted.merge(stored, on=["stock_id", "date"], how="left")
        parts.append(
            joined[["stock_id", "date"]].assign(
                missing_share=joined[list(features)].isna().mean(axis=1)
            )
        )
    return pd.concat(parts, ignore_index=True)


def missing_split(
    experiment: Experiment, scores: pd.DataFrame, shares: pd.DataFrame
) -> pd.DataFrame:
    """Both arms' mean percentile per population and horizon, rows mostly missing or not."""
    merged = _by_row(scores).merge(shares, on=["stock_id", "date"], how="left")
    merged = merged.assign(split=np.where(merged["missing_share"] >= MOSTLY_MISSING, MANY, FEW))
    table = {}
    for population, rows in _populations(merged).items():
        for horizon in HORIZONS:
            for split in (MANY, FEW):
                picked = rows[(rows["horizon"] == horizon) & (rows["split"] == split)]
                first, second = _sides(experiment, picked)
                low, high = float(first["pct"].mean()), float(second["pct"].mean())
                table[(population, horizon, split)] = {
                    "rows": len(first),
                    "first": low,
                    "second": high,
                    "gap": high - low,
                }
    return pd.DataFrame.from_dict(table, orient="index")


def masked_gain_share(
    experiment: Experiment, run_key: str, folds: Mapping[str, list[Fold]], seeds: Sequence[int]
) -> dict[int, float]:
    """The first arm's share of gain on the masked features per horizon, over folds and seeds."""
    plain, masked = experiment.arms
    hidden = set(masked.variant.masked)
    shares = {}
    for horizon in HORIZONS:
        values = []
        for seed in seeds:
            gains = fold_gains(arm_key(run_key, plain, seed), folds[plain.name], horizon)
            for gain in gains.values():
                total = sum(gain.values())
                part = sum(value for name, value in gain.items() if name in hidden)
                values.append(part / total if total > 0 else math.nan)
        shares[horizon] = float(np.nanmean(values)) if values else math.nan
    return shares


def _cell(value: float) -> str:
    return "—" if math.isnan(value) else f"{value:.4f}"


def _signed(value: float, digits: int = 4) -> str:
    return "—" if math.isnan(value) else f"{value:+.{digits}f}"


def _header(
    experiment: Experiment, run_key: str, daily: pd.DataFrame, kept: Sequence[str]
) -> list[str]:
    plain, masked = experiment.arms
    folds = sorted(daily["fold"].unique())
    seeds = ", ".join(map(str, sorted(daily["seed"].unique())))
    groups = [f"  - {name}：{'、'.join(names)}" for name, names in MASKED_GROUPS.items()]
    return [
        *(f"- {arm.name}：{arm_setting(arm)}" for arm in experiment.arms),
        f"- 遮掉下市股缺值偏高的 {len(MASKED_GROUPS)} 類特徵，共 {len(masked.variant.masked)} 個，"
        f"篩選後實際用到 {len(kept)} 個：",
        *groups,
        f"- 比較 {len(folds)} 個研究摺（摺 {folds[0]}–{folds[-1]}），seed {seeds}；"
        f"每組的快取 run_key 為 `{run_key}-<組名>`，seed 不是 0 時再加 `-s<seed>`；"
        f"{plain.name} 和 rank 窗長實驗同名，用同一個 run_key 時直接沿用它的快取",
        "- 這是診斷，不是採用判定：遮掉後之後會下市的股票分數明顯回升，代表模型在用這幾類特徵"
        "壓低它們的分數；靠的是缺值還是特徵值，看「依缺值拆分」。只用現存股票重算 IC 是倖存者樣本"
        "（SPEC §8 第 3 條），不做",
    ]


def _delisting_lines(experiment: Experiment, scores: pd.DataFrame) -> list[str]:
    plain, masked = (arm.name for arm in experiment.arms)
    table = delisting_summary(experiment, scores)
    lines = [
        "",
        "## 之後會下市的股票的分數",
        "",
        "- 每天的分數先在當天的 universe 列裡轉成百分位（越高越看好），各 seed 先逐列平均",
        f"- {POPULATIONS['near']}：從該檔下市前最後一個有報價的交易日往前數；"
        f"{POPULATIONS['all']}：這些股票在比較期內、最後報價日以前的全部測試列",
    ]
    if not table["rows"].any():
        lines.append("- 比較的測試期裡沒有之後會下市的列")
    lines += [
        "",
        f"| 母體 | horizon | {plain} | {masked} | 差 | 逐摺差 NW t | 摺數 | 檔數 | 列數 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    risen = []
    for key, row in table.iterrows():
        population, horizon = cast(tuple[str, int], key)
        label = f"{POPULATIONS[population]} {horizon} 日"
        lines.append(
            f"| {POPULATIONS[population]} | {horizon} 日 | {_cell(row['first'])} | "
            f"{_cell(row['second'])} | {_signed(row['gap'])} | {_signed(row['nw_t'], 2)} | "
            f"{row['folds']:.0f} | {row['stocks']:.0f} | {row['rows']:.0f} |"
        )
        if row["gap"] > 0 and row["nw_t"] > GATE_T:
            risen.append(label)
    lines.append("")
    if risen:
        lines.append(
            f"**判讀：⚠️ 遮掉後 {'、'.join(risen)}的分數回升，逐摺差 NW t 高於 {GATE_T:g}：模型在用"
            "這幾類特徵壓低之後會下市的股票的分數；真資料上的特徵重要度先不下結論**"
        )
    else:
        lines.append(f"**判讀：沒有看到遮掉後分數明顯回升（差 > 0 且逐摺差 NW t > {GATE_T:g}）**")
    return lines


def _split_lines(experiment: Experiment, split: pd.DataFrame | None) -> list[str]:
    lines = ["", "## 依缺值拆分", ""]
    if split is None:
        return [*lines, "- 篩選後沒有被遮掉的特徵，或沒有之後會下市的列：不拆"]
    plain, masked = (arm.name for arm in experiment.arms)
    lines += [
        f"- 「{MANY}」：篩選後被遮掉的特徵至少 {MOSTLY_MISSING:.0%} 缺值的列。回升集中在{MANY}，"
        f"代表模型靠的是缺值本身；兩群都回升，代表靠的是特徵值",
        "",
        f"| 母體 | horizon | 群 | 列數 | {plain} | {masked} | 差 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, row in split.iterrows():
        population, horizon, group = cast(tuple[str, int, str], key)
        lines.append(
            f"| {POPULATIONS[population]} | {horizon} 日 | {group} | {row['rows']:.0f} | "
            f"{_cell(row['first'])} | {_cell(row['second'])} | {_signed(row['gap'])} |"
        )
    return lines


def _gain_lines(experiment: Experiment, shares: Mapping[int, float]) -> list[str]:
    plain = experiment.arms[0].name
    lines = ["", "## 被遮掉特徵的重要度", "", f"| horizon | 占 {plain} 總 gain |", "| --- | --- |"]
    for horizon, share in shares.items():
        lines.append(f"| {horizon} 日 | {'—' if math.isnan(share) else f'{share:.1%}'} |")
    return lines


def mask_report(
    experiment: Experiment,
    run_key: str,
    daily: pd.DataFrame,
    scores: pd.DataFrame,
    split: pd.DataFrame | None,
    shares: Mapping[int, float],
    kept: Sequence[str],
    note: str | None = None,
) -> str:
    """Markdown of the IC, the soon-delisted scores, their missing split and the masked gain."""
    lines = [f"# 缺值遮蔽檢查（{run_key}）", ""]
    if note:
        lines += [f"> {note}", ""]
    lines += [
        *_header(experiment, run_key, daily, kept),
        "",
        "## IC",
        "",
        *ic_table_lines(summary(experiment, daily)),
        *_delisting_lines(experiment, scores),
        *_split_lines(experiment, split),
        *_gain_lines(experiment, shares),
    ]
    if daily["seed"].nunique() > 1:
        lines += seed_lines(experiment, daily)
    lines += fold_lines(experiment, daily)
    return "\n".join(lines) + "\n"
