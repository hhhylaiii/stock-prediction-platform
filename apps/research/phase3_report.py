"""Sections of the Phase 3 report that ic_report does not write (plan 3.8b)."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from apps.market.calendar import TradingDays
from apps.research.metrics.ic import daily_rank_ic

FLAG_RATIO = 3.0
FLAG_GAP = 0.20
ACCEPT_RATIO = 1.5
ACCEPT_GAP = 0.05
MONTHLY_TOP = 10
FUNDAMENTAL_PREFIX = "fund_"
TABLE_COLUMNS = ["current", "delisted", "rows_current", "rows_delisted", "ratio", "gap_pp", "flag"]


def _inside(rows: pd.DataFrame, delisted: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    inside = rows[rows["in_universe"].astype(bool)]
    fate = inside["stock_id"].map(delisted).eq(True)
    return inside, fate


def _split(counts: pd.DataFrame, fate: pd.Series, prefix: str) -> pd.DataFrame:
    sizes = fate.value_counts()
    return pd.DataFrame(
        {
            f"{prefix}_current": counts.get(False, 0),
            f"{prefix}_delisted": counts.get(True, 0),
            "rows_current": int(sizes.get(False, 0)),
            "rows_delisted": int(sizes.get(True, 0)),
        }
    ).astype("int64")


def missing_counts(rows: pd.DataFrame, delisted: pd.Series, names: Sequence[str]) -> pd.DataFrame:
    """Missing values and universe rows per feature for stocks later delisted and the rest."""
    inside, fate = _inside(rows, delisted)
    counts = inside[list(names)].isna().groupby(fate).sum().T
    return _split(counts, fate, "missing")


def _rates(counts: pd.DataFrame, prefix: str) -> tuple[pd.Series, pd.Series]:
    current = counts[f"{prefix}_current"] / counts["rows_current"].where(counts["rows_current"] > 0)
    delisted = counts[f"{prefix}_delisted"] / counts["rows_delisted"].where(
        counts["rows_delisted"] > 0
    )
    return current, delisted


def missing_table(counts: pd.DataFrame) -> pd.DataFrame:
    """Missing shares, their ratio and gap in points, flagged past FLAG_RATIO or FLAG_GAP."""
    current, delisted = _rates(counts, "missing")
    ratio = delisted / current.where(current > 0)
    gap = (delisted - current) * 100
    table = pd.DataFrame(
        {
            "current": current,
            "delisted": delisted,
            "rows_current": counts["rows_current"],
            "rows_delisted": counts["rows_delisted"],
            "ratio": ratio,
            "gap_pp": gap,
            "flag": (ratio > FLAG_RATIO) | (gap > FLAG_GAP * 100),
        }
    )
    return table[TABLE_COLUMNS]


def accepted(table: pd.DataFrame) -> pd.Series:
    """Features meeting the 11/07 bar: ratio at most ACCEPT_RATIO or gap at most ACCEPT_GAP."""
    return (table["ratio"] <= ACCEPT_RATIO) | (table["gap_pp"] <= ACCEPT_GAP * 100)


def fill_counts(
    rows: pd.DataFrame, delisted: pd.Series, fallbacks: Mapping[str, str]
) -> pd.DataFrame:
    """Universe rows whose value equals the fallback source, i.e. rows the fallback filled."""
    inside, fate = _inside(rows, delisted)
    filled = pd.DataFrame(
        {
            name: inside[name].notna() & inside[name].eq(inside[source])
            for name, source in fallbacks.items()
        },
        index=inside.index,
    )
    return _split(filled.groupby(fate).sum().T, fate, "filled")


def fill_table(counts: pd.DataFrame) -> pd.DataFrame:
    """Filled shares per group and their ratio, flagged past FLAG_RATIO."""
    current, delisted = _rates(counts, "filled")
    ratio = delisted / current.where(current > 0)
    return pd.DataFrame(
        {"current": current, "delisted": delisted, "ratio": ratio, "flag": ratio > FLAG_RATIO}
    )


def month_ranges(months: Iterable[str]) -> str:
    """Sorted YYYY-MM months joined into runs of consecutive months."""
    periods = sorted({pd.Period(month, "M") for month in months})
    if not periods:
        return "無"
    runs: list[list[pd.Period]] = [[periods[0]]]
    for period in periods[1:]:
        if period == runs[-1][-1] + 1:
            runs[-1].append(period)
        else:
            runs.append([period])
    return "、".join(str(run[0]) if len(run) == 1 else f"{run[0]}–{run[-1]}" for run in runs)


def monthly_ic(
    frame: pd.DataFrame, names: Sequence[str], label: str, calendar: TradingDays
) -> pd.DataFrame:
    """Monthly mean of each feature's daily rank IC, its IC days and the month's trading days."""
    first, last = min(frame["date"]), max(frame["date"])
    trading_days = pd.Series(pd.to_datetime(calendar.between(first.replace(day=1), last)))
    trading = trading_days.groupby(trading_days.dt.strftime("%Y-%m")).size()
    parts = []
    for name in names:
        ic = daily_rank_ic(frame, name, label).dropna()
        months = pd.to_datetime(pd.Series(ic.index, index=ic.index)).dt.strftime("%Y-%m")
        grouped = ic.groupby(months.to_numpy())
        part = pd.DataFrame({"ic": grouped.mean(), "days": grouped.size()}).reindex(trading.index)
        parts.append(
            part.assign(
                feature=name,
                month=trading.index,
                days=part["days"].fillna(0).astype("int64"),
                trading_days=trading.to_numpy(),
            )
        )
    table = pd.concat(parts, ignore_index=True)
    table["complete"] = table["days"] >= table["trading_days"]
    return table[["feature", "month", "ic", "days", "trading_days", "complete"]]


def strongest(ic: Mapping[str, Mapping[str, float | None]], count: int) -> list[str]:
    """Features ordered by their largest absolute IC over the horizons; none counts as 0."""
    strength = {
        name: max((abs(value) for value in by_horizon.values() if value is not None), default=0.0)
        for name, by_horizon in ic.items()
    }
    return sorted(strength, key=lambda name: -strength[name])[:count]


def statement_coverage(params: Mapping[str, Any]) -> pd.DataFrame:
    """IC days of each fundamental feature in the screening and whether the floor judged it."""
    window, floor = int(params["window_days"]), float(params["min_coverage"])
    rows = {
        name: {"days_h5": int(days["5"]), "days_h20": int(days["20"])}
        for name, days in params["days"].items()
        if name.startswith(FUNDAMENTAL_PREFIX)
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    coverage = table[["days_h5", "days_h20"]].min(axis=1) / window
    return table.assign(coverage=coverage, judged=coverage >= floor)


def _pct(value: float) -> str:
    return "—" if pd.isna(value) else f"{value * 100:.1f}%"


def _num(value: float, digits: int) -> str:
    return "—" if pd.isna(value) else f"{value:.{digits}f}"


def _yes(flag: bool) -> str:
    return "是" if flag else "否"


def _gap_rows(before: pd.DataFrame, after: pd.DataFrame) -> list[str]:
    names = [*after.index, *(name for name in before.index if name not in after.index)]
    passed = accepted(after)
    flagged = [
        name
        for name in names
        if bool(before["flag"].get(name, False))
        or bool(after["flag"].get(name, False))
        or not bool(passed.get(name, True))
    ]
    rows = []
    for name in flagged:
        old = before.loc[name] if name in before.index else None
        new = after.loc[name] if name in after.index else None
        was = "—" if old is None else f"{_pct(old['delisted'])}／{_pct(old['current'])}"
        now = "—" if new is None else f"{_pct(new['delisted'])}／{_pct(new['current'])}"
        tail = (
            "— | — | —"
            if new is None
            else f"{_num(new['ratio'], 2)} | {_num(new['gap_pp'], 1)} | {_yes(bool(passed[name]))}"
        )
        rows.append(f"| {name} | {was} | {now} | {tail} |")
    return rows


def _failed(after: pd.DataFrame) -> str:
    names = [str(name) for name, ok in accepted(after).items() if not ok]
    return f"補後未達驗收 {len(names)} 個：{'、'.join(names)}。" if names else "補後全部達到驗收。"


def _monthly_rows(monthly: pd.DataFrame, top: Sequence[str]) -> list[str]:
    lines = []
    for horizon, part in monthly.groupby("horizon", sort=True):
        cells = part.assign(
            cell=[
                f"{_num(ic, 4)}{'' if complete else '†'}"
                for ic, complete in zip(part["ic"], part["complete"], strict=True)
            ]
        ).pivot(index="month", columns="feature", values="cell")
        shown = [name for name in top if name in cells.columns]
        lines += [
            f"### {horizon} 日",
            "",
            "| 月份 | " + " | ".join(shown) + " |",
            "| --- | " + " | ".join("---" for _ in shown) + " |",
            *(
                f"| {month} | " + " | ".join(str(row[name]) for name in shown) + " |"
                for month, row in cells.iterrows()
            ),
            "",
        ]
    return lines


def render(
    header: Mapping[str, Any],
    coverage: pd.DataFrame,
    gaps: tuple[pd.DataFrame, pd.DataFrame, str, str],
    fills: tuple[pd.DataFrame, Mapping[str, str]],
    masked: Mapping[str, frozenset[str]],
    monthly: pd.DataFrame,
    top: Sequence[str],
) -> str:
    """Markdown of the data-gap, fallback and monthly IC sections."""
    before, after, start, end = gaps
    fill_rates, fallbacks = fills
    rows_delisted = int(after["rows_delisted"].max()) if len(after) else 0
    rows_current = int(after["rows_current"].max()) if len(after) else 0
    lines = [
        "# Phase 3 資料缺口與逐月單因子 IC（3.8b）",
        "",
        f"- 篩選列 `{header['run_id']}`：{header['start']} 到 {header['end']}，"
        f"{header['window_days']} 個交易日；保留 {header['kept']} 個，剔除 {header['dropped']} 個。"
        "單因子 IC 表見 `phase3-ic-report.md`。",
        f"- 缺值對照：{start} 到 {end} 的 universe 內列，依 `stock_master.delist_date` 分組；"
        f"之後下市 {rows_delisted:,} 列、現存 {rows_current:,} 列。",
        "",
        "## 財報類特徵的 IC 覆蓋率（K14、K16）",
        "",
        "財報公告日套用 K14（確定晚交的照實際日延後）與 K16（有重編的用最後一次重編的上傳日）；"
        "月營收查不到歷史公告日，研究期用法定截止日的下一個交易日（K2 保守值）。"
        "覆蓋率是兩個天期中較少的 IC 天數除以篩選期交易日數，低於門檻的不做低 IC 判定。",
        "",
        "| 特徵 | IC 天數（5／20 日） | 覆蓋率 | 低 IC 篩選 |",
        "| --- | --- | --- | --- |",
        *(
            f"| {name} | {row['days_h5']}／{row['days_h20']} | {_pct(row['coverage'])} "
            f"| {'判定' if row['judged'] else '不判定'} |"
            for name, row in coverage.iterrows()
        ),
        "",
        "## 缺值：之後下市 vs 現存（K17 補前、補後）",
        "",
        f"驗收：補後比值 ≤ {ACCEPT_RATIO:g}，或差距 ≤ {ACCEPT_GAP * 100:g} 個百分點。"
        f"{_failed(after)}",
        "",
        f"只列補前或補後超過列表門檻（比值 > {FLAG_RATIO:g}，"
        f"或差距 > {FLAG_GAP * 100:g} 個百分點），或補後未達驗收的特徵。完整表見附錄 CSV。",
        "",
        "| 特徵 | 補前 下市／現存 | 補後 下市／現存 | 補後比值 | 補後差距（百分點） "
        "| 補後達到驗收 |",
        "| --- | --- | --- | --- | --- | --- |",
        *_gap_rows(before, after),
        "",
        "## 殘餘缺口的處理",
        "",
        "C：主來源缺值時，對所有股票一律改用第二來源。補值比例的比值 > "
        f"{FLAG_RATIO:g} 表示「被補過」仍分得出下市與否。",
        "",
        "| 特徵 | 第二來源 | 補值比例 下市／現存 | 比值 | 超標 |",
        "| --- | --- | --- | --- | --- |",
        *(
            f"| {name} | {fallbacks[str(name)]} | {_pct(row['delisted'])}／{_pct(row['current'])} "
            f"| {_num(row['ratio'], 2)} | {_yes(bool(row['flag']))} |"
            for name, row in fill_rates.iterrows()
        ),
        "",
        "A：沒有第二來源，在受影響月份對所有股票遮成空值。",
        "",
        "| 特徵 | 遮罩月份 |",
        "| --- | --- |",
        *(f"| {name} | {month_ranges(months)} |" for name, months in masked.items()),
        "",
        f"## 逐月單因子 rank IC（|IC| 前 {len(top)} 名）",
        "",
        "† 表示 IC 有值的天數少於該月交易日數。全部特徵的逐月表見附錄 CSV。",
        "",
        *_monthly_rows(monthly, top),
    ]
    return "\n".join(lines)
