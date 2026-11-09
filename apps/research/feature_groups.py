"""Second-batch feature groups added one at a time to the rank design (SPEC §6, PR 4.8b-2)."""

import math
from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path

from apps.market.calendar import TradingDays
from apps.research.experiments import Arm, Experiment, experiments, window_arms
from apps.research.features.store import columns_for, read_partitions
from apps.research.features.transforms import STOCK_LEVEL
from apps.research.labels import HORIZONS
from apps.research.screening import Screening, screen, screening_window
from apps.research.training import load_labels
from apps.research.walkforward import WalkForwardConfig, generate_folds

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "tech": (
        "tech_parkinson_vol_20d",
        "tech_gap_1d",
        "tech_close_loc_5d",
        "tech_max_ret_20d",
        "tech_skew_60d",
    ),
    "chip": (
        "chip_day_trade_ratio_5d",
        "chip_fi_holding_chg_20d",
        "chip_it_net_20d",
        "chip_fi_buy_streak",
    ),
    "ind": ("ind_mom_neutral", "ind_ep_neutral"),
    "exdt": ("fund_days_to_ex_date",),
}
BLOCKED = {"ind_sector_id": "要等時點正確（PIT）的產業資料，這次不評估"}
BASES = {"plain": "window", "floor": "stopping", "every5": "sampling"}


def group_names(accepted: Sequence[str], group: str | None = None) -> list[str]:
    """The first-batch stock-level features, then the accepted groups, then the one under test."""
    tags = [*accepted, *([group] if group else [])]
    unknown = [tag for tag in tags if tag not in FEATURE_GROUPS]
    if unknown or len(set(tags)) < len(tags):
        raise ValueError(f"groups must be distinct names from {list(FEATURE_GROUPS)}: {tags}")
    return [*STOCK_LEVEL, *(name for tag in tags for name in FEATURE_GROUPS[tag])]


def screening_span(calendar: TradingDays, last: date) -> tuple[date, date]:
    """The SPEC tuning folds' training span the Phase 3 screening reads, kept for every window."""
    return screening_window(generate_folds(calendar, last, WalkForwardConfig()))


def group_screening(
    names: Sequence[str], directory: Path, calendar: TradingDays, last: date
) -> Screening:
    """SPEC §6 step 7 rerun over names on the screening window."""
    start, end = screening_span(calendar, last)
    features = read_partitions(start, end, directory, columns_for(names))
    labels = {horizon: load_labels(start, end, horizon) for horizon in HORIZONS}
    return screen(features, labels, names)


def base_arm(base: WalkForwardConfig, window: str, start: str) -> Arm:
    """The design the groups build on: the window arm, or its floor or every-5 sibling."""
    if start not in BASES:
        raise ValueError(f"base must be one of {list(BASES)}, not {start!r}")
    if start == "plain":
        return window_arms(base)[window]
    chosen = experiments(base, window)[BASES[start]]
    return next(arm for arm in chosen.arms if arm.name.endswith(start))


def with_features(arm: Arm, tags: Sequence[str], features: Sequence[str]) -> Arm:
    """The arm refit on a screened feature list, named after the groups it carries."""
    return Arm(
        "-".join([arm.name, *tags]),
        arm.config,
        replace(arm.variant, features=tuple(features)),
        target=arm.target,
    )


def group_experiment(
    start: Arm,
    accepted: Sequence[str],
    accepted_kept: Sequence[str] | None,
    group: str,
    group_kept: Sequence[str],
) -> Experiment:
    """The design with the accepted groups against it with the next group; better wins."""
    first = start if not accepted else with_features(start, accepted, accepted_kept or ())
    return Experiment(
        f"group-{group}", "better", (first, with_features(start, [*accepted, group], group_kept))
    )


def screening_lines(result: Screening, group: str, kept_before: Sequence[str]) -> list[str]:
    """Markdown of the group's screening verdicts and the first-batch features it displaced."""
    table = result.table
    lines = ["## 這組的篩選", ""]
    lines += [
        "- 篩選窗沿用 Phase 3 的調參摺訓練窗（36 個月窗下 2014–2015 年的 8 個季摺），"
        "規則同 SPEC §6 第 7 步",
        "",
        "| 特徵 | 5 日 IC | 20 日 IC | 結果 |",
        "| --- | --- | --- | --- |",
    ]
    ics = {horizon: table[f"ic_h{horizon}"].to_dict() for horizon in HORIZONS}
    reasons = table["reason"].to_dict()
    for name in FEATURE_GROUPS[group]:
        cells = " | ".join(_ic(float(ics[horizon][name])) for horizon in HORIZONS)
        lines.append(f"| `{name}` | {cells} | {reasons[name] or '保留'} |")
    displaced = [name for name in kept_before if name in result.dropped]
    lines.append("")
    if displaced:
        listed = "；".join(f"`{name}`：{result.dropped[name]}" for name in displaced)
        lines.append(f"- 加入這組後被剔除的原有特徵：{listed}")
    else:
        lines.append("- 原有特徵沒有因為這組被剔除")
    return [*lines, ""]


def _ic(value: float) -> str:
    return "—" if math.isnan(value) else f"{value:+.4f}"
