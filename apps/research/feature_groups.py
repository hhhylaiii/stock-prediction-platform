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
from apps.research.models import ModelRun
from apps.research.runs import SCREENING_MODEL, feature_hash
from apps.research.screening import Screening, screen, screening_window, verdict
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
BLOCKED = {"ind_sector_id": "受阻：要等時點正確（PIT）的產業資料，這次不評估"}
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
    config = WalkForwardConfig()
    if last <= config.tuning_end:
        raise ValueError(
            f"--last-date {last} falls inside the tuning period ending {config.tuning_end}; "
            "the screening window would be cut short"
        )
    return screening_window(generate_folds(calendar, last, config))


def check_recorded_screening(directory: Path) -> ModelRun:
    """The recorded screening, refused when the partitions it read have changed since."""
    run = ModelRun.objects.filter(model_type=SCREENING_MODEL).first()
    if run is None:
        raise ValueError("no feature screening is recorded; run ic_report first")
    if feature_hash(run.train_start, run.train_end, directory) != run.feature_hash:
        raise ValueError("features changed since the recorded screening; run ic_report again")
    return run


def check_same_ics(result: Screening, run: ModelRun) -> None:
    """Refuse a rescreening whose first-batch ICs differ from the recorded screening's."""
    recorded = run.params.get("ic", {})
    for horizon in HORIZONS:
        now = result.table[f"ic_h{horizon}"].to_dict()
        for name in (name for name in STOCK_LEVEL if name in now):
            before = recorded.get(name, {}).get(str(horizon))
            value = float(now[name])
            same = (
                math.isnan(value) if before is None else math.isclose(value, before, abs_tol=1e-9)
            )
            if not same:
                raise ValueError(
                    f"{name}'s {horizon}-day IC {value:.6f} differs from the recorded screening's "
                    f"{before}; labels or the screening window changed, run ic_report again"
                )


def group_screening(
    names: Sequence[str], directory: Path, calendar: TradingDays, last: date
) -> Screening:
    """SPEC §6 step 7 rerun over names on the screening window, with Phase 3's input checks."""
    start, end = screening_span(calendar, last)
    digest = feature_hash(start, end, directory)
    features = read_partitions(start, end, directory, columns_for(names))
    if feature_hash(start, end, directory) != digest:
        raise RuntimeError("feature partitions changed while reading; run feature_group again")
    expected = calendar.between(start, end)
    missing = sorted(set(expected) - set(features["date"]))
    if missing:
        raise ValueError(
            f"features are missing on {len(missing)} of {len(expected)} trading days "
            f"from {missing[0]}; rebuild the feature partitions over the screening window"
        )
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
    reasons, covered = table["reason"].to_dict(), table["covered"].to_dict()
    for name in FEATURE_GROUPS[group]:
        cells = " | ".join(_ic(float(ics[horizon][name])) for horizon in HORIZONS)
        lines.append(f"| `{name}` | {cells} | {verdict(reasons[name], bool(covered[name]))} |")
    displaced = [name for name in kept_before if name in result.dropped]
    lines.append("")
    if displaced:
        listed = "；".join(f"`{name}`：{result.dropped[name]}" for name in displaced)
        lines.append(f"- 加入這組後被剔除的原有特徵：{listed}")
    else:
        lines.append("- 原有特徵沒有因為這組被剔除")
    grouped = set(FEATURE_GROUPS[group])
    restored = [name for name in result.kept if name not in kept_before and name not in grouped]
    if restored:
        lines.append(
            f"- 加入這組後重新保留的原有特徵：{'、'.join(f'`{name}`' for name in restored)}"
        )
    return [*lines, ""]


def blocked_lines() -> list[str]:
    """Markdown of the second-batch features this evaluation leaves out, with why."""
    return [
        "## 受阻、沒評估的特徵",
        "",
        *(f"- `{name}`：{why}" for name, why in BLOCKED.items()),
        "",
    ]


def _ic(value: float) -> str:
    return "—" if math.isnan(value) else f"{value:+.4f}"
