"""Daily check that live features are filled as often as their training window saw (SPEC §5)."""

from calendar import monthrange
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from django.db import transaction

from apps.etl.alerts import dispatch
from apps.etl.models import DataQualityLog
from apps.market.calendar import OutsideCalendarError, TradingDays
from apps.research.features.residual import ACCEPT_GAP, ACCEPT_RATIO, MASKED_MONTHS, SHARE_DIGITS
from apps.research.features.store import read_partitions
from apps.research.lgbm_runs import design_features
from apps.research.lock import locked
from apps.research.production import DESIGNS
from apps.research.runs import RunError

CHECK = "monitor_feature_coverage"
UNIVERSE = "universe"
FLAGGED = "flagged"
LOW_QUANTILE = 0.05
USUALLY_FILLED = 0.5
HALF = 0.5
MIN_REFERENCE_DAYS = 40
SHOWN_DIGITS = 6
MASKED_MONTH = "masked_month"
SHORT_REFERENCE = "short_reference"
EMPTY_REFERENCE = "empty_reference"
ZERO_LOW = "zero_low"
Severity = DataQualityLog.Severity


class CoverageError(ValueError):
    """The day's coverage cannot be checked; the message never names a host path."""


@dataclass(frozen=True)
class Verdict:
    flagged: list[dict[str, Any]]
    unjudged: dict[str, str]
    appeared: list[str]


def watched() -> tuple[list[str], int]:
    """Every feature a frozen design reads, and the years its longest training window spans."""
    names: dict[str, None] = {}
    months = 0
    for scope, design in DESIGNS.items():
        try:
            listed = design_features(scope, design.variant, design.target)
        except RunError as error:
            raise CoverageError(f"the {scope} design's features cannot be read: {error}") from error
        names.update(dict.fromkeys(listed))
        months = max(months, design.config.train_months)
    return list(names), months // 12


def universe_counts(
    start: date, end: date, directory: Path, features: Sequence[str]
) -> pd.DataFrame:
    """Per day of [start, end], one year: the universe's size and its rows holding each feature."""
    try:
        rows = read_partitions(start, end, directory, ["date", "in_universe", *features])
    except (FileNotFoundError, ValueError) as error:
        raise CoverageError(
            f"the feature partition of {start.year} is missing or lacks a watched feature; "
            "run build_features for it"
        ) from error
    grouped = rows[rows["in_universe"]].groupby("date")[list(features)]
    return pd.concat([grouped.size().rename(UNIVERSE), grouped.count()], axis=1)


def reference_counts(
    day: date, directory: Path, features: Sequence[str], years: int
) -> pd.DataFrame:
    """The day's calendar month in each of the years before it, counted as universe_counts does."""
    parts = [
        universe_counts(
            date(year, day.month, 1),
            date(year, day.month, monthrange(year, day.month)[1]),
            directory,
            features,
        )
        for year in range(day.year - years, day.year)
    ]
    return pd.concat(parts) if parts else pd.DataFrame(columns=[UNIVERSE, *features])


def flagged_days(days: Sequence[date]) -> dict[str, set[date]]:
    """Per feature, the days among these this check flagged; they stay out of a reference."""
    rows = DataQualityLog.objects.filter(check_name=CHECK, date__in=list(days))
    found: dict[str, set[date]] = {}
    for when, detail in rows.values_list("date", "detail"):
        for entry in detail.get(FLAGGED, []):
            found.setdefault(entry["feature"], set()).add(when)
    return found


def _bound(low: float) -> float:
    """Share under which a feature is flagged: half a sparse low, else residual's gap on missing."""
    if low < USUALLY_FILLED:
        return HALF * low
    return low - max(ACCEPT_GAP, (ACCEPT_RATIO - 1) * (1 - low))


def _under(share: float, limit: float) -> bool:
    return bool(round(share - limit, SHARE_DIGITS) < 0)


def _finding(name: str, count: int, share: float, values: pd.Series) -> dict[str, Any] | None:
    """The feature's flagged entry when its share is under the bound of these reference shares."""
    low = float(values.quantile(LOW_QUANTILE))
    bound = _bound(low)
    if not _under(share, bound):
        return None
    shown = {"share": share, "median": float(values.median()), "low": low, "bound": bound}
    return {
        "feature": name,
        "count": count,
        **{key: round(value, SHOWN_DIGITS) for key, value in shown.items()},
        "days": len(values),
        "collapsed": low >= USUALLY_FILLED and _under(share, HALF * low),
    }


def judge(
    today: pd.Series, reference: pd.DataFrame, month: str, left_out: Mapping[str, Set[date]]
) -> Verdict:
    """Each feature's share today against its reference days, less masked months and flagged days.

    A masked month of its own, under MIN_REFERENCE_DAYS or a low of zero leaves a feature unjudged.
    """
    shares = reference.drop(columns=UNIVERSE).div(reference[UNIVERSE], axis=0)
    months = pd.Series([f"{day:%Y-%m}" for day in reference.index], index=reference.index)
    flagged: list[dict[str, Any]] = []
    unjudged: dict[str, str] = {}
    appeared: list[str] = []
    for name in shares.columns:
        masked = MASKED_MONTHS.get(name, frozenset())
        if month in masked:
            unjudged[name] = MASKED_MONTH
            continue
        kept = ~months.isin(list(masked)) & ~reference.index.isin(list(left_out.get(name, ())))
        values = shares.loc[kept, name].astype(float)
        count = int(today[name])
        if len(values) < MIN_REFERENCE_DAYS:
            unjudged[name] = SHORT_REFERENCE
        elif values.quantile(LOW_QUANTILE) <= 0:
            empty = values.max() <= 0
            unjudged[name] = EMPTY_REFERENCE if empty else ZERO_LOW
            appeared.extend([name] if empty and count else [])
        elif (found := _finding(name, count, count / int(today[UNIVERSE]), values)) is not None:
            flagged.append(found)
    return Verdict(flagged, unjudged, appeared)


def _severity(verdict: Verdict) -> str:
    if any(entry["collapsed"] for entry in verdict.flagged):
        return str(Severity.ERROR)
    return str(Severity.WARNING if verdict.flagged else Severity.INFO)


def coverage_row(day: date, directory: Path) -> DataQualityLog:
    """The day's unsaved row: error on a collapsed feature, warning on a flagged one, else info."""
    features, years = watched()
    today = universe_counts(day, day, directory, features)
    if today.empty:
        raise CoverageError(f"no universe feature rows on {day}; run build_features for it")
    reference = reference_counts(day, directory, features, years)
    left_out = flagged_days(list(reference.index))
    verdict = judge(today.iloc[0], reference, f"{day:%Y-%m}", left_out)
    detail = {
        UNIVERSE: int(today.iloc[0][UNIVERSE]),
        "features": len(features),
        "judged": len(features) - len(verdict.unjudged),
        FLAGGED: verdict.flagged,
        "unjudged": verdict.unjudged,
        "appeared": verdict.appeared,
        "reference": {"month": day.month, "years": years, "days": len(reference)},
    }
    return DataQualityLog(date=day, check_name=CHECK, severity=_severity(verdict), detail=detail)


def _names(row: DataQualityLog | None) -> set[str]:
    return set() if row is None else {entry["feature"] for entry in row.detail.get(FLAGGED, [])}


def _flagged_before(day: date, calendar: TradingDays) -> set[str]:
    """The features the session before day had flagged; none without that session or its row."""
    try:
        previous = calendar.shift(day, -1)
    except OutsideCalendarError:
        return set()
    rows = DataQualityLog.objects.filter(date=previous, check_name=CHECK)
    return _names(rows.order_by("-pk").first())


def coverage_summary(row: DataQualityLog) -> str:
    """One line on the row: how many features were judged, which were flagged or collapsed."""
    detail = row.detail
    names = ", ".join(entry["feature"] for entry in detail[FLAGGED])
    collapsed = ", ".join(entry["feature"] for entry in detail[FLAGGED] if entry["collapsed"])
    judged = f"{detail['judged']} of {detail['features']} features judged"
    shown = f"flagged {names}" if names else "none flagged"
    stopped = f"; {collapsed} collapsed, so the day is not predicted" if collapsed else ""
    return f"{judged}, {shown}{stopped}"


@locked("coverage")
def run_coverage(
    day: date, calendar: TradingDays, directory: Path, alert: bool = True
) -> DataQualityLog:
    """Write the day's row over its earlier one; an error always alerts, a warning on a new list.

    An error row shuts the day's prediction gate, which reads data_quality_log.
    """
    if not calendar.is_trading_day(day):
        raise CoverageError(f"{day} is not a trading day")
    row = coverage_row(day, directory)
    with transaction.atomic():
        DataQualityLog.objects.filter(date=day, check_name=CHECK).delete()
        row.save()
    changed = _names(row) != _flagged_before(day, calendar)
    if alert and (row.severity == Severity.ERROR or (row.severity == Severity.WARNING and changed)):
        dispatch([row])
    return row
