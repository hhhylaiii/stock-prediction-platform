"""Daily check that live features are filled as often as their training window saw (SPEC §5)."""

from calendar import monthrange
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
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
PRICE, INDEX, FLOWS, MARGIN, SBL = "daily_price", "index_daily", "institutional", "margin", "sbl"
HOLDING, VALUATION, EX_DATES = "foreign_holding", "daily_valuation", "dividend_event"
SHARES, REVENUE, STATEMENTS = "shares_outstanding", "monthly_revenue", "financial_statement"
MASTER, DAY_TRADE, WEEKLY, ATTENTION = "stock_master", "day_trade", "weekly_holding", "attention"
CARRIED = frozenset({SHARES, REVENUE, STATEMENTS, MASTER})
GAPPED = frozenset({DAY_TRADE, WEEKLY, ATTENTION})
Severity = DataQualityLog.Severity


def _reading(names: str, *datasets: str) -> dict[str, tuple[str, ...]]:
    return dict.fromkeys(names.split(), datasets)


SOURCES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        **_reading("tech_ret_1d tech_ret_5d tech_ret_20d tech_ret_60d tech_mom_250_20", PRICE),
        **_reading("tech_ma_gap_5d tech_ma_gap_20d tech_ma_gap_60d", PRICE),
        **_reading("tech_vol_20d tech_vol_60d tech_volume_ratio_5_20", PRICE),
        **_reading("tech_rsi_14 tech_kd_k tech_kd_d tech_macd_hist_pct", PRICE),
        **_reading("tech_high_52w_gap tech_low_52w_gap", PRICE),
        **_reading("tech_limit_up_t tech_limit_up_cnt_20d tech_limit_down_cnt_20d", PRICE),
        **_reading("tech_parkinson_vol_20d tech_gap_1d tech_close_loc_5d", PRICE),
        **_reading("tech_max_ret_20d tech_skew_60d", PRICE),
        **_reading("liq_dollar_vol_med_20d liq_amihud_20d", PRICE),
        **_reading("tech_turnover_rate_5d size_log_mcap", PRICE, SHARES),
        **_reading("tech_attention_t", ATTENTION),
        **_reading("chip_fi_net_1d chip_fi_net_5d chip_fi_net_20d", FLOWS, SHARES),
        **_reading("chip_it_net_5d chip_it_net_20d", FLOWS, SHARES),
        **_reading("chip_dealer_self_net_5d chip_dealer_hedge_net_5d", FLOWS, SHARES),
        **_reading("chip_fi_buy_streak", FLOWS),
        **_reading("chip_margin_chg_5d", MARGIN, SHARES),
        **_reading("chip_margin_usage chip_short_ratio", MARGIN),
        **_reading("chip_sbl_short_chg_5d", SBL, SHARES),
        **_reading("chip_fi_holding_pct chip_fi_holding_chg_20d", HOLDING),
        **_reading("chip_big_holder_pct_chg_1w chip_holder_count_chg_1w", WEEKLY),
        **_reading("chip_day_trade_ratio_5d", DAY_TRADE, PRICE),
        **_reading("fund_rev_yoy fund_rev_3m_yoy", REVENUE, STATEMENTS),
        **_reading("fund_rev_yoy_q fund_eps_yoy fund_roe_ttm", STATEMENTS),
        **_reading("fund_gross_margin fund_op_margin fund_debt_ratio", STATEMENTS),
        **_reading("fund_gm_yoy_chg fund_opm_yoy_chg fund_cfo_to_ni", STATEMENTS),
        **_reading("fund_days_since_fs", STATEMENTS),
        **_reading("fund_ep", STATEMENTS, PRICE),
        **_reading("fund_bp fund_dividend_yield", VALUATION),
        **_reading("fund_days_to_ex_date", EX_DATES),
        **_reading("risk_beta_250d risk_idio_vol_60d", PRICE, INDEX),
        **_reading("ind_sector_ret_20d ind_rel_ret_20d ind_mom_neutral", PRICE, MASTER),
        **_reading("ind_ep_neutral", STATEMENTS, PRICE, MASTER),
    }
)


class CoverageError(ValueError):
    """The day's coverage cannot be checked; the message never names a host path."""


@dataclass(frozen=True)
class Verdict:
    flagged: list[dict[str, Any]]
    unjudged: dict[str, str]
    appeared: list[str]


def stops_predictions(name: str) -> bool:
    """Whether the feature's collapse is an error, which shuts the day's predictions.

    It only warns when a source is outside the research gate and not carried forward, so one
    missed day leaves the feature empty: GAPPED. Classify a new feature in SOURCES by that rule.
    """
    return not GAPPED.intersection(SOURCES[name])


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
    unlisted = [name for name in names if name not in SOURCES]
    if unlisted:
        raise CoverageError(f"{', '.join(unlisted)} have no sources listed in SOURCES")
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
    halved = low >= USUALLY_FILLED and _under(share, HALF * low)
    return {
        "feature": name,
        "count": count,
        **{key: round(value, SHOWN_DIGITS) for key, value in shown.items()},
        "days": len(values),
        "collapsed": halved and stops_predictions(name),
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
