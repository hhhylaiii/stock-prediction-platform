from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import time_machine
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.api.ops import MODEL_CHECK_PREFIXES
from apps.etl.models import DataQualityLog
from apps.etl.quality import CHECKS, should_run_prediction
from apps.etl.tasks import GATE_DATASETS
from apps.market.calendar import TradingDays
from apps.market.models import TradingCalendar
from apps.research import feature_coverage, production
from apps.research.feature_coverage import (
    CARRIED,
    CHECK,
    GAPPED,
    SOURCES,
    CoverageError,
    coverage_row,
    coverage_summary,
    run_coverage,
    stops_predictions,
    universe_counts,
    watched,
)
from apps.research.features.store import partition_path
from apps.research.features.transforms import STOCK_LEVEL
from apps.research.target_models import MARKET_LEVEL

pytestmark = pytest.mark.django_db
Severity = DataQualityLog.Severity
DENSE, USUAL, SPARSE = "tech_ret_5d", "fund_eps_yoy", "fund_days_to_ex_date"
FEATURES = [DENSE, USUAL, SPARSE]
USUAL_SHARES = {DENSE: 1.0, USUAL: 0.78, SPARSE: 0.06}
UNIVERSE, OUTSIDE = 200, 20
THURSDAY, DAY, MONDAY = date(2026, 11, 5), date(2026, 11, 6), date(2026, 11, 9)
REFERENCE_YEARS = range(2021, 2026)
NOVEMBER_DAYS = 107
Shares = Mapping[str, float]


def weekdays(start: date, end: date) -> list[date]:
    span = (start + timedelta(days=n) for n in range((end - start).days + 1))
    return [day for day in span if day.weekday() < 5]


def november(year: int) -> list[date]:
    return weekdays(date(year, 11, 1), date(year, 11, 30))


def rows(days: Sequence[date], shares: Shares, on: Mapping[date, Shares] | None = None) -> Any:
    """Universe stocks of which the first share hold each feature; the stocks outside hold all."""
    stocks = [f"{n:04d}" for n in range(1, UNIVERSE + OUTSIDE + 1)]
    inside = np.arange(len(stocks)) < UNIVERSE
    parts = []
    for day in days:
        held = {**shares, **(on or {}).get(day, {})}
        values = {
            name: np.where(
                ~inside | (np.arange(len(stocks)) < round(share * UNIVERSE)), 1.0, np.nan
            )
            for name, share in held.items()
        }
        parts.append(
            pd.DataFrame({"stock_id": stocks, "date": day, "in_universe": inside, **values})
        )
    return pd.concat(parts, ignore_index=True)


def write(directory: Path, year: int, frame: Any) -> None:
    path = partition_path(directory, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)


def history(
    directory: Path,
    shares: Shares = USUAL_SHARES,
    years: Mapping[int, Shares] | None = None,
    on: Mapping[date, Shares] | None = None,
) -> None:
    """November of each reference year at the usual shares, a year or a day overriding them."""
    for year in REFERENCE_YEARS:
        write(directory, year, rows(november(year), {**shares, **(years or {}).get(year, {})}, on))


def live(directory: Path, shares: Shares, days: Sequence[date] = (DAY,)) -> None:
    write(directory, 2026, rows(days, {**USUAL_SHARES, **shares}))


def flagged(row: DataQualityLog) -> dict[str, dict[str, Any]]:
    return {entry["feature"]: entry for entry in row.detail["flagged"]}


@pytest.fixture(autouse=True)
def three_features(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three watched features over five years, and no masked month unless a test names one."""
    monkeypatch.setattr(feature_coverage, "watched", lambda: (FEATURES, 5))
    monkeypatch.setattr(feature_coverage, "MASKED_MONTHS", {})


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[tuple[date, str]]:
    calls: list[tuple[date, str]] = []
    monkeypatch.setattr(
        feature_coverage,
        "dispatch",
        lambda found: calls.extend((row.date, row.severity) for row in found),
    )
    return calls


@pytest.fixture
def calendar() -> TradingDays:
    start, end = date(2026, 10, 1), date(2026, 11, 30)
    return TradingDays(weekdays(start, end), start, end)


def test_the_check_is_listed_with_the_model_checks() -> None:
    assert CHECK == "monitor_feature_coverage"
    assert CHECK.startswith(MODEL_CHECK_PREFIXES)


def test_the_watched_features_are_what_the_frozen_designs_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lists = {
        "rank": [DENSE, USUAL, SPARSE],
        "return": [DENSE, "size_log_mcap"],
        "interval": ["size_log_mcap", DENSE],
    }
    monkeypatch.setattr(
        feature_coverage, "design_features", lambda scope, variant, target: lists[scope]
    )

    assert watched() == ([DENSE, USUAL, SPARSE, "size_log_mcap"], 5)

    longer = replace(production.DESIGNS["return"].config, train_months=84)
    designs = {**production.DESIGNS, "return": replace(production.DESIGNS["return"], config=longer)}
    monkeypatch.setattr(feature_coverage, "DESIGNS", designs)
    assert watched() == ([DENSE, USUAL, SPARSE, "size_log_mcap"], 7)


def test_a_design_feature_without_listed_sources_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        feature_coverage, "design_features", lambda scope, variant, target: [DENSE, "tech_new_1d"]
    )

    with pytest.raises(CoverageError, match="tech_new_1d have no sources listed in SOURCES"):
        watched()


def test_every_feature_a_frozen_design_can_read_has_its_sources_listed() -> None:
    readable = set(STOCK_LEVEL)
    for scope, design in production.DESIGNS.items():
        readable |= set(design.variant.features) | set(design.target.extra)
        if scope != "rank" and design.target.market:
            readable |= set(MARKET_LEVEL)

    assert readable <= set(SOURCES)
    assert len(production.DESIGNS["rank"].variant.features) == 60


def test_each_source_holds_the_gate_is_carried_forward_or_leaves_a_gap() -> None:
    read = {dataset for datasets in SOURCES.values() for dataset in datasets}

    assert read <= GATE_DATASETS | CARRIED | GAPPED
    assert not (CARRIED | GAPPED) & GATE_DATASETS
    assert not CARRIED & GAPPED
    assert all(SOURCES.values())


def test_only_features_a_missed_day_empties_are_spared_the_error() -> None:
    spared = {name for name in SOURCES if not stops_predictions(name)}

    assert spared == {
        "chip_day_trade_ratio_5d",
        "chip_big_holder_pct_chg_1w",
        "chip_holder_count_chg_1w",
        "tech_attention_t",
    }
    assert sorted(GAPPED) == ["attention", "day_trade", "weekly_holding"]
    assert sorted(CARRIED) == [
        "financial_statement",
        "monthly_revenue",
        "shares_outstanding",
        "stock_master",
    ]


def test_designs_whose_features_cannot_be_read_are_refused() -> None:
    with pytest.raises(CoverageError, match="design's features cannot be read") as refused:
        watched()

    assert "no feature screening is recorded" in str(refused.value)


def test_only_universe_rows_are_counted(tmp_path: Path) -> None:
    live(tmp_path, {SPARSE: 0.02}, days=(THURSDAY, DAY))

    counts = universe_counts(THURSDAY, DAY, tmp_path, FEATURES)

    assert list(counts.index) == [THURSDAY, DAY]
    assert counts.loc[DAY].to_dict() == {"universe": 200, DENSE: 200, USUAL: 156, SPARSE: 4}


def test_a_day_inside_its_reference_writes_an_info_row(tmp_path: Path) -> None:
    history(tmp_path)
    live(tmp_path, {})

    row = coverage_row(DAY, tmp_path)

    assert (row.date, row.check_name, row.severity) == (DAY, CHECK, Severity.INFO)
    assert row.detail == {
        "universe": 200,
        "features": 3,
        "judged": 3,
        "flagged": [],
        "unjudged": {},
        "appeared": [],
        "reference": {"month": 11, "years": 5, "days": NOVEMBER_DAYS},
    }
    assert coverage_summary(row) == "3 of 3 features judged, none flagged"


@pytest.mark.parametrize(
    ("feature", "kept", "lost", "bound"),
    [(DENSE, 0.95, 0.945, 0.95), (USUAL, 0.67, 0.665, 0.67), (SPARSE, 0.03, 0.025, 0.03)],
)
def test_a_feature_is_flagged_only_under_its_bound(
    tmp_path: Path, feature: str, kept: float, lost: float, bound: float
) -> None:
    history(tmp_path)
    live(tmp_path, {feature: kept}, days=(THURSDAY,))
    assert coverage_row(THURSDAY, tmp_path).severity == Severity.INFO

    live(tmp_path, {feature: lost})
    row = coverage_row(DAY, tmp_path)

    assert row.severity == Severity.WARNING
    assert flagged(row) == {
        feature: {
            "feature": feature,
            "share": lost,
            "count": round(lost * UNIVERSE),
            "median": USUAL_SHARES[feature],
            "low": USUAL_SHARES[feature],
            "bound": bound,
            "days": NOVEMBER_DAYS,
            "collapsed": False,
        }
    }
    assert coverage_summary(row) == f"3 of 3 features judged, flagged {feature}"


@pytest.mark.parametrize(("feature", "held"), [(DENSE, 0.5), (USUAL, 0.39)])
def test_half_the_low_of_a_usually_filled_feature_is_still_a_warning(
    tmp_path: Path, feature: str, held: float
) -> None:
    history(tmp_path)
    live(tmp_path, {feature: held})

    row = coverage_row(DAY, tmp_path)

    assert row.severity == Severity.WARNING
    assert flagged(row)[feature]["collapsed"] is False


@pytest.mark.parametrize(("feature", "held"), [(DENSE, 0.495), (USUAL, 0.385)])
def test_a_usually_filled_feature_under_half_its_low_is_an_error(
    tmp_path: Path, feature: str, held: float
) -> None:
    history(tmp_path)
    live(tmp_path, {feature: held, SPARSE: 0.0})

    row = coverage_row(DAY, tmp_path)

    assert row.severity == Severity.ERROR
    assert {name: entry["collapsed"] for name, entry in flagged(row).items()} == {
        feature: True,
        SPARSE: False,
    }
    assert coverage_summary(row) == (
        f"3 of 3 features judged, flagged {feature}, {SPARSE}; "
        f"{feature} collapsed, so the day is not predicted"
    )


def test_a_collapse_only_warns_for_a_feature_one_missed_day_would_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gapped, carried = "chip_day_trade_ratio_5d", "size_log_mcap"
    monkeypatch.setattr(feature_coverage, "watched", lambda: ([gapped, carried], 5))
    for year in REFERENCE_YEARS:
        write(tmp_path, year, rows(november(year), {gapped: 1.0, carried: 1.0}))

    write(tmp_path, 2026, rows([THURSDAY], {gapped: 0.0, carried: 1.0}))
    spared = coverage_row(THURSDAY, tmp_path)
    write(tmp_path, 2026, rows([DAY], {gapped: 0.0, carried: 0.0}))
    stopped = coverage_row(DAY, tmp_path)

    assert spared.severity == Severity.WARNING
    assert flagged(spared)[gapped]["collapsed"] is False
    assert stopped.severity == Severity.ERROR
    assert {name: entry["collapsed"] for name, entry in flagged(stopped).items()} == {
        gapped: False,
        carried: True,
    }


def test_an_empty_sparse_feature_is_a_warning_never_an_error(tmp_path: Path) -> None:
    history(tmp_path)
    live(tmp_path, {SPARSE: 0.0})

    row = coverage_row(DAY, tmp_path)

    assert row.severity == Severity.WARNING
    assert flagged(row)[SPARSE] == {
        "feature": SPARSE,
        "share": 0.0,
        "count": 0,
        "median": 0.06,
        "low": 0.06,
        "bound": 0.03,
        "days": NOVEMBER_DAYS,
        "collapsed": False,
    }


def test_the_low_is_the_fifth_percentile_so_a_few_empty_days_do_not_blind_it(
    tmp_path: Path,
) -> None:
    empty = {day: {SPARSE: 0.0} for day in november(2024)[:3]}
    history(tmp_path, on=empty)
    live(tmp_path, {SPARSE: 0.0})

    entry = flagged(coverage_row(DAY, tmp_path))[SPARSE]

    assert (entry["low"], entry["bound"], entry["days"]) == (0.06, 0.03, NOVEMBER_DAYS)


def test_a_month_that_is_often_empty_is_not_judged(tmp_path: Path) -> None:
    history(tmp_path, years={2022: {SPARSE: 0.0}, 2023: {SPARSE: 0.0}})
    live(tmp_path, {SPARSE: 0.0})

    row = coverage_row(DAY, tmp_path)

    assert row.severity == Severity.INFO
    assert row.detail["unjudged"] == {SPARSE: "zero_low"}
    assert (row.detail["judged"], row.detail["appeared"]) == (2, [])


def test_masked_months_stay_out_of_the_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history(tmp_path, years={2022: {SPARSE: 0.0}})
    live(tmp_path, {SPARSE: 0.0})
    assert coverage_row(DAY, tmp_path).detail["unjudged"] == {SPARSE: "zero_low"}

    monkeypatch.setattr(feature_coverage, "MASKED_MONTHS", {SPARSE: frozenset({"2022-11"})})
    row = coverage_row(DAY, tmp_path)

    assert row.severity == Severity.WARNING
    entry = flagged(row)[SPARSE]
    assert (entry["low"], entry["days"]) == (0.06, NOVEMBER_DAYS - len(november(2022)))


def test_a_feature_masked_in_the_day_s_own_month_is_not_judged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history(tmp_path)
    live(tmp_path, {SPARSE: 0.0, DENSE: 0.0})
    masked = {SPARSE: frozenset({"2026-11"}), DENSE: frozenset({"2026-11"})}
    monkeypatch.setattr(feature_coverage, "MASKED_MONTHS", masked)

    row = coverage_row(DAY, tmp_path)

    assert row.severity == Severity.INFO
    assert row.detail["unjudged"] == {DENSE: "masked_month", SPARSE: "masked_month"}


def test_days_this_check_flagged_stay_out_of_the_reference(tmp_path: Path) -> None:
    history(tmp_path, years={2025: {SPARSE: 0.0, DENSE: 0.4}})
    live(tmp_path, {SPARSE: 0.0, DENSE: 0.3})
    assert coverage_row(DAY, tmp_path).detail["unjudged"] == {SPARSE: "zero_low"}

    DataQualityLog.objects.bulk_create(
        DataQualityLog(
            date=day,
            check_name=CHECK,
            severity=Severity.WARNING,
            detail={"flagged": [{"feature": SPARSE}]},
        )
        for day in november(2025)
    )
    DataQualityLog.objects.create(
        date=november(2024)[0], check_name=CHECK, severity=Severity.WARNING, detail={"error": "x"}
    )
    row = coverage_row(DAY, tmp_path)

    entry = flagged(row)[SPARSE]
    assert (entry["low"], entry["days"]) == (0.06, NOVEMBER_DAYS - len(november(2025)))
    assert row.detail["unjudged"] == {}
    assert DENSE not in flagged(row)


def test_a_reference_shorter_than_forty_days_is_not_judged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(feature_coverage, "watched", lambda: (FEATURES, 1))
    write(tmp_path, 2025, rows(november(2025), USUAL_SHARES))
    live(tmp_path, {DENSE: 0.0})

    row = coverage_row(DAY, tmp_path)

    assert row.severity == Severity.INFO
    assert row.detail["unjudged"] == dict.fromkeys(FEATURES, "short_reference")
    assert row.detail["reference"] == {"month": 11, "years": 1, "days": 20}
    assert coverage_summary(row) == "0 of 3 features judged, none flagged"


@pytest.mark.parametrize(("held", "appeared"), [(0.01, [SPARSE]), (0.0, [])])
def test_values_where_the_reference_never_had_any_are_noted_without_a_warning(
    tmp_path: Path, held: float, appeared: list[str]
) -> None:
    history(tmp_path, shares={**USUAL_SHARES, SPARSE: 0.0})
    live(tmp_path, {SPARSE: held})

    row = coverage_row(DAY, tmp_path)

    assert row.severity == Severity.INFO
    assert row.detail["unjudged"] == {SPARSE: "empty_reference"}
    assert row.detail["appeared"] == appeared


def test_a_missing_reference_partition_is_refused_without_a_host_path(tmp_path: Path) -> None:
    history(tmp_path)
    live(tmp_path, {})
    partition_path(tmp_path, 2023).unlink()

    with pytest.raises(CoverageError, match="feature partition of 2023") as refused:
        coverage_row(DAY, tmp_path)

    assert "/" not in str(refused.value)


def test_a_partition_without_a_watched_feature_is_refused_without_a_host_path(
    tmp_path: Path,
) -> None:
    history(tmp_path)
    live(tmp_path, {})
    write(tmp_path, 2022, rows(november(2022), {DENSE: 1.0, USUAL: 0.78}))

    with pytest.raises(CoverageError, match="feature partition of 2022") as refused:
        coverage_row(DAY, tmp_path)

    assert "/" not in str(refused.value)


def test_a_day_without_universe_rows_is_refused(tmp_path: Path) -> None:
    history(tmp_path)
    live(tmp_path, {}, days=(THURSDAY,))

    with pytest.raises(CoverageError, match="no universe feature rows on 2026-11-06"):
        coverage_row(DAY, tmp_path)


def test_a_rerun_replaces_the_day_s_row(
    tmp_path: Path, calendar: TradingDays, sent: list[tuple[date, str]]
) -> None:
    history(tmp_path)
    live(tmp_path, {SPARSE: 0.0})
    assert run_coverage(DAY, calendar, tmp_path).severity == Severity.WARNING

    live(tmp_path, {})
    row = run_coverage(DAY, calendar, tmp_path)

    stored = DataQualityLog.objects.get(date=DAY, check_name=CHECK)
    assert (stored.pk, stored.severity, stored.detail) == (row.pk, Severity.INFO, row.detail)
    assert sent == [(DAY, Severity.WARNING)]


def test_a_warning_is_sent_only_when_the_flagged_features_change(
    tmp_path: Path, calendar: TradingDays, sent: list[tuple[date, str]]
) -> None:
    history(tmp_path)
    empty = rows([THURSDAY, DAY], {**USUAL_SHARES, SPARSE: 0.0})
    wider = rows([MONDAY], {**USUAL_SHARES, SPARSE: 0.0, DENSE: 0.9})
    write(tmp_path, 2026, pd.concat([empty, wider], ignore_index=True))

    for day in (THURSDAY, DAY, MONDAY):
        run_coverage(day, calendar, tmp_path)

    assert sent == [(THURSDAY, Severity.WARNING), (MONDAY, Severity.WARNING)]
    assert DataQualityLog.objects.filter(check_name=CHECK, severity=Severity.WARNING).count() == 3


def test_a_warning_is_sent_again_after_a_day_the_check_could_not_run(
    tmp_path: Path, calendar: TradingDays, sent: list[tuple[date, str]]
) -> None:
    history(tmp_path)
    live(tmp_path, {SPARSE: 0.0})
    DataQualityLog.objects.create(
        date=THURSDAY, check_name=CHECK, severity=Severity.WARNING, detail={"error": "KeyError"}
    )

    run_coverage(DAY, calendar, tmp_path)

    assert sent == [(DAY, Severity.WARNING)]


def test_an_error_is_sent_every_day(
    tmp_path: Path, calendar: TradingDays, sent: list[tuple[date, str]]
) -> None:
    history(tmp_path)
    live(tmp_path, {DENSE: 0.1}, days=(THURSDAY, DAY))

    for day in (THURSDAY, DAY):
        run_coverage(day, calendar, tmp_path)

    assert sent == [(THURSDAY, Severity.ERROR), (DAY, Severity.ERROR)]


def test_a_quiet_run_writes_the_row_without_alerting(
    tmp_path: Path, calendar: TradingDays, sent: list[tuple[date, str]]
) -> None:
    history(tmp_path)
    write(
        tmp_path,
        2026,
        pd.concat(
            [
                rows([THURSDAY], {**USUAL_SHARES, SPARSE: 0.0}),
                rows([DAY], {**USUAL_SHARES, DENSE: 0.1}),
            ],
            ignore_index=True,
        ),
    )

    found = [run_coverage(day, calendar, tmp_path, alert=False) for day in (THURSDAY, DAY)]

    assert [row.severity for row in found] == [Severity.WARNING, Severity.ERROR]
    assert DataQualityLog.objects.filter(check_name=CHECK).count() == 2
    assert sent == []


def test_the_first_calendar_day_has_no_earlier_row_to_compare(
    tmp_path: Path, sent: list[tuple[date, str]]
) -> None:
    history(tmp_path)
    live(tmp_path, {SPARSE: 0.0})
    short = TradingDays([DAY], DAY, DAY)

    run_coverage(DAY, short, tmp_path)

    assert sent == [(DAY, Severity.WARNING)]


def test_a_day_the_market_is_closed_is_refused(tmp_path: Path, calendar: TradingDays) -> None:
    with pytest.raises(CoverageError, match="2026-11-07 is not a trading day"):
        run_coverage(date(2026, 11, 7), calendar, tmp_path)

    assert not DataQualityLog.objects.exists()


def test_an_error_row_shuts_the_day_s_prediction_gate(
    tmp_path: Path, calendar: TradingDays, sent: list[tuple[date, str]]
) -> None:
    history(tmp_path)
    live(tmp_path, {DENSE: 0.1})
    DataQualityLog.objects.bulk_create(
        DataQualityLog(date=DAY, check_name=check, severity=Severity.INFO) for check in CHECKS
    )
    assert should_run_prediction(DAY)

    run_coverage(DAY, calendar, tmp_path)

    assert not should_run_prediction(DAY)
    assert sent == [(DAY, Severity.ERROR)]


@pytest.fixture
def stored_calendar(settings: Any, tmp_path: Path) -> Path:
    first, last = date(2026, 10, 1), date(2026, 11, 30)
    span = (first + timedelta(days=n) for n in range((last - first).days + 1))
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=day.weekday() < 5) for day in span
    )
    settings.FEATURE_PARQUET_DIR = tmp_path
    return tmp_path


def test_the_command_writes_a_row_for_each_trading_day_of_a_range(stored_calendar: Path) -> None:
    history(stored_calendar)
    live(stored_calendar, {SPARSE: 0.0}, days=(THURSDAY, DAY, MONDAY))
    out = StringIO()

    call_command("feature_coverage", start=THURSDAY, end=MONDAY, stdout=out)

    assert out.getvalue().splitlines() == [
        f"feature coverage {day}: warning, 3 of 3 features judged, flagged {SPARSE}"
        for day in (THURSDAY, DAY, MONDAY)
    ]
    written = DataQualityLog.objects.filter(check_name=CHECK).order_by("date")
    assert [row.date for row in written] == [THURSDAY, DAY, MONDAY]


def test_the_command_backfills_quietly_when_asked(
    stored_calendar: Path, sent: list[tuple[date, str]]
) -> None:
    history(stored_calendar)
    live(stored_calendar, {SPARSE: 0.0}, days=(THURSDAY, DAY))

    call_command("feature_coverage", start=THURSDAY, end=DAY, quiet=True, stdout=StringIO())
    assert sent == []

    call_command("feature_coverage", date=THURSDAY, stdout=StringIO())
    assert sent == [(THURSDAY, Severity.WARNING)]


def test_the_command_takes_one_day(stored_calendar: Path) -> None:
    history(stored_calendar)
    live(stored_calendar, {})
    out = StringIO()

    call_command("feature_coverage", date=DAY, stdout=out)

    assert out.getvalue().strip() == (
        "feature coverage 2026-11-06: info, 3 of 3 features judged, none flagged"
    )


@time_machine.travel(datetime(2026, 11, 7, 10, tzinfo=ZoneInfo("Asia/Taipei")), tick=False)
def test_the_command_defaults_to_the_last_trading_day_before_today(stored_calendar: Path) -> None:
    history(stored_calendar)
    live(stored_calendar, {})
    out = StringIO()

    call_command("feature_coverage", stdout=out)

    assert out.getvalue().startswith("feature coverage 2026-11-06: info")


def test_the_command_refuses_a_date_beside_a_range_and_half_a_range(stored_calendar: Path) -> None:
    with pytest.raises(CommandError, match="--date cannot be combined"):
        call_command("feature_coverage", date=DAY, start=THURSDAY, end=DAY)
    with pytest.raises(CommandError, match="--start and --end go together"):
        call_command("feature_coverage", start=THURSDAY)


def test_the_command_reports_a_day_it_cannot_check(stored_calendar: Path) -> None:
    history(stored_calendar)

    with pytest.raises(CommandError, match="feature partition of 2026"):
        call_command("feature_coverage", date=DAY)
