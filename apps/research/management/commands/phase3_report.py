"""Write the 3.8b sections ic_report does not: gaps by later delisting, fallbacks, monthly IC."""

from argparse import ArgumentParser
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.etl.management.options import as_date
from apps.market.calendar import TradingDays
from apps.market.models import StockMaster
from apps.research.features.build import FRAME_COLUMNS, MODEL_COLUMNS
from apps.research.features.residual import FALLBACKS, MASKED_MONTHS, rules_signature
from apps.research.features.store import partition_rules, read_partitions
from apps.research.labels import HORIZONS
from apps.research.models import ModelRun
from apps.research.phase3_report import (
    MONTHLY_TOP,
    TABLE_COLUMNS,
    fill_counts,
    fill_table,
    missing_counts,
    missing_table,
    monthly_ic,
    render,
    statement_coverage,
    strongest,
)
from apps.research.runs import SCREENING_MODEL, feature_hash
from apps.research.screening import RANK_LABEL
from apps.research.training import load_labels, training_rows

REPORT = Path("docs") / "reports" / "phase3-data-report.md"
DEFAULT_START, DEFAULT_END = date(2010, 1, 1), date(2020, 12, 31)
APPENDIX = Path("docs") / "reports" / "phase3"


def _screening(run_key: str) -> ModelRun:
    run = ModelRun.objects.filter(run_id=f"{run_key}.screening", model_type=SCREENING_MODEL).first()
    if run is None:
        raise CommandError(f"no screening run {run_key}.screening; run ic_report first")
    return run


def _unchanged(run: ModelRun) -> None:
    digest = feature_hash(run.train_start, run.train_end, settings.FEATURE_PARQUET_DIR)
    if digest != run.feature_hash:
        raise CommandError("features changed since the screening; run ic_report again")


def _before(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise CommandError(f"{path} does not exist")
    table = pd.read_csv(path, index_col=0)
    missing = [column for column in TABLE_COLUMNS if column not in table.columns]
    if missing:
        raise CommandError(f"{path} lacks {', '.join(missing)}")
    return table


def _delisted() -> pd.Series:
    rows = StockMaster.objects.values_list("stock_id", "delist_date")
    return pd.Series({stock: delisted is not None for stock, delisted in rows}, dtype=bool)


def _ruled(start: date, end: date) -> None:
    """Refuse partitions exported under source-gap rules other than this code's."""
    for year in range(start.year, end.year + 1):
        if partition_rules(settings.FEATURE_PARQUET_DIR, year) != rules_signature():
            raise CommandError(
                f"the {year} partition was exported under other source-gap rules than this "
                "code's; rebuild features"
            )


def _gaps(start: date, end: date, delisted: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing: pd.DataFrame | None = None
    filled: pd.DataFrame | None = None
    directory = settings.FEATURE_PARQUET_DIR
    for year in range(start.year, end.year + 1):
        low, high = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
        rows = read_partitions(low, high, directory, columns=FRAME_COLUMNS)
        counts = missing_counts(rows, delisted, MODEL_COLUMNS)
        fills = fill_counts(rows, delisted, FALLBACKS)
        missing = counts if missing is None else missing.add(counts, fill_value=0)
        filled = fills if filled is None else filled.add(fills, fill_value=0)
    if missing is None or filled is None:
        raise CommandError(f"no years between {start} and {end}")
    return missing_table(missing), fill_table(filled)


def _kept(run: ModelRun) -> dict[str, dict[str, float | None]]:
    kept = set(run.feature_list)
    return {name: ic for name, ic in run.params["ic"].items() if name in kept}


def _monthly(run: ModelRun, calendar: TradingDays) -> pd.DataFrame:
    names = list(run.params["ic"])
    features = read_partitions(run.train_start, run.train_end, settings.FEATURE_PARQUET_DIR)
    _unchanged(run)
    parts = []
    for horizon in HORIZONS:
        labels = load_labels(run.train_start, run.train_end, horizon)
        rows = training_rows(features, labels, RANK_LABEL)
        parts.append(monthly_ic(rows, names, RANK_LABEL, calendar).assign(horizon=horizon))
    return pd.concat(parts, ignore_index=True)


class Command(BaseCommand):
    help = "Write the data-gap, fallback and monthly IC sections of the Phase 3 report."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--run-key", required=True)
        parser.add_argument("--before", required=True)
        parser.add_argument("--start", default=DEFAULT_START.isoformat())
        parser.add_argument("--end", default=DEFAULT_END.isoformat())
        parser.add_argument("--output")
        parser.add_argument("--appendix")

    def handle(self, *args: Any, **options: Any) -> None:
        run = _screening(options["run_key"])
        before = _before(Path(options["before"]))
        start = as_date(options["start"], "--start") or DEFAULT_START
        end = as_date(options["end"], "--end") or DEFAULT_END
        if start > end:
            raise CommandError(f"--start {start} is after --end {end}")
        if (start, end) != (DEFAULT_START, DEFAULT_END):
            self.stdout.write(
                f"warning: --start/--end differ from {DEFAULT_START}..{DEFAULT_END}; "
                "compare against a before table of the same period"
            )
        try:
            _unchanged(run)
            _ruled(start, end)
            _ruled(run.train_start, run.train_end)
            after, fills = _gaps(start, end, _delisted())
            monthly = _monthly(run, TradingDays.from_db())
        except (ValueError, FileNotFoundError) as error:
            raise CommandError(str(error)) from error
        base = Path(settings.BASE_DIR)
        output = Path(options["output"]) if options["output"] else base / REPORT
        appendix = Path(options["appendix"]) if options["appendix"] else base / APPENDIX
        appendix.mkdir(parents=True, exist_ok=True)
        before.rename_axis("feature").to_csv(appendix / "missing-before.csv")
        after.rename_axis("feature").to_csv(appendix / "missing-after.csv")
        fills.rename_axis("feature").to_csv(appendix / "fills.csv")
        monthly.to_csv(appendix / "monthly-ic.csv", index=False)
        header = {
            "run_id": run.run_id,
            "start": run.train_start,
            "end": run.train_end,
            "window_days": run.params["window_days"],
            "kept": len(run.feature_list),
            "dropped": len(run.params["dropped"]),
            "before": Path(options["before"]).name,
        }
        text = render(
            header=header,
            coverage=statement_coverage(run.params),
            gaps=(before, after, str(start), str(end)),
            fills=(fills, FALLBACKS),
            masked=MASKED_MONTHS,
            monthly=monthly,
            top=strongest(_kept(run), MONTHLY_TOP),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        self.stdout.write(f"{run.run_id}: report at {output}, appendix in {appendix}")
