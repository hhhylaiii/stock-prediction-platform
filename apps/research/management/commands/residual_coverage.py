"""Monthly coverage of features for stocks later delisted and the rest, before any month mask."""

from argparse import ArgumentParser
from datetime import date
from typing import Any

import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from apps.etl.management.options import as_date
from apps.market.calendar import TradingDays
from apps.market.models import StockMaster
from apps.research.features.build import FEATURE_COLUMNS, build_feature_frame
from apps.research.features.residual import (
    FALLBACKS,
    MASKED_MONTHS,
    MAX_MASKED_SHARE,
    affected_months,
    monthly_coverage,
)
from apps.research.ranges import years

DEFAULT_START, DEFAULT_END = date(2010, 1, 1), date(2020, 12, 31)
TABLE_COLUMNS = ["column", "month", "current", "delisted"]


def _delisted() -> pd.Series:
    rows = StockMaster.objects.values_list("stock_id", "delist_date")
    return pd.Series({stock: delisted is not None for stock, delisted in rows}, dtype=bool)


class Command(BaseCommand):
    help = (
        "Build features in memory without month masks and print each column's affected months "
        "for MASKED_MONTHS; writes nothing to the database or the partitions."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--columns", help="Comma list of feature columns; defaults to the rules' columns."
        )
        parser.add_argument("--start", help=f"First day; defaults to {DEFAULT_START}.")
        parser.add_argument("--end", help=f"Last day; defaults to {DEFAULT_END}.")
        parser.add_argument("--output", help="CSV of the monthly coverage of every column.")

    def _columns(self, text: str | None) -> list[str]:
        chosen = [*MASKED_MONTHS, *FALLBACKS] if text is None else text.split(",")
        columns = list(dict.fromkeys(part.strip() for part in chosen))
        unknown = [name for name in columns if name not in FEATURE_COLUMNS]
        if unknown:
            raise CommandError(f"{unknown} are not feature columns")
        return columns

    def _report(self, column: str, coverage: pd.DataFrame) -> None:
        if coverage.empty:
            self.stdout.write(f"{column}: no universe rows in the period")
            return
        affected = affected_months(coverage)
        share = len(affected) / len(coverage)
        self.stdout.write(
            f"{column}: {len(affected)} of {len(coverage)} months affected ({share:.0%})"
        )
        if share > MAX_MASKED_SHARE:
            self.stdout.write(
                f"    warning: over {MAX_MASKED_SHARE:.0%} of the months; "
                "report before filling MASKED_MONTHS"
            )
        months = ", ".join(f'"{month}"' for month in affected)
        self.stdout.write(f'    "{column}": frozenset({{{months}}}),')

    def handle(self, *args: Any, **options: Any) -> None:
        columns = self._columns(options["columns"])
        start = as_date(options["start"], "--start") or DEFAULT_START
        end = as_date(options["end"], "--end") or DEFAULT_END
        if start > end:
            raise CommandError(f"--start {start} is after --end {end}")
        calendar, delisted = TradingDays.from_db(), _delisted()
        parts: dict[str, list[pd.DataFrame]] = {column: [] for column in columns}
        for low, high in years(start, end):
            frame = build_feature_frame(calendar, low, high, masked={})
            for column in columns:
                parts[column].append(monthly_coverage(frame, column, delisted))
        tables = {column: pd.concat(chunks) for column, chunks in parts.items()}
        for column, coverage in tables.items():
            self._report(column, coverage)
        if options["output"]:
            rows = [
                coverage.rename_axis("month").reset_index().assign(column=column)
                for column, coverage in tables.items()
            ]
            pd.concat(rows)[TABLE_COLUMNS].to_csv(options["output"], index=False)
            self.stdout.write(f"wrote {options['output']}")
