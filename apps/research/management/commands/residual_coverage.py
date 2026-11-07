"""Monthly coverage of features for stocks later delisted and the rest, before any month mask.

A gap whose cause is known on its day (residual.OBSERVABLE, LISTING_WINDOWS) counts as covered.
"""

from argparse import ArgumentParser
from datetime import date
from typing import Any

import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from apps.etl.management.options import as_date
from apps.market.calendar import TradingDays
from apps.market.models import StockMaster
from apps.research.features.build import FEATURE_COLUMNS, build_feature_frame, observable_frame
from apps.research.features.fundamental import BASIS_SIGNALS
from apps.research.features.residual import (
    ACCEPT_RATIO,
    BASIS_GAP,
    FALLBACKS,
    MASKED_MONTHS,
    MAX_FILL_RATIO,
    MAX_MASKED_SHARE,
    affected_months,
    basis_counts,
    fill_flags,
    monthly_coverage,
    unmasked_fills,
)
from apps.research.ranges import years

DEFAULT_START, DEFAULT_END = date(2010, 1, 1), date(2020, 12, 31)
TABLE_COLUMNS = [
    "column",
    "month",
    "current",
    "delisted",
    "present_current",
    "present_delisted",
    "filled_current",
    "filled_delisted",
    "rows_current",
    "rows_delisted",
]


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
        if column in FALLBACKS:
            self._fills(column, coverage, affected)

    def _fills(self, column: str, coverage: pd.DataFrame, affected: list[str]) -> None:
        shares = unmasked_fills(coverage, affected)
        current, delisted = shares["current"], shares["delisted"]
        if shares.isna().all():
            self.stdout.write(f"{column}: every month is affected, so no fills are left unmasked")
            return
        self.stdout.write(
            f"{column}: filled from {FALLBACKS[column]} in the months left unmasked, "
            f"current {current:.2%}, delisted {delisted:.2%}"
        )
        if fill_flags(pd.Series([current]), pd.Series([delisted])).iloc[0]:
            self.stdout.write(
                f"    warning: fills still lean on stocks later delisted (over {MAX_FILL_RATIO:g} "
                "times, or delisted only); report before filling MASKED_MONTHS"
            )

    def _basis(self, column: str, counts: pd.Series) -> None:
        current = counts["current"] / max(counts["rows_current"], 1)
        delisted = counts["delisted"] / max(counts["rows_delisted"], 1)
        self.stdout.write(
            f"{column}: gaps read across report bases, "
            f"current {current:.2%}, delisted {delisted:.2%}"
        )
        if delisted > ACCEPT_RATIO * current and delisted - current > BASIS_GAP:
            self.stdout.write(
                "    warning: these gaps lean on stocks later delisted; "
                "report before keeping them as having a known cause"
            )

    def handle(self, *args: Any, **options: Any) -> None:
        columns = self._columns(options["columns"])
        start = as_date(options["start"], "--start") or DEFAULT_START
        end = as_date(options["end"], "--end") or DEFAULT_END
        if start > end:
            raise CommandError(f"--start {start} is after --end {end}")
        calendar, delisted = TradingDays.from_db(), _delisted()
        parts: dict[str, list[pd.DataFrame]] = {column: [] for column in columns}
        crossed = [column for column in columns if column in BASIS_SIGNALS]
        bases = dict.fromkeys(crossed, pd.Series(0, index=["current", "delisted"]))
        for low, high in years(start, end):
            built = build_feature_frame(calendar, low, high, masked={})
            gaps = observable_frame(calendar, low, high)
            frame = built.merge(gaps, on=["stock_id", "date"], how="left")
            for column in columns:
                parts[column].append(monthly_coverage(frame, column, delisted))
            for column in crossed:
                bases[column] = basis_counts(frame, column, delisted).add(
                    bases[column], fill_value=0
                )
        tables = {column: pd.concat(chunks) for column, chunks in parts.items()}
        for column, coverage in tables.items():
            self._report(column, coverage)
        for column, counts in bases.items():
            self._basis(column, counts)
        if options["output"]:
            rows = [
                coverage.rename_axis("month").reset_index().assign(column=column)
                for column, coverage in tables.items()
            ]
            table = pd.concat(rows).reindex(columns=TABLE_COLUMNS)
            table.to_csv(options["output"], index=False)
            self.stdout.write(f"wrote {options['output']}")
