"""Replace FinMind statement figures with each quarter's MOPS filing, from raw (plan K18)."""

from argparse import ArgumentParser
from typing import Any

import pandas as pd
from django.core.management.base import BaseCommand
from django.db import transaction

from apps.etl.cumulative_margins import clear_cumulative_margins
from apps.etl.statement_overwrite import (
    overwrite_statements,
    summary_frame,
    viewer_frame_from_raw,
)


class Command(BaseCommand):
    help = "Overwrite financial_statement figures that differ from MOPS, without fetching again."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--stock", action="append", dest="stocks")
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args: Any, **options: Any) -> None:
        stocks = options["stocks"]
        summaries = summary_frame()
        if stocks:
            summaries = summaries[summaries["stock_id"].isin(stocks)]
        viewers = viewer_frame_from_raw(stocks)
        found = pd.concat([summaries, viewers], ignore_index=True)
        with transaction.atomic():
            changed = overwrite_statements(found) if not found.empty else 0
            cleared = clear_cumulative_margins(stocks)
            if options["dry_run"]:
                transaction.set_rollback(True)
        verb = "would change" if options["dry_run"] else "changed"
        self.stdout.write(f"{changed} statements {verb} from {len(found)} MOPS rows")
        kinds = ", ".join(f"{kind} {count}" for kind, count in sorted(cleared.kinds.items()))
        done = "would be cleared" if options["dry_run"] else "cleared"
        self.stdout.write(f"{sum(cleared.kinds.values())} margins {done} ({kinds or 'none'})")
        self.stdout.write(
            f"{cleared.unmatched} margins match no span and stay, "
            f"{cleared.unchecked} have no MOPS figures to judge"
        )
        if viewers.attrs["unreadable"]:
            codes = ", ".join(viewers.attrs["unreadable"])
            self.stdout.write(f"set aside {codes}: their kept viewer pages do not parse")
