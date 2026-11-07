"""Replace FinMind's zero foreign holdings that the exchanges' stored pages contradict (K21)."""

import json
from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand

from apps.etl.zero_holdings import replace_zero_holdings


class Command(BaseCommand):
    help = "Replace wrong zero foreign holdings with the exchanges' stored values."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args: Any, **options: Any) -> None:
        report = replace_zero_holdings(dry_run=options["dry_run"])
        self.stdout.write(json.dumps(report, ensure_ascii=False, indent=1))
