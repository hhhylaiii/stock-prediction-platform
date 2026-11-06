"""Fill rows the standard tables lack from the exchanges' whole-market daily tables (plan K17c)."""

from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from apps.etl.adapters import build_adapter
from apps.etl.exchange_fill import FILLS, fill_missing
from apps.etl.management.options import required_date
from apps.etl.models import EtlRun

SOURCES = ("twse", "tpex")


class Command(BaseCommand):
    help = "Fill missing holdings, share counts or valuations from the exchanges' daily tables."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--source", choices=SOURCES, required=True)
        parser.add_argument("--fill", choices=sorted(FILLS), required=True)
        parser.add_argument("--start")
        parser.add_argument("--end")
        parser.add_argument("--resume", type=int, help="etl_run id of a stopped fill to go on with")

    def handle(self, *args: Any, **options: Any) -> None:
        resume = self._resumed(options)
        start = resume.start_date if resume else required_date(options["start"], "--start")
        end = resume.end_date if resume else required_date(options["end"], "--end")
        run = fill_missing(build_adapter(options["source"]), options["fill"], start, end, resume)
        for target, codes in run.checkpoint["filled"].items():
            self.stdout.write(f"{target} {sum(count for _, _, count in codes.values())}")

    def _resumed(self, options: dict[str, Any]) -> EtlRun | None:
        if options["resume"] is None:
            return None
        run = EtlRun.objects.filter(
            run_id=options["resume"], source=options["source"], dataset=f"fill:{options['fill']}"
        ).first()
        if run is None or run.status == EtlRun.Status.SUCCESS:
            raise CommandError(f"--resume {options['resume']} is not a stopped fill of this kind")
        if run.status == EtlRun.Status.RUNNING:
            raise CommandError(
                f"etl_run {run.run_id} is still marked running; stop that process first, "
                "or set its status to failed if it died"
            )
        return run
