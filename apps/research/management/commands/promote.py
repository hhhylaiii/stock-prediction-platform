"""Swap a production build in once its promotion gates pass (SPEC §7 每月重訓)."""

from argparse import ArgumentParser
from datetime import date
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.market.calendar import CalendarError
from apps.research.holdout import HoldoutError
from apps.research.lock import ResearchBusyError
from apps.research.predict import PredictionGateError
from apps.research.prediction_cache import FoldMismatchError, MissingFoldsError
from apps.research.promotion import Promoted, first_day, promote
from apps.research.runs import RunError


def failures(result: Promoted) -> list[str]:
    """Each failed slot with its failed gates, as the report prints them."""
    return [
        f"  {item['slot']} {item['run_id']}："
        + "、".join(f"{gate['name']}（{gate['reason']}）" for gate in item["gates"])
        for item in result.failed
    ]


def left_out(result: Promoted) -> list[str]:
    """Each run kept out of a partial swap, with the gates it failed."""
    failed = {item["slot"]: item["gates"] for item in result.failed}
    run_ids = {str(checked.slot): checked.run_id for checked in result.checked}
    return [
        f"  {slot} {run_ids[slot]}："
        + "、".join(f"{gate['name']}（{gate['reason']}）" for gate in failed.get(slot, []))
        for slot in result.held
    ]


class Command(BaseCommand):
    help = (
        "Rerun the gates of a build_run run_key's candidates and, for each horizon that passes, "
        "retire its serving run, make N active and write its rows for --date; a failed horizon "
        "keeps its serving run. --force with --reason overrides failed gates, never a refused "
        "build."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--run-key", required=True, help="The run_key build_run fitted.")
        parser.add_argument(
            "--date",
            help=(
                "First day N serves, YYYY-MM-DD; by default the latest day predictions are "
                "served for (the latest research-ready day when none are), never before the "
                "build's as_of or its slots' latest rows."
            ),
        )
        parser.add_argument("--force", action="store_true", help="Swap in despite failed gates.")
        parser.add_argument("--reason", default="", help="Why --force overrides the gates.")

    def handle(self, *args: Any, **options: Any) -> None:
        run_key = options["run_key"]
        try:
            day = date.fromisoformat(options["date"]) if options["date"] else first_day(run_key)
            result = promote(
                run_key,
                day,
                settings.FEATURE_PARQUET_DIR,
                force=options["force"],
                reason=options["reason"],
            )
        except (
            RunError,
            ResearchBusyError,
            HoldoutError,
            CalendarError,
            PredictionGateError,
            ValueError,
            FileNotFoundError,
            MissingFoldsError,
            FoldMismatchError,
        ) as error:
            raise CommandError(str(error)) from error
        if not result.promoted:
            self.stdout.write(f"{run_key} 未換模，舊模型續用；沒過的門檻：")
            self.stdout.write("\n".join(failures(result)))
            return
        self.stdout.write(f"{run_key} 於 {day} 換模")
        for run_id, count in sorted(result.written.items()):
            self.stdout.write(f"  {run_id}：active，寫入 {count} 列")
        if result.retired:
            self.stdout.write(f"  retired：{', '.join(result.retired)}")
        if result.held:
            self.stdout.write("  沒過門檻、未換上（舊模型續用）：")
            self.stdout.write("\n".join(left_out(result)))
        elif result.failed:
            self.stdout.write(f"  強制換模（{options['reason'].strip()}），沒過的門檻：")
            self.stdout.write("\n".join(failures(result)))
