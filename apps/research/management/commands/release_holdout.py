"""Record the user's decision to build on a holdout judged short of its criteria (SPEC §7)."""

from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from apps.research.holdout import DEGRADED, FAILED, HoldoutError, release_holdout
from apps.research.models import HoldoutScope


class Command(BaseCommand):
    help = (
        "Release the verdict of a scope's latest holdout (未達標 or 明顯退化) so build_run and "
        "the monthly retrain may build on it. 只有使用者明確同意後才能執行; --reason keeps the "
        "user's reason with the release, and every build on it carries the release and alerts."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--scope", required=True, choices=HoldoutScope.values)
        parser.add_argument(
            "--status",
            required=True,
            choices=[FAILED, DEGRADED],
            help="The verdict its report gives.",
        )
        parser.add_argument("--reason", required=True, help="Why the user accepts the verdict.")

    def handle(self, *args: Any, **options: Any) -> None:
        try:
            release = release_holdout(options["scope"], options["status"], options["reason"])
        except HoldoutError as error:
            raise CommandError(str(error)) from error
        self.stdout.write(
            f"{release.holdout.run_key}: released {release.status} "
            f"at {timezone.localtime(release.released_at).isoformat()}"
        )
