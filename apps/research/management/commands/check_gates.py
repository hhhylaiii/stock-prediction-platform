"""Check a production build's promotion gates against the serving runs (SPEC §7)."""

from argparse import ArgumentParser
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.research.gates import Gate, check_gates
from apps.research.prediction_cache import FoldMismatchError, MissingFoldsError
from apps.research.runs import RunError


def verdict(gate: Gate) -> str:
    """One gate as the report prints it."""
    if gate.passed is None:
        return gate.reason
    shown = "—" if gate.value is None else f"{gate.value:.4f}"
    detail = f"值 {shown}，門檻 {gate.threshold}，樣本 {gate.sample}"
    if gate.passed:
        return f"通過，{detail}"
    listed = gate.extra.get("errors_total")
    where = f'；{listed} 組 error 的清單在 N 的 params["gates"]' if listed else ""
    return f"不過（{gate.reason}），{detail}{where}"


class Command(BaseCommand):
    help = "Record and print each gate of a build_run run_key's model N; no status changes."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--run-key", required=True, help="The run_key build_run fitted.")

    def handle(self, *args: Any, **options: Any) -> None:
        try:
            results = check_gates(options["run_key"], settings.FEATURE_PARQUET_DIR)
        except (
            RunError,
            ValueError,
            FileNotFoundError,
            MissingFoldsError,
            FoldMismatchError,
        ) as error:
            raise CommandError(str(error)) from error
        for result in results:
            against = "（沒有服務中的 run）" if result.serving is None else f" 對 {result.serving}"
            self.stdout.write(f"{result.slot} {result.run_id}{against}")
            for gate in result.gates:
                self.stdout.write(f"  {gate.name}：{verdict(gate)}")
        failed = [str(result.slot) for result in results if not result.passed]
        self.stdout.write("全部通過" if not failed else f"未通過：{', '.join(failed)}")
