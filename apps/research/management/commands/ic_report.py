"""Screen features on the tuning folds, record the kept list and write the IC report."""

import math
from argparse import ArgumentParser
from datetime import date
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.etl.management.options import as_date
from apps.market.calendar import TradingDays
from apps.research.features.store import read_partitions
from apps.research.labels import HORIZONS, latest_price_date
from apps.research.models import HoldoutRun, ModelObjective, ModelRun
from apps.research.runs import RunError, feature_hash, lib_versions, record_run
from apps.research.screening import (
    CORRELATION_CEILING,
    IC_FLOOR,
    Screening,
    screen,
    screening_window,
)
from apps.research.training import load_labels
from apps.research.walkforward import WalkForwardConfig, generate_folds

DEFAULT_CONFIG = WalkForwardConfig()
SCREENING_VERSION = "1"
REPORT = Path("docs") / "reports" / "phase3-ic-report.md"


def _plain(value: float) -> float | None:
    return None if math.isnan(value) else float(value)


def _cell(value: float, digits: int) -> str:
    return "—" if math.isnan(value) else f"{value:.{digits}f}"


def render(result: Screening, start: date, end: date, run_id: str) -> str:
    """Markdown table of every screened feature's IC and verdict."""
    header = [
        "# Phase 3 單因子 rank IC 報表",
        "",
        f"- 篩選窗：{start} 到 {end}（測試期落在 2014 到 2015 年的調參摺的訓練窗）",
        f"- `model_run`：`{run_id}`；保留 {len(result.kept)} 個，剔除 {len(result.dropped)} 個",
        f"- 規則：訊號型特徵在每個 horizon 的 |rank IC| 都低於 {IC_FLOOR} 就剔除，條件型豁免；"
        f"日均橫截面 Spearman 絕對值高於 {CORRELATION_CEILING} 的一對留 |IC| 較高者",
        "",
        "| 特徵 | 類型 | IC 5 日 | NW t | IC 20 日 | NW t | 結果 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    rows = [
        f"| {name} | {'條件' if row['conditional'] else '訊號'} "
        f"| {_cell(row['ic_h5'], 4)} | {_cell(row['nw_t_h5'], 2)} "
        f"| {_cell(row['ic_h20'], 4)} | {_cell(row['nw_t_h20'], 2)} "
        f"| {'剔除：' + row['reason'] if row['reason'] else '保留'} |"
        for name, row in result.table.iterrows()
    ]
    return "\n".join([*header, *rows, ""])


class Command(BaseCommand):
    help = (
        "Screen stock-level features by single-factor rank IC and correlation; record and report."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--run-key", required=True)
        parser.add_argument("--last-date")
        parser.add_argument("--output")

    def handle(self, *args: Any, **options: Any) -> None:
        if HoldoutRun.objects.exists():
            raise CommandError("a holdout run exists, so the screening is frozen (SPEC §6 步驟 7)")
        last = as_date(options["last_date"], "--last-date") or latest_price_date()
        if last is None:
            raise CommandError("daily_price is empty")
        start, end = screening_window(generate_folds(TradingDays.from_db(), last, DEFAULT_CONFIG))
        directory = settings.FEATURE_PARQUET_DIR
        digest = feature_hash(start, end, directory)
        labels = {horizon: load_labels(start, end, horizon) for horizon in HORIZONS}
        result = screen(read_partitions(start, end, directory), labels)
        run_key = options["run_key"]
        run = ModelRun(
            run_id=f"{run_key}.screening",
            run_key=run_key,
            model_type="screening",
            objective=ModelObjective.RANK,
            version=SCREENING_VERSION,
            train_start=start,
            train_end=end,
            params={
                "dropped": result.dropped,
                "ic": {
                    name: {str(h): _plain(row[f"ic_h{h}"]) for h in HORIZONS}
                    for name, row in result.table.iterrows()
                },
                "ic_floor": IC_FLOOR,
                "correlation_ceiling": CORRELATION_CEILING,
            },
            feature_list=result.kept,
            lib_versions=lib_versions(),
            feature_hash=digest,
        )
        try:
            record_run(run)
        except (RunError, ValueError) as error:
            raise CommandError(str(error)) from error
        output = Path(options["output"] or Path(settings.BASE_DIR) / REPORT)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render(result, start, end, run.run_id), encoding="utf-8")
        kept, total = len(result.kept), len(result.table)
        self.stdout.write(f"{run.run_id}: kept {kept} of {total} features; report at {output}")
