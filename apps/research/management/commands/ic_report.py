"""Screen features on the tuning folds, record the kept list and write the IC report."""

import math
from argparse import ArgumentParser
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.etl.management.options import as_date
from apps.market.calendar import TradingDays
from apps.research.features.registry import FEATURES, FeatureSpec
from apps.research.features.store import read_partitions
from apps.research.holdout import require_key
from apps.research.labels import HORIZONS, latest_price_date
from apps.research.models import HoldoutRun, ModelObjective, ModelRun
from apps.research.runs import SCREENING_MODEL, RunError, feature_hash, lib_versions, record_run
from apps.research.screening import (
    CORRELATION_CEILING,
    IC_FLOOR,
    MIN_COVERAGE,
    Screening,
    screen,
    screening_window,
)
from apps.research.training import load_labels
from apps.research.walkforward import WalkForwardConfig, generate_folds

DEFAULT_CONFIG = WalkForwardConfig()
SCREENING_VERSION = "1"
REPORT = Path("docs") / "reports" / "phase3-ic-report.md"
UNSCREENED = [
    spec.name for spec in FEATURES if spec.batch == 1 and (spec.market_level or spec.categorical)
]


def _plain(value: float) -> float | None:
    return None if math.isnan(value) else float(value)


def _cell(value: float, digits: int) -> str:
    return "—" if math.isnan(value) else f"{value:.{digits}f}"


def _text(value: str) -> str:
    return value.replace("|", "\\|")


def _verdict(row: pd.Series) -> str:
    if row["reason"]:
        return f"剔除：{_text(row['reason'])}"
    if not row["covered"]:
        return f"保留：IC 有值的天數不到篩選窗的 {MIN_COVERAGE:.0%}，未做低 IC 篩選"
    return "保留"


def _unscreened(spec: FeatureSpec) -> str:
    kind, why = ("類別", "類別值沒有大小順序") if spec.categorical else ("市場", "同一天全市場同值")
    return f"| {spec.name} | {kind} | — | — | — | — | — | 不篩選：{why}，橫截面 IC 沒有定義 |"


def render(result: Screening, start: date, end: date, run_id: str) -> str:
    """Markdown table of every screened feature's IC and verdict."""
    header = [
        "# Phase 3 單因子 rank IC 報表",
        "",
        f"- 篩選窗：{start} 到 {end}，{result.days} 個交易日"
        "（測試期落在 2014 到 2015 年的調參摺的訓練窗）",
        f"- `model_run`：`{run_id}`；保留 {len(result.kept)} 個，剔除 {len(result.dropped)} 個",
        f"- 規則：訊號型特徵在每個 horizon 的 |rank IC| 都低於 {IC_FLOOR} 就剔除，條件型豁免，"
        f"IC 有值的天數不到篩選窗 {MIN_COVERAGE:.0%} 的不判定；"
        f"日均橫截面 Spearman 絕對值高於 {CORRELATION_CEILING} 的一對留 |IC| 較高者",
        "",
        "| 特徵 | 類型 | IC 5 日 | NW t | IC 20 日 | NW t | 天數（5／20 日） | 結果 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rows = [
        f"| {name} | {'條件' if row['conditional'] else '訊號'} "
        f"| {_cell(row['ic_h5'], 4)} | {_cell(row['nw_t_h5'], 2)} "
        f"| {_cell(row['ic_h20'], 4)} | {_cell(row['nw_t_h20'], 2)} "
        f"| {int(row['days_h5'])}／{int(row['days_h20'])} | {_verdict(row)} |"
        for name, row in result.table.iterrows()
    ]
    skipped = [_unscreened(spec) for spec in FEATURES if spec.name in UNSCREENED]
    return "\n".join([*header, *rows, *skipped, ""])


class Command(BaseCommand):
    help = (
        "Screen stock-level features by single-factor rank IC and correlation; record and report."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--run-key", required=True)
        parser.add_argument("--last-date")
        parser.add_argument("--output")

    def handle(self, *args: Any, **options: Any) -> None:
        run_key = options["run_key"]
        try:
            require_key(f"{run_key}.screening")
            self._refuse_frozen(run_key)
            calendar = TradingDays.from_db()
            start, end = self._window(options["last_date"], calendar)
            result, digest = self._screen(start, end, calendar)
            self._refuse_frozen(run_key)
            run = self._record(run_key, result, start, end, digest)
        except (RunError, ValueError, FileNotFoundError) as error:
            raise CommandError(str(error)) from error
        output = Path(options["output"] or Path(settings.BASE_DIR) / REPORT)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render(result, start, end, run.run_id), encoding="utf-8")
        kept, total = len(result.kept), len(result.table)
        self.stdout.write(f"{run.run_id}: kept {kept} of {total} features; report at {output}")

    def _refuse_frozen(self, run_key: str) -> None:
        if HoldoutRun.objects.exists():
            raise CommandError("a holdout run exists, so the screening is frozen (SPEC §6 步驟 7)")
        other = (
            ModelRun.objects.filter(model_type=SCREENING_MODEL)
            .exclude(run_key=run_key)
            .values_list("run_id", "run_key")
            .first()
        )
        if other is not None:
            raise CommandError(
                f"{other[0]} is the recorded screening; run drop_run {other[1]} to screen again"
            )

    def _window(self, last_date: str | None, calendar: TradingDays) -> tuple[date, date]:
        last = as_date(last_date, "--last-date") or latest_price_date()
        if last is None:
            raise CommandError("daily_price is empty")
        if last <= DEFAULT_CONFIG.tuning_end:
            raise CommandError(
                f"--last-date {last} falls inside the tuning period ending "
                f"{DEFAULT_CONFIG.tuning_end}; the screening window would be cut short"
            )
        return screening_window(generate_folds(calendar, last, DEFAULT_CONFIG))

    def _screen(self, start: date, end: date, calendar: TradingDays) -> tuple[Screening, str]:
        directory = settings.FEATURE_PARQUET_DIR
        digest = feature_hash(start, end, directory)
        features = read_partitions(start, end, directory)
        if feature_hash(start, end, directory) != digest:
            raise CommandError("feature partitions changed while reading; run ic_report again")
        expected = calendar.between(start, end)
        missing = sorted(set(expected) - set(features["date"]))
        if missing:
            raise CommandError(
                f"features are missing on {len(missing)} of {len(expected)} trading days "
                f"from {missing[0]}; rebuild the feature partitions over the screening window"
            )
        labels = {horizon: load_labels(start, end, horizon) for horizon in HORIZONS}
        return screen(features, labels), digest

    def _record(
        self, run_key: str, result: Screening, start: date, end: date, digest: str
    ) -> ModelRun:
        run = ModelRun(
            run_id=f"{run_key}.screening",
            run_key=run_key,
            model_type=SCREENING_MODEL,
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
                "days": {
                    name: {str(h): int(row[f"days_h{h}"]) for h in HORIZONS}
                    for name, row in result.table.iterrows()
                },
                "window_days": result.days,
                "min_coverage": MIN_COVERAGE,
                "ic_floor": IC_FLOOR,
                "correlation_ceiling": CORRELATION_CEILING,
                "unscreened": UNSCREENED,
            },
            feature_list=result.kept,
            lib_versions=lib_versions(),
            feature_hash=digest,
        )
        record_run(run)
        return run
