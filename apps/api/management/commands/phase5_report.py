"""Write the S5 report: data volume, the SPEC section 9 checks and the endpoint timings."""

import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError, CommandParser
from django.db import connection
from django.db.models import Max, Model
from django.test import Client
from django.utils import timezone

from apps.api.management.commands.measure_api import ENDPOINTS, HEADER, Timing, measure
from apps.etl.models import DataQualityLog, EtlRun
from apps.market.models import DailyPrice, StockMaster
from apps.research.models import (
    OUTPUT_HORIZONS,
    BacktestMetric,
    FeatureDaily,
    LabelDaily,
    ModelMonitor,
    ModelRun,
    Prediction,
    PredictionContrib,
)

if TYPE_CHECKING:
    from playwright.sync_api import Page

LIMIT_MS = 500
THROTTLED = 429
VIEW_TIMEOUT_MS = 15_000
LOADING = "載入中"
SHOWN_VIEW = "section.view:not([hidden])"
VIEWS = (
    ("today.png", "#today", ("#rows .trow",)),
    ("stock.png", "#stock-{stock}", ("#price-chart rect.candle", "#price-chart path.fan-median")),
    ("models.png", "#models", ("#model-table .trow",)),
)

Row = dict[str, Any]


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def fetch(client: Client, path: str) -> dict[str, Any]:
    """The decoded envelope, or an empty one when the body is not a JSON object."""
    response = client.get(path)
    if response.status_code == THROTTLED:
        raise CommandError(f"{path} 被 throttle 擋下（429）；照文件把 API_THROTTLE_RATE 調高再跑")
    try:
        body = json.loads(response.content)
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def ranked(row: Row) -> bool:
    return row.get("rank") is not None


def priced(row: Row) -> bool:
    return row.get("price_pred") is not None


def banded(row: Row) -> bool:
    return row.get("interval") is not None


def likely(row: Row) -> bool:
    return row.get("prob_up_5pct") is not None


@dataclass(frozen=True)
class Table:
    rows: list[Row]
    fresh: bool
    head: str

    def gaps(self, test: Callable[[Row], bool]) -> int:
        return sum(not test(row) for row in self.rows)

    def check(self, name: str, test: Callable[[Row], bool], extra: str = "") -> Check:
        missing = self.gaps(test)
        detail = f"缺 {missing} 檔" if self.fresh else "沒有最新交易日的表"
        return Check(name, self.fresh and not missing, detail + extra)


def served(client: Client, horizon: int, day: str | None) -> Table:
    """Every in-universe row of the horizon's table, read page by page at the largest page size."""
    rows: list[Row] = []
    first: dict[str, Any] = {}
    page, pages = 1, 1
    while page <= pages:
        body = fetch(
            client,
            f"/api/predictions?horizon={horizon}&page_size={settings.API_MAX_PAGE_SIZE}&page={page}",
        )
        meta = body.get("meta") or {}
        first = first or meta
        rows += [row for row in body.get("data") or [] if row.get("in_universe")]
        pages, page = meta.get("pages") or 0, page + 1
    universe = first.get("universe")
    fresh = day is not None and first.get("date") == day and bool(rows) and len(rows) == universe
    return Table(rows, fresh, f"date {first.get('date')}，{len(rows)} 檔，universe {universe}")


def table_checks(client: Client, day: str | None) -> list[Check]:
    long, short = served(client, 20, day), served(client, 5, day)
    sources = Counter(str(row.get("interval_source")) for row in long.rows)
    source_text = "、".join(f"{name} {n}" for name, n in sorted(sources.items())) or "無"
    return [
        long.check("20 日排名", ranked, f"；{long.head}"),
        long.check("20 日預測價格", priced),
        long.check("20 日價格區間", banded, f"；來源 {source_text}"),
        long.check("20 日上漲機率", likely),
        short.check(
            "5 日排名與預測價格", lambda row: ranked(row) and priced(row), f"；{short.head}"
        ),
    ]


def shown(entry: Row) -> bool:
    """What the stock page draws: rank and price, plus band and odds on the 20-day horizon."""
    row = entry.get("prediction")
    if not row or not (ranked(row) and priced(row)):
        return False
    return entry.get("horizon") != 20 or (banded(row) and likely(row))


def stock_checks(client: Client, stock: str, day: str | None) -> list[Check]:
    forecast = fetch(client, f"/api/stocks/{stock}/predictions")
    entries = (forecast.get("data") or {}).get("horizons") or []
    whole = sorted(
        entry["horizon"]
        for entry in entries
        if day is not None and entry.get("date") == day and shown(entry)
    )
    features = fetch(client, f"/api/stocks/{stock}/features")
    items = (features.get("data") or {}).get("items") or []
    feature_day = (features.get("meta") or {}).get("date")
    valued = sum(item.get("value") is not None for item in items)
    return [
        Check(
            f"{stock} 個股頁兩個期間",
            whole == sorted(OUTPUT_HORIZONS),
            "齊全的期間 " + ("、".join(map(str, whole)) or "無"),
        ),
        Check(
            f"{stock} 的特徵",
            day is not None and feature_day == day and valued > 0,
            f"date {feature_day}，{len(items)} 項貢獻，{valued} 項有原始值",
        ),
    ]


def timing_checks(timings: list[Timing]) -> list[Check]:
    odd = [f"`{timing.path}` {timing.status}" for timing in timings if timing.status != "200"]
    slowest = max(timings, key=lambda timing: timing.p95)
    return [
        Check(
            f"{len(ENDPOINTS)} 個端點都回 200",
            not odd and len(timings) == len(ENDPOINTS),
            "、".join(odd) or "全部 200",
        ),
        Check(
            f"p95 都在 {LIMIT_MS} ms 內",
            slowest.p95 < LIMIT_MS,
            f"最大 p95 {slowest.p95:.1f} ms（`{slowest.path}`）",
        ),
    ]


def acceptance_checks(client: Client, stock: str, timings: list[Timing]) -> list[Check]:
    status = fetch(client, "/api/status").get("data") or {}
    day, predicted = status.get("price_date"), status.get("prediction_date")
    return [
        Check(
            "最新交易日有預測",
            day is not None and predicted == day,
            f"price_date {day}，prediction_date {predicted}",
        ),
        *table_checks(client, day),
        *stock_checks(client, stock, day),
        *timing_checks(timings),
    ]


def latest(model: type[Model]) -> str:
    day = model._default_manager.aggregate(latest=Max("date"))["latest"]
    return "無" if day is None else day.isoformat()


def volume(stock: str) -> list[str]:
    statuses = Counter(ModelRun.objects.values_list("status", flat=True))
    status_text = "、".join(f"{name} {n}" for name, n in sorted(statuses.items())) or "無"
    days = Prediction.objects.values("date").distinct().count()
    return [
        f"- stock_master {StockMaster.objects.count():,} 檔。",
        f"- daily_price {DailyPrice.objects.count():,} 列，到 {latest(DailyPrice)}；"
        f"{stock} 有 {DailyPrice.objects.filter(stock_id=stock).count():,} 列。",
        f"- feature_daily {FeatureDaily.objects.count():,} 列，到 {latest(FeatureDaily)}；"
        f"{stock} 的 label_daily 有 {LabelDaily.objects.filter(stock_id=stock).count():,} 列。",
        f"- model_run {sum(statuses.values())} 筆：{status_text}。",
        f"- prediction {Prediction.objects.count():,} 列，{days} 個預測日，"
        f"到 {latest(Prediction)}；prediction_contrib {PredictionContrib.objects.count():,} 列。",
        f"- model_monitor {ModelMonitor.objects.count():,} 列，"
        f"backtest_metric {BacktestMetric.objects.count():,} 列。",
        f"- etl_run {EtlRun.objects.count():,} 筆，"
        f"data_quality_log {DataQualityLog.objects.count():,} 筆。",
    ]


def report_text(stock: str, checks: list[Check], timings: list[Timing]) -> str:
    lines = [
        "# S5 重量測",
        "",
        f"量測時間 {timezone.localtime():%Y-%m-%d %H:%M}，"
        f"資料庫 `{connection.settings_dict['NAME']}`。",
        "",
        "## 資料量",
        "",
        *volume(stock),
        "",
        "## 驗收檢查",
        "",
        "| 條件 | 結果 | 實際值 |",
        "| --- | --- | --- |",
        *(f"| {c.name} | {'通過' if c.passed else '未通過'} | {c.detail} |" for c in checks),
        "",
        "## 端點耗時",
        "",
        *HEADER,
        *(timing.line() for timing in timings),
    ]
    return "\n".join(lines) + "\n"


def wait_for(page: "Page", name: str, ready: tuple[str, ...], timeout_ms: float) -> None:
    """Each ready selector attached, then no 載入中 left in the status strip or the shown view."""
    from playwright.sync_api import expect

    for selector in ready:
        try:
            expect(page.locator(selector).first).to_be_attached(timeout=timeout_ms)
        except AssertionError as error:
            raise CommandError(f"{name}：等不到 {selector}") from error
    pending = page.locator(f"#status, {SHOWN_VIEW}").get_by_text(LOADING)
    try:
        expect(pending).to_have_count(0, timeout=timeout_ms)
    except AssertionError as error:
        raise CommandError(f"{name}：還有區塊停在「{LOADING}」") from error


def capture_views(
    page: "Page", base_url: str, stock: str, out_dir: Path, timeout_ms: float = VIEW_TIMEOUT_MS
) -> list[Path]:
    """Screenshot every view once drawn; out_dir gets all three or none of them."""
    page.set_viewport_size({"width": 1440, "height": 900})
    with tempfile.TemporaryDirectory() as scratch:
        for name, anchor, ready in VIEWS:
            page.goto(f"{base_url.rstrip('/')}/{anchor.format(stock=stock)}")
            wait_for(page, name, ready, timeout_ms)
            page.screenshot(path=Path(scratch) / name, full_page=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        return [Path(shutil.move(Path(scratch) / name, out_dir / name)) for name, _, _ in VIEWS]


def browse(base_url: str, stock: str, out_dir: Path) -> list[Path]:
    """Capture the views in a headless Chromium; only the dev install has Playwright."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise CommandError(
            "--views 需要 Playwright，只有 dev 依賴裡有；請在主機上用 uv run 跑"
        ) from error
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            return capture_views(browser.new_page(), base_url, stock, out_dir)
        finally:
            browser.close()


class Command(BaseCommand):
    help = (
        "Check the S5 conditions through the API, time each endpoint and write a markdown report."
    )

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--stock", default="2330")
        parser.add_argument("--repeat", type=int, default=20)
        parser.add_argument("--views", metavar="URL")

    def handle(self, *args: Any, **options: Any) -> None:
        out, stock = options["out"], options["stock"]
        client = Client(HTTP_HOST="localhost", raise_request_exception=False)
        timings = measure(client, stock, options["repeat"])
        checks = acceptance_checks(client, stock, timings)
        text = report_text(stock, checks, timings)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        self.stdout.write(text, ending="")
        failed = [check.name for check in checks if not check.passed]
        if failed:
            raise CommandError("未通過：" + "、".join(failed))
        if options["views"]:
            browse(options["views"], stock, out.parent)
