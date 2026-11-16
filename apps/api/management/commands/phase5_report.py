"""Write the S5 report: data volume, the SPEC section 9 checks and the endpoint timings."""

import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError, CommandParser
from django.db import connection
from django.db.models import Max, Model
from django.test import Client
from django.utils import timezone

from apps.api.management.commands.measure_api import ENDPOINTS, HEADER, PATHS, Timing, measure
from apps.api.serving import Slot, roles_for
from apps.etl.models import DataQualityLog, EtlRun
from apps.market.models import DailyPrice, StockMaster
from apps.research.models import (
    OUTPUT_HORIZONS,
    BacktestMetric,
    FeatureDaily,
    LabelDaily,
    ModelMonitor,
    ModelObjective,
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

PASSED, FAILED, NOT_LIVE = "通過", "未通過", "未上線"
HORIZONS = sorted(OUTPUT_HORIZONS, reverse=True)

Row = dict[str, Any]


@dataclass(frozen=True)
class Check:
    name: str
    result: str
    detail: str


def judged(name: str, passed: bool, detail: str) -> Check:
    return Check(name, PASSED if passed else FAILED, detail)


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


FIELDS: tuple[tuple[str, tuple[str, ...], Callable[[Row], bool]], ...] = (
    ("排名", ("rank",), ranked),
    ("預測價格", ("rank", "return"), priced),
    ("價格區間", ("rank",), banded),
    ("跑贏大盤機率", ("rank", "probability"), likely),
)


def gate_text(gate: dict[str, Any]) -> str:
    value = gate.get("value")
    shown = f"{value:.4f}" if isinstance(value, int | float) else "—"
    return (
        f"{gate.get('name')}：{gate.get('reason')}，實際 {shown}，門檻 {gate.get('threshold')}，"
        f"樣本 {gate.get('sample')} {gate.get('sample_unit')}"
    )


@dataclass(frozen=True)
class Live:
    """The page slots /api/models lists as serving; None when unreadable, so every one counts."""

    serving: frozenset[Slot] | None
    held: dict[Slot, dict[str, Any]]

    def missing(self, slots: Iterable[Slot]) -> list[Slot]:
        wanted = list(dict.fromkeys(slots))
        return [] if self.serving is None else [s for s in wanted if s not in self.serving]

    def why(self, slot: Slot) -> str:
        name = f"{ModelObjective(slot.objective).label} {slot.horizon} 日沒有上線中的模型"
        entry = self.held.get(slot)
        if entry is None:
            return name + "，也沒有換模紀錄"
        gates = "、".join(gate_text(gate) for gate in entry.get("gates") or [])
        return f"{name}：{entry.get('run_key')} 沒過換模門檻（{gates}）"

    def not_live(self, name: str, missing: list[Slot]) -> Check:
        return Check(name, NOT_LIVE, "未上線：" + "；".join(self.why(slot) for slot in missing))


def live_models(client: Client) -> Live:
    body = fetch(client, "/api/models")
    rows, meta = body.get("data"), body.get("meta") or {}
    if body.get("success") is not True or not isinstance(rows, list):
        return Live(None, {})
    entries = meta.get("not_live") or []
    held = {Slot(entry["objective"], entry["horizon"]): entry for entry in entries}
    return Live(frozenset(Slot(row["objective"], row["horizon"]) for row in rows), held)


def needed(horizon: int, needs: tuple[str, ...]) -> list[Slot] | None:
    """The slots a field reads at the horizon; None when the horizon has no such field."""
    roles = roles_for(horizon)
    return [roles[role] for role in needs] if all(role in roles for role in needs) else None


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
        return judged(name, self.fresh and not missing, detail + extra)


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


def sources_of(table: Table) -> str:
    sources = Counter(str(row.get("interval_source")) for row in table.rows)
    return "、".join(f"{name} {n}" for name, n in sorted(sources.items())) or "無"


def table_checks(client: Client, day: str | None, live: Live) -> list[Check]:
    """Each field of each horizon's table, unless a model it reads is not serving."""
    checks = []
    for horizon in HORIZONS:
        table = served(client, horizon, day)
        extras = {"排名": f"；{table.head}", "價格區間": f"；來源 {sources_of(table)}"}
        for label, needs, test in FIELDS:
            slots = needed(horizon, needs)
            if slots is None:
                continue
            name, missing = f"{horizon} 日{label}", live.missing(slots)
            if missing:
                checks.append(live.not_live(name, missing))
            else:
                checks.append(table.check(name, test, extras.get(label, "")))
    return checks


def page_check(stock: str, horizon: int, entry: Row, day: str | None, live: Live) -> Check:
    """The stock page's horizon carries every field whose models serve."""
    name = f"{stock} 個股頁 {horizon} 日"
    missing = live.missing([roles_for(horizon)["rank"]])
    if missing:
        return live.not_live(name, missing)
    fields = [(label, needed(horizon, needs), test) for label, needs, test in FIELDS]
    shown = [(label, test) for label, slots, test in fields if slots and not live.missing(slots)]
    off = [label for label, slots, _ in fields if slots and live.missing(slots)]
    note = f"；未上線的欄位 {'、'.join(off)}" if off else ""
    row = entry.get("prediction") or {}
    if day is None or entry.get("date") != day or not row:
        return judged(name, False, f"沒有最新交易日的預測（date {entry.get('date')}）{note}")
    lacking = [label for label, test in shown if not test(row)]
    return judged(name, not lacking, ("缺 " + "、".join(lacking) if lacking else "齊全") + note)


def feature_check(client: Client, stock: str, horizon: int, day: str | None, live: Live) -> Check:
    name = f"{stock} 的 {horizon} 日特徵"
    missing = live.missing([roles_for(horizon)["rank"]])
    if missing:
        return live.not_live(name, missing)
    features = fetch(client, f"/api/stocks/{stock}/features?horizon={horizon}")
    items = (features.get("data") or {}).get("items") or []
    feature_day = (features.get("meta") or {}).get("date")
    valued = sum(item.get("value") is not None for item in items)
    return judged(
        name,
        day is not None and feature_day == day and valued > 0,
        f"date {feature_day}，{len(items)} 項貢獻，{valued} 項有原始值",
    )


def stock_checks(client: Client, stock: str, day: str | None, live: Live) -> list[Check]:
    forecast = fetch(client, f"/api/stocks/{stock}/predictions")
    entries = (forecast.get("data") or {}).get("horizons") or []
    by_horizon = {entry.get("horizon"): entry for entry in entries}
    return [
        *(page_check(stock, h, by_horizon.get(h) or {}, day, live) for h in HORIZONS),
        *(feature_check(client, stock, h, day, live) for h in HORIZONS),
    ]


def timing_checks(timings: list[Timing]) -> list[Check]:
    odd = [f"`{timing.path}` {timing.status}" for timing in timings if timing.status != "200"]
    slowest = max(timings, key=lambda timing: timing.p95)
    return [
        judged(
            f"{len(ENDPOINTS)} 個端點的 {len(PATHS)} 條路徑都回 200",
            not odd and len(timings) == len(PATHS),
            "、".join(odd) or "全部 200",
        ),
        judged(
            f"p95 都在 {LIMIT_MS} ms 內",
            slowest.p95 < LIMIT_MS,
            f"最大 p95 {slowest.p95:.1f} ms（`{slowest.path}`）",
        ),
    ]


def fresh_check(status: dict[str, Any], live: Live) -> Check:
    """Every horizon whose rank model serves has caught up; horizons without one are named."""
    day, predicted = status.get("price_date"), status.get("prediction_date")
    horizons = status.get("horizons") or []
    serving = [e for e in horizons if not live.missing([roles_for(e["horizon"])["rank"]])]
    behind = [f"{entry['horizon']} 日" for entry in serving if not entry["ready"]]
    off = [f"{entry['horizon']} 日" for entry in horizons if entry not in serving]
    return judged(
        "最新交易日有預測",
        day is not None and bool(serving) and not behind,
        f"price_date {day}，prediction_date {predicted}"
        + (f"，沒跟上的期間 {'、'.join(behind)}" if behind else "")
        + (f"，未上線的期間 {'、'.join(off)}" if off else ""),
    )


def acceptance_checks(client: Client, stock: str, timings: list[Timing]) -> list[Check]:
    status = fetch(client, "/api/status").get("data") or {}
    day, live = status.get("price_date"), live_models(client)
    return [
        fresh_check(status, live),
        *table_checks(client, day, live),
        *stock_checks(client, stock, day, live),
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
        *(f"| {c.name} | {c.result} | {c.detail} |" for c in checks),
        "",
        "、".join(
            f"{result} {sum(c.result == result for c in checks)} 項"
            for result in (PASSED, FAILED, NOT_LIVE)
        )
        + "。",
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
        failed = [check.name for check in checks if check.result == FAILED]
        if failed:
            raise CommandError("未通過：" + "、".join(failed))
        if options["views"]:
            browse(options["views"], stock, out.parent)
