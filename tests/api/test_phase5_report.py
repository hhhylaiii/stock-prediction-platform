import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import DatabaseError, connection, transaction
from django.test import Client

from apps.api.management.commands import phase5_report
from apps.api.management.commands.measure_api import ENDPOINTS, Timing
from apps.api.status import StatusView
from apps.market.models import StockMaster
from apps.research.models import FeatureDaily, ModelRun, Prediction, PredictionContrib
from tests.api.conftest import OCT_30, Runs, add_bars, add_predictions, seed_market

pytestmark = pytest.mark.django_db

CHECKS = [
    "最新交易日有預測",
    "20 日排名",
    "20 日預測價格",
    "20 日價格區間",
    "20 日跑贏大盤機率",
    "5 日排名、預測價格與區間",
    "2330 個股頁兩個期間",
    "2330 的特徵",
    "13 個端點都回 200",
    "p95 都在 500 ms 內",
]
TIMED = CHECKS[-2:]
URL = "http://127.0.0.1:8000"
NOV_2 = date(2026, 11, 2)


@pytest.fixture
def complete(runs: Runs, calendar_in_db: list[date]) -> Runs:
    """Every ranked stock carries each displayed field, and 2330 has its contributions."""
    add_predictions(runs.quantile, OCT_30, 20, {"6488": {"q10": -0.04, "q50": 0.02, "q90": 0.09}})
    PredictionContrib.objects.create(
        run=runs.rank20,
        stock_id="2330",
        date=OCT_30,
        horizon=20,
        feature="tech_ret_5d",
        position=1,
        contrib=0.02,
    )
    FeatureDaily.objects.create(stock_id="2330", date=OCT_30, in_universe=True, tech_ret_5d=-0.012)
    return runs


def report(tmp_path: Path, *extra: str) -> tuple[Path, str]:
    target = tmp_path / "reports" / "twp.md"
    out = StringIO()
    call_command("phase5_report", "--out", str(target), "--repeat", "1", *extra, stdout=out)
    return target, out.getvalue()


def failing(tmp_path: Path, *extra: str) -> tuple[CommandError, str]:
    with pytest.raises(CommandError) as raised:
        report(tmp_path, *extra)
    return raised.value, (tmp_path / "reports" / "twp.md").read_text()


def checks_of(text: str) -> dict[str, tuple[str, str]]:
    section = text.split("## 驗收檢查\n\n", 1)[1].split("\n\n", 1)[0]
    rows = [line.strip("| ").split(" | ") for line in section.splitlines()[2:]]
    return {name: (result, detail) for name, result, detail in rows}


def failed(text: str) -> set[str]:
    return {name for name, (result, _) in checks_of(text).items() if result == "未通過"}


def test_every_check_passes_when_each_field_is_served(complete: Runs, tmp_path: Path) -> None:
    target, printed = report(tmp_path)

    text = target.read_text()
    assert printed == text
    assert [line for line in text.splitlines() if line.startswith("## ")] == [
        "## 資料量",
        "## 驗收檢查",
        "## 端點耗時",
    ]
    assert f"資料庫 `{connection.settings_dict['NAME']}`" in text
    checks = checks_of(text)
    assert list(checks) == CHECKS
    assert {result for result, _ in checks.values()} == {"通過"}
    assert checks["最新交易日有預測"][1] == "price_date 2026-10-30，prediction_date 2026-10-30"
    assert checks["20 日排名"][1] == "缺 0 檔；date 2026-10-30，3 檔，universe 3"
    assert checks["20 日價格區間"][1] == "缺 0 檔；來源 quantile 3"
    assert checks["2330 個股頁兩個期間"][1] == "齊全的期間 5、20"
    assert checks["2330 的特徵"][1] == "date 2026-10-30，1 項貢獻，1 項有原始值"
    assert sum(line.startswith("| `/api/") for line in text.splitlines()) == len(ENDPOINTS)


def test_the_volume_counts_each_table_and_the_stock(complete: Runs, tmp_path: Path) -> None:
    target, _ = report(tmp_path)

    section = target.read_text().split("## 資料量\n\n", 1)[1].split("\n\n", 1)[0]
    assert section.splitlines() == [
        "- stock_master 4 檔。",
        "- daily_price 8 列，到 2026-10-30；2330 有 2 列。",
        "- feature_daily 1 列，到 2026-10-30；2330 的 label_daily 有 0 列。",
        "- model_run 6 筆：active 6。",
        "- prediction 24 列，2 個預測日，到 2026-10-30；prediction_contrib 1 列。",
        "- model_monitor 0 列，backtest_metric 0 列。",
        "- etl_run 0 筆，data_quality_log 0 筆。",
    ]


def test_a_missing_probability_fails_only_its_check(complete: Runs, tmp_path: Path) -> None:
    Prediction.objects.filter(run=complete.prob, stock_id="6488").delete()

    error, text = failing(tmp_path)

    assert failed(text) == {"20 日跑贏大盤機率"}
    assert checks_of(text)["20 日跑贏大盤機率"][1] == "缺 1 檔"
    assert str(error) == "未通過：20 日跑贏大盤機率"


def test_a_five_day_row_without_a_band_fails(complete: Runs, tmp_path: Path) -> None:
    Prediction.objects.filter(run=complete.vol, stock_id="2330").delete()

    _, text = failing(tmp_path)

    assert failed(text) == {"5 日排名、預測價格與區間", "2330 個股頁兩個期間"}
    assert checks_of(text)["2330 個股頁兩個期間"][1] == "齊全的期間 20"


def test_a_twenty_day_row_without_a_band_fails_only_the_bands(
    complete: Runs, tmp_path: Path
) -> None:
    Prediction.objects.filter(run=complete.quantile, stock_id="6488").update(q10=None, q90=None)
    Prediction.objects.filter(run=complete.vol, stock_id="6488").delete()

    _, text = failing(tmp_path)

    assert failed(text) == {"20 日價格區間", "5 日排名、預測價格與區間"}
    assert checks_of(text)["20 日價格區間"][1] == "缺 1 檔；來源 None 1、quantile 2"


def test_a_ranked_stock_without_a_rank_fails_the_ranking(complete: Runs, tmp_path: Path) -> None:
    Prediction.objects.filter(run=complete.rank20, stock_id="2317", date=OCT_30).update(
        rank=None, pct=None
    )

    _, text = failing(tmp_path)

    assert failed(text) == {"20 日排名"}
    assert checks_of(text)["20 日排名"][1] == "缺 1 檔；date 2026-10-30，3 檔，universe 3"


def test_a_page_that_fails_mid_scan_leaves_no_table(
    complete: Runs, tmp_path: Path, settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.API_MAX_PAGE_SIZE = 2
    real_get = Client.get

    def broken(self: Client, path: str, *args: Any, **kwargs: Any) -> Any:
        response = real_get(self, path, *args, **kwargs)
        if path.endswith("&page=2"):
            response.status_code, response.content = 500, b""
        return response

    monkeypatch.setattr(Client, "get", broken)

    _, text = failing(tmp_path)

    lacking = {"20 日排名", "20 日預測價格", "20 日價格區間", "20 日跑贏大盤機率"}
    assert failed(text) == lacking | {"5 日排名、預測價格與區間"}
    assert {checks_of(text)[name][1].split("；")[0] for name in lacking} == {"沒有最新交易日的表"}


def test_a_stock_missing_its_twenty_day_odds_lacks_that_horizon(
    complete: Runs, tmp_path: Path
) -> None:
    Prediction.objects.filter(run=complete.prob, stock_id="2330").delete()

    _, text = failing(tmp_path)

    assert failed(text) == {"20 日跑贏大盤機率", "2330 個股頁兩個期間"}
    assert checks_of(text)["2330 個股頁兩個期間"][1] == "齊全的期間 5"


def test_predictions_a_session_behind_the_prices_are_not_the_latest(
    complete: Runs, tmp_path: Path
) -> None:
    for code, close in {"2330": 1010.0, "2317": 201.0, "6488": 505.0, "8299": 810.0}.items():
        add_bars(code, [NOV_2], [close])

    _, text = failing(tmp_path)

    assert failed(text) == set(CHECKS) - set(TIMED)
    assert (
        checks_of(text)["最新交易日有預測"][1]
        == "price_date 2026-11-02，prediction_date 2026-10-30，沒跟上的期間 5 日、20 日"
    )
    assert checks_of(text)["20 日預測價格"][1] == "沒有最新交易日的表"
    assert checks_of(text)["2330 個股頁兩個期間"][1] == "齊全的期間 無"


def test_features_without_a_raw_value_fail(complete: Runs, tmp_path: Path) -> None:
    FeatureDaily.objects.all().delete()

    _, text = failing(tmp_path)

    assert failed(text) == {"2330 的特徵"}
    assert checks_of(text)["2330 的特徵"][1] == "date 2026-10-30，1 項貢獻，0 項有原始值"


def test_every_page_of_the_table_is_checked(complete: Runs, tmp_path: Path, settings: Any) -> None:
    settings.API_MAX_PAGE_SIZE = 2
    Prediction.objects.filter(run=complete.quantile, stock_id="2317").delete()

    _, text = failing(tmp_path)

    assert failed(text) == {"20 日預測價格"}
    assert checks_of(text)["20 日預測價格"][1] == "缺 1 檔"
    assert checks_of(text)["20 日排名"][1] == "缺 0 檔；date 2026-10-30，3 檔，universe 3"


def test_a_throttled_check_says_to_raise_the_rate(
    complete: Runs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_get = Client.get

    def throttled(self: Client, path: str, *args: Any, **kwargs: Any) -> Any:
        response = real_get(self, path, *args, **kwargs)
        if path == "/api/status":
            response.status_code = 429
        return response

    monkeypatch.setattr(Client, "get", throttled)

    with pytest.raises(
        CommandError, match=r"^/api/status 被 throttle 擋下（429）；.*API_THROTTLE_RATE"
    ):
        report(tmp_path)


def test_one_horizon_keeping_up_is_not_every_horizon(complete: Runs, tmp_path: Path) -> None:
    ModelRun.objects.filter(pk=complete.rank20.pk).update(status="candidate")

    _, text = failing(tmp_path)

    assert "最新交易日有預測" in failed(text)
    assert (
        checks_of(text)["最新交易日有預測"][1]
        == "price_date 2026-10-30，prediction_date 2026-10-30，沒跟上的期間 20 日"
    )


def test_prices_without_predictions_fail_every_prediction_check(
    calendar_in_db: list[date], tmp_path: Path
) -> None:
    seed_market()

    error, text = failing(tmp_path)

    assert failed(text) == set(CHECKS) - set(TIMED)
    assert (
        checks_of(text)["最新交易日有預測"][1]
        == "price_date 2026-10-30，prediction_date None，沒跟上的期間 5 日、20 日"
    )
    assert str(error).startswith("未通過：最新交易日有預測、20 日排名")


def test_two_missing_dates_are_not_the_same_day(calendar_in_db: list[date], tmp_path: Path) -> None:
    _, text = failing(tmp_path)

    assert {"最新交易日有預測", "20 日排名", "5 日排名、預測價格與區間"} <= failed(text)
    assert (
        checks_of(text)["最新交易日有預測"][1]
        == "price_date None，prediction_date None，沒跟上的期間 5 日、20 日"
    )


def test_a_non_200_status_and_a_slow_p95_fail_the_timing_checks(
    complete: Runs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def slow(*args: Any) -> list[Timing]:
        fast = [Timing(path, "200", 1.0, 2.0, 3.0) for path in ENDPOINTS[:10]]
        return [
            *fast,
            Timing("/api/etl/runs", "200/429×2", 1.0, 2.0, 3.0),
            Timing("/api/data-quality", "200", 300.0, 612.0, 700.0),
            Timing("/api/data-quality/corrections", "200", 1.0, 2.0, 3.0),
        ]

    monkeypatch.setattr(phase5_report, "measure", slow)

    _, text = failing(tmp_path)

    assert failed(text) == set(TIMED)
    assert checks_of(text)["13 個端點都回 200"][1] == "`/api/etl/runs` 200/429×2"
    assert checks_of(text)["p95 都在 500 ms 內"][1] == "最大 p95 612.0 ms（`/api/data-quality`）"


def test_a_server_error_is_reported_rather_than_raised(
    complete: Runs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(StatusView, "get", broken)

    _, text = failing(tmp_path)

    assert failed(text) == set(CHECKS) - {"p95 都在 500 ms 內"}
    assert checks_of(text)["13 個端點都回 200"][1] == "`/api/status` 500"
    assert checks_of(text)["最新交易日有預測"][1] == "price_date None，prediction_date None"


def test_the_report_runs_inside_a_read_only_transaction(complete: Runs, tmp_path: Path) -> None:
    with connection.cursor() as cursor:
        cursor.execute("SET CONSTRAINTS ALL IMMEDIATE")
        cursor.execute("SET TRANSACTION READ ONLY")

    report(tmp_path)

    with pytest.raises(DatabaseError, match="read-only"), transaction.atomic():
        StockMaster.objects.create(stock_id="9998", name="寫入", market="twse", industry="其他業")


def test_the_report_never_imports_playwright_without_views(
    complete: Runs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "playwright", None)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    importlib.reload(phase5_report)

    target, _ = report(tmp_path)

    assert target.exists()


def test_views_without_playwright_say_where_to_run_them(
    complete: Runs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)

    error, text = failing(tmp_path, "--views", URL)

    assert "Playwright" in str(error)
    assert "uv run" in str(error)
    assert failed(text) == set()


class FakeBrowser:
    def __init__(self) -> None:
        self.closed = False

    def new_page(self) -> str:
        return "page"

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def browser(monkeypatch: pytest.MonkeyPatch) -> FakeBrowser:
    fake = FakeBrowser()

    @contextmanager
    def sync_playwright() -> Iterator[SimpleNamespace]:
        yield SimpleNamespace(chromium=SimpleNamespace(launch=lambda: fake))

    monkeypatch.setattr("playwright.sync_api.sync_playwright", sync_playwright)
    return fake


def test_views_open_one_browser_beside_the_report_and_close_it(
    complete: Runs, tmp_path: Path, browser: FakeBrowser, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = []

    def capture(page: str, base_url: str, stock: str, out_dir: Path) -> list[Path]:
        seen.append((page, base_url, stock, out_dir))
        return [out_dir / "today.png"]

    monkeypatch.setattr(phase5_report, "capture_views", capture)

    target, _ = report(tmp_path, "--views", URL, "--stock", "2330")

    assert seen == [("page", URL, "2330", target.parent)]
    assert browser.closed


def test_a_view_that_never_draws_still_closes_the_browser(
    complete: Runs, tmp_path: Path, browser: FakeBrowser, monkeypatch: pytest.MonkeyPatch
) -> None:
    def capture(*args: Any) -> list[Path]:
        raise CommandError("today.png：等不到 #rows .trow")

    monkeypatch.setattr(phase5_report, "capture_views", capture)

    error, _ = failing(tmp_path, "--views", URL)

    assert str(error) == "today.png：等不到 #rows .trow"
    assert browser.closed


def test_views_wait_until_every_check_passes(
    runs: Runs, calendar_in_db: list[date], tmp_path: Path, browser: FakeBrowser
) -> None:
    error, _ = failing(tmp_path, "--views", URL)

    assert str(error).startswith("未通過：")
    assert not browser.closed
