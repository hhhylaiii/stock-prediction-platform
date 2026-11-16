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
from apps.api.management.commands.measure_api import ENDPOINTS, PATHS, Timing
from apps.api.model_runs import ModelListView
from apps.api.status import StatusView
from apps.etl.models import DataQualityLog
from apps.market.models import StockMaster
from apps.research.models import FeatureDaily, ModelRun, Prediction, PredictionContrib
from tests.api.conftest import OCT_30, Runs, add_bars, add_predictions, seed_market

pytestmark = pytest.mark.django_db

TIMING = f"{len(ENDPOINTS)} 個端點的 {len(PATHS)} 條路徑都回 200"
TWENTY = [
    "20 日排名",
    "20 日預測價格",
    "20 日價格區間",
    "20 日跑贏大盤機率",
]
FIVE = ["5 日排名", "5 日預測價格", "5 日價格區間"]
CHECKS = [
    "最新交易日有預測",
    *TWENTY,
    *FIVE,
    "2330 個股頁 20 日",
    "2330 個股頁 5 日",
    "2330 的 20 日特徵",
    "2330 的 5 日特徵",
    TIMING,
    "p95 都在 500 ms 內",
]
TIMED = CHECKS[-2:]
TWENTY_ALL = {*TWENTY, "2330 個股頁 20 日", "2330 的 20 日特徵"}
SHADOW_GATE = {
    "name": "shadow_oos",
    "reason": "未達門檻",
    "value": -0.013045199326777494,
    "threshold": "rank IC > 0",
    "sample": 105,
    "sample_unit": "days",
}
HELD_RANK = (
    "未上線：排名 20 日沒有上線中的模型：rank-2026-11 沒過換模門檻："
    "影子模型最近 6 個月 OOS（未達門檻，實際 -0.013，門檻 rank IC > 0，樣本 105 天）"
)
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
    PredictionContrib.objects.create(
        run=runs.rank5,
        stock_id="2330",
        date=OCT_30,
        horizon=5,
        feature="tech_ret_5d",
        position=1,
        contrib=0.01,
    )
    FeatureDaily.objects.create(stock_id="2330", date=OCT_30, in_universe=True, tech_ret_5d=-0.012)
    return runs


def hold_back_rank_twenty(runs: Runs, gates: list[dict[str, Any]] | None = None) -> None:
    """The 20-day rank run stays a candidate and its scope's newest promotion row says why."""
    ModelRun.objects.filter(pk=runs.rank20.pk).update(status="candidate")
    gates = [SHADOW_GATE] if gates is None else gates
    held = {"objective": "rank", "horizon": 20, "run_id": "rank-2026-11.h20", "gates": gates}
    detail = {"scope": "rank", "run_key": "rank-2026-11", "held_slots": [held]}
    DataQualityLog.objects.create(
        date=OCT_30, check_name="model_promotion", severity="warning", detail=detail
    )


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


def not_live(text: str) -> set[str]:
    return {name for name, (result, _) in checks_of(text).items() if result == "未上線"}


def summary_of(text: str) -> str:
    return text.split("## 驗收檢查\n\n", 1)[1].split("\n\n", 2)[1]


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
    assert summary_of(text) == "通過 14 項、未通過 0 項、未上線 0 項。"
    assert checks["最新交易日有預測"][1] == "price_date 2026-10-30，prediction_date 2026-10-30"
    assert checks["20 日排名"][1] == "缺 0 檔；date 2026-10-30，3 檔，universe 3"
    assert checks["20 日價格區間"][1] == "缺 0 檔；來源 quantile 3"
    assert checks["5 日排名"][1] == "缺 0 檔；date 2026-10-30，3 檔，universe 3"
    assert checks["5 日價格區間"][1] == "缺 0 檔；來源 volatility 3"
    assert checks["2330 個股頁 20 日"][1] == "齊全"
    assert checks["2330 個股頁 5 日"][1] == "齊全"
    assert checks["2330 的 20 日特徵"][1] == "date 2026-10-30，1 項貢獻，1 項有原始值"
    assert checks["2330 的 5 日特徵"][1] == "date 2026-10-30，1 項貢獻，1 項有原始值"
    assert sum(line.startswith("| `/api/") for line in text.splitlines()) == len(PATHS)
    assert "回的是空表" not in text


def test_the_volume_counts_each_table_and_the_stock(complete: Runs, tmp_path: Path) -> None:
    target, _ = report(tmp_path)

    section = target.read_text().split("## 資料量\n\n", 1)[1].split("\n\n", 1)[0]
    assert section.splitlines() == [
        "- stock_master 4 檔。",
        "- daily_price 8 列，到 2026-10-30；2330 有 2 列。",
        "- feature_daily 1 列，到 2026-10-30；2330 的 label_daily 有 0 列。",
        "- model_run 6 筆：active 6。",
        "- prediction 24 列，2 個預測日，到 2026-10-30；prediction_contrib 2 列。",
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

    assert failed(text) == {"5 日價格區間", "2330 個股頁 5 日"}
    assert checks_of(text)["2330 個股頁 5 日"][1] == "缺 價格區間"


def test_a_twenty_day_row_without_a_band_fails_only_the_bands(
    complete: Runs, tmp_path: Path
) -> None:
    Prediction.objects.filter(run=complete.quantile, stock_id="6488").update(q10=None, q90=None)
    Prediction.objects.filter(run=complete.vol, stock_id="6488").delete()

    _, text = failing(tmp_path)

    assert failed(text) == {"20 日價格區間", "5 日價格區間"}
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

    lacking = {*TWENTY, *FIVE}
    assert failed(text) == lacking
    assert {checks_of(text)[name][1].split("；")[0] for name in lacking} == {"沒有最新交易日的表"}


def test_a_stock_missing_its_twenty_day_odds_lacks_that_horizon(
    complete: Runs, tmp_path: Path
) -> None:
    Prediction.objects.filter(run=complete.prob, stock_id="2330").delete()

    _, text = failing(tmp_path)

    assert failed(text) == {"20 日跑贏大盤機率", "2330 個股頁 20 日"}
    assert checks_of(text)["2330 個股頁 20 日"][1] == "缺 跑贏大盤機率"


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
    assert checks_of(text)["2330 個股頁 20 日"][1] == "沒有最新交易日的預測（date 2026-10-30）"


def test_features_without_a_raw_value_fail(complete: Runs, tmp_path: Path) -> None:
    FeatureDaily.objects.all().delete()

    _, text = failing(tmp_path)

    assert failed(text) == {"2330 的 20 日特徵", "2330 的 5 日特徵"}
    assert checks_of(text)["2330 的 5 日特徵"][1] == "date 2026-10-30，1 項貢獻，0 項有原始值"


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
    Prediction.objects.filter(run=complete.rank20, date=OCT_30).delete()

    _, text = failing(tmp_path)

    assert {"最新交易日有預測", "20 日排名"} <= failed(text)
    assert (
        checks_of(text)["最新交易日有預測"][1]
        == "price_date 2026-10-30，prediction_date 2026-10-30，沒跟上的期間 20 日"
    )


def test_prices_without_predictions_fail_freshness_and_leave_the_rest_not_live(
    calendar_in_db: list[date], tmp_path: Path
) -> None:
    seed_market()

    error, text = failing(tmp_path)

    assert failed(text) == {"最新交易日有預測"}
    assert not_live(text) == set(CHECKS) - set(TIMED) - {"最新交易日有預測"}
    assert (
        checks_of(text)["最新交易日有預測"][1]
        == "price_date 2026-10-30，prediction_date None，未上線的期間 5 日、20 日"
    )
    assert checks_of(text)["20 日預測價格"][1] == (
        "未上線：排名 20 日沒有上線中的模型，也沒有換模紀錄；"
        "報酬區間 20 日沒有上線中的模型，也沒有換模紀錄"
    )
    assert "- 5 日沒有上線中的排名模型：帶 `horizon=5` 的路徑回的是空表" in text
    assert str(error) == "未通過：最新交易日有預測"


def test_two_missing_dates_are_not_the_same_day(calendar_in_db: list[date], tmp_path: Path) -> None:
    _, text = failing(tmp_path)

    assert "最新交易日有預測" in failed(text)
    assert {"20 日排名", "5 日排名"} <= not_live(text)
    assert (
        checks_of(text)["最新交易日有預測"][1]
        == "price_date None，prediction_date None，未上線的期間 5 日、20 日"
    )


def test_a_non_200_status_and_a_slow_p95_fail_the_timing_checks(
    complete: Runs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def slow(*args: Any) -> list[Timing]:
        odd = {
            "/api/etl/runs": Timing("/api/etl/runs", "200/429×2", 1.0, 2.0, 3.0),
            "/api/data-quality": Timing("/api/data-quality", "200", 300.0, 612.0, 700.0),
        }
        return [odd.get(path, Timing(path, "200", 1.0, 2.0, 3.0)) for path in PATHS]

    monkeypatch.setattr(phase5_report, "measure", slow)

    _, text = failing(tmp_path)

    assert failed(text) == set(TIMED)
    assert checks_of(text)[TIMING][1] == "`/api/etl/runs` 200/429×2"
    assert checks_of(text)["p95 都在 500 ms 內"][1] == "最大 p95 612.0 ms（`/api/data-quality`）"


def test_a_server_error_is_reported_rather_than_raised(
    complete: Runs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(StatusView, "get", broken)

    _, text = failing(tmp_path)

    assert failed(text) == set(CHECKS) - {"p95 都在 500 ms 內"}
    assert checks_of(text)[TIMING][1] == "`/api/status` 500"
    assert checks_of(text)["最新交易日有預測"][1] == "price_date None，prediction_date None"


def test_a_horizon_without_a_serving_rank_model_is_not_live_and_the_report_passes(
    complete: Runs, tmp_path: Path
) -> None:
    hold_back_rank_twenty(complete)

    target, _ = report(tmp_path)

    text = target.read_text()
    assert failed(text) == set()
    assert not_live(text) == TWENTY_ALL
    assert summary_of(text) == "通過 8 項、未通過 0 項、未上線 6 項。"
    assert checks_of(text)["20 日排名"][1] == HELD_RANK
    assert checks_of(text)["20 日預測價格"][1] == HELD_RANK
    assert checks_of(text)["2330 的 20 日特徵"][1] == HELD_RANK
    assert checks_of(text)["最新交易日有預測"] == (
        "通過",
        "price_date 2026-10-30，prediction_date 2026-10-30，未上線的期間 20 日",
    )
    assert text.endswith(
        "\n\n- 20 日沒有上線中的排名模型：帶 `horizon=20` 的路徑，以及不帶期間的預測表、特徵與歷史"
        "回的是空表，耗時只代表沒有資料時的速度。\n"
    )


def test_a_held_slot_whose_own_gates_passed_says_its_group_failed(
    complete: Runs, tmp_path: Path
) -> None:
    hold_back_rank_twenty(complete, gates=[])

    target, _ = report(tmp_path)

    assert checks_of(target.read_text())["20 日排名"][1] == (
        "未上線：排名 20 日沒有上線中的模型：rank-2026-11 沒有換上；"
        "它自己的門檻都過了，是同一組有模型沒過"
    )


def test_a_live_horizon_missing_a_rank_fails_beside_one_not_live(
    complete: Runs, tmp_path: Path
) -> None:
    hold_back_rank_twenty(complete)
    Prediction.objects.filter(run=complete.rank5, stock_id="6488").update(rank=None, pct=None)

    error, text = failing(tmp_path)

    assert failed(text) == {"5 日排名"}
    assert not_live(text) == TWENTY_ALL
    assert checks_of(text)["5 日排名"][1] == "缺 1 檔；date 2026-10-30，3 檔，universe 3"
    assert str(error) == "未通過：5 日排名"


def test_a_field_without_its_model_is_not_live_on_a_live_horizon(
    complete: Runs, tmp_path: Path
) -> None:
    ModelRun.objects.filter(pk=complete.prob.pk).update(status="candidate")

    target, _ = report(tmp_path)

    text = target.read_text()
    assert failed(text) == set()
    assert not_live(text) == {"20 日跑贏大盤機率"}
    assert checks_of(text)["20 日跑贏大盤機率"][1] == (
        "未上線：跑贏大盤機率 20 日沒有上線中的模型，也沒有換模紀錄"
    )
    assert checks_of(text)["2330 個股頁 20 日"] == ("通過", "齊全；未上線的欄位 跑贏大盤機率")


def test_a_band_without_a_return_model_still_comes_from_volatility(
    complete: Runs, tmp_path: Path
) -> None:
    ModelRun.objects.filter(pk=complete.quantile.pk).update(status="candidate")

    target, _ = report(tmp_path)

    text = target.read_text()
    assert not_live(text) == {"20 日預測價格"}
    assert checks_of(text)["20 日價格區間"] == ("通過", "缺 0 檔；來源 volatility 3")
    assert checks_of(text)["2330 個股頁 20 日"] == ("通過", "齊全；未上線的欄位 預測價格")


def test_an_unreadable_model_list_judges_every_horizon_as_serving(
    complete: Runs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hold_back_rank_twenty(complete)

    def broken(*args: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(ModelListView, "get", broken)

    _, text = failing(tmp_path)

    assert not_live(text) == set()
    assert failed(text) == TWENTY_ALL | {"最新交易日有預測", TIMING}
    assert checks_of(text)[TIMING][1] == "`/api/models` 500"


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

    def capture(page: str, base_url: str, stock: str, out_dir: Path, fan: bool) -> list[Path]:
        seen.append((page, base_url, stock, out_dir, fan))
        return [out_dir / "today.png"]

    monkeypatch.setattr(phase5_report, "capture_views", capture)

    target, _ = report(tmp_path, "--views", URL, "--stock", "2330")

    assert seen == [("page", URL, "2330", target.parent, True)]
    assert browser.closed


@pytest.mark.parametrize(
    ("offline", "fan"),
    [
        (["quantile"], False),
        (["rank20"], True),
        (["rank20", "reg5"], False),
        (["reg5"], True),
    ],
)
def test_views_wait_for_the_median_line_only_where_the_stock_page_draws_one(
    complete: Runs,
    tmp_path: Path,
    browser: FakeBrowser,
    monkeypatch: pytest.MonkeyPatch,
    offline: list[str],
    fan: bool,
) -> None:
    ModelRun.objects.filter(pk__in=[getattr(complete, name).pk for name in offline]).update(
        status="candidate"
    )
    seen = []

    def capture(page: str, base_url: str, stock: str, out_dir: Path, fan: bool) -> list[Path]:
        seen.append(fan)
        return []

    monkeypatch.setattr(phase5_report, "capture_views", capture)

    report(tmp_path, "--views", URL)

    assert seen == [fan]


def test_a_view_that_never_draws_still_closes_the_browser(
    complete: Runs, tmp_path: Path, browser: FakeBrowser, monkeypatch: pytest.MonkeyPatch
) -> None:
    def capture(*args: Any, **kwargs: Any) -> list[Path]:
        raise CommandError("today.png：等不到 #rows .trow")

    monkeypatch.setattr(phase5_report, "capture_views", capture)

    error, _ = failing(tmp_path, "--views", URL)

    assert str(error) == "today.png：等不到 #rows .trow"
    assert browser.closed


def test_views_wait_until_no_check_fails(
    runs: Runs, calendar_in_db: list[date], tmp_path: Path, browser: FakeBrowser
) -> None:
    error, _ = failing(tmp_path, "--views", URL)

    assert str(error).startswith("未通過：")
    assert not browser.closed
