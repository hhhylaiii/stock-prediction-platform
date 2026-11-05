import pytest
from playwright.sync_api import Page, expect
from pytest_django.live_server_helper import LiveServer

from apps.etl.models import DataQualityLog
from tests.e2e.conftest import XSS_NAME, Seed

pytestmark = pytest.mark.e2e


def log(seed: Seed, check: str, severity: str, detail: dict[str, object]) -> None:
    DataQualityLog.objects.create(
        date=seed.last_session, check_name=check, severity=severity, detail=detail
    )


@pytest.fixture
def checked(seeded: Seed) -> Seed:
    """Every kind of model output check on the last session, all written as warnings."""
    log(
        seeded,
        "output_width_ratio",
        "warning",
        {"share": 0.12, "flagged": 220, "n_stocks": 1842, "stocks": ["1101"]},
    )
    log(
        seeded,
        "output_odds_vs_q50",
        "warning",
        {"share": 0.0065, "flagged": 12, "n_stocks": 1842, "stocks": ["2330"]},
    )
    log(
        seeded,
        "monitor_unscored",
        "warning",
        {"runs": [{"run_id": "rank-h20", "date": "2026-10-01", "horizon": 20}] * 2},
    )
    log(seeded, "model_alert", "warning", {"reason": "below_threshold", "run_id": XSS_NAME})
    return seeded


def test_model_output_checks_have_their_own_block(
    page: Page, live_server: LiveServer, checked: Seed
) -> None:
    page.goto(f"{live_server.url}/#models")

    rows = page.locator("#model-checks .check")
    expect(rows).to_have_count(4)
    width = rows.filter(has_text="區間寬度比異常")
    expect(width).to_contain_text("220／1,842 檔")
    expect(width.locator("i")).to_have_class("warn")
    odds = rows.filter(has_text="機率與中位數矛盾")
    expect(odds).to_contain_text("少量標記，12／1,842 檔")
    expect(odds.locator("i")).not_to_have_class("warn")
    unscored = rows.filter(has_text="到期預測缺 label")
    expect(unscored).to_contain_text("2 筆")
    expect(unscored.locator("i")).to_have_class("warn")
    alert = rows.filter(has_text="模型告警")
    expect(alert).to_contain_text("滾動 IC 低於門檻")
    expect(alert.locator("i")).to_have_class("warn")
    expect(page.locator("#model-checks img")).to_have_count(0)
    expect(page.locator("#model-checks-sub")).to_contain_text(checked.last_session.isoformat())


def test_model_output_checks_leave_the_data_verdict_alone(
    page: Page, live_server: LiveServer, checked: Seed
) -> None:
    page.goto(f"{live_server.url}/#models")

    expect(page.locator("#quality-sub")).to_contain_text("檢查通過，但有 1 項警告")
    expect(page.locator("#quality")).not_to_contain_text("區間寬度比異常")
    expect(page.locator("#status")).to_contain_text("品質檢查通過（1 項警告）")


def test_no_model_output_checks_yet(page: Page, live_server: LiveServer, seeded: Seed) -> None:
    page.goto(f"{live_server.url}/#models")

    expect(page.locator("#model-checks")).to_contain_text("還沒有模型輸出檢查。")
