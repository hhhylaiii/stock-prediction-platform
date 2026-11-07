import pytest
from playwright.sync_api import Locator, Page, expect
from pytest_django.live_server_helper import LiveServer

from apps.etl.models import DataQualityLog
from tests.e2e.conftest import XSS_NAME, Seed

pytestmark = pytest.mark.e2e


def log(seed: Seed, check: str, severity: str, detail: dict[str, object], back: int = 1) -> None:
    DataQualityLog.objects.create(
        date=seed.sessions[-back], check_name=check, severity=severity, detail=detail
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
    alerts = [
        {
            "run_id": XSS_NAME,
            "objective": "rank",
            "horizon": 20,
            "metric": "ic",
            "reason": "below_threshold",
            "value": 0.0123,
            "threshold": 0.02,
            "days": 60,
        },
        {
            "run_id": "quantile-h20",
            "objective": "quantile",
            "horizon": 20,
            "metric": "coverage",
            "reason": "outside_range",
            "value": 0.652,
            "threshold": [0.7, 0.9],
            "days": 60,
        },
        {
            "run_id": "prob-h20",
            "objective": "classification",
            "horizon": 20,
            "metric": "brier",
            "reason": "missing_reference",
            "value": 0.231,
            "threshold": None,
            "days": 60,
        },
    ]
    log(seeded, "model_alert", "warning", {"runs": alerts})
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
    expect(alert.locator("i")).to_have_class("warn")
    lines = alert.locator(".run")
    expect(lines).to_have_count(3)
    expect(lines.nth(0)).to_have_text(
        f"{XSS_NAME}（排名 20 日）：IC 低於門檻，近 60 日 0.012，門檻 0.020"
    )
    expect(lines.nth(1)).to_have_text(
        "quantile-h20（報酬區間 20 日）：覆蓋率超出範圍，近 60 日 65.2%，門檻 70.0%–90.0%"
    )
    expect(lines.nth(2)).to_have_text("prob-h20（上漲機率 20 日）：Brier 缺參考值，近 60 日 0.231")
    expect(page.locator("#model-checks img")).to_have_count(0)
    expect(page.locator("#model-checks-sub")).to_contain_text(checked.last_session.isoformat())


ALERT = {
    "objective": "rank",
    "horizon": 20,
    "metric": "ic",
    "reason": "below_threshold",
    "value": 0.0123,
    "threshold": 0.02,
    "days": 60,
}
ALERT_TEXT = "（排名 20 日）：IC 低於門檻，近 60 日 0.012，門檻 0.020"
UNSET = {
    "grace_start": None,
    "grace_end": None,
    "restored_on": None,
    "reset_rows": None,
    "reset_needed": None,
}


def promotion_lines(page: Page) -> Locator:
    return page.locator("#model-checks .check").filter(has_text="換模").locator(".run")


def test_alert_entries_say_how_each_run_was_judged(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    judged = {**ALERT, **UNSET, "run_id": "rank-a", "mode": "judged", "own_rows": 45}
    graced = {
        **ALERT,
        **UNSET,
        "run_id": "rank-b",
        "mode": "grace",
        "own_rows": 12,
        "mostly_predecessors": True,
        "grace_start": "2026-10-01",
        "grace_end": "2026-10-28",
    }
    reset = {
        **ALERT,
        **UNSET,
        "run_id": "rank-c",
        "mode": "reset",
        "own_rows": 60,
        "mostly_predecessors": False,
        "restored_on": "2026-10-20",
        "reset_rows": 12,
        "reset_needed": 60,
    }
    log(seeded, "model_alert", "warning", {"runs": [judged, graced, reset]})

    page.goto(f"{live_server.url}/#models")

    lines = page.locator("#model-checks .check").filter(has_text="模型告警").locator(".run")
    expect(lines).to_have_count(3)
    expect(lines.nth(0)).to_have_text(f"rank-a{ALERT_TEXT}；窗口 45／60 列為本 run（75%）")
    expect(lines.nth(1)).to_have_text(
        f"rank-b{ALERT_TEXT}；窗口 12／60 列為本 run（20%），多為前任 run 的列"
        "；寬限期 2026-10-01–2026-10-28，只告警不降級"
    )
    expect(lines.nth(2)).to_have_text(
        f"rank-c{ALERT_TEXT}；窗口 60／60 列為本 run（100%）"
        "；此 slot 在 restore 重設期（2026-10-20 restore，已累積 12／60 列），只告警不降級"
    )


def test_a_mode_without_its_dates_still_says_it_only_warns(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    graced = {**ALERT, **UNSET, "run_id": "rank-b", "mode": "grace", "grace_start": "2026-10-01"}
    reset = {**ALERT, **UNSET, "run_id": "rank-c", "mode": "reset", "reset_rows": 12}
    log(seeded, "model_alert", "warning", {"runs": [graced, reset]})

    page.goto(f"{live_server.url}/#models")

    lines = page.locator("#model-checks .check").filter(has_text="模型告警").locator(".run")
    expect(lines.nth(0)).to_have_text(f"rank-b{ALERT_TEXT}；寬限期內，只告警不降級")
    expect(lines.nth(1)).to_have_text(
        f"rank-c{ALERT_TEXT}；此 slot 在 restore 重設期（已累積 12／— 列），只告警不降級"
    )


def test_a_refused_build_names_each_failed_slot_and_gate(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    failed = [
        {
            "slot": "rank 20 日",
            "run_id": "rank-n",
            "gates": [
                {"name": "agreement", "reason": "相關太低"},
                {"name": "shadow_oos", "reason": XSS_NAME},
            ],
        },
        {
            "slot": "quantile 20 日",
            "run_id": "quantile-n",
            "gates": [{"name": "data_quality", "reason": "2 天有 error"}],
        },
    ]
    detail = {"run_key": "prod-2026-11", "forced": False, "reason": None, "failed": failed}
    log(seeded, "model_promotion", "warning", detail)

    page.goto(f"{live_server.url}/#models")

    promotion = page.locator("#model-checks .check").filter(has_text="換模")
    expect(promotion.locator("div > div").first).to_contain_text("換模　")
    expect(promotion.locator(".pinned")).to_have_text(seeded.last_session.isoformat())
    expect(promotion.locator("i")).to_have_class("warn")
    lines = promotion_lines(page)
    expect(lines).to_have_count(3)
    expect(lines.nth(0)).to_have_text("換模門檻未過，續用現役模型（prod-2026-11）")
    expect(lines.nth(1)).to_have_text(
        "排名 20 日（rank-n）：與現役模型的預測相關（相關太低）"
        f"、影子模型最近 6 個月 OOS（{XSS_NAME}）"
    )
    expect(lines.nth(2)).to_have_text(
        "報酬區間 20 日（quantile-n）：訓練窗內的資料品質（2 天有 error）"
    )
    expect(page.locator("#model-checks img")).to_have_count(0)
    expect(page.locator("#quality")).not_to_contain_text("換模")


def promotion(run_key: str, gates: list[dict[str, object]]) -> dict[str, object]:
    failed = [{"slot": "rank 20 日", "run_id": "rank-n", "gates": gates}]
    return {"run_key": run_key, "forced": False, "reason": None, "failed": failed}


def test_the_latest_promotion_stays_on_top_for_thirty_sessions(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    gate = {
        "name": "agreement",
        "reason": "相關太低",
        "value": 0.62,
        "threshold": ">= 0.7",
        "sample": 1842,
        "sample_unit": "stocks",
    }
    log(seeded, "model_promotion", "warning", promotion("prod-old", [gate]), back=12)
    log(seeded, "model_promotion", "warning", promotion("prod-new", [gate]), back=5)
    log(seeded, "model_promotion", "warning", promotion("prod-gone", [gate]), back=31)
    clean = {"share": 0.0, "flagged": 0, "n_stocks": 1842, "stocks": []}
    log(seeded, "output_width_ratio", "info", clean)

    page.goto(f"{live_server.url}/#models")

    checks = page.locator("#model-checks .check")
    expect(checks).to_have_count(2)
    expect(checks.nth(0).locator(".pinned")).to_have_text(
        f"{seeded.sessions[-5].isoformat()}，近 30 個交易日共 2 筆"
    )
    lines = checks.nth(0).locator(".run")
    expect(lines.nth(0)).to_have_text("換模門檻未過，續用現役模型（prod-new）")
    expect(lines.nth(1)).to_have_text(
        "排名 20 日（rank-n）：與現役模型的預測相關"
        "（相關太低，實際 0.620，門檻 >= 0.7，樣本 1,842 檔）"
    )
    expect(checks.nth(1)).to_contain_text("區間寬度比異常")
    expect(page.locator("#model-checks-sub")).to_contain_text(seeded.last_session.isoformat())


def test_a_forced_swap_leads_with_its_reason(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    gates = [
        {
            "name": "gain_overlap",
            "reason": "重疊太少",
            "value": 4.0,
            "threshold": 6,
            "sample": 10,
            "sample_unit": "features",
        },
        {
            "name": "shadow_oos",
            "reason": "",
            "value": -0.01,
            "threshold": 0,
            "sample": 120,
            "sample_unit": "days",
        },
    ]
    failed = [{"slot": "rank 5 日", "run_id": "rank5-n", "gates": gates}]
    detail = {"run_key": "prod-2026-11", "forced": True, "reason": XSS_NAME, "failed": failed}
    log(seeded, "model_promotion", "warning", detail)

    page.goto(f"{live_server.url}/#models")

    lines = promotion_lines(page)
    expect(lines).to_have_count(2)
    expect(lines.nth(0)).to_have_text(f"強制換模：{XSS_NAME}（prod-2026-11）")
    expect(lines.nth(1)).to_have_text(
        "排名 5 日（rank5-n）：與現役模型的重要特徵重疊（重疊太少，實際 4，門檻 6，樣本 10 個特徵）"
        "、影子模型最近 6 個月 OOS（實際 -0.010，門檻 0，樣本 120 天）"
    )
    expect(page.locator("#model-checks img")).to_have_count(0)


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


def test_the_block_reads_the_last_day_monitored(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    clean = {"share": 0.0, "flagged": 0, "n_stocks": 1842, "stocks": []}
    log(seeded, "output_width_ratio", "info", clean, back=2)
    edge = {"share": 0.1, "flagged": 184, "n_stocks": 1842, "stocks": ["2330"]}
    log(seeded, "output_odds_vs_q50", "warning", edge, back=2)
    log(seeded, "output_width_ratio", "info", clean, back=3)

    page.goto(f"{live_server.url}/#models")

    expect(page.locator("#model-checks-sub")).to_contain_text(seeded.sessions[-2].isoformat())
    rows = page.locator("#model-checks .check")
    expect(rows).to_have_count(2)
    expect(rows.filter(has_text="區間寬度比異常")).to_contain_text("沒有標記")
    odds = rows.filter(has_text="機率與中位數矛盾")
    expect(odds).to_contain_text("少量標記，184／1,842 檔")
    expect(odds.locator("i")).not_to_have_class("warn")


def test_a_model_check_error_fails_the_strip_like_the_gate(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    log(seeded, "output_width_ratio", "error", {"share": 0.05, "flagged": 5, "n_stocks": 100})

    page.goto(f"{live_server.url}/#models")

    expect(page.locator("#status")).to_contain_text("品質檢查未通過")
    expect(page.locator("#quality-sub")).to_contain_text("模型輸出檢查有錯誤。")
    expect(page.locator("#quality-days .trow").first).to_contain_text("模型輸出檢查有錯誤")
    expect(page.locator("#quality-days .trow").first).not_to_contain_text("最嚴重")
    expect(page.locator("#model-checks .check")).to_contain_text("錯誤")
    expect(page.locator("#model-checks .check i")).to_have_class("warn")


def test_the_block_reads_the_alert_share_from_the_api(
    page: Page, live_server: LiveServer, checked: Seed, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.api.ops.ALERT_SHARE", 0.2)

    page.goto(f"{live_server.url}/#models")

    width = page.locator("#model-checks .check").filter(has_text="區間寬度比異常")
    expect(width).to_contain_text("少量標記，220／1,842 檔")
    expect(width.locator("i")).not_to_have_class("warn")
    expect(page.locator("#model-checks-sub")).to_contain_text("超過 20% 才算警告")
