from datetime import date, timedelta
from pathlib import Path

import pytest
from playwright.sync_api import Locator, Page, expect
from pytest_django.live_server_helper import LiveServer

from apps.etl.models import DataQualityLog
from apps.market.adjust import ADJ_CHECK, _merged
from apps.market.calendar import TradingDays
from apps.research import production, tasks
from apps.research.gates import SAMPLE_UNITS, Gate, SlotGates
from apps.research.management.commands.build_features import EX_DATE_CHECK
from apps.research.monitor import MODE_FIELDS, MODES, alert_problems
from apps.research.predict import Slot
from apps.research.promotion import Promoted, _failed, _note
from apps.research.runs import RunError
from tests.e2e.conftest import XSS_NAME, Seed

pytestmark = pytest.mark.e2e


def log(seed: Seed, check: str, severity: str, detail: dict[str, object], back: int = 1) -> None:
    DataQualityLog.objects.create(
        date=seed.sessions[-back], check_name=check, severity=severity, detail=detail
    )


@pytest.fixture
def checked(seeded: Seed) -> Seed:
    """Every model output check on the last session as warnings; its alerts predate the modes."""
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
    assert all(alert_problems(entry) for entry in alerts)
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
    expect(odds).to_contain_text("參考，12／1,842 檔")
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
    expect(lines.nth(2)).to_have_text(
        "prob-h20（跑贏大盤機率 20 日）：Brier 缺參考值，近 60 日 0.231"
    )
    expect(page.locator("#model-checks img")).to_have_count(0)
    expect(page.locator("#model-checks-sub")).to_contain_text(checked.last_session.isoformat())


COVERAGE = "monitor_feature_coverage"
SPARSE_LOW = {
    "feature": "fund_days_to_ex_date",
    "share": 0.0,
    "count": 0,
    "median": 0.0054,
    "low": 0.001,
    "bound": 0.0005,
    "days": 85,
    "collapsed": False,
}
COLLAPSED = {
    "feature": XSS_NAME,
    "share": 0.31,
    "count": 303,
    "median": 1.0,
    "low": 0.998,
    "bound": 0.948,
    "days": 107,
    "collapsed": True,
}
SPARSE_LINE = (
    "偏低：fund_days_to_ex_date 線上有值 0.00%（0 檔），參考中位數 0.54%、下緣 0.10%、門檻 0.05%"
)


def coverage(flagged: list[dict[str, object]], appeared: list[str]) -> dict[str, object]:
    return {
        "universe": 979,
        "features": 60,
        "judged": 58,
        "flagged": flagged,
        "unjudged": dict.fromkeys(["chip_a", "chip_b"], "empty_reference"),
        "appeared": appeared,
        "reference": {"month": 11, "years": 5, "days": 107},
    }


def test_a_coverage_error_lists_the_collapsed_features_before_the_low_ones(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    log(seeded, COVERAGE, "error", coverage([SPARSE_LOW, COLLAPSED], []))

    page.goto(f"{live_server.url}/#models")

    row = page.locator("#model-checks .check").filter(has_text="線上特徵覆蓋率")
    expect(row.locator("i")).to_have_class("warn")
    lines = row.locator(".run")
    expect(lines).to_have_count(3)
    expect(lines.nth(0)).to_have_text(
        "錯誤，1 個特徵崩落，當天不產出預測；另有 1 個偏低（判 58／60 個特徵）"
    )
    expect(lines.nth(1)).to_have_text(
        f"崩落：{XSS_NAME} 線上有值 31.00%（303 檔），參考中位數 100.00%、下緣 99.80%、門檻 94.80%"
    )
    expect(lines.nth(2)).to_have_text(SPARSE_LINE)
    expect(page.locator("#model-checks img")).to_have_count(0)


@pytest.mark.parametrize(
    ("severity", "detail", "lines"),
    [
        (
            "warning",
            coverage([SPARSE_LOW], []),
            ["1 個特徵偏低（判 58／60 個特徵）", SPARSE_LINE],
        ),
        (
            "info",
            coverage([], ["chip_a", "chip_b"]),
            ["都在參考範圍內（判 58／60 個特徵）", "參考期全空、線上有值：chip_a、chip_b"],
        ),
        (
            "warning",
            {"error": "the feature partition of 2023 is missing", "flagged": [SPARSE_LOW]},
            [
                "檢查沒有跑成：the feature partition of 2023 is missing",
                "當天先前標記：fund_days_to_ex_date",
            ],
        ),
    ],
)
def test_a_coverage_row_reads_as_low_in_range_or_not_run(
    page: Page,
    live_server: LiveServer,
    seeded: Seed,
    severity: str,
    detail: dict[str, object],
    lines: list[str],
) -> None:
    log(seeded, COVERAGE, severity, detail)

    page.goto(f"{live_server.url}/#models")

    row = page.locator("#model-checks .check").filter(has_text="線上特徵覆蓋率")
    expect(row.locator(".run")).to_have_text(lines)
    if severity == "info":
        expect(row.locator("i")).not_to_have_class("warn")
    else:
        expect(row.locator("i")).to_have_class("warn")


def test_ex_dates_past_the_calendar_read_their_count_the_latest_and_the_calendar_s_end(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    detail = {"count": 1234, "latest_ex_date": "2027-01-12", "calendar_end": "2026-12-31"}
    log(seeded, EX_DATE_CHECK, "warning", detail)

    page.goto(f"{live_server.url}/#models")

    row = page.locator("#model-checks .check").filter(has_text="除息日超出交易日曆")
    expect(row).to_contain_text("1,234 筆，最晚 2027-01-12；交易日曆只到 2026-12-31")
    expect(row.locator("i")).to_have_class("warn")


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
UNSET = dict.fromkeys(name for names in MODE_FIELDS.values() for name in names)


def promotion_lines(page: Page) -> Locator:
    return page.locator("#model-checks .check").filter(has_text="換模").locator(".run")


def test_alert_entries_say_how_each_run_was_judged(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    judged = {
        **ALERT,
        **UNSET,
        "run_id": "rank-a",
        "mode": "judged",
        "own_rows": 45,
        "mostly_predecessors": False,
    }
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
    entries = [judged, graced, reset]
    assert {entry["mode"] for entry in entries} == set(MODES)
    assert [alert_problems(entry) for entry in entries] == [[], [], []]
    log(seeded, "model_alert", "warning", {"runs": entries})

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
    assert all(alert_problems(entry) for entry in (graced, reset))
    log(seeded, "model_alert", "warning", {"runs": [graced, reset]})

    page.goto(f"{live_server.url}/#models")

    lines = page.locator("#model-checks .check").filter(has_text="模型告警").locator(".run")
    expect(lines.nth(0)).to_have_text(f"rank-b{ALERT_TEXT}；寬限期內，只告警不降級")
    expect(lines.nth(1)).to_have_text(
        f"rank-c{ALERT_TEXT}；此 slot 在 restore 重設期（已累積 12／— 列），只告警不降級"
    )


def test_a_reset_alert_dated_before_its_restore_says_it_came_before(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    restored = (seeded.last_session + timedelta(days=3)).isoformat()
    started = seeded.last_session.isoformat()
    earlier = {
        **ALERT,
        **UNSET,
        "run_id": "rank-c",
        "mode": "reset",
        "own_rows": 60,
        "mostly_predecessors": False,
        "restored_on": restored,
        "reset_rows": 0,
        "reset_needed": 60,
    }
    fresh = {**earlier, "run_id": "rank-d", "restored_on": started}
    assert [alert_problems(entry) for entry in (earlier, fresh)] == [[], []]
    log(seeded, "model_alert", "warning", {"runs": [earlier, fresh]})

    page.goto(f"{live_server.url}/#models")

    lines = page.locator("#model-checks .check").filter(has_text="模型告警").locator(".run")
    expect(lines.nth(0)).to_have_text(
        f"rank-c{ALERT_TEXT}；窗口 60／60 列為本 run（100%）；恢復前的告警（{restored} restore）"
    )
    expect(lines.nth(1)).to_have_text(
        f"rank-d{ALERT_TEXT}；窗口 60／60 列為本 run（100%）"
        f"；此 slot 在 restore 重設期（{started} restore，已累積 0／60 列），只告警不降級"
    )


AGREEMENT = Gate("agreement", False, 0.62, "相關 >= 0.7", 1842, "相關太低")
AGREEMENT_TEXT = "與現役模型的預測相關（相關太低，實際 0.620，門檻 相關 >= 0.7，樣本 1,842 檔）"
DATA_QUALITY = Gate("data_quality", False, 2.0, "無 error", 250, "2 天有 error")


def refused(slot: Slot, run_id: str, *gates: Gate) -> SlotGates:
    passing = Gate("gain_overlap", True, 8.0, "前 10 名重疊 >= 6", 10, "")
    return SlotGates(slot, run_id, "serving-old", [passing, *gates])


def promotion_alert(
    seed: Seed,
    run_key: str,
    results: list[SlotGates],
    reason: str = "",
    back: int = 1,
    scope: str = "rank",
) -> None:
    day = seed.sessions[-back]
    _note(day, scope, run_key, _failed(results), reason, promoted=bool(reason))


def swapped_in(seed: Seed, run_key: str, back: int = 1, scope: str = "rank") -> None:
    _note(seed.sessions[-back], scope, run_key, [], "", promoted=True)


def test_a_passed_promotion_reads_as_swapped_in_not_as_a_warning(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    promotion_alert(
        seeded, "rank-2026-11", [refused(Slot("rank", 20), "rank-n", AGREEMENT)], back=3
    )
    swapped_in(seeded, "rank-2026-12")

    page.goto(f"{live_server.url}/#models")

    promotion = page.locator("#model-checks .check").filter(has_text="換模")
    expect(promotion).to_have_count(1)
    expect(promotion.locator("i")).not_to_have_class("warn")
    expect(promotion.locator(".pinned")).to_have_text(
        f"{seeded.last_session.isoformat()}，近 30 個交易日共 2 筆"
    )
    expect(promotion_lines(page)).to_have_count(1)
    expect(promotion_lines(page).first).to_have_text("已換上 rank-2026-12")
    expect(page.locator("#model-checks")).not_to_contain_text("續用現役模型")


def test_the_last_written_promotion_of_a_scope_wins_over_its_date(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    promotion_alert(seeded, "rank-new", [refused(Slot("rank", 20), "rank-n", AGREEMENT)])
    swapped_in(seeded, "rank-backfill", back=3)

    page.goto(f"{live_server.url}/#models")

    promotion = page.locator("#model-checks .check").filter(has_text="換模")
    expect(promotion).to_have_count(1)
    expect(promotion.locator(".pinned")).to_have_text(
        f"{seeded.sessions[-3].isoformat()}，近 30 個交易日共 2 筆"
    )
    expect(promotion_lines(page).first).to_have_text("已換上 rank-backfill")


def test_each_scope_keeps_its_own_latest_promotion_across_days(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    promotion_alert(
        seeded, "rank-2026-10", [refused(Slot("rank", 20), "rank-n", AGREEMENT)], back=10
    )
    interval = [refused(Slot("quantile", 20), "q-n", DATA_QUALITY)]
    promotion_alert(seeded, "interval-2026-11", interval, back=2, scope="interval")

    page.goto(f"{live_server.url}/#models")

    promotions = page.locator("#model-checks .check").filter(has_text="換模")
    expect(promotions).to_have_count(2)
    expect(promotions.nth(0).locator(".pinned")).to_have_text(seeded.sessions[-2].isoformat())
    expect(promotions.nth(0).locator(".run").first).to_contain_text("interval-2026-11")
    expect(promotions.nth(1).locator(".pinned")).to_have_text(seeded.sessions[-10].isoformat())
    expect(promotions.nth(1).locator(".run").first).to_contain_text("rank-2026-10")


def test_a_refused_build_names_each_failed_slot_and_gate(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    shadow = Gate("shadow_oos", False, None, "rank IC > 0", 120, XSS_NAME)
    rank = refused(Slot("rank", 20), "rank-n", AGREEMENT, shadow)
    interval = refused(Slot("quantile", 20), "quantile-n", DATA_QUALITY)
    promotion_alert(seeded, "prod-2026-11", [rank, interval])

    page.goto(f"{live_server.url}/#models")

    promotion = page.locator("#model-checks .check").filter(has_text="換模")
    expect(promotion.locator("div > div").first).to_contain_text("換模　")
    expect(promotion.locator(".pinned")).to_have_text(seeded.last_session.isoformat())
    expect(promotion.locator("i")).to_have_class("warn")
    lines = promotion_lines(page)
    expect(lines).to_have_count(3)
    expect(lines.nth(0)).to_have_text("換模門檻未過，續用現役模型（prod-2026-11）")
    expect(lines.nth(1)).to_have_text(
        f"排名 20 日（rank-n）：{AGREEMENT_TEXT}"
        f"、影子模型最近 6 個月 OOS（{XSS_NAME}，門檻 rank IC > 0，樣本 120 天）"
    )
    expect(lines.nth(2)).to_have_text(
        "報酬區間 20 日（quantile-n）：訓練窗內的資料品質"
        "（2 天有 error，實際 2，門檻 無 error，樣本 250 天）"
    )
    expect(page.locator("#model-checks img")).to_have_count(0)
    expect(page.locator("#quality")).not_to_contain_text("換模")


def test_the_latest_promotion_stays_on_top_for_thirty_sessions(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    rank = [refused(Slot("rank", 20), "rank-n", AGREEMENT)]
    promotion_alert(seeded, "prod-old", rank, back=12)
    promotion_alert(seeded, "prod-new", rank, back=5)
    promotion_alert(seeded, "prod-gone", rank, back=31)
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
    expect(lines.nth(1)).to_have_text(f"排名 20 日（rank-n）：{AGREEMENT_TEXT}")
    expect(checks.nth(1)).to_contain_text("區間寬度比異常")
    expect(page.locator("#model-checks-sub")).to_contain_text(seeded.last_session.isoformat())


def test_a_same_day_force_replaces_the_refusal_and_other_builds_stay(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    rank = [refused(Slot("rank", 20), "rank-n", AGREEMENT)]
    promotion_alert(seeded, "rank-2026-11", rank)
    promotion_alert(seeded, "rank-2026-11", rank, reason="人工確認後強制換模")
    promotion_alert(
        seeded,
        "interval-2026-11",
        [refused(Slot("quantile", 20), "q-n", DATA_QUALITY)],
        scope="interval",
    )

    page.goto(f"{live_server.url}/#models")

    checks = page.locator("#model-checks .check")
    expect(checks).to_have_count(2)
    note = f"{seeded.last_session.isoformat()}，近 30 個交易日共 3 筆"
    expect(checks.nth(0).locator(".pinned")).to_have_text(note)
    expect(checks.nth(0).locator(".run").first).to_have_text(
        "換模門檻未過，續用現役模型（interval-2026-11）"
    )
    expect(checks.nth(1).locator(".run").first).to_have_text(
        "強制換模：人工確認後強制換模（rank-2026-11）"
    )
    expect(page.locator("#model-checks")).not_to_contain_text("續用現役模型（rank-2026-11）")
    expect(page.locator("#model-checks-sub")).to_have_text("")


def test_a_promotion_newer_than_the_last_monitor_keeps_that_day_s_checks(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    promotion_alert(seeded, "prod-new", [refused(Slot("rank", 20), "rank-n", AGREEMENT)])
    clean = {"share": 0.0, "flagged": 0, "n_stocks": 1842, "stocks": []}
    log(seeded, "output_width_ratio", "info", clean, back=2)

    page.goto(f"{live_server.url}/#models")

    checks = page.locator("#model-checks .check")
    expect(checks).to_have_count(2)
    expect(checks.nth(1)).to_contain_text("區間寬度比異常")
    expect(page.locator("#model-checks-sub")).to_contain_text(seeded.sessions[-2].isoformat())


def test_a_forced_swap_leads_with_its_reason(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    shadow = Gate("shadow_oos", False, -0.01, "rank IC > 0", 120, "IC 為負")
    rank5 = refused(Slot("rank", 5), "rank5-n", shadow)
    promotion_alert(seeded, "prod-2026-11", [rank5], reason=XSS_NAME)

    page.goto(f"{live_server.url}/#models")

    lines = promotion_lines(page)
    expect(lines).to_have_count(2)
    expect(lines.nth(0)).to_have_text(f"強制換模：{XSS_NAME}（prod-2026-11）")
    expect(lines.nth(1)).to_have_text(
        "排名 5 日（rank5-n）：影子模型最近 6 個月 OOS"
        "（IC 為負，實際 -0.010，門檻 rank IC > 0，樣本 120 天）"
    )
    expect(page.locator("#model-checks img")).to_have_count(0)


def test_a_numeric_threshold_off_the_contract_still_shows_zero(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    gate = {
        "name": "shadow_oos",
        "reason": "",
        "value": -0.01,
        "threshold": 0,
        "sample": 120,
        "sample_unit": SAMPLE_UNITS["shadow_oos"],
    }
    failed = [{"slot": "rank 5 日", "run_id": "rank5-n", "gates": [gate]}]
    detail = {"run_key": "prod-x", "forced": False, "reason": None, "failed": failed}
    log(seeded, "model_promotion", "warning", detail)

    page.goto(f"{live_server.url}/#models")

    expect(promotion_lines(page).nth(1)).to_have_text(
        "排名 5 日（rank5-n）：影子模型最近 6 個月 OOS（實際 -0.010，門檻 0，樣本 120 天）"
    )


def test_research_schedule_checks_sit_with_the_model_checks(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    log(seeded, "research_morning", "info", {}, back=2)
    clean = {"share": 0.0, "flagged": 0, "n_stocks": 1842, "stocks": []}
    log(seeded, "output_width_ratio", "info", clean, back=2)
    log(seeded, "research_evening", "warning", {})

    page.goto(f"{live_server.url}/#models")

    checks = page.locator("#model-checks .check")
    expect(checks).to_have_count(3)
    evening = checks.nth(0)
    expect(evening).to_contain_text("研究晚間排程")
    expect(evening).to_contain_text("警告")
    expect(evening.locator(".pinned")).to_have_text(seeded.last_session.isoformat())
    expect(evening.locator("i")).to_have_class("warn")
    morning = checks.nth(1)
    expect(morning).to_contain_text("研究早盤排程")
    expect(morning.locator(".pinned")).to_have_text(seeded.sessions[-2].isoformat())
    expect(morning.locator("i")).not_to_have_class("warn")
    expect(checks.nth(2)).to_contain_text("區間寬度比異常")
    expect(page.locator("#model-checks-sub")).to_contain_text(seeded.sessions[-2].isoformat())
    expect(page.locator("#quality")).not_to_contain_text("研究")
    expect(page.locator("#status")).to_contain_text("品質檢查通過（1 項警告）")


BUILT = {"rank": "rank-2026-11", "return": "return-2026-11"}
HAND_NOTE = "人工重建後，這列會在 30 個交易日內或出現新的人工重建紀錄時消失"
NOT_FROZEN = {"interval": tasks.NOT_FROZEN}


def retrained(
    seed: Seed,
    back: int,
    built: dict[str, str],
    refused: dict[str, str],
    failed: dict[str, str] | None = None,
) -> None:
    detail = tasks._retrain_detail(refused, built, failed, 80.0, {})
    log(seed, tasks.RETRAIN_CHECK, "warning" if refused or failed else "info", detail, back=back)


def promote_step(
    seed: Seed, monkeypatch: pytest.MonkeyPatch, back: int, errors: dict[str, str]
) -> None:
    day = seed.sessions[-back]

    def promoted(run_key: str, on: date, directory: Path) -> Promoted:
        if run_key in errors:
            raise RunError(errors[run_key])
        return Promoted(True, [])

    monkeypatch.setattr(tasks, "first_day", lambda run_key: day)
    monkeypatch.setattr(tasks, "promote", promoted)
    monkeypatch.setattr(tasks, "dispatch", lambda rows: None)
    tasks._promote(day)


def rebuild_step(
    seed: Seed,
    monkeypatch: pytest.MonkeyPatch,
    error: str | None = None,
    build: bool = True,
) -> None:
    monkeypatch.setattr(tasks, "_command", lambda name, **options: name)
    monkeypatch.setattr(tasks, "dispatch", lambda rows: None)
    if error is not None:

        def broken(day: date, build: bool) -> str:
            raise RuntimeError(error)

        monkeypatch.setattr(tasks, "_rebuild_notes", broken)
    tasks._rebuild(seed.last_session, build)


def test_the_monthly_retrain_stays_pinned_beside_the_day_s_checks(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    retrained(seeded, 15, {}, NOT_FROZEN, {"rank": "MemoryError: "})
    clean = {"share": 0.0, "flagged": 0, "n_stocks": 1842, "stocks": []}
    log(seeded, "output_width_ratio", "info", clean)

    page.goto(f"{live_server.url}/#models")

    checks = page.locator("#model-checks .check")
    expect(checks).to_have_count(2)
    expect(checks.nth(0)).to_contain_text("每月重訓")
    expect(checks.nth(0)).to_contain_text("建 0、拒 1、失敗 1")
    expect(checks.nth(0).locator(".pinned")).to_have_text(seeded.sessions[-15].isoformat())
    expect(checks.nth(0).locator("i")).to_have_class("warn")
    expect(checks.nth(1)).to_contain_text("區間寬度比異常")
    expect(page.locator("#model-checks-sub")).to_contain_text(seeded.last_session.isoformat())


def test_research_checks_keep_a_fixed_order_whatever_ran_last(
    page: Page, live_server: LiveServer, seeded: Seed, monkeypatch: pytest.MonkeyPatch
) -> None:
    log(seeded, "research_evening", "info", {}, back=2)
    log(seeded, "research_morning", "warning", {})
    rebuild_step(seeded, monkeypatch, error="boom")
    retrained(seeded, 1, BUILT, {})
    promote_step(seeded, monkeypatch, 1, {"rank-2026-11": "boom"})

    page.goto(f"{live_server.url}/#models")

    checks = page.locator("#model-checks .check")
    expect(checks).to_have_count(5)
    expected = (
        ("每月重訓", 1),
        ("自動換模", 1),
        ("研究資料重建", 1),
        ("研究晚間排程", 2),
        ("研究早盤排程", 1),
    )
    for n, (name, back) in enumerate(expected):
        expect(checks.nth(n)).to_contain_text(name)
        expect(checks.nth(n).locator(".pinned")).to_have_text(seeded.sessions[-back].isoformat())


def test_a_rebuild_shows_every_row_of_its_day_the_one_left_by_hand_first(
    page: Page, live_server: LiveServer, seeded: Seed, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = TradingDays.from_db().shift(seeded.last_session, -80)
    recent = seeded.sessions[-5]
    log(seeded, ADJ_CHECK, "info", _merged({}, "2330", old), back=3)
    log(seeded, ADJ_CHECK, "info", _merged(_merged({}, "2330", recent), "9999", recent), back=2)
    rebuild_step(seeded, monkeypatch)

    page.goto(f"{live_server.url}/#models")

    rebuilds = page.locator("#model-checks .check")
    expect(rebuilds).to_have_count(2)
    expect(rebuilds.nth(0)).to_contain_text("研究資料重建")
    expect(rebuilds.nth(0).locator(".muted")).to_have_text(
        [
            f"最近 {tasks.RECENT_SESSIONS} 個交易日以外，未自動重建：1 檔，自 {old.isoformat()} 起",
            HAND_NOTE,
        ]
    )
    expect(rebuilds.nth(0).locator("i")).to_have_class("warn")
    rebuilt = f"已重建 2 檔，自 {recent.isoformat()} 起"
    expect(rebuilds.nth(1).locator(".muted")).to_have_text(rebuilt)
    expect(rebuilds.nth(1).locator("i")).not_to_have_class("warn")
    for row in (rebuilds.nth(0), rebuilds.nth(1)):
        expect(row.locator(".pinned")).to_have_text(seeded.last_session.isoformat())


def test_a_rebuild_the_nightly_recompute_covered_says_so(
    page: Page, live_server: LiveServer, seeded: Seed, monkeypatch: pytest.MonkeyPatch
) -> None:
    log(seeded, ADJ_CHECK, "info", _merged({}, "2330", seeded.sessions[-5]), back=2)
    rebuild_step(seeded, monkeypatch, build=False)

    page.goto(f"{live_server.url}/#models")

    covered = page.locator("#model-checks .check")
    expect(covered).to_have_count(1)
    expect(covered.locator(".muted")).to_have_text(f"已由每晚 {tasks.RECENT_SESSIONS} 日重算涵蓋")
    expect(covered.locator("i")).not_to_have_class("warn")


def test_a_rebuild_row_from_before_the_window_names_only_the_limit(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    earliest = seeded.sessions[-40].isoformat()
    legacy = {"earliest": earliest, "stocks": 1, "notes": {}, "limit": 30}
    log(seeded, tasks.REBUILD_CHECK, "warning", legacy)

    page.goto(f"{live_server.url}/#models")

    expect(page.locator("#model-checks .check .muted")).to_have_text(
        [f"超過上限，未自動重建：1 檔，自 {earliest} 起", HAND_NOTE]
    )


def test_a_rebuild_row_from_before_the_handling_reads_as_rebuilt(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    earliest = seeded.sessions[-5].isoformat()
    log(seeded, tasks.REBUILD_CHECK, "info", {"earliest": earliest, "stocks": 2, "notes": {}})

    page.goto(f"{live_server.url}/#models")

    rebuilt = page.locator("#model-checks .check")
    expect(rebuilt.locator(".muted")).to_have_text(f"重建 2 檔，自 {earliest} 起")
    expect(rebuilt.locator("i")).not_to_have_class("warn")


def moved(check: str, day: date) -> None:
    """Date the newest row of check on day, as if an earlier chain had written it."""
    row = DataQualityLog.objects.filter(check_name=check).latest("pk")
    DataQualityLog.objects.filter(pk=row.pk).update(date=day)


def test_only_the_latest_day_of_a_research_check_is_pinned(
    page: Page, live_server: LiveServer, seeded: Seed, monkeypatch: pytest.MonkeyPatch
) -> None:
    rebuild_step(seeded, monkeypatch, error="boom")
    moved(tasks.REBUILD_CHECK, seeded.sessions[-3])
    monkeypatch.undo()
    old = TradingDays.from_db().shift(seeded.last_session, -80)
    log(seeded, ADJ_CHECK, "info", _merged({}, "2330", old), back=2)
    log(seeded, ADJ_CHECK, "info", _merged({}, "9999", seeded.sessions[-5]), back=2)
    rebuild_step(seeded, monkeypatch)

    page.goto(f"{live_server.url}/#models")

    rebuilds = page.locator("#model-checks .check")
    expect(rebuilds).to_have_count(2)
    expect(rebuilds.locator(".pinned")).to_have_text([seeded.last_session.isoformat()] * 2)
    expect(page.locator("#model-checks")).not_to_contain_text("重建失敗")


def test_a_rebuild_left_by_hand_stays_pinned_after_a_later_rebuild(
    page: Page, live_server: LiveServer, seeded: Seed, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = TradingDays.from_db().shift(seeded.last_session, -80)
    log(seeded, ADJ_CHECK, "info", _merged({}, "2330", old), back=4)
    rebuild_step(seeded, monkeypatch)
    moved(tasks.REBUILD_CHECK, seeded.sessions[-3])
    recent = seeded.sessions[-2]
    log(seeded, ADJ_CHECK, "info", _merged({}, "9999", recent), back=2)
    rebuild_step(seeded, monkeypatch)

    page.goto(f"{live_server.url}/#models")

    rebuilds = page.locator("#model-checks .check")
    expect(rebuilds).to_have_count(2)
    expect(rebuilds.nth(0).locator(".muted").nth(0)).to_contain_text("未自動重建")
    expect(rebuilds.nth(0).locator(".muted").nth(1)).to_have_text(HAND_NOTE)
    expect(rebuilds.nth(0).locator(".pinned")).to_have_text(seeded.sessions[-3].isoformat())
    expect(rebuilds.nth(0).locator("i")).to_have_class("warn")
    rebuilt = f"已重建 1 檔，自 {recent.isoformat()} 起"
    expect(rebuilds.nth(1).locator(".muted")).to_have_text(rebuilt)
    expect(rebuilds.nth(1).locator(".pinned")).to_have_text(seeded.last_session.isoformat())


def test_an_evening_failure_and_a_morning_rebuild_of_one_day_both_stay(
    page: Page, live_server: LiveServer, seeded: Seed, monkeypatch: pytest.MonkeyPatch
) -> None:
    recent = seeded.sessions[-2]
    log(seeded, ADJ_CHECK, "info", _merged({}, "9999", recent), back=2)
    rebuild_step(seeded, monkeypatch, error="boom")
    monkeypatch.undo()
    rebuild_step(seeded, monkeypatch)

    page.goto(f"{live_server.url}/#models")

    rebuilds = page.locator("#model-checks .check")
    expect(rebuilds).to_have_count(2)
    expect(rebuilds.nth(0).locator(".muted")).to_have_text("重建失敗：RuntimeError: boom")
    expect(rebuilds.nth(0).locator("i")).to_have_class("warn")
    rebuilt = f"已重建 1 檔，自 {recent.isoformat()} 起"
    expect(rebuilds.nth(1).locator(".muted")).to_have_text(rebuilt)
    expect(rebuilds.locator(".pinned")).to_have_text([seeded.last_session.isoformat()] * 2)


def test_a_failed_rebuild_shows_its_error(
    page: Page, live_server: LiveServer, seeded: Seed, monkeypatch: pytest.MonkeyPatch
) -> None:
    rebuild_step(seeded, monkeypatch, error=XSS_NAME)

    page.goto(f"{live_server.url}/#models")

    failed = page.locator("#model-checks .check")
    expect(failed).to_have_count(1)
    expect(failed.locator(".muted")).to_have_text(f"重建失敗：RuntimeError: {XSS_NAME}")
    expect(failed.locator("i")).to_have_class("warn")
    assert page.evaluate("window.__xss") is None


def test_an_adj_factor_change_is_a_data_check_counting_its_stocks(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    named = _merged(_merged({}, "2330", seeded.sessions[-2]), "9999", seeded.sessions[-4])
    log(seeded, ADJ_CHECK, "info", named)

    page.goto(f"{live_server.url}/#models")

    change = page.locator("#quality .check", has_text="還原因子更新")
    expect(change).to_contain_text(f"2 檔，最早 {seeded.sessions[-4].isoformat()}")
    expect(change.locator("i")).not_to_have_class("warn")
    expect(page.locator("#model-checks")).not_to_contain_text("還原因子")


def test_a_collapsed_adj_factor_change_reads_its_count(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    collapsed = {"earliest": seeded.sessions[-9].isoformat(), "count": 249}
    log(seeded, ADJ_CHECK, "info", _merged(collapsed, "2330", seeded.sessions[-3]))

    page.goto(f"{live_server.url}/#models")

    change = page.locator("#quality .check", has_text="還原因子更新")
    expect(change).to_contain_text(f"250 檔，最早 {seeded.sessions[-9].isoformat()}")


def test_a_failed_automatic_promotion_names_each_build_and_its_error(
    page: Page, live_server: LiveServer, seeded: Seed, monkeypatch: pytest.MonkeyPatch
) -> None:
    retrained(seeded, 1, BUILT, NOT_FROZEN)
    promote_step(seeded, monkeypatch, 1, {"rank-2026-11": XSS_NAME})

    page.goto(f"{live_server.url}/#models")

    checks = page.locator("#model-checks .check")
    retrain, promote = checks.nth(0), checks.nth(1)
    expect(retrain).to_contain_text("建 2、拒 1、失敗 0")
    expect(retrain.locator("i")).to_have_class("warn")
    expect(promote).to_contain_text("自動換模")
    expect(promote.locator(".muted")).to_have_text(
        ["失敗 1、完成 1", f"rank-2026-11：RunError: {XSS_NAME}"]
    )
    expect(promote.locator("i")).to_have_class("warn")
    assert page.evaluate("window.__xss") is None


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
    expect(odds).to_contain_text("參考，184／1,842 檔")
    expect(odds.locator("i")).not_to_have_class("warn")


def test_the_odds_check_is_a_reference_even_above_the_alert_share(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    many = {"share": 0.5, "flagged": 921, "n_stocks": 1842, "stocks": ["2330"]}
    log(seeded, "output_odds_vs_q50", "info", many, back=2)

    page.goto(f"{live_server.url}/#models")

    odds = page.locator("#model-checks .check").filter(has_text="機率與中位數矛盾")
    expect(odds).to_contain_text("參考，921／1,842 檔")
    expect(odds.locator("i")).not_to_have_class("warn")
    expect(page.locator("#model-checks-sub")).to_contain_text("機率矛盾只做參考")


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


RELEASE_REASON = "差距在 seed 間標準差內，先上線觀察"
RELEASE: dict[str, object] = {
    "status": "未達標",
    "failed": ["排名 20 日 IC"],
    "flagged": ["排名 5 日 IC"],
    "reason": RELEASE_REASON,
    "released_at": "2026-11-12T10:03:00.123456+08:00",
    "report": "/srv/host/reports/rank-holdout-20261111.md",
    "sha256": "ab" * 32,
}


def built_on_release(
    seed: Seed,
    monkeypatch: pytest.MonkeyPatch,
    back: int = 1,
    scope: str = "rank",
    run_key: str = "rank-2026-11",
    **changed: object,
) -> None:
    monkeypatch.setattr(production, "dispatch", lambda rows: None)
    production._alert_release(
        scope, run_key, seed.sessions[-back], f"{scope}-holdout-20261111", {**RELEASE, **changed}
    )


def release_rows(page: Page) -> Locator:
    return page.locator("#model-checks .check").filter(has_text="holdout 放行後建模")


def test_a_build_on_a_released_holdout_names_its_scope_verdict_and_what_fell_short(
    page: Page, live_server: LiveServer, seeded: Seed, monkeypatch: pytest.MonkeyPatch
) -> None:
    built_on_release(seeded, monkeypatch, failed=["排名 20 日 IC", XSS_NAME])

    page.goto(f"{live_server.url}/#models")

    row = release_rows(page)
    expect(row.locator("i")).to_have_class("warn")
    expect(row.locator(".run")).to_have_text(
        [
            "排名（rank-2026-11）：holdout 判定「未達標」，經使用者放行後建模",
            f"未達的門檻：排名 20 日 IC、{XSS_NAME}",
            "退化旗標：排名 5 日 IC",
            "2026-11-12 10:03 放行，holdout rank-holdout-20261111",
        ]
    )
    expect(row.locator(".pinned")).to_have_text(seeded.last_session.isoformat())
    block = page.locator("#model-checks")
    expect(block).not_to_contain_text(RELEASE_REASON)
    expect(block).not_to_contain_text("/srv/host")
    expect(block.locator("img")).to_have_count(0)
    expect(page.locator("#model-checks-sub")).to_have_text("")
    assert page.evaluate("window.__xss") is None


def test_each_scope_keeps_its_latest_release_pinned_after_the_day_it_was_built(
    page: Page, live_server: LiveServer, seeded: Seed, monkeypatch: pytest.MonkeyPatch
) -> None:
    built_on_release(seeded, monkeypatch, back=25, run_key="rank-2026-10")
    built_on_release(seeded, monkeypatch, back=4, run_key="rank-2026-11")
    built_on_release(
        seeded,
        monkeypatch,
        back=4,
        scope="interval",
        run_key="interval-2026-11",
        status="明顯退化",
        failed=[],
    )
    swapped_in(seeded, "rank-2026-11", back=3)
    clean = {"share": 0.0, "flagged": 0, "n_stocks": 1842, "stocks": []}
    log(seeded, "output_width_ratio", "info", clean)

    page.goto(f"{live_server.url}/#models")

    checks = page.locator("#model-checks .check")
    expect(checks).to_have_count(4)
    expect(checks.nth(0)).to_contain_text("換模")
    expect(checks.nth(3)).to_contain_text("區間寬度比異常")
    rows = release_rows(page)
    expect(rows).to_have_count(2)
    expect(rows.nth(0).locator(".run")).to_have_text(
        [
            "區間與機率（interval-2026-11）：holdout 判定「明顯退化」，經使用者放行後建模",
            "退化旗標：排名 5 日 IC",
            "2026-11-12 10:03 放行，holdout interval-holdout-20261111",
        ]
    )
    expect(rows.nth(1).locator(".run").first).to_contain_text("排名（rank-2026-11）")
    for row in (rows.nth(0), rows.nth(1)):
        expect(row.locator(".pinned")).to_have_text(seeded.sessions[-4].isoformat())
    expect(page.locator("#model-checks-sub")).to_contain_text(seeded.last_session.isoformat())


def test_a_retrain_says_why_each_scope_was_refused_or_failed(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    verdict = (
        "HoldoutError: holdout rank-holdout-20261111 was judged 未達標 (report "
        "rank-holdout-20261111.md); the build is refused until release_holdout records the "
        "user's decision"
    )
    retrained(seeded, 1, {"return": "return-2026-11"}, NOT_FROZEN, {"rank": verdict})

    page.goto(f"{live_server.url}/#models")

    row = page.locator("#model-checks .check").filter(has_text="每月重訓")
    expect(row.locator(".muted")).to_have_text(
        [
            "建 1、拒 1、失敗 1",
            f"區間與機率拒建：{tasks.NOT_FROZEN}",
            f"排名失敗：{verdict}",
        ]
    )
    expect(row.locator("i")).to_have_class("warn")
