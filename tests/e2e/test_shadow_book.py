from typing import Any

import pytest
from playwright.sync_api import Locator, Page, expect
from pytest_django.live_server_helper import LiveServer

from apps.etl.models import DataQualityLog
from tests.e2e.conftest import Seed

pytestmark = pytest.mark.e2e

BOOK = "H70N30-C0"
TIMEOUT = "OperationalError: canceling statement due to statement timeout"
NOTHING = "nothing; the next morning starts again from the same day"
ONE_DAY = {"listed": ["2026-11-20"], "filled": ["2026-11-19"], "waiting": "2026-11-20"}
ONE_DAY_LINE = f"{BOOK}：寫入 11/20 名單、11/19 成交，1.8 秒"


def shadow(seed: Seed, severity: str, detail: dict[str, Any]) -> None:
    DataQualityLog.objects.create(
        date=seed.last_session, check_name="research_shadow", severity=severity, detail=detail
    )


def written(books: dict[str, Any]) -> dict[str, Any]:
    return {"books": books, "notes": [], "seconds": 1.84}


def noted(note: dict[str, Any]) -> dict[str, Any]:
    return {
        **written({BOOK: ONE_DAY}),
        "notes": [{"book": BOOK, "signal_date": "2026-11-19", **note}],
    }


def shadow_row(page: Page, live_server: LiveServer) -> Locator:
    page.goto(f"{live_server.url}/#models")
    row = page.locator("#model-checks .check").filter(has_text="持有規則影子帳本")
    expect(row).to_have_count(1)
    return row


@pytest.mark.parametrize(
    ("books", "lines"),
    [
        ({BOOK: ONE_DAY}, [ONE_DAY_LINE]),
        (
            {
                BOOK: {
                    "listed": ["2026-11-18", "2026-11-19"],
                    "filled": [],
                    "waiting": "2026-11-19",
                },
                "H100N50-C0": {"listed": [], "filled": ["2026-11-19"], "waiting": None},
            },
            [
                "H100N50-C0：沒有新名單、11/19 成交",
                f"{BOOK}：寫入 11/18、11/19 名單、沒有新成交，1.8 秒",
            ],
        ),
        ({}, ["沒有帳本要寫，1.8 秒"]),
    ],
)
def test_a_written_shadow_book_says_which_days_it_wrote(
    page: Page, live_server: LiveServer, seeded: Seed, books: dict[str, Any], lines: list[str]
) -> None:
    shadow(seeded, "info", written(books))

    row = shadow_row(page, live_server)

    expect(row.locator(".muted.small")).to_have_text(lines)
    expect(row.locator(".pinned")).to_have_text(seeded.last_session.isoformat())
    expect(row.locator("i")).not_to_have_class("warn")
    expect(page.locator("#model-checks")).not_to_contain_text("research_shadow")


@pytest.mark.parametrize(
    ("note", "line"),
    [
        (
            {
                "predictions_rewritten": {
                    "universe_added": 0,
                    "universe_removed": 1,
                    "scores_changed": 37,
                    "first": ["1007"],
                }
            },
            "預測被改寫過：universe 新增 0 檔、移除 1 檔，分數變動 37 檔",
        ),
        ({"in_universe_without_score": 1, "first": ["1001"]}, "1 檔在 universe 內卻沒有分數"),
        (
            {"trade_date_moved": {"listed": "2026-11-19", "filled": "2026-11-20"}},
            "成交日有變動：名單 11/19、成交 11/20",
        ),
        ({"no_taiex_tr": ["2026-11-19"]}, "缺加權報酬指數 11/19"),
        ({"brand_new": {"x": 1}, "first": ["1001"]}, "其他備註 brand_new"),
    ],
)
def test_each_shadow_note_gets_a_sentence_without_stock_codes(
    page: Page, live_server: LiveServer, seeded: Seed, note: dict[str, Any], line: str
) -> None:
    shadow(seeded, "warning", noted(note))

    row = shadow_row(page, live_server)

    expect(row.locator(".muted.small")).to_have_text([ONE_DAY_LINE, f"{BOOK} 11/19：{line}"])
    expect(row.locator("i")).to_have_class("warn")
    expect(row).not_to_contain_text("1007")
    expect(row).not_to_contain_text("1001")
    expect(page.locator("#status")).to_contain_text("品質檢查通過（1 項警告）")


@pytest.mark.parametrize(
    ("where", "head"),
    [
        ({"book": BOOK, "signal_date": "2026-11-19"}, f"{BOOK} 卡在 11/19"),
        ({"book": None, "signal_date": None}, "帳本之外失敗"),
    ],
)
def test_a_failed_shadow_book_says_where_it_stopped_and_that_nothing_was_written(
    page: Page, live_server: LiveServer, seeded: Seed, where: dict[str, Any], head: str
) -> None:
    error = TIMEOUT + " <b>near</b>"
    shadow(seeded, "warning", {**where, "error": error, "written": NOTHING, "seconds": 60.02})

    row = shadow_row(page, live_server)

    expect(row.locator(".muted.small")).to_have_text(
        [f"{head}：{error}", "這次沒有寫入，下一個早上從同一天重來，60.0 秒"]
    )
    expect(row.locator("b")).to_have_count(0)
    expect(row.locator("i")).to_have_class("warn")


def test_a_shadow_detail_of_the_wrong_shape_still_shows_the_row(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    shadow(seeded, "warning", {"books": [1, 2], "notes": "x", "seconds": "slow"})

    row = shadow_row(page, live_server)

    expect(row.locator(".muted.small")).to_have_text(["警告"])
    expect(row.locator("i")).to_have_class("warn")
