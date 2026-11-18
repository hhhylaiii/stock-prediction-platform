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
BOOK_LINE = f"{BOOK}：寫入 11/20 名單、11/19 成交"
SECONDS_LINE = "整步 1.8 秒"
BUSY = "ResearchBusyError: the research lock is held by predict"


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
        ({BOOK: ONE_DAY}, [BOOK_LINE, SECONDS_LINE]),
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
                f"{BOOK}：寫入 11/18、11/19 名單、沒有新成交",
                SECONDS_LINE,
            ],
        ),
        ({}, ["沒有帳本要寫", SECONDS_LINE]),
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
            "成交日改期：名單記 11/19，實際 11/20（交易日曆更正）",
        ),
        (
            {"no_taiex_tr": ["2026-11-19"]},
            "寫入時還沒有加權報酬指數 11/19 的收盤，大盤報酬留空",
        ),
        ({"brand_new": {"x": 1}, "first": ["1001"]}, "其他備註 brand_new"),
    ],
)
def test_each_shadow_note_gets_a_sentence_without_stock_codes(
    page: Page, live_server: LiveServer, seeded: Seed, note: dict[str, Any], line: str
) -> None:
    shadow(seeded, "warning", noted(note))

    row = shadow_row(page, live_server)

    expect(row.locator(".muted.small")).to_have_text(
        [BOOK_LINE, f"{BOOK} 11/19：{line}", SECONDS_LINE]
    )
    expect(row.locator(".warn-line")).to_have_count(0)
    expect(row.locator("i")).to_have_class("warn")
    expect(row).not_to_contain_text("1007")
    expect(row).not_to_contain_text("1001")
    expect(page.locator("#status")).to_contain_text("品質檢查通過（1 項警告）")


@pytest.mark.parametrize(
    ("where", "said", "head"),
    [
        pytest.param({"book": None, "signal_date": None}, BUSY, "帳本之外失敗", id="busy"),
        pytest.param(
            {"book": None, "signal_date": None}, TIMEOUT, "帳本之外失敗", id="outside-the-books"
        ),
        pytest.param(
            {"book": BOOK, "signal_date": "2026-11-19"},
            TIMEOUT,
            f"{BOOK} 卡在 11/19",
            id="stuck-book-of-the-eighth-image",
        ),
    ],
)
def test_a_failed_shadow_step_says_where_it_stopped_and_that_nothing_was_written(
    page: Page,
    live_server: LiveServer,
    seeded: Seed,
    where: dict[str, Any],
    said: str,
    head: str,
) -> None:
    error = said + " <b>near</b>"
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


PAIRED = "H70N30-C0-20261201"
CANDIDATE = "H70N30-C1"
SHA_MISMATCH = "C1 rank-2026-12.lightgbm.h5.f000 g1's file no longer matches its frozen sha256"
STOPPED = "stopped: the shadow budget ran out after 1 days"
SKIPPED = "skipped: the shadow budget ran out"


@pytest.mark.parametrize(
    ("note", "line"),
    [
        pytest.param(
            {"book": PAIRED, "role": "paired", "signal_date": "2026-12-02", "book_failed": "x <i>"},
            f"{PAIRED}（對照帳本）12/2：帳本這次沒寫入：x <i>",
            id="paired-book-failed",
        ),
        pytest.param(
            {"book": CANDIDATE, "role": "candidate", "signal_date": None, "book_failed": SKIPPED},
            f"{CANDIDATE}（候選）：帳本這次沒寫入：{SKIPPED}",
            id="candidate-book-skipped",
        ),
        pytest.param(
            {
                "book": "C1",
                "role": "candidate",
                "signal_date": "2026-12-02",
                "scores_failed": STOPPED,
            },
            f"C1（候選）12/2：打分沒完成：{STOPPED}",
            id="scores-stopped-on-a-day",
        ),
        pytest.param(
            {"book": "C1", "role": "candidate", "scores_failed": "ArithmeticError: no features"},
            "C1（候選）：打分沒完成：ArithmeticError: no features",
            id="scores-failed-without-a-day",
        ),
        pytest.param(
            {
                "book": "C1",
                "role": "candidate",
                "signal_date": "2026-12-02",
                "artifact_invalid": SHA_MISMATCH,
            },
            f"C1（候選）12/2：模型檔不見或 sha256 對不上，從那天起這次都不打分：{SHA_MISMATCH}",
            id="artifact-invalid",
        ),
    ],
)
def test_a_note_of_a_paired_or_candidate_book_names_it_its_role_and_the_original_text(
    page: Page, live_server: LiveServer, seeded: Seed, note: dict[str, Any], line: str
) -> None:
    shadow(seeded, "warning", {**written({BOOK: ONE_DAY}), "notes": [note]})

    row = shadow_row(page, live_server)

    expect(row.locator(".muted.small")).to_have_text([BOOK_LINE, line, SECONDS_LINE])
    expect(row.locator(".warn-line")).to_have_count(0)
    expect(row.locator(".muted.small i")).to_have_count(0)
    expect(row).not_to_contain_text("其他備註")
    expect(row.locator("i")).to_have_class("warn")


def test_the_main_book_failing_alone_is_a_warning_line_not_nothing_to_write(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    failed = {"book": BOOK, "role": "main", "signal_date": "2026-11-19", "book_failed": TIMEOUT}
    shadow(seeded, "warning", {**written({}), "notes": [failed]})

    row = shadow_row(page, live_server)

    main = f"{BOOK}（主紀錄）11/19：帳本這次沒寫入：{TIMEOUT}"
    expect(row.locator(".muted.small")).to_have_text([main, SECONDS_LINE])
    expect(row.locator(".warn-line")).to_have_text([main])
    expect(row.locator(".warn-line")).to_have_css("color", "rgb(224, 142, 138)")
    expect(row).not_to_contain_text("沒有帳本要寫")
    expect(row.locator("i")).to_have_class("warn")


@pytest.mark.parametrize(
    ("artifacts", "shown"),
    [
        pytest.param(
            ["rank-2026-11.lightgbm.h5.f000 g1", "rank-2026-12.lightgbm.h5.f000 g1"],
            "rank-2026-11 第 1 代、rank-2026-12 第 1 代",
            id="across-a-monthly-swap",
        ),
        pytest.param(
            [
                "rank-2026-12.lightgbm.h5.f000 g1",
                "rank-2026-12.lightgbm.h5.f001 g1",
                "rank-2026-12.lightgbm.h5.f000 g2",
                "unreadable",
            ],
            "rank-2026-12 第 1 代、rank-2026-12 第 2 代",
            id="repeats-merged-after-a-replace",
        ),
    ],
)
def test_a_candidates_scores_say_which_days_and_the_serving_run_key_and_generation(
    page: Page, live_server: LiveServer, seeded: Seed, artifacts: list[str], shown: str
) -> None:
    scores = {
        "C2": {"scored": [], "artifacts": []},
        "C1": {"scored": ["2026-12-01", "2026-12-02"], "artifacts": artifacts},
    }
    shadow(seeded, "info", {**written({BOOK: ONE_DAY}), "scores": scores})

    row = shadow_row(page, live_server)

    expect(row.locator(".muted.small")).to_have_text(
        [BOOK_LINE, f"候選 C1：打分 12/1、12/2（{shown}）", "候選 C2：這次沒打分", SECONDS_LINE]
    )
    expect(row).not_to_contain_text("lightgbm")
    expect(row.locator("i")).not_to_have_class("warn")


def test_an_unknown_note_lists_its_own_keys_but_not_the_book_role_or_day(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    note = {"book": CANDIDATE, "role": "candidate", "signal_date": "2026-12-02", "brand_new": 1}
    shadow(seeded, "warning", {**written({BOOK: ONE_DAY}), "notes": [note]})

    row = shadow_row(page, live_server)

    expect(row.locator(".muted.small")).to_have_text(
        [BOOK_LINE, f"{CANDIDATE}（候選）12/2：其他備註 brand_new", SECONDS_LINE]
    )


def test_a_candidates_scores_come_before_its_notes_and_the_time_comes_last(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    invalid = {
        "book": "C1",
        "role": "candidate",
        "signal_date": "2026-12-02",
        "artifact_invalid": SHA_MISMATCH,
    }
    detail = {
        **written({BOOK: ONE_DAY}),
        "scores": {"C1": {"scored": [], "artifacts": []}},
        "notes": [invalid],
    }
    shadow(seeded, "warning", detail)

    row = shadow_row(page, live_server)

    expect(row.locator(".muted.small")).to_have_text(
        [
            BOOK_LINE,
            "候選 C1：這次沒打分",
            f"C1（候選）12/2：模型檔不見或 sha256 對不上，從那天起這次都不打分：{SHA_MISMATCH}",
            SECONDS_LINE,
        ]
    )
