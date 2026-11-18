import time
from collections.abc import Iterator
from typing import Any

import pytest
from django.core.signals import request_finished, request_started
from playwright.sync_api import Page
from pytest_django.live_server_helper import LiveServer

from apps.api.status import StatusView
from tests.e2e.conftest import Seed, Serving, settle

pytestmark = pytest.mark.e2e

SLOW_FETCH = "fetch('/api/status?slow=1'); 1"


@pytest.fixture
def slow_status(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """`/api/status?slow=1` sleeps 0.6 s on the server and notes when it is done."""
    done: list[str] = []
    original = StatusView.get

    def get(self: StatusView, request: Any, *args: Any, **kwargs: Any) -> Any:
        if request.GET.get("slow"):
            time.sleep(0.6)
            done.append("slow")
        return original(self, request, *args, **kwargs)

    monkeypatch.setattr(StatusView, "get", get)
    return done


@pytest.fixture
def seen_at_flush(slow_status: list[str]) -> Iterator[None]:
    """Torn down after `page` and just before the flush, so the slow request must be done."""
    yield
    assert slow_status == ["slow"]


@pytest.fixture
def serving() -> Iterator[Serving]:
    watched = Serving()
    request_started.connect(watched.started, weak=False, dispatch_uid="test-serving-started")
    request_finished.connect(watched.finished, weak=False, dispatch_uid="test-serving-finished")
    yield watched
    request_started.disconnect(dispatch_uid="test-serving-started")
    request_finished.disconnect(dispatch_uid="test-serving-finished")


def test_settle_waits_for_a_request_still_running_on_the_server(
    page: Page, live_server: LiveServer, seeded: Seed, slow_status: list[str], serving: Serving
) -> None:
    page.goto(f"{live_server.url}/api/status")
    page.evaluate(SLOW_FETCH)
    page.wait_for_timeout(100)

    settle(page, serving)

    assert slow_status == ["slow"]
    assert serving.count == 0


def test_after_settle_the_page_reaches_the_server_no_more(
    page: Page, live_server: LiveServer, seeded: Seed, serving: Serving
) -> None:
    page.goto(f"{live_server.url}/api/status")
    settle(page, serving)
    starts: list[object] = []

    def note(sender: object, **kwargs: Any) -> None:
        starts.append(sender)

    request_started.connect(note, weak=False, dispatch_uid="test-starts")
    try:
        outcome = page.evaluate("fetch('/api/status').then(() => 'reached', () => 'blocked')")
    finally:
        request_started.disconnect(dispatch_uid="test-starts")

    assert outcome == "blocked"
    assert starts == []


def test_a_request_outlasting_the_wait_fails_with_how_many_are_left(
    page: Page,
    live_server: LiveServer,
    seeded: Seed,
    slow_status: list[str],
    serving: Serving,
) -> None:
    page.goto(f"{live_server.url}/api/status")
    page.evaluate(SLOW_FETCH)
    page.wait_for_timeout(100)

    with pytest.raises(AssertionError, match=r"requests still running after 0\.2 s: 1"):
        settle(page, serving, seconds=0.2)


def test_the_page_fixture_lets_a_request_finish_before_the_flush(
    seen_at_flush: None, page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    page.goto(f"{live_server.url}/api/status")
    page.evaluate(SLOW_FETCH)
