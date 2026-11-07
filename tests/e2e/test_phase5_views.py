from pathlib import Path
from typing import Any

import pytest
from django.core.management.base import CommandError
from playwright.sync_api import Page
from pytest_django.live_server_helper import LiveServer

from apps.api.management.commands.phase5_report import capture_views
from apps.research.models import Prediction
from tests.e2e.conftest import Seed

pytestmark = pytest.mark.e2e


def test_each_view_is_captured_once_its_data_has_drawn(
    page: Page, live_server: LiveServer, predicted: Seed, tmp_path: Path
) -> None:
    shots = capture_views(page, live_server.url, "2330", tmp_path)

    assert [shot.name for shot in shots] == ["today.png", "stock.png", "models.png"]
    assert all(shot.read_bytes().startswith(b"\x89PNG") for shot in shots)


def test_a_view_without_predictions_names_what_it_waited_for(
    page: Page, live_server: LiveServer, seeded: Seed, tmp_path: Path
) -> None:
    with pytest.raises(CommandError, match=r"^today\.png：等不到 #rows \.trow$"):
        capture_views(page, live_server.url, "2330", tmp_path, timeout_ms=1_000)

    assert list(tmp_path.iterdir()) == []


def test_a_stock_without_a_forecast_is_not_captured(
    page: Page, live_server: LiveServer, predicted: Seed, tmp_path: Path
) -> None:
    Prediction.objects.filter(stock_id="1101").delete()

    with pytest.raises(CommandError, match=r"^stock\.png：等不到 #price-chart path\.fan-median$"):
        capture_views(page, live_server.url, "1101", tmp_path, timeout_ms=1_000)

    assert list(tmp_path.iterdir()) == []


SLOW_SECTIONS = """
const real = window.fetch;
window.fetch = (url, options) => /features|history|ic-monthly|data-quality/.test(String(url))
  ? new Promise(done => setTimeout(done, 1500)).then(() => real(url, options))
  : real(url, options);
"""


def test_no_view_is_captured_while_a_section_is_still_loading(
    page: Page,
    live_server: LiveServer,
    predicted: Seed,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = []
    real_screenshot = page.screenshot

    def screenshot(**kwargs: Any) -> bytes:
        seen.append(" ".join(page.locator("#status, section.view:not([hidden])").all_inner_texts()))
        return real_screenshot(**kwargs)

    monkeypatch.setattr(page, "screenshot", screenshot)
    page.add_init_script(SLOW_SECTIONS)

    capture_views(page, live_server.url, "2330", tmp_path)

    assert len(seen) == 3
    assert not [text for text in seen if "載入中" in text]
