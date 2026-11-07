import logging
from datetime import date

import pandas as pd
import pytest

from apps.research.markets import HISTORY_COLUMNS, markets_on


def d(text: str) -> date:
    return date.fromisoformat(text)


def rows(*keyed: tuple[str, str, str]) -> pd.DataFrame:
    """Panel rows as stock, day and the stock's current market."""
    frame = pd.DataFrame(keyed, columns=["stock_id", "date", "market"])
    return frame.assign(date=frame["date"].map(d)).set_index(
        pd.Index([10 * k for k in range(len(frame))])
    )


def history(*spans: tuple[str, str, str | None, str]) -> pd.DataFrame:
    frame = pd.DataFrame(spans, columns=HISTORY_COLUMNS)
    return frame.assign(
        start_date=frame["start_date"].map(d),
        end_date=frame["end_date"].map(lambda text: None if text is None else d(text)),
    )


def test_each_row_takes_the_market_of_the_span_holding_its_day() -> None:
    panel = rows(
        ("3092", "2012-03-16", "TWSE"),
        ("3092", "2014-06-30", "TWSE"),
        ("3092", "2014-07-01", "TWSE"),
        ("3092", "2026-11-06", "TWSE"),
        ("2330", "2012-03-16", "TWSE"),
    )
    spans = history(
        ("3092", "2010-01-04", "2014-06-30", "TPEx"),
        ("3092", "2014-07-01", None, "TWSE"),
        ("2330", "2010-01-04", None, "TWSE"),
    )
    before = panel.copy()

    found = markets_on(panel, spans)

    assert found["market"].tolist() == ["TPEx", "TPEx", "TWSE", "TWSE", "TWSE"]
    assert found["listed"].tolist() == [True] * 5
    assert found.index.equals(panel.index)
    pd.testing.assert_frame_equal(panel, before)


def test_a_day_no_span_holds_is_unlisted_and_takes_the_nearest_known_market() -> None:
    panel = rows(
        ("1593", "2011-05-02", "TWSE"),
        ("1593", "2013-04-30", "TWSE"),
        ("1593", "2013-05-02", "TWSE"),
        ("1593", "2013-05-03", "TWSE"),
        ("6177", "2013-05-03", "TWSE"),
    )
    spans = history(
        ("1593", "2012-01-02", "2013-04-30", "TPEx"),
        ("1593", "2013-05-03", None, "TWSE"),
    )

    found = markets_on(panel, spans)

    assert found["market"].tolist() == ["TPEx", "TPEx", "TPEx", "TWSE", "TWSE"]
    assert found["listed"].tolist() == [False, True, False, True, False]


def test_a_day_after_the_last_span_ended_keeps_that_span_market_and_is_unlisted() -> None:
    panel = rows(("8299", "2015-06-30", "TWSE"), ("8299", "2015-07-01", "TWSE"))

    found = markets_on(panel, history(("8299", "2010-01-04", "2015-06-30", "TPEx")))

    assert found["market"].tolist() == ["TPEx", "TPEx"]
    assert found["listed"].tolist() == [True, False]


def test_without_any_history_every_row_keeps_the_current_market_and_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    panel = rows(("2330", "2016-03-01", "TWSE"), ("6488", "2016-03-01", "TPEx"))

    with caplog.at_level(logging.WARNING, logger="apps.research.markets"):
        found = markets_on(panel, history())

    assert found["market"].tolist() == ["TWSE", "TPEx"]
    assert not found["listed"].any()
    assert [record.getMessage() for record in caplog.records] == [
        "market history has no span for any of 2 stocks; none of their rows is listed"
    ]


def test_some_spans_log_nothing(caplog: pytest.LogCaptureFixture) -> None:
    panel = rows(("2330", "2016-03-01", "TWSE"), ("6794", "2016-03-01", "TWSE"))

    with caplog.at_level(logging.WARNING, logger="apps.research.markets"):
        found = markets_on(panel, history(("2330", "2010-01-04", None, "TWSE")))

    assert found["listed"].tolist() == [True, False]
    assert caplog.records == []


def test_an_empty_panel_gives_empty_columns() -> None:
    found = markets_on(rows(), history(("2330", "2010-01-04", None, "TWSE")))

    assert list(found.columns) == ["market", "listed"]
    assert found.empty
