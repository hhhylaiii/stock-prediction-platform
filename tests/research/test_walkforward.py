from dataclasses import replace
from datetime import date, timedelta
from itertools import pairwise
from typing import Any

import pytest

from apps.market.calendar import CalendarError, TradingDays
from apps.research.walkforward import (
    Fold,
    WalkForwardConfig,
    WalkForwardError,
    add_months,
    generate_folds,
    research_folds,
    time_decay_weights,
)

LAST = date(2026, 10, 30)
DEFAULT = WalkForwardConfig()


def d(text: str) -> date:
    return date.fromisoformat(text)


@pytest.fixture(scope="module")
def folds(long_calendar: TradingDays) -> list[Fold]:
    return generate_folds(long_calendar, LAST)


def test_default_layout_rolls_quarterly_from_2014_to_the_last_date(folds: list[Fold]) -> None:
    assert len(folds) == 52
    assert [fold.index for fold in folds] == list(range(52))
    assert (folds[0].test_start, folds[0].test_end) == (d("2014-01-01"), d("2014-03-31"))
    assert (folds[1].test_start, folds[1].test_end) == (d("2014-04-01"), d("2014-06-30"))
    assert (folds[-1].test_start, folds[-1].test_end) == (d("2026-10-01"), LAST)


def test_first_fold_boundaries_match_hand_count(folds: list[Fold]) -> None:
    first = folds[0]
    assert first.train_start == d("2010-12-06")
    assert first.train_end == d("2013-12-03")
    assert first.valid_start == d("2013-06-04")
    assert first.fit_end == d("2013-05-06")


def test_label_windows_never_reach_the_next_period(
    folds: list[Fold], long_calendar: TradingDays
) -> None:
    for fold in folds:
        assert fold.train_start <= fold.fit_end < fold.valid_start <= fold.train_end
        assert fold.train_end < fold.test_start <= fold.test_end
        assert long_calendar.shift(fold.train_end, DEFAULT.embargo) < fold.test_start
        assert long_calendar.shift(fold.fit_end, DEFAULT.embargo) < fold.valid_start
        assert len(long_calendar.between(fold.train_end, fold.test_start)) == DEFAULT.embargo + 2
        assert len(long_calendar.between(fold.fit_end, fold.valid_start)) == DEFAULT.embargo + 2


def test_test_periods_are_contiguous_and_disjoint(
    folds: list[Fold], long_calendar: TradingDays
) -> None:
    for before, after in pairwise(folds):
        assert after.test_start == long_calendar.shift(before.test_end, 1)


def test_rolling_window_covers_train_months_before_the_embargo(folds: list[Fold]) -> None:
    for fold in folds:
        window_start = add_months(fold.train_end, -DEFAULT.train_months) + timedelta(days=1)
        assert window_start <= fold.train_start < window_start + timedelta(days=4)


def test_validation_split_is_the_last_six_months_of_the_window(folds: list[Fold]) -> None:
    for fold in folds:
        assert add_months(fold.train_end, -6) < fold.valid_start
        assert fold.valid_start <= add_months(fold.train_end, -6) + timedelta(days=4)


def test_tuning_folds_are_the_ones_tested_in_2014_and_2015(folds: list[Fold]) -> None:
    tuning = [fold for fold in folds if fold.tuning]
    assert [fold.index for fold in tuning] == list(range(8))
    assert all(fold.test_end <= d("2015-12-31") for fold in tuning)


def test_semiannual_layout_has_exactly_four_tuning_folds(long_calendar: TradingDays) -> None:
    config = WalkForwardConfig(test_months=6, step_months=6)
    folds = generate_folds(long_calendar, LAST, config)
    assert len(folds) == 26
    assert [(f.test_start, f.test_end) for f in folds if f.tuning] == [
        (d("2014-01-01"), d("2014-06-30")),
        (d("2014-07-01"), d("2014-12-31")),
        (d("2015-01-01"), d("2015-06-30")),
        (d("2015-07-01"), d("2015-12-31")),
    ]


def test_holdout_folds_are_the_ones_tested_from_2025_on(folds: list[Fold]) -> None:
    holdout = [fold for fold in folds if fold.holdout]
    assert len(holdout) == 8
    assert holdout[0].test_start == d("2025-01-01")
    assert all(not fold.holdout for fold in folds if fold.test_end < d("2025-01-01"))


def test_a_fold_overlapping_a_boundary_takes_the_stricter_flag(
    long_calendar: TradingDays,
) -> None:
    config = WalkForwardConfig(first_test=d("2015-11-01"), tuning_end=d("2015-12-31"))
    folds = generate_folds(long_calendar, d("2016-06-30"), config)
    assert [fold.tuning for fold in folds] == [True, False, False]
    config = WalkForwardConfig(first_test=d("2024-11-01"), holdout_start=d("2025-01-01"))
    folds = generate_folds(long_calendar, d("2025-06-30"), config)
    assert [fold.holdout for fold in folds] == [True, True, True]


def test_research_folds_drop_tuning_and_holdout(folds: list[Fold]) -> None:
    research = research_folds(folds)
    assert len(research) == 36
    assert research[0].test_start == d("2016-01-01")
    assert research[-1].test_end == d("2024-12-31")
    assert all(fold.research for fold in research)


def test_sixty_month_window_starts_in_2015(long_calendar: TradingDays) -> None:
    folds = generate_folds(long_calendar, LAST, WalkForwardConfig(train_months=60))
    assert folds[0].test_start == d("2015-04-01")
    assert folds[0].index == 5
    assert DEFAULT.data_start <= folds[0].train_start < d("2010-04-01")
    assert len(folds) == 47


def test_fold_numbers_follow_the_test_period_across_window_lengths(
    folds: list[Fold], long_calendar: TradingDays
) -> None:
    sixty = generate_folds(long_calendar, LAST, WalkForwardConfig(train_months=60))
    by_start = {fold.test_start: fold.index for fold in folds}
    assert all(by_start[fold.test_start] == fold.index for fold in sixty)


def test_expanding_window_always_starts_at_data_start(long_calendar: TradingDays) -> None:
    folds = generate_folds(long_calendar, LAST, WalkForwardConfig(window="expanding"))
    assert len(folds) == 52
    assert {fold.train_start for fold in folds} == {d("2010-01-01")}


def test_real_calendar_holidays_shift_every_boundary(twse_calendar: TradingDays) -> None:
    config = WalkForwardConfig(
        train_months=12, valid_months=3, first_test=d("2015-10-01"), data_start=d("2014-01-01")
    )
    folds = generate_folds(twse_calendar, d("2016-12-31"), config)
    january = next(fold for fold in folds if fold.test_start.year == 2016)
    assert january.test_start == d("2016-01-04")
    assert len(twse_calendar.between(january.train_end, january.test_start)) == 22
    assert folds[-1].test_end == d("2016-12-30")


def test_last_date_is_snapped_to_a_trading_day(long_calendar: TradingDays) -> None:
    folds = generate_folds(long_calendar, d("2022-01-02"))
    assert folds[-1].test_start == d("2021-10-01")
    assert folds[-1].test_end == d("2021-12-31")


def test_last_date_before_first_test_is_refused(long_calendar: TradingDays) -> None:
    with pytest.raises(WalkForwardError, match="before the first test period"):
        generate_folds(long_calendar, d("2013-12-31"))


def test_calendar_that_misses_the_training_window_is_refused(
    twse_calendar: TradingDays,
) -> None:
    with pytest.raises(CalendarError):
        generate_folds(twse_calendar, d("2016-12-31"))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"embargo": 19}, "embargo"),
        ({"train_months": 0}, "train_months"),
        ({"test_months": 0}, "test_months"),
        ({"step_months": 0}, "step_months"),
        ({"valid_months": 36}, "valid_months"),
        ({"tuning_end": d("2025-01-01")}, "tuning_end"),
        ({"first_test": d("2009-12-01")}, "first_test"),
        ({"window": "Expanding"}, "window"),
        ({"test_months": 6}, "overlap"),
    ],
)
def test_invalid_config_is_refused(change: dict[str, Any], message: str) -> None:
    with pytest.raises(WalkForwardError, match=message):
        replace(DEFAULT, **change)


def test_time_decay_halves_every_24_months() -> None:
    end = d("2013-12-03")
    weights = time_decay_weights([end, add_months(end, -24), add_months(end, -48)], end)
    assert weights[0] == 1.0
    assert weights[1] == pytest.approx(0.5, rel=2e-3)
    assert weights[2] == pytest.approx(0.25, rel=2e-3)


def test_time_decay_refuses_dates_after_the_window() -> None:
    with pytest.raises(WalkForwardError, match="after train_end"):
        time_decay_weights([d("2014-01-02")], d("2013-12-03"))


@pytest.mark.parametrize(
    ("start", "months", "expected"),
    [
        ("2014-01-01", -36, "2011-01-01"),
        ("2013-12-03", -6, "2013-06-03"),
        ("2014-08-31", -6, "2014-02-28"),
        ("2016-05-31", -3, "2016-02-29"),
        ("2014-11-01", 3, "2015-02-01"),
    ],
)
def test_add_months_clamps_to_month_end(start: str, months: int, expected: str) -> None:
    assert add_months(d(start), months) == d(expected)


def test_window_too_short_for_validation_is_refused(long_calendar: TradingDays) -> None:
    with pytest.raises(WalkForwardError, match="no fit rows"):
        generate_folds(long_calendar, LAST, WalkForwardConfig(train_months=7, valid_months=6))


def test_expanding_window_without_days_before_the_embargo_is_refused(
    long_calendar: TradingDays,
) -> None:
    config = WalkForwardConfig(window="expanding", data_start=d("2013-12-20"))
    with pytest.raises(WalkForwardError, match="no training days"):
        generate_folds(long_calendar, LAST, config)


def test_time_decay_refuses_a_non_positive_half_life() -> None:
    with pytest.raises(WalkForwardError, match="half_life_months"):
        time_decay_weights([d("2013-12-03")], d("2013-12-03"), half_life_months=0)


def test_non_default_layout_keeps_every_boundary(long_calendar: TradingDays) -> None:
    config = WalkForwardConfig(embargo=25, valid_months=3, test_months=3, step_months=6)
    folds = generate_folds(long_calendar, LAST, config)
    assert [fold.index for fold in folds[:3]] == [0, 1, 2]
    for fold in folds[:-1]:
        assert len(long_calendar.between(fold.train_end, fold.test_start)) == 27
        assert len(long_calendar.between(fold.fit_end, fold.valid_start)) == 27
        assert add_months(fold.train_end, -3) < fold.valid_start
        assert fold.valid_start <= add_months(fold.train_end, -3) + timedelta(days=4)
        assert fold.test_end < add_months(fold.test_start, 3)
    for before, after in pairwise(folds):
        months = (after.test_start.year - before.test_start.year) * 12
        assert months + after.test_start.month - before.test_start.month == 6


def test_month_end_start_stays_anchored(long_calendar: TradingDays) -> None:
    config = WalkForwardConfig(first_test=d("2014-01-31"), test_months=1, step_months=1)
    folds = generate_folds(long_calendar, d("2014-06-30"), config)
    assert [fold.test_start for fold in folds] == [
        d("2014-01-31"),
        d("2014-02-28"),
        d("2014-03-31"),
        d("2014-04-30"),
        d("2014-06-02"),
        d("2014-06-30"),
    ]
    for before, after in pairwise(folds):
        assert after.test_start == long_calendar.shift(before.test_end, 1)


def test_eval_end_keeps_labels_out_of_the_next_era(
    folds: list[Fold], long_calendar: TradingDays
) -> None:
    last_research = next(fold for fold in folds if fold.test_start == d("2024-10-01"))
    last_tuning = next(fold for fold in folds if fold.test_start == d("2015-10-01"))
    for fold, era_start in ((last_research, d("2025-01-01")), (last_tuning, d("2016-01-01"))):
        assert fold.eval_end < fold.test_end
        assert long_calendar.shift(fold.eval_end, DEFAULT.embargo) < era_start
        assert long_calendar.shift(fold.eval_end, DEFAULT.embargo + 1) == era_start
    trimmed = {last_research.index, last_tuning.index}
    assert all(fold.eval_end == fold.test_end for fold in folds if fold.index not in trimmed)


@pytest.mark.parametrize("first_test", [d("2015-11-01"), d("2024-08-01"), d("2014-04-01")])
def test_eval_end_stops_before_the_next_era_when_folds_straddle_a_boundary(
    first_test: date, long_calendar: TradingDays
) -> None:
    months = 6 if first_test.month == 4 else 3
    config = WalkForwardConfig(first_test=first_test, test_months=months, step_months=months)
    folds = generate_folds(long_calendar, LAST, config)
    changes = [
        (before, after)
        for before, after in pairwise(folds)
        if (before.tuning, before.holdout) != (after.tuning, after.holdout)
    ]
    assert changes
    for before, after in changes:
        assert long_calendar.shift(before.eval_end, DEFAULT.embargo) < after.test_start
