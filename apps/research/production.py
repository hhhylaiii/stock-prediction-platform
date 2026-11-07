"""Production runs (SPEC §7): the frozen design fit through the latest labels, and its shadow."""

from datetime import date, timedelta

from apps.market.calendar import TradingDays
from apps.research.walkforward import (
    Fold,
    WalkForwardConfig,
    WalkForwardError,
    add_months,
    training_window,
)

SHADOW_MONTHS = 6
PRODUCTION_FOLD = 0


def _fold(
    calendar: TradingDays,
    config: WalkForwardConfig,
    train_end: date,
    test: tuple[date, date],
    eval_end: date,
) -> Fold:
    window = training_window(calendar, config, train_end, test[0])
    if window is None:
        raise WalkForwardError(
            f"the training window before {test[0]} starts before data_start {config.data_start}"
        )
    train_start, fit_end, valid_start = window
    return Fold(
        index=PRODUCTION_FOLD,
        train_start=train_start,
        fit_end=fit_end,
        valid_start=valid_start,
        train_end=train_end,
        test_start=test[0],
        test_end=test[1],
        eval_end=eval_end,
        tuning=False,
        holdout=False,
    )


def production_fold(calendar: TradingDays, as_of: date, config: WalkForwardConfig) -> Fold:
    """Fold of model N: training ends where as_of's labels end, the test is as_of alone."""
    train_end = calendar.shift(as_of, -config.embargo)
    return _fold(calendar, config, train_end, (as_of, as_of), train_end)


def shadow_fold(calendar: TradingDays, as_of: date, config: WalkForwardConfig) -> Fold:
    """Fold of shadow N′: it tests the SHADOW_MONTHS through as_of after the usual embargo."""
    test_start = calendar.between(add_months(as_of, -SHADOW_MONTHS) + timedelta(days=1), as_of)[0]
    train_end = calendar.shift(test_start, -(config.embargo + 1))
    eval_end = calendar.shift(as_of, -config.embargo)
    return _fold(calendar, config, train_end, (test_start, as_of), eval_end)
