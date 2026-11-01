"""Walk-forward folds with embargo, tuning and holdout flags; never random splits (SPEC §7, §8)."""

from calendar import monthrange
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import numpy as np
import numpy.typing as npt

from apps.market.calendar import TradingDays

MAX_LABEL_HORIZON = 20
DAYS_PER_MONTH = 365.25 / 12


class WalkForwardError(ValueError):
    """A fold layout cannot be built from this configuration or date range."""


@dataclass(frozen=True)
class WalkForwardConfig:
    """Fold layout; SPEC §7 defaults: 36-month window, 20-day embargo, quarterly test periods."""

    window: Literal["rolling", "expanding"] = "rolling"
    train_months: int = 36
    embargo: int = MAX_LABEL_HORIZON
    test_months: int = 3
    step_months: int = 3
    valid_months: int = 6
    data_start: date = date(2010, 1, 1)
    first_test: date = date(2014, 1, 1)
    tuning_end: date = date(2015, 12, 31)
    holdout_start: date = date(2025, 1, 1)

    def __post_init__(self) -> None:
        if self.embargo < MAX_LABEL_HORIZON:
            raise WalkForwardError(f"embargo must cover the {MAX_LABEL_HORIZON}-day label window")
        for name in ("train_months", "test_months", "step_months", "valid_months"):
            if getattr(self, name) < 1:
                raise WalkForwardError(f"{name} must be at least 1")
        if self.window not in ("rolling", "expanding"):
            raise WalkForwardError(f"window must be rolling or expanding, got {self.window!r}")
        if self.test_months > self.step_months:
            raise WalkForwardError("test_months longer than step_months would overlap test periods")
        if self.valid_months >= self.train_months:
            raise WalkForwardError("valid_months must be shorter than train_months")
        if self.first_test <= self.data_start:
            raise WalkForwardError("first_test must come after data_start")
        if self.tuning_end >= self.holdout_start:
            raise WalkForwardError("tuning_end must come before holdout_start")


@dataclass(frozen=True)
class Fold:
    """Inclusive trading-day boundaries; metrics count test_start..eval_end, labels stay inside."""

    index: int
    train_start: date
    fit_end: date
    valid_start: date
    train_end: date
    test_start: date
    test_end: date
    eval_end: date
    tuning: bool
    holdout: bool

    @property
    def research(self) -> bool:
        return not self.tuning and not self.holdout


def add_months(day: date, months: int) -> date:
    """Same day of the month months later or earlier, clamped to the month's last day."""
    year, month = divmod(day.year * 12 + day.month - 1 + months, 12)
    return date(year, month + 1, min(day.day, monthrange(year, month + 1)[1]))


def _eval_end(calendar: TradingDays, config: WalkForwardConfig, test_end: date) -> date:
    end = test_end
    for boundary in (config.tuning_end + timedelta(days=1), config.holdout_start):
        if test_end < boundary <= calendar.end:
            following = calendar.between(boundary, calendar.end)
            if following:
                end = min(end, calendar.shift(following[0], -(config.embargo + 1)))
    return end


def _fold(calendar: TradingDays, config: WalkForwardConfig, index: int, last: date) -> Fold | None:
    period = add_months(config.first_test, index * config.step_months)
    period_end = add_months(config.first_test, index * config.step_months + config.test_months)
    test_days = calendar.between(period, min(period_end - timedelta(days=1), last))
    train_end = calendar.shift(test_days[0], -(config.embargo + 1))
    window_start = (
        config.data_start
        if config.window == "expanding"
        else add_months(train_end, -config.train_months) + timedelta(days=1)
    )
    if window_start < config.data_start:
        return None
    if window_start > train_end:
        raise WalkForwardError(f"fold at {period} has no training days before the embargo")
    train_start = calendar.between(window_start, train_end)[0]
    valid_from = add_months(train_end, -config.valid_months) + timedelta(days=1)
    valid_start = calendar.between(valid_from, train_end)[0]
    fit_end = calendar.shift(valid_start, -(config.embargo + 1))
    if fit_end < train_start:
        raise WalkForwardError(f"fold at {period} leaves no fit rows before validation")
    return Fold(
        index=index,
        train_start=train_start,
        fit_end=fit_end,
        valid_start=valid_start,
        train_end=train_end,
        test_start=test_days[0],
        test_end=test_days[-1],
        eval_end=_eval_end(calendar, config, test_days[-1]),
        tuning=test_days[0] <= config.tuning_end,
        holdout=test_days[-1] >= config.holdout_start,
    )


def generate_folds(
    calendar: TradingDays, last_date: date, config: WalkForwardConfig | None = None
) -> list[Fold]:
    """Folds numbered by test period from first_test, every step_months through last_date."""
    config = config or WalkForwardConfig()
    last = calendar.latest_on_or_before(last_date)
    if last < config.first_test:
        raise WalkForwardError(f"{last} is before the first test period {config.first_test}")
    folds: list[Fold] = []
    number = 0
    while add_months(config.first_test, number * config.step_months) <= last:
        fold = _fold(calendar, config, number, last)
        if fold is not None:
            folds.append(fold)
        number += 1
    return folds


def research_folds(folds: Iterable[Fold]) -> list[Fold]:
    """Folds whose metrics count toward research results: not tuning, not holdout."""
    return [fold for fold in folds if fold.research]


def time_decay_weights(
    dates: Iterable[date], train_end: date, half_life_months: float = 24.0
) -> npt.NDArray[np.float64]:
    """Sample weights 0.5 ** (age / half-life) for the expanding window, age in months."""
    if half_life_months <= 0:
        raise WalkForwardError("half_life_months must be positive")
    ages = np.array([(train_end - day).days for day in dates], dtype=np.float64) / DAYS_PER_MONTH
    if (ages < 0).any():
        raise WalkForwardError("time decay weights got dates after train_end")
    return np.power(0.5, ages / half_life_months)
