import math
from datetime import date
from typing import Any

import pandas as pd
import pytest
from django.db import IntegrityError, transaction

from apps.etl.upsert import upsert
from apps.research.models import BacktestMetric, MetricGrain

pytestmark = pytest.mark.django_db
KEY = [
    "run_key",
    "model_type",
    "objective",
    "horizon",
    "metric",
    "grain",
    "period_start",
    "segment",
]


def metric(**changes: Any) -> BacktestMetric:
    fields: dict[str, Any] = {
        "run_key": "rank-wf36",
        "model_type": "lightgbm",
        "objective": "rank",
        "horizon": 20,
        "metric": "rank_ic",
        "grain": MetricGrain.MONTH,
        "period_start": date(2024, 1, 2),
        "period_end": date(2024, 1, 31),
        "value": 0.041,
    }
    return BacktestMetric(**{**fields, **changes})


def test_grains_cover_total_fold_year_and_month() -> None:
    assert set(MetricGrain.values) == {"total", "fold", "year", "month"}


def test_monthly_ic_series_reads_per_model_in_period_order() -> None:
    for model_type, values in [("lightgbm", [0.05, -0.01]), ("ridge", [0.03, -0.02])]:
        for (start, end), value in zip(
            [(date(2024, 2, 1), date(2024, 2, 29)), (date(2024, 1, 2), date(2024, 1, 31))],
            values,
            strict=True,
        ):
            metric(model_type=model_type, period_start=start, period_end=end, value=value).save()
    metric(grain=MetricGrain.TOTAL, period_end=date(2024, 12, 31), value=0.03).save()

    monthly = BacktestMetric.objects.filter(
        run_key="rank-wf36", objective="rank", horizon=20, metric="rank_ic", grain="month"
    ).order_by("model_type", "period_start")
    assert list(monthly.values_list("model_type", "period_start", "value")) == [
        ("lightgbm", date(2024, 1, 2), -0.01),
        ("lightgbm", date(2024, 2, 1), 0.05),
        ("ridge", date(2024, 1, 2), -0.02),
        ("ridge", date(2024, 2, 1), 0.03),
    ]


def test_segments_and_null_values_keep_their_own_rows() -> None:
    metric(segment="liquidity:3", value=0.02).save()
    metric(segment="liquidity:1", value=None).save()
    metric().save()

    assert BacktestMetric.objects.count() == 3
    assert BacktestMetric.objects.get(segment="liquidity:1").value is None


def test_upsert_on_the_metric_key_is_idempotent() -> None:
    frame = pd.DataFrame(
        [
            {
                "run_key": "rank-wf36",
                "model_type": model_type,
                "objective": "rank",
                "horizon": 20,
                "metric": "rank_ic",
                "grain": "total",
                "period_start": date(2016, 1, 4),
                "period_end": date(2024, 12, 31),
                "segment": "",
                "value": value,
            }
            for model_type, value in [("lightgbm", 0.031), ("ridge", 0.022)]
        ]
    )

    upsert(BacktestMetric, frame, KEY, ["period_end", "value"])
    upsert(BacktestMetric, frame.assign(value=[0.033, 0.021]), KEY, ["period_end", "value"])
    upsert(BacktestMetric, frame.assign(value=[0.033, 0.021]), KEY, ["period_end", "value"])

    rows = BacktestMetric.objects.order_by("model_type").values_list("model_type", "value")
    assert list(rows) == [("lightgbm", 0.033), ("ridge", 0.021)]


def test_the_metric_key_is_unique() -> None:
    metric().save()

    with (
        pytest.raises(IntegrityError, match="backtest_metric_key"),
        transaction.atomic(),
    ):
        metric(value=0.5).save()


@pytest.mark.parametrize(
    ("changes", "constraint"),
    [
        ({"objective": "lambdarank"}, "backtest_metric_objective_known"),
        ({"grain": "week"}, "backtest_metric_grain_known"),
        ({"horizon": 60}, "backtest_metric_horizon_known"),
        ({"period_end": date(2024, 1, 1)}, "backtest_metric_period_ordered"),
        ({"value": math.nan}, "backtest_metric_finite"),
        ({"value": math.inf}, "backtest_metric_finite"),
        ({"metric": " "}, "backtest_metric_names_present"),
        ({"run_key": ""}, "backtest_metric_names_present"),
        ({"model_type": ""}, "backtest_metric_names_present"),
    ],
)
def test_backtest_metric_check_constraints(changes: dict[str, Any], constraint: str) -> None:
    with pytest.raises(IntegrityError, match=constraint), transaction.atomic():
        metric(**changes).save()
