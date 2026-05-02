from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class FunctionCategory(StrEnum):
    TIME_RANGE_HELPERS = "time_range_helpers"
    AGGREGATION = "aggregation"
    CHANGE_TREND = "change_trend"
    CONDITION_OVER_TIME = "condition_over_time"
    NUMERIC = "numeric"


TR = FunctionCategory.TIME_RANGE_HELPERS
AGG = FunctionCategory.AGGREGATION
TREND = FunctionCategory.CHANGE_TREND
COND = FunctionCategory.CONDITION_OVER_TIME
NUM = FunctionCategory.NUMERIC


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    category: FunctionCategory
    arity: tuple[int, ...]
    signature: str
    description: str
    examples: tuple[str, ...] = ()
    metric_arg: int | None = None
    time_range_arg: int | None = None
    previous_snapshot_arg: int | None = None
    returns_time_range: bool = False
    condition_arg: int | None = None

    def allows_arity(self, arity: int) -> bool:
        return arity in self.arity


def _spec(
    name: str,
    category: FunctionCategory,
    arity: int | tuple[int, ...],
    signature: str,
    description: str,
    *,
    examples: tuple[str, ...] = (),
    metric_arg: int | None = None,
    time_range_arg: int | None = None,
    previous_snapshot_arg: int | None = None,
    returns_time_range: bool = False,
    condition_arg: int | None = None,
) -> FunctionSpec:
    normalized_arity = (arity,) if isinstance(arity, int) else arity
    return FunctionSpec(
        name=name,
        category=category,
        arity=normalized_arity,
        signature=signature,
        description=description,
        examples=examples,
        metric_arg=metric_arg,
        time_range_arg=time_range_arg,
        previous_snapshot_arg=previous_snapshot_arg,
        returns_time_range=returns_time_range,
        condition_arg=condition_arg,
    )


FUNCTION_SPECS: Final[tuple[FunctionSpec, ...]] = (
    _spec(
        "now", TR, 0, "now()", "Current instant as a forecast time range.", returns_time_range=True
    ),
    _spec(
        "today", TR, 0, "today()", "Current local day in Europe/Warsaw.", returns_time_range=True
    ),
    _spec(
        "tomorrow", TR, 0, "tomorrow()", "Next local day in Europe/Warsaw.", returns_time_range=True
    ),
    _spec(
        "weekend", TR, 0, "weekend()", "The next Saturday-Sunday window.", returns_time_range=True
    ),
    _spec(
        "next_hours",
        TR,
        1,
        "next_hours(6)",
        "Window from now through the next N hours.",
        returns_time_range=True,
    ),
    _spec(
        "date_range",
        TR,
        2,
        'date_range("2026-05-02T00:00:00+02:00", "2026-05-02T23:59:00+02:00")',
        "Explicit ISO datetime range.",
        returns_time_range=True,
    ),
    _spec(
        "between",
        TR,
        3,
        'between(time_range, "0800", "1700")',
        "Constrain a time range to a local HHMM-HHMM interval.",
        time_range_arg=0,
        returns_time_range=True,
    ),
    _spec(
        "minutes",
        TR,
        1,
        "minutes(30)",
        "Convert minutes to a numeric duration.",
    ),
    _spec(
        "hours",
        TR,
        1,
        "hours(2)",
        "Convert hours to minutes as a numeric duration.",
    ),
    _spec(
        "previous_snapshot",
        TR,
        0,
        "previous_snapshot()",
        "Reference the previous forecast snapshot for trend functions.",
    ),
    _spec(
        "min_metric",
        AGG,
        2,
        'min_metric("metric_name", time_range)',
        "Minimum metric value in a time range.",
        metric_arg=0,
        time_range_arg=1,
    ),
    _spec(
        "max_metric",
        AGG,
        2,
        'max_metric("metric_name", time_range)',
        "Maximum metric value in a time range.",
        metric_arg=0,
        time_range_arg=1,
    ),
    _spec(
        "avg_metric",
        AGG,
        2,
        'avg_metric("metric_name", time_range)',
        "Average metric value in a time range.",
        metric_arg=0,
        time_range_arg=1,
    ),
    _spec(
        "sum_metric",
        AGG,
        2,
        'sum_metric("metric_name", time_range)',
        "Sum of metric values in a time range.",
        metric_arg=0,
        time_range_arg=1,
    ),
    _spec(
        "median_metric",
        AGG,
        2,
        'median_metric("metric_name", time_range)',
        "Median metric value in a time range.",
        metric_arg=0,
        time_range_arg=1,
    ),
    _spec(
        "stddev_metric",
        AGG,
        2,
        'stddev_metric("metric_name", time_range)',
        "Population standard deviation for metric values in a time range.",
        metric_arg=0,
        time_range_arg=1,
    ),
    _spec(
        "pctl_metric",
        AGG,
        3,
        'pctl_metric("metric_name", time_range, percentile)',
        "Percentile metric value in a time range.",
        metric_arg=0,
        time_range_arg=1,
    ),
    _spec(
        "delta_metric",
        TREND,
        2,
        'delta_metric("metric_name", time_range)',
        "Last minus first metric value in a time range.",
        metric_arg=0,
        time_range_arg=1,
    ),
    _spec(
        "abs_delta_metric",
        TREND,
        2,
        'abs_delta_metric("metric_name", time_range)',
        "Absolute change in metric value over a time range.",
        metric_arg=0,
        time_range_arg=1,
    ),
    _spec(
        "rate_of_change_metric",
        TREND,
        2,
        'rate_of_change_metric("metric_name", time_range)',
        "Relative change in metric value over a time range.",
        metric_arg=0,
        time_range_arg=1,
    ),
    _spec(
        "forecast_delta_metric",
        TREND,
        3,
        'forecast_delta_metric("metric_name", time_range, previous_snapshot())',
        "Average metric delta between current and previous forecast snapshots.",
        metric_arg=0,
        time_range_arg=1,
        previous_snapshot_arg=2,
    ),
    _spec(
        "points_between",
        COND,
        1,
        "points_between(time_range)",
        "Forecast points in a time range for CEL collection macros.",
        time_range_arg=0,
    ),
    _spec(
        "duration_minutes",
        COND,
        1,
        "duration_minutes(points)",
        "Convert a point list to minutes using hourly forecast resolution.",
    ),
    _spec("abs", NUM, 1, "abs(value)", "Absolute numeric value."),
    _spec("round", NUM, (1, 2), "round(value, digits)", "Round a numeric value."),
    _spec(
        "clamp",
        NUM,
        3,
        "clamp(value, min_value, max_value)",
        "Clamp a numeric value into an inclusive range.",
    ),
)

FUNCTION_REGISTRY: Final[dict[str, FunctionSpec]] = {spec.name: spec for spec in FUNCTION_SPECS}

FUNCTION_NAMES_BY_CATEGORY: Final[dict[str, list[str]]] = {
    category.value: [spec.name for spec in FUNCTION_SPECS if spec.category == category]
    for category in FunctionCategory
}

ALL_FUNCTION_NAMES: Final[frozenset[str]] = frozenset(FUNCTION_REGISTRY)

METRIC_FUNCTION_NAMES: Final[frozenset[str]] = frozenset(
    spec.name for spec in FUNCTION_SPECS if spec.metric_arg is not None
)

TIME_RANGE_FUNCTION_NAMES: Final[frozenset[str]] = frozenset(
    spec.name for spec in FUNCTION_SPECS if spec.returns_time_range
)
