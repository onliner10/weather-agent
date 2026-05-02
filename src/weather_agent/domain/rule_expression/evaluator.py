from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

import cel
from pydantic import BaseModel

from weather_agent.domain.rule_expression.allowlist import ALLOWED_METRICS
from weather_agent.domain.rule_expression.registry import FUNCTION_REGISTRY
from weather_agent.domain.rule_expression.validation import validate_expression
from weather_agent.domain.time import WARSAW_TZ, ensure_aware, parse_datetime
from weather_agent.domain.weather import TimeRange

_WARSAW = WARSAW_TZ
_HOURLY_RESOLUTION_MINUTES = 60
_DataDict = dict[str, Any]
_TimeRangeDict = dict[str, str]
_Function = Callable[..., Any]
_FUNCTION_NAME_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_STRING_LITERAL_RE = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"|\'([^\'\\]*(?:\\.[^\'\\]*)*)\'')
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


class RuleExpressionEvaluationResult(BaseModel):
    expression: str
    result: bool | float | int | str | None = None
    error: str | None = None
    evaluated_metrics: list[str] = []
    evaluated_functions: list[str] = []

    @property
    def valid(self) -> bool:
        return self.error is None


class RuleExpressionEvalError(Exception):
    pass


def _time_range(start: datetime, end: datetime) -> _TimeRangeDict:
    return {
        "start": ensure_aware(start, _WARSAW).isoformat(),
        "end": ensure_aware(end, _WARSAW).isoformat(),
    }


def _now() -> _TimeRangeDict:
    now = datetime.now(tz=_WARSAW)
    return _time_range(now, now)


def _today() -> _TimeRangeDict:
    now = datetime.now(tz=_WARSAW)
    return _time_range(
        now.replace(hour=0, minute=0, second=0, microsecond=0),
        now.replace(hour=23, minute=59, second=59, microsecond=0),
    )


def _tomorrow() -> _TimeRangeDict:
    now = datetime.now(tz=_WARSAW)
    tomorrow_date = (now + timedelta(days=1)).date()
    return _time_range(
        datetime.combine(tomorrow_date, datetime.min.time(), tzinfo=_WARSAW),
        datetime.combine(tomorrow_date, datetime.max.time().replace(microsecond=0), tzinfo=_WARSAW),
    )


def _weekend() -> _TimeRangeDict:
    today = datetime.now(tz=_WARSAW).date()
    weekday = today.weekday()
    if weekday == 6:
        saturday = today - timedelta(days=1)
    elif weekday == 5:
        saturday = today
    else:
        saturday = today + timedelta(days=5 - weekday)
    sunday = saturday + timedelta(days=1)
    return _time_range(
        datetime.combine(saturday, datetime.min.time(), tzinfo=_WARSAW),
        datetime.combine(sunday, datetime.max.time().replace(microsecond=0), tzinfo=_WARSAW),
    )


def _next_hours(n: int | float) -> _TimeRangeDict:
    now = datetime.now(tz=_WARSAW)
    return _time_range(now, now + timedelta(hours=float(n)))


def _date_range(start_iso: str, end_iso: str) -> _TimeRangeDict:
    return _time_range(parse_datetime(start_iso, _WARSAW), parse_datetime(end_iso, _WARSAW))


def _between(time_range: Mapping[str, Any], start_hhmm: str, end_hhmm: str) -> _TimeRangeDict:
    start = _coerce_datetime(time_range["start"])
    start_h = int(start_hhmm[:2])
    start_m = int(start_hhmm[2:4])
    end_h = int(end_hhmm[:2])
    end_m = int(end_hhmm[2:4])
    return _time_range(
        start.replace(hour=start_h, minute=start_m),
        start.replace(hour=end_h, minute=end_m),
    )


def _minutes(n: int | float) -> float:
    return float(n)


def _hours(n: int | float) -> float:
    return float(n) * 60


def _previous_snapshot() -> str:
    return "previous"


def _coerce_datetime(value: object) -> datetime:
    if isinstance(value, str):
        return parse_datetime(value, _WARSAW)
    if isinstance(value, datetime):
        return ensure_aware(value, _WARSAW)
    raise RuleExpressionEvalError(f"Expected datetime value, got {type(value).__name__}")


def _points_in_range(
    points: Sequence[Mapping[str, Any]], time_range: Mapping[str, Any]
) -> list[_DataDict]:
    start = _coerce_datetime(time_range["start"])
    end = _coerce_datetime(time_range["end"])
    result: list[_DataDict] = []
    for point in points:
        target_time = point.get("target_time")
        if target_time is None:
            result.append(dict(point))
            continue
        coerced_target_time = _coerce_datetime(target_time)
        if start <= coerced_target_time <= end:
            result.append(dict(point))
    return result


def _metric_values(
    points: Sequence[Mapping[str, Any]],
    metric_key: str,
    time_range: Mapping[str, Any],
) -> list[float]:
    if metric_key not in ALLOWED_METRICS:
        raise RuleExpressionEvalError(f"Unknown metric: {metric_key}")
    values: list[float] = []
    for point in _points_in_range(points, time_range):
        value = point.get(metric_key)
        if value is not None:
            values.append(float(value))
    return values


def _ensure_time_range(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or "start" not in value or "end" not in value:
        raise RuleExpressionEvalError("Expected a time range with start and end")
    return value


def _require_values(values: Sequence[float], metric_key: str) -> Sequence[float]:
    if not values:
        raise RuleExpressionEvalError(f"No data points for metric '{metric_key}'")
    return values


def _percentile(values: Sequence[float], percentile: int | float) -> float:
    sorted_values = sorted(values)
    rank = (float(percentile) / 100.0) * (len(sorted_values) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_values[lower]
    fraction = rank - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def _build_functions(data: _DataDict, *, validation_mode: bool = False) -> dict[str, _Function]:
    def points_between(time_range: Mapping[str, Any]) -> list[_DataDict]:
        _ensure_time_range(time_range)
        if validation_mode:
            return [_sample_point(datetime.now(tz=_WARSAW))]
        return _points_in_range(data.get("points", []), time_range)

    def duration_minutes(points: Sequence[Mapping[str, Any]]) -> float:
        return float(len(points) * _HOURLY_RESOLUTION_MINUTES)

    def metric_values(metric_key: str, time_range: Mapping[str, Any]) -> list[float]:
        _ensure_time_range(time_range)
        if validation_mode:
            if metric_key not in ALLOWED_METRICS:
                raise RuleExpressionEvalError(f"Unknown metric: {metric_key}")
            return [1.0, 2.0]
        return _metric_values(data.get("points", []), metric_key, time_range)

    def min_metric(metric_key: str, time_range: Mapping[str, Any]) -> float:
        return min(_require_values(metric_values(metric_key, time_range), metric_key))

    def max_metric(metric_key: str, time_range: Mapping[str, Any]) -> float:
        return max(_require_values(metric_values(metric_key, time_range), metric_key))

    def avg_metric(metric_key: str, time_range: Mapping[str, Any]) -> float:
        values = _require_values(metric_values(metric_key, time_range), metric_key)
        return sum(values) / len(values)

    def sum_metric(metric_key: str, time_range: Mapping[str, Any]) -> float:
        return sum(_require_values(metric_values(metric_key, time_range), metric_key))

    def median_metric(metric_key: str, time_range: Mapping[str, Any]) -> float:
        values = sorted(_require_values(metric_values(metric_key, time_range), metric_key))
        midpoint = len(values) // 2
        if len(values) % 2 == 0:
            return (values[midpoint - 1] + values[midpoint]) / 2.0
        return values[midpoint]

    def stddev_metric(metric_key: str, time_range: Mapping[str, Any]) -> float:
        values = _require_values(metric_values(metric_key, time_range), metric_key)
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))

    def pctl_metric(
        metric_key: str, time_range: Mapping[str, Any], percentile: int | float
    ) -> float:
        return _percentile(
            _require_values(metric_values(metric_key, time_range), metric_key), percentile
        )

    def delta_metric(metric_key: str, time_range: Mapping[str, Any]) -> float:
        values = _require_values(metric_values(metric_key, time_range), metric_key)
        if len(values) < 2:
            raise RuleExpressionEvalError(
                f"Need at least 2 data points for delta, got {len(values)}"
            )
        return values[-1] - values[0]

    def abs_delta_metric(metric_key: str, time_range: Mapping[str, Any]) -> float:
        return abs(delta_metric(metric_key, time_range))

    def rate_of_change_metric(metric_key: str, time_range: Mapping[str, Any]) -> float:
        values = _require_values(metric_values(metric_key, time_range), metric_key)
        if len(values) < 2:
            raise RuleExpressionEvalError(
                f"Need at least 2 data points for rate_of_change, got {len(values)}"
            )
        if values[0] == 0:
            raise RuleExpressionEvalError(
                "Cannot compute rate of change when initial value is zero"
            )
        return (values[-1] - values[0]) / abs(values[0])

    def forecast_delta_metric(
        metric_key: str,
        time_range: Mapping[str, Any],
        snapshot_ref: str,
    ) -> float:
        if snapshot_ref != "previous":
            raise RuleExpressionEvalError(f"Unsupported snapshot reference: {snapshot_ref}")
        current_values = metric_values(metric_key, time_range)
        if validation_mode:
            return 1.0
        previous_values = _metric_values(data.get("previous_points", []), metric_key, time_range)
        if not current_values or not previous_values:
            raise RuleExpressionEvalError(f"Insufficient data for forecast_delta on '{metric_key}'")
        return (sum(current_values) / len(current_values)) - (
            sum(previous_values) / len(previous_values)
        )

    def clamp(value: int | float, min_value: int | float, max_value: int | float) -> float:
        return float(max(min_value, min(max_value, value)))

    return {
        "now": _now,
        "today": _today,
        "tomorrow": _tomorrow,
        "weekend": _weekend,
        "next_hours": _next_hours,
        "date_range": _date_range,
        "between": _between,
        "minutes": _minutes,
        "hours": _hours,
        "previous_snapshot": _previous_snapshot,
        "points_between": points_between,
        "duration_minutes": duration_minutes,
        "min_metric": min_metric,
        "max_metric": max_metric,
        "avg_metric": avg_metric,
        "sum_metric": sum_metric,
        "median_metric": median_metric,
        "stddev_metric": stddev_metric,
        "pctl_metric": pctl_metric,
        "delta_metric": delta_metric,
        "abs_delta_metric": abs_delta_metric,
        "rate_of_change_metric": rate_of_change_metric,
        "forecast_delta_metric": forecast_delta_metric,
        "abs": abs,
        "round": round,
        "clamp": clamp,
    }


def _sample_point(target_time: datetime) -> _DataDict:
    return {"target_time": target_time.isoformat(), **{metric: 1.0 for metric in ALLOWED_METRICS}}


def _context(data: _DataDict) -> cel.Context:
    points = [_cel_point(point) for point in data.get("points", [])]
    first_point = points[0] if points else {}
    variables = {
        metric: first_point.get(metric)
        for metric in ALLOWED_METRICS
        if isinstance(first_point, Mapping) and metric in first_point
    }
    variables["points"] = points
    cel_data = {
        **data,
        "points": points,
        "previous_points": [_cel_point(point) for point in data.get("previous_points", [])],
    }
    context = cel.Context(variables=variables, functions=_build_functions(cel_data))
    return context


def _validation_context() -> cel.Context:
    point = _sample_point(datetime.now(tz=_WARSAW))
    return cel.Context(
        variables={**point, "points": [point]},
        functions=_build_functions(
            {"points": [point], "previous_points": [point]},
            validation_mode=True,
        ),
    )


def _cel_point(point: Mapping[str, Any]) -> _DataDict:
    normalized = dict(point)
    target_time = normalized.get("target_time")
    if isinstance(target_time, datetime):
        normalized["target_time"] = ensure_aware(target_time, _WARSAW).isoformat()
    fetched_at = normalized.get("fetched_at")
    if isinstance(fetched_at, datetime):
        normalized["fetched_at"] = ensure_aware(fetched_at, _WARSAW).isoformat()
    return normalized


def _referenced_functions(expression: str) -> set[str]:
    names = set(_FUNCTION_NAME_RE.findall(expression))
    return {name for name in names if name in FUNCTION_REGISTRY}


def _referenced_metrics(expression: str) -> set[str]:
    metrics: set[str] = set()
    for match in _STRING_LITERAL_RE.finditer(expression):
        value = match.group(1) if match.group(1) is not None else match.group(2)
        if value in ALLOWED_METRICS:
            metrics.add(value)
    for name in _IDENTIFIER_RE.findall(expression):
        if name in ALLOWED_METRICS:
            metrics.add(name)
    return metrics


class RuleExpressionEvaluator:
    def __init__(self, data: _DataDict | None = None) -> None:
        self._data = data or {"points": []}

    def validate(self, expression: str) -> RuleExpressionEvaluationResult:
        validation = validate_expression(expression)
        if not validation.valid:
            return RuleExpressionEvaluationResult(expression=expression, error=validation.error)

        try:
            cel.compile(expression.strip()).execute(_validation_context())
        except Exception as exc:
            return RuleExpressionEvaluationResult(
                expression=expression,
                error=f"CEL validation error: {exc}",
                evaluated_metrics=sorted(_referenced_metrics(expression)),
                evaluated_functions=sorted(_referenced_functions(expression)),
            )

        return RuleExpressionEvaluationResult(
            expression=expression,
            result=None,
            evaluated_metrics=sorted(_referenced_metrics(expression)),
            evaluated_functions=sorted(_referenced_functions(expression)),
        )

    def evaluate(
        self, expression: str, data: _DataDict | None = None
    ) -> RuleExpressionEvaluationResult:
        evaluation_data = data if data is not None else self._data
        validation = self.validate(expression)
        if not validation.valid:
            return RuleExpressionEvaluationResult(expression=expression, error=validation.error)

        try:
            program = cel.compile(expression.strip())
            result = program.execute(_context(evaluation_data))
        except Exception as exc:
            return RuleExpressionEvaluationResult(
                expression=expression,
                error=f"Evaluation error: {exc}",
                evaluated_metrics=validation.evaluated_metrics,
                evaluated_functions=validation.evaluated_functions,
            )

        return RuleExpressionEvaluationResult(
            expression=expression,
            result=result,
            evaluated_metrics=validation.evaluated_metrics,
            evaluated_functions=validation.evaluated_functions,
        )


def time_range_to_domain(value: Mapping[str, Any]) -> TimeRange:
    return TimeRange(
        start=_coerce_datetime(value["start"]),
        end=_coerce_datetime(value["end"]),
    )
