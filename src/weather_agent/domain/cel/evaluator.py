from __future__ import annotations

import ast
import math
import operator
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from weather_agent.domain.cel.allowlist import ALL_ALLOWED_FUNCTION_NAMES, ALLOWED_METRICS
from weather_agent.domain.cel.validation import ValidationResult, validate_expression
from weather_agent.domain.weather import TimeRange

_WARSAW = ZoneInfo("Europe/Warsaw")

_DataDict = dict[str, Any]

_AGGREGATION_METRIC_FUNCTIONS = frozenset(
    {
        "min",
        "max",
        "avg",
        "sum",
        "median",
        "stddev",
        "pctl",
        "delta",
        "abs_delta",
        "rate_of_change",
        "forecast_delta",
    }
)

_HOURLY_RESOLUTION_MINUTES = 60


class CELEvaluationResult(BaseModel):
    expression: str
    result: bool | float | int | str | None = None
    error: str | None = None
    evaluated_metrics: list[str] = []
    evaluated_functions: list[str] = []

    @property
    def valid(self) -> bool:
        return self.error is None


class TimeRangeValue:
    def __init__(self, start: datetime, end: datetime) -> None:
        self.start = start
        self.end = end

    def to_time_range(self) -> TimeRange:
        return TimeRange(start=self.start, end=self.end)

    def __repr__(self) -> str:
        return f"TimeRangeValue({self.start.isoformat()}, {self.end.isoformat()})"


class SnapshotRef:
    def __init__(self, label: str = "previous") -> None:
        self.label = label

    def __repr__(self) -> str:
        return f"SnapshotRef({self.label})"


def _now() -> TimeRangeValue:
    now = datetime.now(tz=_WARSAW)
    return TimeRangeValue(now, now)


def _today() -> TimeRangeValue:
    now = datetime.now(tz=_WARSAW)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = now.replace(hour=23, minute=59, second=59, microsecond=0)
    return TimeRangeValue(start, end)


def _tomorrow() -> TimeRangeValue:
    now = datetime.now(tz=_WARSAW)
    tomorrow_date = (now + timedelta(days=1)).date()
    start = datetime.combine(tomorrow_date, datetime.min.time(), tzinfo=_WARSAW)
    end = datetime.combine(
        tomorrow_date,
        datetime.max.time().replace(microsecond=0),
        tzinfo=_WARSAW,
    )
    return TimeRangeValue(start, end)


def _weekend() -> TimeRangeValue:
    now = datetime.now(tz=_WARSAW)
    today = now.date()
    weekday = today.weekday()
    if weekday == 6:
        saturday = today - timedelta(days=1)
    elif weekday == 5:
        saturday = today
    else:
        saturday = today + timedelta(days=5 - weekday)
    sunday = saturday + timedelta(days=1)
    start = datetime.combine(saturday, datetime.min.time(), tzinfo=_WARSAW)
    end = datetime.combine(sunday, datetime.max.time().replace(microsecond=0), tzinfo=_WARSAW)
    return TimeRangeValue(start, end)


def _next_hours(n: int | float) -> TimeRangeValue:
    now = datetime.now(tz=_WARSAW)
    end = now + timedelta(hours=int(n))
    return TimeRangeValue(now, end)


def _date_range(start_iso: str, end_iso: str) -> TimeRangeValue:
    start = datetime.fromisoformat(start_iso).replace(tzinfo=_WARSAW)
    end = datetime.fromisoformat(end_iso).replace(tzinfo=_WARSAW)
    return TimeRangeValue(start, end)


def _between(time_range: TimeRangeValue, start_hhmm: str, end_hhmm: str) -> TimeRangeValue:
    start_h = int(start_hhmm[:2])
    start_m = int(start_hhmm[2:4])
    end_h = int(end_hhmm[:2])
    end_m = int(end_hhmm[2:4])
    constrained_start = time_range.start.replace(hour=start_h, minute=start_m)
    constrained_end = time_range.start.replace(hour=end_h, minute=end_m)
    return TimeRangeValue(constrained_start, constrained_end)


def _minutes(n: int | float) -> float:
    return float(int(n))


def _hours(n: int | float) -> float:
    return float(int(n) * 60)


def _previous_snapshot() -> SnapshotRef:
    return SnapshotRef("previous")


def _points_in_range(points: list[_DataDict], time_range: TimeRangeValue) -> list[_DataDict]:
    result: list[_DataDict] = []
    for point in points:
        target_time = point.get("target_time")
        if target_time is None:
            result.append(point)
            continue
        if isinstance(target_time, str):
            target_time = datetime.fromisoformat(target_time)
        if time_range.start <= target_time <= time_range.end:
            result.append(point)
    return result if result else points


def _metric_agg(metric_key: str, time_range: TimeRangeValue, data: _DataDict) -> list[float]:
    if not isinstance(metric_key, str):
        raise CELEvalError(f"Metric key must be a string, got {type(metric_key).__name__}")
    points = data.get("points", [])
    filtered = _points_in_range(points, time_range)
    values: list[float] = []
    for point in filtered:
        val = point.get(metric_key)
        if val is not None:
            values.append(float(val))
    return values


def _min_(metric_key: str, time_range: TimeRangeValue, data: _DataDict) -> float:
    values = _metric_agg(metric_key, time_range, data)
    if not values:
        raise CELEvalError(f"No data points for metric '{metric_key}'")
    return min(values)


def _max_(metric_key: str, time_range: TimeRangeValue, data: _DataDict) -> float:
    values = _metric_agg(metric_key, time_range, data)
    if not values:
        raise CELEvalError(f"No data points for metric '{metric_key}'")
    return max(values)


def _avg_(metric_key: str, time_range: TimeRangeValue, data: _DataDict) -> float:
    values = _metric_agg(metric_key, time_range, data)
    if not values:
        raise CELEvalError(f"No data points for metric '{metric_key}'")
    return sum(values) / len(values)


def _sum_(metric_key: str, time_range: TimeRangeValue, data: _DataDict) -> float:
    values = _metric_agg(metric_key, time_range, data)
    if not values:
        raise CELEvalError(f"No data points for metric '{metric_key}'")
    return sum(values)


def _median_(metric_key: str, time_range: TimeRangeValue, data: _DataDict) -> float:
    values = _metric_agg(metric_key, time_range, data)
    if not values:
        raise CELEvalError(f"No data points for metric '{metric_key}'")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    return sorted_vals[mid]


def _stddev_(metric_key: str, time_range: TimeRangeValue, data: _DataDict) -> float:
    values = _metric_agg(metric_key, time_range, data)
    if not values:
        raise CELEvalError(f"No data points for metric '{metric_key}'")
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _pctl_(
    metric_key: str,
    time_range: TimeRangeValue,
    percentile: int | float,
    data: _DataDict,
) -> float:
    values = _metric_agg(metric_key, time_range, data)
    if not values:
        raise CELEvalError(f"No data points for metric '{metric_key}'")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    rank = (float(percentile) / 100.0) * (n - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_vals[lower]
    frac = rank - lower
    return sorted_vals[lower] + frac * (sorted_vals[upper] - sorted_vals[lower])


def _delta_(metric_key: str, time_range: TimeRangeValue, data: _DataDict) -> float:
    values = _metric_agg(metric_key, time_range, data)
    if len(values) < 2:
        raise CELEvalError(f"Need at least 2 data points for delta, got {len(values)}")
    return values[-1] - values[0]


def _abs_delta_(metric_key: str, time_range: TimeRangeValue, data: _DataDict) -> float:
    return abs(_delta_(metric_key, time_range, data))


def _rate_of_change_(metric_key: str, time_range: TimeRangeValue, data: _DataDict) -> float:
    values = _metric_agg(metric_key, time_range, data)
    if len(values) < 2:
        raise CELEvalError(f"Need at least 2 data points for rate_of_change, got {len(values)}")
    if values[0] == 0:
        raise CELEvalError("Cannot compute rate of change when initial value is zero")
    return (values[-1] - values[0]) / abs(values[0])


def _forecast_delta_(
    metric_key: str,
    time_range: TimeRangeValue,
    snapshot_ref: SnapshotRef,
    data: _DataDict,
) -> float:
    current_values = _metric_agg(metric_key, time_range, data)
    previous_points = data.get("previous_points", [])
    prev_data: _DataDict = {**data, "points": previous_points}
    previous_values = _metric_agg(metric_key, time_range, prev_data)

    if not current_values or not previous_values:
        raise CELEvalError(f"Insufficient data for forecast_delta on '{metric_key}'")

    curr_avg = sum(current_values) / len(current_values)
    prev_avg = sum(previous_values) / len(previous_values)
    return curr_avg - prev_avg


def _filter_points_by_condition(
    condition_expr: ast.Expression,
    time_range: TimeRangeValue,
    data: _DataDict,
    namespace: dict[str, Any],
) -> list[_DataDict]:
    points = data.get("points", [])
    filtered = _points_in_range(points, time_range)
    matching: list[_DataDict] = []
    for point in filtered:
        point_ns: dict[str, Any] = {**namespace}
        for k, v in point.items():
            if k not in ("target_time", "fetched_at", "raw_payload"):
                point_ns[k] = v
        try:
            result = _safe_eval_node(condition_expr, point_ns)
            if result:
                matching.append(point)
        except Exception:
            continue
    return matching


def _duration_where_(
    condition_expr: ast.Expression,
    time_range: TimeRangeValue,
    data: _DataDict,
    namespace: dict[str, Any],
) -> float:
    matching = _filter_points_by_condition(condition_expr, time_range, data, namespace)
    return float(len(matching) * _HOURLY_RESOLUTION_MINUTES)


def _count_where_(
    condition_expr: ast.Expression,
    time_range: TimeRangeValue,
    data: _DataDict,
    namespace: dict[str, Any],
) -> int:
    matching = _filter_points_by_condition(condition_expr, time_range, data, namespace)
    return len(matching)


def _any_(
    condition_expr: ast.Expression,
    time_range: TimeRangeValue,
    data: _DataDict,
    namespace: dict[str, Any],
) -> bool:
    matching = _filter_points_by_condition(condition_expr, time_range, data, namespace)
    return len(matching) > 0


def _all_(
    condition_expr: ast.Expression,
    time_range: TimeRangeValue,
    data: _DataDict,
    namespace: dict[str, Any],
) -> bool:
    points = data.get("points", [])
    filtered = _points_in_range(points, time_range)
    if not filtered:
        return False
    matching = 0
    for point in filtered:
        point_ns: dict[str, Any] = {**namespace}
        for k, v in point.items():
            if k not in ("target_time", "fetched_at", "raw_payload"):
                point_ns[k] = v
        try:
            result = _safe_eval_node(condition_expr, point_ns)
            if result:
                matching += 1
        except Exception:
            return False
    return matching == len(filtered)


class CELEvalError(Exception):
    pass


_BIN_OPS: dict[type[ast.AST], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_CMP_OPS: dict[type[ast.AST], Any] = {
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}

_BOOL_OPS: dict[type[ast.AST], Any] = {
    ast.And: lambda a, b: a and b,
    ast.Or: lambda a, b: a or b,
}


class _SafeEvaluator(ast.NodeVisitor):
    def __init__(self, namespace: dict[str, Any], data: _DataDict) -> None:
        self._namespace = namespace
        self._data = data

    def evaluate(self, node: ast.AST) -> Any:
        return self._eval_node(node)

    def _eval_node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self._namespace:
                return self._namespace[node.id]
            if node.id == "null":
                return None
            if node.id == "True":
                return True
            if node.id == "False":
                return False
            raise CELEvalError(f"Unknown name: {node.id}")
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.Not):
                return not operand
            raise CELEvalError(f"Unsupported unary op: {type(node.op).__name__}")
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_func = _BIN_OPS.get(type(node.op))
            if op_func is None:
                raise CELEvalError(f"Unsupported binary op: {type(node.op).__name__}")
            return op_func(left, right)
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left)
            result = True
            current = left
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                right = self._eval_node(comparator)
                cmp_func = _CMP_OPS.get(type(op))
                if cmp_func is None:
                    raise CELEvalError(f"Unsupported comparison: {type(op).__name__}")
                result = result and cmp_func(current, right)
                current = right
            return result
        if isinstance(node, ast.BoolOp):
            values = [self._eval_node(v) for v in node.values]
            op_func = _BOOL_OPS.get(type(node.op))
            if op_func is None:
                raise CELEvalError(f"Unsupported boolean op: {type(node.op).__name__}")
            from functools import reduce

            return reduce(op_func, values)
        if isinstance(node, ast.Call):
            return self._eval_call(node)
        if isinstance(node, ast.IfExp):
            test = self._eval_node(node.test)
            if test:
                return self._eval_node(node.body)
            return self._eval_node(node.orelse)
        if isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elt) for elt in node.elts)
        if isinstance(node, ast.List):
            return [self._eval_node(elt) for elt in node.elts]
        raise CELEvalError(f"Unsupported AST node: {type(node).__name__}")

    def _eval_call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        else:
            raise CELEvalError(f"Unsupported call type: {type(node.func).__name__}")

        if func_name not in ALL_ALLOWED_FUNCTION_NAMES:
            raise CELEvalError(f"Unknown function: {func_name}")

        args = [self._eval_node(a) for a in node.args]

        return self._dispatch_function(func_name, args, node)

    def _dispatch_function(self, name: str, args: list[Any], call_node: ast.Call) -> Any:
        if name == "now":
            return _now()
        if name == "today":
            return _today()
        if name == "tomorrow":
            return _tomorrow()
        if name == "weekend":
            return _weekend()
        if name == "next_hours":
            return _next_hours(args[0])
        if name == "date_range":
            return _date_range(args[0], args[1])
        if name == "between":
            return _between(args[0], args[1], args[2])
        if name == "minutes":
            return _minutes(args[0])
        if name == "hours":
            return _hours(args[0])
        if name == "previous_snapshot":
            return _previous_snapshot()

        if name == "min":
            return _min_(args[0], args[1], self._data)
        if name == "max":
            return _max_(args[0], args[1], self._data)
        if name == "avg":
            return _avg_(args[0], args[1], self._data)
        if name == "sum":
            return _sum_(args[0], args[1], self._data)
        if name == "median":
            return _median_(args[0], args[1], self._data)
        if name == "stddev":
            return _stddev_(args[0], args[1], self._data)
        if name == "pctl":
            return _pctl_(args[0], args[1], args[2], self._data)

        if name == "delta":
            return _delta_(args[0], args[1], self._data)
        if name == "abs_delta":
            return _abs_delta_(args[0], args[1], self._data)
        if name == "rate_of_change":
            return _rate_of_change_(args[0], args[1], self._data)
        if name == "forecast_delta":
            return _forecast_delta_(args[0], args[1], args[2], self._data)

        if name == "duration_where":
            return self._eval_condition_over_time(_duration_where_, call_node)
        if name == "count_where":
            return self._eval_condition_over_time(_count_where_, call_node)
        if name == "any":
            return self._eval_condition_over_time(_any_, call_node)
        if name == "all":
            return self._eval_condition_over_time(_all_, call_node)

        if name == "abs":
            return abs(args[0])
        if name == "round":
            if len(args) >= 2:
                return round(args[0], int(args[1]))
            return round(args[0])
        if name == "clamp":
            return max(args[1], min(args[2], args[0]))

        raise CELEvalError(f"Unimplemented function: {name}")

    def _eval_condition_over_time(self, func: Any, call_node: ast.Call) -> Any:
        if len(call_node.args) < 2:
            raise CELEvalError(f"{call_node.func} requires (condition, range) arguments")
        condition_arg = call_node.args[0]
        range_arg = self._eval_node(call_node.args[1])
        condition_expr = ast.fix_missing_locations(ast.Expression(body=condition_arg))
        return func(condition_expr, range_arg, self._data, self._namespace)


def _safe_eval_node(tree: ast.Expression, namespace: dict[str, Any]) -> Any:
    evaluator = _SafeEvaluator(namespace, {})
    return evaluator.evaluate(tree)


def _extract_referenced_names(node: ast.AST) -> tuple[set[str], set[str]]:
    functions: set[str] = set()
    metrics: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, call_node: ast.Call) -> None:
            if isinstance(call_node.func, ast.Name):
                functions.add(call_node.func.id)
                if call_node.func.id in _AGGREGATION_METRIC_FUNCTIONS:
                    for arg in call_node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            if arg.value in ALLOWED_METRICS:
                                metrics.add(arg.value)
            self.generic_visit(call_node)

        def visit_Name(self, name_node: ast.Name) -> None:
            if name_node.id in ALLOWED_METRICS:
                metrics.add(name_node.id)
            self.generic_visit(name_node)

    visitor = Visitor()
    visitor.visit(node)
    return functions, metrics


class CELEvaluator:
    def __init__(self, data: _DataDict | None = None) -> None:
        self._data = data or {"points": []}

    def validate(self, expression: str) -> CELEvaluationResult:
        validation: ValidationResult = validate_expression(expression)
        if not validation.valid:
            return CELEvaluationResult(
                expression=expression,
                error=validation.error,
            )

        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except SyntaxError as exc:
            return CELEvaluationResult(
                expression=expression,
                error=f"Syntax error: {exc.msg}",
            )

        functions, metrics = _extract_referenced_names(tree)
        return CELEvaluationResult(
            expression=expression,
            result=None,
            evaluated_metrics=sorted(metrics),
            evaluated_functions=sorted(functions),
        )

    def evaluate(self, expression: str, data: _DataDict | None = None) -> CELEvaluationResult:
        evaluation_data: _DataDict = data if data is not None else self._data

        validation = self.validate(expression)
        if not validation.valid:
            return CELEvaluationResult(
                expression=expression,
                error=validation.error,
            )

        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except SyntaxError as exc:
            val_metrics = getattr(validation, "evaluated_metrics", [])
            val_funcs = getattr(validation, "evaluated_functions", [])
            return CELEvaluationResult(
                expression=expression,
                error=f"Syntax error: {exc.msg}",
                evaluated_metrics=val_metrics,
                evaluated_functions=val_funcs,
            )

        points = evaluation_data.get("points", [])
        first_point = points[0] if points else {}

        namespace: dict[str, Any] = {}
        for k, v in first_point.items():
            if k not in ("target_time", "fetched_at", "raw_payload"):
                namespace[k] = v

        for metric in validation.evaluated_metrics:
            if metric not in namespace:
                namespace[metric] = None

        evaluator = _SafeEvaluator(namespace, evaluation_data)

        try:
            result = evaluator.evaluate(tree)
        except CELEvalError as exc:
            return CELEvaluationResult(
                expression=expression,
                error=str(exc),
                evaluated_metrics=validation.evaluated_metrics,
                evaluated_functions=validation.evaluated_functions,
            )
        except Exception as exc:
            return CELEvaluationResult(
                expression=expression,
                error=f"Evaluation error: {exc}",
                evaluated_metrics=validation.evaluated_metrics,
                evaluated_functions=validation.evaluated_functions,
            )

        return CELEvaluationResult(
            expression=expression,
            result=result,
            evaluated_metrics=validation.evaluated_metrics,
            evaluated_functions=validation.evaluated_functions,
        )
