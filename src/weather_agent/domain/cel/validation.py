from __future__ import annotations

import ast
from dataclasses import dataclass, field

from weather_agent.domain.cel.allowlist import ALL_ALLOWED_FUNCTION_NAMES, ALLOWED_METRICS

_TIME_RANGE_HELPERS = frozenset(
    {
        "now",
        "today",
        "tomorrow",
        "weekend",
        "next_hours",
        "date_range",
        "between",
    }
)
_AGGREGATION_FUNCTIONS = frozenset(
    {
        "min",
        "max",
        "avg",
        "sum",
        "median",
        "stddev",
        "delta",
        "abs_delta",
        "rate_of_change",
    }
)
_THREE_ARG_METRIC_FUNCTIONS = frozenset({"pctl", "forecast_delta"})


@dataclass
class ValidationResult:
    expression: str
    valid: bool
    error: str | None = None
    unknown_functions: list[str] = field(default_factory=list)
    unknown_metrics: list[str] = field(default_factory=list)


def _extract_node_names(node: ast.AST) -> tuple[set[str], set[str]]:
    functions: set[str] = set()
    names: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, call_node: ast.Call) -> None:
            if isinstance(call_node.func, ast.Name):
                functions.add(call_node.func.id)
            elif isinstance(call_node.func, ast.Attribute):
                attr = call_node.func
                functions.add(attr.attr)
            self.generic_visit(call_node)

        def visit_Name(self, name_node: ast.Name) -> None:
            names.add(name_node.id)
            self.generic_visit(name_node)

    visitor = Visitor()
    visitor.visit(node)
    return functions, names


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _is_metric_literal(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in ALLOWED_METRICS
    )


def _is_time_range_arg(node: ast.AST) -> bool:
    name = _call_name(node)
    if name is None:
        return False
    if name in _TIME_RANGE_HELPERS and name != "between":
        return True
    if isinstance(node, ast.Call) and name == "between" and node.args:
        return _is_time_range_arg(node.args[0])
    return False


def _semantic_signature_errors(node: ast.AST) -> list[str]:
    errors: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, call_node: ast.Call) -> None:
            if not isinstance(call_node.func, ast.Name):
                self.generic_visit(call_node)
                return

            name = call_node.func.id
            if name in _AGGREGATION_FUNCTIONS:
                if len(call_node.args) != 2:
                    errors.append(
                        f'{name} expects exactly 2 arguments: {name}("metric_name", time_range)'
                    )
                else:
                    metric_arg, range_arg = call_node.args
                    if not _is_metric_literal(metric_arg):
                        errors.append(f"{name} first argument must be a quoted allowed metric name")
                    if not _is_time_range_arg(range_arg):
                        errors.append(f"{name} second argument must be a time range helper call")
            elif name == "pctl":
                if len(call_node.args) != 3:
                    errors.append(
                        "pctl expects exactly 3 arguments: "
                        'pctl("metric_name", time_range, percentile)'
                    )
                else:
                    metric_arg, range_arg, _percentile_arg = call_node.args
                    if not _is_metric_literal(metric_arg):
                        errors.append("pctl first argument must be a quoted allowed metric name")
                    if not _is_time_range_arg(range_arg):
                        errors.append("pctl second argument must be a time range helper call")
            elif name == "forecast_delta":
                if len(call_node.args) != 3:
                    errors.append(
                        "forecast_delta expects exactly 3 arguments: "
                        'forecast_delta("metric_name", time_range, previous_snapshot())'
                    )
                else:
                    metric_arg, range_arg, previous_arg = call_node.args
                    if not _is_metric_literal(metric_arg):
                        errors.append(
                            "forecast_delta first argument must be a quoted allowed metric name"
                        )
                    if not _is_time_range_arg(range_arg):
                        errors.append(
                            "forecast_delta second argument must be a time range helper call"
                        )
                    if _call_name(previous_arg) != "previous_snapshot":
                        errors.append("forecast_delta third argument must be previous_snapshot()")
            elif name == "next_hours":
                if len(call_node.args) != 1:
                    errors.append("next_hours expects exactly 1 numeric argument")

            self.generic_visit(call_node)

    Visitor().visit(node)
    return errors


def validate_expression(expression: str) -> ValidationResult:
    stripped = expression.strip()
    if not stripped:
        return ValidationResult(
            expression=expression,
            valid=False,
            error="Empty expression",
        )

    try:
        tree = ast.parse(stripped, mode="eval")
    except SyntaxError as exc:
        return ValidationResult(
            expression=expression,
            valid=False,
            error=f"Syntax error: {exc.msg}",
        )

    functions, names = _extract_node_names(tree)

    python_builtins_and_keywords = {
        "True",
        "False",
        "None",
        "null",
        "and",
        "or",
        "not",
        "in",
        "is",
    }

    identifier_names = names - python_builtins_and_keywords
    unknown_metrics = sorted(
        n
        for n in identifier_names
        if n not in ALLOWED_METRICS and n not in ALL_ALLOWED_FUNCTION_NAMES
    )

    all_called = functions | {n for n in identifier_names if n in ALL_ALLOWED_FUNCTION_NAMES}
    unknown_functions = sorted(f for f in all_called if f not in ALL_ALLOWED_FUNCTION_NAMES)

    combined_unknown = sorted(set(unknown_functions) | set(unknown_metrics))

    if combined_unknown:
        parts: list[str] = []
        if unknown_functions:
            parts.append(f"Unknown functions: {unknown_functions}")
        if unknown_metrics:
            parts.append(f"Unknown metrics/variables: {unknown_metrics}")
        return ValidationResult(
            expression=expression,
            valid=False,
            error="; ".join(parts),
            unknown_functions=unknown_functions,
            unknown_metrics=unknown_metrics,
        )

    signature_errors = _semantic_signature_errors(tree)
    if signature_errors:
        return ValidationResult(
            expression=expression,
            valid=False,
            error="Invalid function signature: " + "; ".join(signature_errors),
        )

    return ValidationResult(
        expression=expression,
        valid=True,
    )
