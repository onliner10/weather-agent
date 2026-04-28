from __future__ import annotations

import ast
from dataclasses import dataclass, field

from weather_agent.domain.cel.allowlist import ALL_ALLOWED_FUNCTION_NAMES, ALLOWED_METRICS


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

    all_called = functions | {
        n for n in identifier_names if n in ALL_ALLOWED_FUNCTION_NAMES
    }
    unknown_functions = sorted(
        f for f in all_called if f not in ALL_ALLOWED_FUNCTION_NAMES
    )

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

    return ValidationResult(
        expression=expression,
        valid=True,
    )