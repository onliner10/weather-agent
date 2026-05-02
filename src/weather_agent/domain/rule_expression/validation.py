from __future__ import annotations

import re
from dataclasses import dataclass, field

import cel

from weather_agent.domain.rule_expression.allowlist import (
    ALL_ALLOWED_FUNCTION_NAMES,
    ALLOWED_METRICS,
)

_FUNCTION_NAME_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_STRING_LITERAL_RE = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"|\'([^\'\\]*(?:\\.[^\'\\]*)*)\'')
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_MACRO_VAR_RE = re.compile(r"\.(?:exists|all|filter|map)\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,")
_CEL_RESERVED_WORDS = frozenset(
    {
        "true",
        "false",
        "null",
        "in",
        "has",
        "exists",
        "all",
        "filter",
        "map",
        "size",
    }
)


@dataclass
class ValidationResult:
    expression: str
    valid: bool
    error: str | None = None
    unknown_functions: list[str] = field(default_factory=list)
    unknown_metrics: list[str] = field(default_factory=list)


def _unknown_names(expression: str) -> tuple[list[str], list[str]]:
    functions = set(_FUNCTION_NAME_RE.findall(expression))
    unknown_functions = sorted(
        name
        for name in functions
        if name not in ALL_ALLOWED_FUNCTION_NAMES and name not in _CEL_RESERVED_WORDS
    )
    macro_variables = set(_MACRO_VAR_RE.findall(expression))
    stripped = _STRING_LITERAL_RE.sub("", expression)
    unknown_metrics = sorted(
        name
        for match in _IDENTIFIER_RE.finditer(stripped)
        if (
            (name := match.group(0)) not in ALLOWED_METRICS
            and name not in ALL_ALLOWED_FUNCTION_NAMES
            and name not in _CEL_RESERVED_WORDS
            and name not in functions
            and name not in macro_variables
            and (match.start() == 0 or stripped[match.start() - 1] != ".")
        )
    )
    return unknown_functions, unknown_metrics


def validate_expression(expression: str) -> ValidationResult:
    stripped = expression.strip()
    if not stripped:
        return ValidationResult(expression=expression, valid=False, error="Empty expression")

    unknown_functions, unknown_metrics = _unknown_names(stripped)
    if unknown_functions or unknown_metrics:
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

    try:
        cel.compile(stripped)
    except Exception as exc:
        return ValidationResult(
            expression=expression,
            valid=False,
            error=f"CEL syntax error: {exc}",
        )

    return ValidationResult(expression=expression, valid=True)
