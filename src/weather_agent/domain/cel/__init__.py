"""CEL expression engine with allowlist validation and safe evaluation."""

from __future__ import annotations

from weather_agent.domain.cel.allowlist import (
    ALL_ALLOWED_FUNCTION_NAMES,
    ALLOWED_FUNCTIONS,
    ALLOWED_METRICS,
    get_allowlist_for_prompt,
)
from weather_agent.domain.cel.evaluator import CELEvalError, CELEvaluationResult, CELEvaluator
from weather_agent.domain.cel.validation import ValidationResult, validate_expression

__all__ = [
    "ALL_ALLOWED_FUNCTION_NAMES",
    "ALLOWED_FUNCTIONS",
    "ALLOWED_METRICS",
    "CELEvalError",
    "CELEvaluationResult",
    "CELEvaluator",
    "ValidationResult",
    "get_allowlist_for_prompt",
    "validate_expression",
]
