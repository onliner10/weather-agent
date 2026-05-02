from __future__ import annotations

from weather_agent.domain.rule_expression.registry import (
    ALL_FUNCTION_NAMES,
    FUNCTION_NAMES_BY_CATEGORY,
    FUNCTION_SPECS,
)
from weather_agent.domain.weather import WeatherVariable  # noqa: I001

ALLOWED_FUNCTIONS: dict[str, list[str]] = dict(FUNCTION_NAMES_BY_CATEGORY)

ALLOWED_METRICS: list[str] = [v.value for v in WeatherVariable]

ALL_ALLOWED_FUNCTION_NAMES: set[str] = set(ALL_FUNCTION_NAMES)

CEL_SIGNATURES: dict[str, str] = {spec.name: spec.signature for spec in FUNCTION_SPECS}

CEL_RULES: list[str] = [
    "Aggregation functions always use a quoted metric first, "
    'e.g. max_metric("wind_gusts_10m_ms", weekend()).',
    "Aggregation functions always include a time range as the second argument.",
    "Use today(), tomorrow(), weekend(), next_hours(n), or date_range(...) "
    "to scope weather conditions.",
    "Use CEL boolean operators &&, ||, ! and null.",
    "Use CEL collection macros for point-level conditions, e.g. "
    "points_between(next_hours(12)).exists(p, p.precipitation_mm > 0.2).",
    "For scheduled weather checks, the schedule controls when to notify; "
    "the CEL expression still needs the forecast time range being checked.",
    "Do not write max(weekend, metric), min(metric), sum(metric, range), "
    "or a naked metric for future forecast checks.",
]

CEL_EXAMPLES: list[str] = [
    'max_metric("wind_gusts_10m_ms", weekend()) > 12.0',
    'min_metric("temperature_2m_c", today()) < -10.0',
    'sum_metric("precipitation_mm", next_hours(6)) > 5.0',
    'max_metric("wind_speed_10m_ms", tomorrow()) > 10.0',
    'max_metric("relative_humidity_2m_pct", today()) > 90.0',
    'min_metric("pressure_msl_hpa", today()) < 1000.0',
    "points_between(next_hours(12)).exists(p, p.precipitation_mm > 0.2)",
]


def get_allowlist_for_prompt() -> dict[str, object]:
    return {
        "functions": ALLOWED_FUNCTIONS,
        "metrics": ALLOWED_METRICS,
        "signatures": CEL_SIGNATURES,
        "rules": CEL_RULES,
        "examples": CEL_EXAMPLES,
    }
