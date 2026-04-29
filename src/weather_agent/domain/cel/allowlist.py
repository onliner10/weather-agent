from __future__ import annotations

from weather_agent.domain.weather import WeatherVariable  # noqa: I001

ALLOWED_FUNCTIONS: dict[str, list[str]] = {
    "time_range_helpers": [
        "now",
        "today",
        "tomorrow",
        "weekend",
        "next_hours",
        "date_range",
        "between",
        "minutes",
        "hours",
        "previous_snapshot",
    ],
    "aggregation": [
        "min",
        "max",
        "avg",
        "sum",
        "median",
        "stddev",
        "pctl",
    ],
    "change_trend": [
        "delta",
        "abs_delta",
        "rate_of_change",
        "forecast_delta",
    ],
    "condition_over_time": [
        "duration_where",
        "count_where",
        "any",
        "all",
    ],
    "numeric": [
        "abs",
        "round",
        "clamp",
    ],
}

ALLOWED_METRICS: list[str] = [v.value for v in WeatherVariable]

ALL_ALLOWED_FUNCTION_NAMES: set[str] = {
    name for names in ALLOWED_FUNCTIONS.values() for name in names
}


def get_allowlist_for_prompt() -> dict[str, object]:
    return {
        "functions": ALLOWED_FUNCTIONS,
        "metrics": ALLOWED_METRICS,
    }
