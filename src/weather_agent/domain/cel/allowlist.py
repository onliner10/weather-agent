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

CEL_SIGNATURES: dict[str, str] = {
    "min": 'min("metric_name", time_range)',
    "max": 'max("metric_name", time_range)',
    "avg": 'avg("metric_name", time_range)',
    "sum": 'sum("metric_name", time_range)',
    "median": 'median("metric_name", time_range)',
    "stddev": 'stddev("metric_name", time_range)',
    "pctl": 'pctl("metric_name", time_range, percentile)',
    "delta": 'delta("metric_name", time_range)',
    "abs_delta": 'abs_delta("metric_name", time_range)',
    "rate_of_change": 'rate_of_change("metric_name", time_range)',
    "forecast_delta": 'forecast_delta("metric_name", time_range, previous_snapshot())',
    "today": "today()",
    "tomorrow": "tomorrow()",
    "weekend": "weekend()",
    "next_hours": "next_hours(6)",
    "date_range": 'date_range("2026-05-02T00:00:00+02:00", "2026-05-02T23:59:00+02:00")',
    "between": 'between(time_range, "0800", "1700")',
}

CEL_RULES: list[str] = [
    "Aggregation functions always use a quoted metric first, "
    'e.g. max("wind_gusts_10m_ms", weekend()).',
    "Aggregation functions always include a time range as the second argument.",
    "Use today(), tomorrow(), weekend(), next_hours(n), or date_range(...) "
    "to scope weather conditions.",
    "For scheduled weather checks, the schedule controls when to notify; "
    "the CEL still needs the forecast time range being checked.",
    "Do not write max(weekend, metric), min(metric), sum(metric, range), "
    "or a naked metric for future forecast checks.",
]

CEL_EXAMPLES: list[str] = [
    'max("wind_gusts_10m_ms", weekend()) > 12.0',
    'min("temperature_2m_c", today()) < -10.0',
    'sum("precipitation_mm", next_hours(6)) > 5.0',
    'max("wind_speed_10m_ms", tomorrow()) > 10.0',
    'max("relative_humidity_2m_pct", today()) > 90.0',
    'min("pressure_msl_hpa", today()) < 1000.0',
]


def get_allowlist_for_prompt() -> dict[str, object]:
    return {
        "functions": ALLOWED_FUNCTIONS,
        "metrics": ALLOWED_METRICS,
        "signatures": CEL_SIGNATURES,
        "rules": CEL_RULES,
        "examples": CEL_EXAMPLES,
    }
