from __future__ import annotations

import json

from weather_agent.domain.cel.allowlist import (
    ALL_ALLOWED_FUNCTION_NAMES,
    ALLOWED_FUNCTIONS,
    ALLOWED_METRICS,
    get_allowlist_for_prompt,
)
from weather_agent.domain.weather import WeatherVariable


class TestAllowlistCompleteness:
    def test_all_weather_variables_in_metrics(self) -> None:
        for var in WeatherVariable:
            assert var.value in ALLOWED_METRICS, f"Missing metric: {var.value}"

    def test_metrics_match_weather_variables(self) -> None:
        expected = [v.value for v in WeatherVariable]
        assert sorted(ALLOWED_METRICS) == sorted(expected)

    def test_all_categories_present(self) -> None:
        expected_categories = [
            "time_range_helpers",
            "aggregation",
            "change_trend",
            "condition_over_time",
            "numeric",
        ]
        for cat in expected_categories:
            assert cat in ALLOWED_FUNCTIONS, f"Missing category: {cat}"

    def test_all_function_names_deduplicated(self) -> None:
        all_names: list[str] = []
        for names in ALLOWED_FUNCTIONS.values():
            all_names.extend(names)
        assert len(all_names) == len(set(all_names)), "Duplicate function names found"

    def test_all_allowed_names_set_matches(self) -> None:
        expected = {name for names in ALLOWED_FUNCTIONS.values() for name in names}
        assert ALL_ALLOWED_FUNCTION_NAMES == expected

    def test_time_range_helpers(self) -> None:
        expected = [
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
        ]
        assert ALLOWED_FUNCTIONS["time_range_helpers"] == expected

    def test_aggregation_helpers(self) -> None:
        expected = ["min", "max", "avg", "sum", "median", "stddev", "pctl"]
        assert ALLOWED_FUNCTIONS["aggregation"] == expected

    def test_change_trend_helpers(self) -> None:
        expected = ["delta", "abs_delta", "rate_of_change", "forecast_delta"]
        assert ALLOWED_FUNCTIONS["change_trend"] == expected

    def test_condition_over_time_helpers(self) -> None:
        expected = ["duration_where", "count_where", "any", "all"]
        assert ALLOWED_FUNCTIONS["condition_over_time"] == expected

    def test_numeric_helpers(self) -> None:
        expected = ["abs", "round", "clamp"]
        assert ALLOWED_FUNCTIONS["numeric"] == expected


class TestGetAllowlistForPrompt:
    def test_returns_dict(self) -> None:
        result = get_allowlist_for_prompt()
        assert isinstance(result, dict)
        assert "functions" in result
        assert "metrics" in result

    def test_serializable_to_json(self) -> None:
        result = get_allowlist_for_prompt()
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed["functions"] == ALLOWED_FUNCTIONS
        assert parsed["metrics"] == ALLOWED_METRICS

    def test_functions_contain_all_categories(self) -> None:
        result = get_allowlist_for_prompt()
        functions = result["functions"]
        assert len(functions) == len(ALLOWED_FUNCTIONS)

    def test_metrics_contain_all_variables(self) -> None:
        result = get_allowlist_for_prompt()
        assert len(result["metrics"]) == len(WeatherVariable)
