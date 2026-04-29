from __future__ import annotations

from weather_agent.domain.cel.validation import validate_expression


class TestValidateSyntax:
    def test_valid_simple_comparison(self) -> None:
        result = validate_expression("temperature_2m_c > 20")
        assert result.valid

    def test_valid_function_call(self) -> None:
        result = validate_expression('max("wind_gusts_10m_ms", weekend()) >= 12')
        assert result.valid

    def test_valid_avg_comparison(self) -> None:
        result = validate_expression('avg("wind_speed_10m_ms", next_hours(24)) >= 7')
        assert result.valid

    def test_valid_numeric(self) -> None:
        result = validate_expression("abs(-5) == 5")
        assert result.valid

    def test_empty_expression(self) -> None:
        result = validate_expression("")
        assert not result.valid
        assert "Empty" in result.error

    def test_whitespace_only(self) -> None:
        result = validate_expression("   ")
        assert not result.valid

    def test_syntax_error(self) -> None:
        result = validate_expression("temperature_2m_c >>> 20")
        assert not result.valid
        assert "Syntax error" in result.error

    def test_unclosed_paren(self) -> None:
        result = validate_expression("max(temperature_2m_c, weekend()")
        assert not result.valid

    def test_valid_boolean_expression(self) -> None:
        result = validate_expression("temperature_2m_c > 20 and rain_mm < 5")
        assert result.valid


class TestUnknownFunctionRejection:
    def test_unknown_function(self) -> None:
        result = validate_expression('foo("temperature_2m_c") > 0')
        assert not result.valid
        assert "foo" in result.error
        assert "Unknown functions" in result.error

    def test_known_function_passes(self) -> None:
        result = validate_expression('max("temperature_2m_c", weekend()) > 10')
        assert result.valid

    def test_all_aggregation_functions_pass(self) -> None:
        for func in ["min", "max", "avg", "sum", "median", "stddev", "pctl"]:
            result = validate_expression(f'{func}("temperature_2m_c", weekend()) > 0')
            assert result.valid, f"Function {func} failed validation"

    def test_all_time_functions_pass(self) -> None:
        for func in ["now", "today", "tomorrow", "weekend", "previous_snapshot"]:
            result = validate_expression(f"{func}() != None")
            assert result.valid, f"Function {func} failed validation"

    def test_all_change_trend_functions_pass(self) -> None:
        for func in ["delta", "abs_delta", "rate_of_change"]:
            result = validate_expression(f'{func}("temperature_2m_c", weekend()) > 0')
            assert result.valid, f"Function {func} failed validation"

    def test_forecast_delta_passes(self) -> None:
        result = validate_expression(
            'forecast_delta("temperature_2m_c", weekend(), previous_snapshot()) > 0'
        )
        assert result.valid


class TestUnknownMetricRejection:
    def test_unknown_metric(self) -> None:
        result = validate_expression("unknown_metric > 20")
        assert not result.valid
        assert "unknown_metric" in result.error

    def test_known_metrics_pass(self) -> None:
        for metric in [
            "temperature_2m_c",
            "apparent_temperature_c",
            "precipitation_mm",
            "precipitation_probability_pct",
            "rain_mm",
            "snowfall_cm",
            "cloud_cover_pct",
            "wind_speed_10m_ms",
            "wind_gusts_10m_ms",
            "wind_direction_10m_deg",
            "pressure_msl_hpa",
            "relative_humidity_2m_pct",
            "weather_code",
        ]:
            result = validate_expression(f"{metric} > 0")
            assert result.valid, f"Metric {metric} failed validation"

    def test_mixed_unknown_function_and_metric(self) -> None:
        result = validate_expression("unknown_func(unknown_metric) > 0")
        assert not result.valid
        assert "Unknown functions" in result.error
        assert "unknown_func" in result.error
        assert "unknown_metric" in result.error

    def test_string_metric_in_function(self) -> None:
        result = validate_expression('max("wind_gusts_10m_ms", weekend()) >= 12')
        assert result.valid

    def test_numeric_literal_only(self) -> None:
        result = validate_expression("42")
        assert result.valid

    def test_boolean_literal(self) -> None:
        result = validate_expression("True")
        assert result.valid
