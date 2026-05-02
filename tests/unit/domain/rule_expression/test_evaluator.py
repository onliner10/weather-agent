from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from weather_agent.domain.rule_expression.evaluator import (
    RuleExpressionEvaluationResult,
    RuleExpressionEvaluator,
)

_WARSAW = ZoneInfo("Europe/Warsaw")


def _make_data(points: list[dict], previous_points: list[dict] | None = None) -> dict:
    data: dict = {"points": points}
    if previous_points is not None:
        data["previous_points"] = previous_points
    return data


def _make_point(
    target_time: datetime,
    temperature_2m_c: float | None = None,
    apparent_temperature_c: float | None = None,
    precipitation_mm: float | None = None,
    precipitation_probability_pct: float | None = None,
    rain_mm: float | None = None,
    snowfall_cm: float | None = None,
    cloud_cover_pct: float | None = None,
    wind_speed_10m_ms: float | None = None,
    wind_gusts_10m_ms: float | None = None,
    wind_direction_10m_deg: float | None = None,
    pressure_msl_hpa: float | None = None,
    relative_humidity_2m_pct: float | None = None,
    weather_code: str | None = None,
) -> dict:
    point: dict = {"target_time": target_time}
    if temperature_2m_c is not None:
        point["temperature_2m_c"] = temperature_2m_c
    if apparent_temperature_c is not None:
        point["apparent_temperature_c"] = apparent_temperature_c
    if precipitation_mm is not None:
        point["precipitation_mm"] = precipitation_mm
    if precipitation_probability_pct is not None:
        point["precipitation_probability_pct"] = precipitation_probability_pct
    if rain_mm is not None:
        point["rain_mm"] = rain_mm
    if snowfall_cm is not None:
        point["snowfall_cm"] = snowfall_cm
    if cloud_cover_pct is not None:
        point["cloud_cover_pct"] = cloud_cover_pct
    if wind_speed_10m_ms is not None:
        point["wind_speed_10m_ms"] = wind_speed_10m_ms
    if wind_gusts_10m_ms is not None:
        point["wind_gusts_10m_ms"] = wind_gusts_10m_ms
    if wind_direction_10m_deg is not None:
        point["wind_direction_10m_deg"] = wind_direction_10m_deg
    if pressure_msl_hpa is not None:
        point["pressure_msl_hpa"] = pressure_msl_hpa
    if relative_humidity_2m_pct is not None:
        point["relative_humidity_2m_pct"] = relative_humidity_2m_pct
    if weather_code is not None:
        point["weather_code"] = weather_code
    return point


_NOW = datetime(2026, 4, 28, 12, 0, tzinfo=_WARSAW)

_WIDE_START = datetime(2020, 1, 1, 0, 0, tzinfo=_WARSAW)
_WIDE_END = datetime(2030, 12, 31, 23, 59, tzinfo=_WARSAW)
_WIDE_RANGE_EXPR = 'date_range("2020-01-01T00:00:00", "2030-12-31T23:59:00")'


class TestRuleExpressionEvaluatorSimpleComparisons:
    def test_temperature_greater_than(self) -> None:
        evaluator = RuleExpressionEvaluator()
        result = evaluator.evaluate(
            "temperature_2m_c > 20",
            data=_make_data([_make_point(_NOW, temperature_2m_c=25.0)]),
        )
        assert result.error is None
        assert result.result is True
        assert "temperature_2m_c" in result.evaluated_metrics

    def test_temperature_less_than(self) -> None:
        evaluator = RuleExpressionEvaluator()
        result = evaluator.evaluate(
            "temperature_2m_c < 20",
            data=_make_data([_make_point(_NOW, temperature_2m_c=15.0)]),
        )
        assert result.error is None
        assert result.result is True

    def test_temperature_equal(self) -> None:
        evaluator = RuleExpressionEvaluator()
        result = evaluator.evaluate(
            "temperature_2m_c == 20",
            data=_make_data([_make_point(_NOW, temperature_2m_c=20.0)]),
        )
        assert result.error is None
        assert result.result is True

    def test_temperature_not_equal(self) -> None:
        evaluator = RuleExpressionEvaluator()
        result = evaluator.evaluate(
            "temperature_2m_c != 20",
            data=_make_data([_make_point(_NOW, temperature_2m_c=25.0)]),
        )
        assert result.error is None
        assert result.result is True

    def test_temperature_greater_equal(self) -> None:
        evaluator = RuleExpressionEvaluator()
        result = evaluator.evaluate(
            "temperature_2m_c >= 20",
            data=_make_data([_make_point(_NOW, temperature_2m_c=20.0)]),
        )
        assert result.error is None
        assert result.result is True

    def test_temperature_less_equal(self) -> None:
        evaluator = RuleExpressionEvaluator()
        result = evaluator.evaluate(
            "temperature_2m_c <= 20",
            data=_make_data([_make_point(_NOW, temperature_2m_c=15.0)]),
        )
        assert result.error is None
        assert result.result is True

    def test_boolean_and(self) -> None:
        evaluator = RuleExpressionEvaluator()
        result = evaluator.evaluate(
            "temperature_2m_c > 20 && rain_mm < 5",
            data=_make_data([_make_point(_NOW, temperature_2m_c=25.0, rain_mm=2.0)]),
        )
        assert result.error is None
        assert result.result is True

    def test_boolean_or(self) -> None:
        evaluator = RuleExpressionEvaluator()
        result = evaluator.evaluate(
            "temperature_2m_c > 30 || rain_mm > 5",
            data=_make_data([_make_point(_NOW, temperature_2m_c=25.0, rain_mm=10.0)]),
        )
        assert result.error is None
        assert result.result is True

    def test_accepts_cel_arithmetic_in_expression(self) -> None:
        evaluator = RuleExpressionEvaluator()
        result = evaluator.evaluate(
            "temperature_2m_c + 5.0 > 30.0",
            data=_make_data([_make_point(_NOW, temperature_2m_c=28.0)]),
        )
        assert result.error is None
        assert result.result is True

    def test_negative_value(self) -> None:
        evaluator = RuleExpressionEvaluator()
        result = evaluator.evaluate(
            "temperature_2m_c < 0",
            data=_make_data([_make_point(_NOW, temperature_2m_c=-5.0)]),
        )
        assert result.error is None
        assert result.result is True


class TestRuleExpressionEvaluatorAggregation:
    def test_max(self) -> None:
        points = [
            _make_point(_NOW - timedelta(hours=i), wind_gusts_10m_ms=float(10 + i))
            for i in range(24)
        ]
        data = _make_data(points)
        result = RuleExpressionEvaluator(data).evaluate(
            f'max_metric("wind_gusts_10m_ms", {_WIDE_RANGE_EXPR}) >= 12',
            data=data,
        )
        assert result.error is None
        assert result.result is True

    def test_min(self) -> None:
        points = [
            _make_point(_NOW - timedelta(hours=i), temperature_2m_c=float(15 + i))
            for i in range(10)
        ]
        data = _make_data(points)
        result = RuleExpressionEvaluator(data).evaluate(
            f'min_metric("temperature_2m_c", {_WIDE_RANGE_EXPR}) >= 15',
            data=data,
        )
        assert result.error is None
        assert result.result is True

    def test_avg(self) -> None:
        points = [
            _make_point(_NOW - timedelta(hours=i), wind_speed_10m_ms=float(5 + i))
            for i in range(24)
        ]
        data = _make_data(points)
        result = RuleExpressionEvaluator(data).evaluate(
            f'avg_metric("wind_speed_10m_ms", {_WIDE_RANGE_EXPR}) >= 7',
            data=data,
        )
        assert result.error is None
        assert result.result is True

    def test_sum(self) -> None:
        points = [
            _make_point(_NOW - timedelta(hours=i), precipitation_mm=float(i + 1)) for i in range(5)
        ]
        data = _make_data(points)
        result = RuleExpressionEvaluator(data).evaluate(
            f'sum_metric("precipitation_mm", {_WIDE_RANGE_EXPR}) > 10',
            data=data,
        )
        assert result.error is None
        assert result.result is True

    def test_median(self) -> None:
        points = [
            _make_point(_NOW - timedelta(hours=i), temperature_2m_c=float(10 + i * 2))
            for i in range(5)
        ]
        data = _make_data(points)
        result = RuleExpressionEvaluator(data).evaluate(
            f'median_metric("temperature_2m_c", {_WIDE_RANGE_EXPR}) >= 12',
            data=data,
        )
        assert result.error is None
        assert result.result is True

    def test_stddev(self) -> None:
        points = [_make_point(_NOW - timedelta(hours=i), temperature_2m_c=20.0) for i in range(5)]
        data = _make_data(points)
        result = RuleExpressionEvaluator(data).evaluate(
            f'stddev_metric("temperature_2m_c", {_WIDE_RANGE_EXPR}) < 1',
            data=data,
        )
        assert result.error is None
        assert result.result is True

    def test_pctl(self) -> None:
        points = [
            _make_point(_NOW - timedelta(hours=i), wind_gusts_10m_ms=float(i * 2))
            for i in range(10)
        ]
        data = _make_data(points)
        result = RuleExpressionEvaluator(data).evaluate(
            f'pctl_metric("wind_gusts_10m_ms", {_WIDE_RANGE_EXPR}, 90) > 10',
            data=data,
        )
        assert result.error is None
        assert result.result is True

    def test_empty_time_window_does_not_fall_back_to_all_points(self) -> None:
        data = _make_data(
            [
                _make_point(
                    datetime(2026, 4, 28, 12, 0, tzinfo=_WARSAW),
                    wind_gusts_10m_ms=30.0,
                )
            ]
        )
        result = RuleExpressionEvaluator(data).evaluate(
            'max_metric("wind_gusts_10m_ms", date_range("2026-05-01T00:00:00", '
            '"2026-05-01T23:59:00")) >= 12',
            data=data,
        )

        assert "No data points for metric 'wind_gusts_10m_ms'" in result.error
        assert result.result is None

    def test_any_empty_time_window_returns_false(self) -> None:
        data = _make_data(
            [
                _make_point(
                    datetime(2026, 4, 28, 12, 0, tzinfo=_WARSAW),
                    wind_gusts_10m_ms=30.0,
                )
            ]
        )
        result = RuleExpressionEvaluator(data).evaluate(
            'points_between(date_range("2026-05-01T00:00:00", '
            '"2026-05-01T23:59:00")).exists(p, p.wind_gusts_10m_ms >= 12)',
            data=data,
        )

        assert result.error is None
        assert result.result is False

    def test_count_empty_time_window_returns_zero(self) -> None:
        data = _make_data(
            [
                _make_point(
                    datetime(2026, 4, 28, 12, 0, tzinfo=_WARSAW),
                    wind_gusts_10m_ms=30.0,
                )
            ]
        )
        result = RuleExpressionEvaluator(data).evaluate(
            'points_between(date_range("2026-05-01T00:00:00", '
            '"2026-05-01T23:59:00")).filter(p, p.wind_gusts_10m_ms >= 12).size()',
            data=data,
        )

        assert result.error is None
        assert result.result == 0


class TestRuleExpressionEvaluatorTimeRangeHelpers:
    def test_now(self) -> None:
        result = RuleExpressionEvaluator().evaluate("now() != null")
        assert result.error is None

    def test_today(self) -> None:
        result = RuleExpressionEvaluator().evaluate("today() != null")
        assert result.error is None

    def test_tomorrow(self) -> None:
        result = RuleExpressionEvaluator().evaluate("tomorrow() != null")
        assert result.error is None

    def test_weekend(self) -> None:
        result = RuleExpressionEvaluator().evaluate("weekend() != null")
        assert result.error is None

    def test_next_hours(self) -> None:
        result = RuleExpressionEvaluator().evaluate("next_hours(24) != null")
        assert result.error is None

    def test_minutes(self) -> None:
        result = RuleExpressionEvaluator().evaluate("minutes(60) == 60.0")
        assert result.error is None
        assert result.result is True

    def test_hours(self) -> None:
        result = RuleExpressionEvaluator().evaluate("hours(2) == 120.0")
        assert result.error is None
        assert result.result is True

    def test_previous_snapshot(self) -> None:
        result = RuleExpressionEvaluator().evaluate("previous_snapshot() != null")
        assert result.error is None


class TestRuleExpressionEvaluatorChangeTrend:
    def test_delta(self) -> None:
        points = [
            _make_point(_NOW - timedelta(hours=2), temperature_2m_c=15.0),
            _make_point(_NOW - timedelta(hours=1), temperature_2m_c=20.0),
            _make_point(_NOW, temperature_2m_c=25.0),
        ]
        data = _make_data(points)
        result = RuleExpressionEvaluator(data).evaluate(
            f'delta_metric("temperature_2m_c", {_WIDE_RANGE_EXPR}) > 5',
            data=data,
        )
        assert result.error is None
        assert result.result is True

    def test_abs_delta(self) -> None:
        points = [
            _make_point(_NOW - timedelta(hours=2), temperature_2m_c=25.0),
            _make_point(_NOW - timedelta(hours=1), temperature_2m_c=15.0),
            _make_point(_NOW, temperature_2m_c=10.0),
        ]
        data = _make_data(points)
        result = RuleExpressionEvaluator(data).evaluate(
            f'abs_delta_metric("temperature_2m_c", {_WIDE_RANGE_EXPR}) > 10',
            data=data,
        )
        assert result.error is None
        assert result.result is True

    def test_rate_of_change(self) -> None:
        points = [
            _make_point(_NOW - timedelta(hours=2), temperature_2m_c=10.0),
            _make_point(_NOW, temperature_2m_c=20.0),
        ]
        data = _make_data(points)
        result = RuleExpressionEvaluator(data).evaluate(
            f'rate_of_change_metric("temperature_2m_c", {_WIDE_RANGE_EXPR}) > 0.5',
            data=data,
        )
        assert result.error is None

    def test_forecast_delta(self) -> None:
        points = [
            _make_point(_NOW - timedelta(hours=2), apparent_temperature_c=15.0),
            _make_point(_NOW, apparent_temperature_c=8.0),
        ]
        prev_points = [
            _make_point(_NOW - timedelta(hours=2), apparent_temperature_c=15.0),
            _make_point(_NOW, apparent_temperature_c=15.0),
        ]
        data = _make_data(points, previous_points=prev_points)
        result = RuleExpressionEvaluator(data).evaluate(
            (
                f'forecast_delta_metric("apparent_temperature_c", {_WIDE_RANGE_EXPR}, '
                "previous_snapshot()) < -5"
            ),
            data=data,
        )
        assert result.error is None


class TestRuleExpressionEvaluatorNumericHelpers:
    def test_abs(self) -> None:
        result = RuleExpressionEvaluator().evaluate("abs(-7) == 7")
        assert result.error is None
        assert result.result is True

    def test_abs_in_expression(self) -> None:
        points = [_make_point(_NOW, temperature_2m_c=-5.0)]
        data = _make_data(points)
        result = RuleExpressionEvaluator(data).evaluate(
            "abs(temperature_2m_c) > 3",
            data=data,
        )
        assert result.error is None
        assert result.result is True

    def test_round(self) -> None:
        result = RuleExpressionEvaluator().evaluate("round(3.14159, 2) == 3.14")
        assert result.error is None
        assert result.result is True

    def test_clamp(self) -> None:
        result = RuleExpressionEvaluator().evaluate("clamp(15, 0, 10) == 10")
        assert result.error is None
        assert result.result is True

    def test_clamp_below_min(self) -> None:
        result = RuleExpressionEvaluator().evaluate("clamp(-5, 0, 100) == 0")
        assert result.error is None
        assert result.result is True

    def test_clamp_within_range(self) -> None:
        result = RuleExpressionEvaluator().evaluate("clamp(50, 0, 100) == 50")
        assert result.error is None
        assert result.result is True


class TestRuleExpressionEvaluatorConditionOverTime:
    def test_duration_minutes_for_filtered_points(self) -> None:
        points = [
            _make_point(_NOW - timedelta(hours=2), precipitation_mm=0.5),
            _make_point(_NOW - timedelta(hours=1), precipitation_mm=0.1),
            _make_point(_NOW, precipitation_mm=0.3),
        ]
        data = _make_data(points)
        result = RuleExpressionEvaluator(data).evaluate(
            f"duration_minutes(points_between({_WIDE_RANGE_EXPR}).filter("
            "p, p.precipitation_mm > 0.2)) >= minutes(60)",
            data=data,
        )
        assert result.error is None
        assert result.result is True

    def test_filter_count(self) -> None:
        points = [
            _make_point(_NOW - timedelta(hours=2), cloud_cover_pct=80.0),
            _make_point(_NOW - timedelta(hours=1), cloud_cover_pct=30.0),
            _make_point(_NOW, cloud_cover_pct=90.0),
        ]
        data = _make_data(points)
        result = RuleExpressionEvaluator(data).evaluate(
            f"points_between({_WIDE_RANGE_EXPR}).filter(p, p.cloud_cover_pct > 50).size() >= 2",
            data=data,
        )
        assert result.error is None
        assert result.result is True

    def test_any_condition(self) -> None:
        points = [
            _make_point(_NOW - timedelta(hours=2), wind_gusts_10m_ms=8.0),
            _make_point(_NOW - timedelta(hours=1), wind_gusts_10m_ms=15.0),
            _make_point(_NOW, wind_gusts_10m_ms=10.0),
        ]
        data = _make_data(points)
        result = RuleExpressionEvaluator(data).evaluate(
            f"points_between({_WIDE_RANGE_EXPR}).exists(p, p.wind_gusts_10m_ms >= 12)",
            data=data,
        )
        assert result.error is None
        assert result.result is True

    def test_all_condition(self) -> None:
        points = [
            _make_point(_NOW - timedelta(hours=2), relative_humidity_2m_pct=80.0),
            _make_point(_NOW - timedelta(hours=1), relative_humidity_2m_pct=85.0),
            _make_point(_NOW, relative_humidity_2m_pct=90.0),
        ]
        data = _make_data(points)
        result = RuleExpressionEvaluator(data).evaluate(
            f"points_between({_WIDE_RANGE_EXPR}).all(p, p.relative_humidity_2m_pct > 70)",
            data=data,
        )
        assert result.error is None
        assert result.result is True

    def test_all_condition_false(self) -> None:
        points = [
            _make_point(_NOW - timedelta(hours=2), relative_humidity_2m_pct=80.0),
            _make_point(_NOW - timedelta(hours=1), relative_humidity_2m_pct=40.0),
            _make_point(_NOW, relative_humidity_2m_pct=90.0),
        ]
        data = _make_data(points)
        result = RuleExpressionEvaluator(data).evaluate(
            f"points_between({_WIDE_RANGE_EXPR}).all(p, p.relative_humidity_2m_pct > 70)",
            data=data,
        )
        assert result.error is None
        assert result.result is False

    def test_any_condition_false(self) -> None:
        points = [
            _make_point(_NOW - timedelta(hours=2), wind_gusts_10m_ms=5.0),
            _make_point(_NOW - timedelta(hours=1), wind_gusts_10m_ms=6.0),
            _make_point(_NOW, wind_gusts_10m_ms=7.0),
        ]
        data = _make_data(points)
        result = RuleExpressionEvaluator(data).evaluate(
            f"points_between({_WIDE_RANGE_EXPR}).exists(p, p.wind_gusts_10m_ms >= 12)",
            data=data,
        )
        assert result.error is None
        assert result.result is False


class TestRuleExpressionEvaluatorErrorCases:
    def test_unknown_function(self) -> None:
        result = RuleExpressionEvaluator().evaluate('foobar("temperature_2m_c") > 0')
        assert result.error is not None
        assert "Unknown function" in result.error or "foobar" in result.error

    def test_unknown_metric_in_expression(self) -> None:
        result = RuleExpressionEvaluator().evaluate("unknown_metric > 20")
        assert result.error is not None
        assert "unknown_metric" in result.error

    def test_syntax_error(self) -> None:
        result = RuleExpressionEvaluator().evaluate("temperature_2m_c >>> 20")
        assert result.error is not None
        assert "syntax" in result.error.lower()

    def test_evaluator_does_not_call_llm(self) -> None:
        evaluator = RuleExpressionEvaluator()
        result = evaluator.validate("temperature_2m_c > 20")
        assert result.valid

    def test_validation_result_without_data(self) -> None:
        result = RuleExpressionEvaluator().validate("temperature_2m_c > 20")
        assert result.valid

    def test_validation_captures_metrics_and_functions(self) -> None:
        result = RuleExpressionEvaluator().validate(
            'max_metric("wind_gusts_10m_ms", weekend()) >= 12'
        )
        assert result.valid
        assert "wind_gusts_10m_ms" in result.evaluated_metrics
        assert "max_metric" in result.evaluated_functions

    def test_empty_expression_error(self) -> None:
        result = RuleExpressionEvaluator().evaluate("")
        assert result.error is not None

    def test_no_data_points_for_metric(self) -> None:
        data = _make_data([_make_point(_NOW, precipitation_mm=1.0)])
        result = RuleExpressionEvaluator(data).evaluate(
            'avg_metric("temperature_2m_c", weekend()) > 0',
            data=data,
        )
        assert result.error is not None
        assert "No data points" in result.error


class TestRuleExpressionEvaluatorShortCircuiting:
    def test_and_short_circuits_false_left_side(self) -> None:
        result = RuleExpressionEvaluator().evaluate(
            'false && max_metric("wind_gusts_10m_ms", today()) > 0',
            data=_make_data([]),
        )
        assert result.error is None
        assert result.result is False

    def test_or_short_circuits_true_left_side(self) -> None:
        result = RuleExpressionEvaluator().evaluate(
            'true || max_metric("wind_gusts_10m_ms", today()) > 0',
            data=_make_data([]),
        )
        assert result.error is None
        assert result.result is True

    def test_cel_and_short_circuits_after_false_comparison(self) -> None:
        result = RuleExpressionEvaluator().evaluate(
            'temperature_2m_c > 0 && max_metric("wind_gusts_10m_ms", today()) > 0',
            data=_make_data([_make_point(_NOW, temperature_2m_c=-5.0)]),
        )
        assert result.error is None
        assert result.result is False


class TestRuleExpressionEvaluationResultModel:
    def test_result_fields(self) -> None:
        result = RuleExpressionEvaluationResult(
            expression="temperature_2m_c > 20",
            result=True,
            evaluated_metrics=["temperature_2m_c"],
            evaluated_functions=[],
        )
        assert result.expression == "temperature_2m_c > 20"
        assert result.result is True
        assert result.error is None
        assert result.evaluated_metrics == ["temperature_2m_c"]
        assert result.evaluated_functions == []

    def test_error_result(self) -> None:
        result = RuleExpressionEvaluationResult(
            expression="bad_expr",
            error="Unknown function: bad_func",
        )
        assert result.result is None
        assert result.error is not None

    def test_valid_property(self) -> None:
        result_ok = RuleExpressionEvaluationResult(
            expression="1 + 1",
            result=2,
        )
        assert result_ok.valid

        result_err = RuleExpressionEvaluationResult(
            expression="bad_expr",
            error="Some error",
        )
        assert not result_err.valid
