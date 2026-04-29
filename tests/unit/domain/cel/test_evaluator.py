from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from weather_agent.domain.cel.evaluator import CELEvaluationResult, CELEvaluator

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


class TestCELEvaluatorSimpleComparisons:
    def test_temperature_greater_than(self) -> None:
        evaluator = CELEvaluator()
        result = evaluator.evaluate(
            "temperature_2m_c > 20",
            data=_make_data([_make_point(_NOW, temperature_2m_c=25.0)]),
        )
        assert result.error is None
        assert result.result is True
        assert "temperature_2m_c" in result.evaluated_metrics

    def test_temperature_less_than(self) -> None:
        evaluator = CELEvaluator()
        result = evaluator.evaluate(
            "temperature_2m_c < 20",
            data=_make_data([_make_point(_NOW, temperature_2m_c=15.0)]),
        )
        assert result.error is None
        assert result.result is True

    def test_temperature_equal(self) -> None:
        evaluator = CELEvaluator()
        result = evaluator.evaluate(
            "temperature_2m_c == 20",
            data=_make_data([_make_point(_NOW, temperature_2m_c=20.0)]),
        )
        assert result.error is None
        assert result.result is True

    def test_temperature_not_equal(self) -> None:
        evaluator = CELEvaluator()
        result = evaluator.evaluate(
            "temperature_2m_c != 20",
            data=_make_data([_make_point(_NOW, temperature_2m_c=25.0)]),
        )
        assert result.error is None
        assert result.result is True

    def test_temperature_greater_equal(self) -> None:
        evaluator = CELEvaluator()
        result = evaluator.evaluate(
            "temperature_2m_c >= 20",
            data=_make_data([_make_point(_NOW, temperature_2m_c=20.0)]),
        )
        assert result.error is None
        assert result.result is True

    def test_temperature_less_equal(self) -> None:
        evaluator = CELEvaluator()
        result = evaluator.evaluate(
            "temperature_2m_c <= 20",
            data=_make_data([_make_point(_NOW, temperature_2m_c=15.0)]),
        )
        assert result.error is None
        assert result.result is True

    def test_boolean_and(self) -> None:
        evaluator = CELEvaluator()
        result = evaluator.evaluate(
            "temperature_2m_c > 20 and rain_mm < 5",
            data=_make_data([_make_point(_NOW, temperature_2m_c=25.0, rain_mm=2.0)]),
        )
        assert result.error is None
        assert result.result is True

    def test_boolean_or(self) -> None:
        evaluator = CELEvaluator()
        result = evaluator.evaluate(
            "temperature_2m_c > 30 or rain_mm > 5",
            data=_make_data([_make_point(_NOW, temperature_2m_c=25.0, rain_mm=10.0)]),
        )
        assert result.error is None
        assert result.result is True

    def test_arithmetic_in_expression(self) -> None:
        evaluator = CELEvaluator()
        result = evaluator.evaluate(
            "temperature_2m_c + 5 > 30",
            data=_make_data([_make_point(_NOW, temperature_2m_c=28.0)]),
        )
        assert result.error is None
        assert result.result is True

    def test_negative_value(self) -> None:
        evaluator = CELEvaluator()
        result = evaluator.evaluate(
            "temperature_2m_c < 0",
            data=_make_data([_make_point(_NOW, temperature_2m_c=-5.0)]),
        )
        assert result.error is None
        assert result.result is True


class TestCELEvaluatorAggregation:
    def test_max(self) -> None:
        points = [
            _make_point(_NOW - timedelta(hours=i), wind_gusts_10m_ms=float(10 + i))
            for i in range(24)
        ]
        data = _make_data(points)
        result = CELEvaluator(data).evaluate(
            'max("wind_gusts_10m_ms", weekend()) >= 12',
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
        result = CELEvaluator(data).evaluate(
            'min("temperature_2m_c", weekend()) >= 15',
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
        result = CELEvaluator(data).evaluate(
            'avg("wind_speed_10m_ms", next_hours(24)) >= 7',
            data=data,
        )
        assert result.error is None
        assert result.result is True

    def test_sum(self) -> None:
        points = [
            _make_point(_NOW - timedelta(hours=i), precipitation_mm=float(i + 1)) for i in range(5)
        ]
        data = _make_data(points)
        result = CELEvaluator(data).evaluate(
            'sum("precipitation_mm", next_hours(24)) > 10',
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
        result = CELEvaluator(data).evaluate(
            'median("temperature_2m_c", weekend()) >= 12',
            data=data,
        )
        assert result.error is None
        assert result.result is True

    def test_stddev(self) -> None:
        points = [_make_point(_NOW - timedelta(hours=i), temperature_2m_c=20.0) for i in range(5)]
        data = _make_data(points)
        result = CELEvaluator(data).evaluate(
            'stddev("temperature_2m_c", weekend()) < 1',
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
        result = CELEvaluator(data).evaluate(
            'pctl("wind_gusts_10m_ms", weekend(), 90) > 10',
            data=data,
        )
        assert result.error is None
        assert result.result is True


class TestCELEvaluatorTimeRangeHelpers:
    def test_now(self) -> None:
        result = CELEvaluator().evaluate("now() != None")
        assert result.error is None

    def test_today(self) -> None:
        result = CELEvaluator().evaluate("today() != None")
        assert result.error is None

    def test_tomorrow(self) -> None:
        result = CELEvaluator().evaluate("tomorrow() != None")
        assert result.error is None

    def test_weekend(self) -> None:
        result = CELEvaluator().evaluate("weekend() != None")
        assert result.error is None

    def test_next_hours(self) -> None:
        result = CELEvaluator().evaluate("next_hours(24) != None")
        assert result.error is None

    def test_minutes(self) -> None:
        result = CELEvaluator().evaluate("minutes(60) == 60.0")
        assert result.error is None
        assert result.result is True

    def test_hours(self) -> None:
        result = CELEvaluator().evaluate("hours(2) == 120.0")
        assert result.error is None
        assert result.result is True

    def test_previous_snapshot(self) -> None:
        result = CELEvaluator().evaluate("previous_snapshot() != None")
        assert result.error is None


class TestCELEvaluatorChangeTrend:
    def test_delta(self) -> None:
        points = [
            _make_point(_NOW - timedelta(hours=2), temperature_2m_c=15.0),
            _make_point(_NOW - timedelta(hours=1), temperature_2m_c=20.0),
            _make_point(_NOW, temperature_2m_c=25.0),
        ]
        data = _make_data(points)
        result = CELEvaluator(data).evaluate(
            'delta("temperature_2m_c", weekend()) > 5',
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
        result = CELEvaluator(data).evaluate(
            'abs_delta("temperature_2m_c", weekend()) > 10',
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
        result = CELEvaluator(data).evaluate(
            'rate_of_change("temperature_2m_c", weekend()) > 0.5',
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
        result = CELEvaluator(data).evaluate(
            'forecast_delta("apparent_temperature_c", weekend(), previous_snapshot()) < -5',
            data=data,
        )
        assert result.error is None


class TestCELEvaluatorNumericHelpers:
    def test_abs(self) -> None:
        result = CELEvaluator().evaluate("abs(-7) == 7")
        assert result.error is None
        assert result.result is True

    def test_abs_in_expression(self) -> None:
        points = [_make_point(_NOW, temperature_2m_c=-5.0)]
        data = _make_data(points)
        result = CELEvaluator(data).evaluate(
            "abs(temperature_2m_c) > 3",
            data=data,
        )
        assert result.error is None
        assert result.result is True

    def test_round(self) -> None:
        result = CELEvaluator().evaluate("round(3.14159, 2) == 3.14")
        assert result.error is None
        assert result.result is True

    def test_clamp(self) -> None:
        result = CELEvaluator().evaluate("clamp(15, 0, 10) == 10")
        assert result.error is None
        assert result.result is True

    def test_clamp_below_min(self) -> None:
        result = CELEvaluator().evaluate("clamp(-5, 0, 100) == 0")
        assert result.error is None
        assert result.result is True

    def test_clamp_within_range(self) -> None:
        result = CELEvaluator().evaluate("clamp(50, 0, 100) == 50")
        assert result.error is None
        assert result.result is True


class TestCELEvaluatorConditionOverTime:
    def test_duration_where(self) -> None:
        points = [
            _make_point(_NOW - timedelta(hours=2), precipitation_mm=0.5),
            _make_point(_NOW - timedelta(hours=1), precipitation_mm=0.1),
            _make_point(_NOW, precipitation_mm=0.3),
        ]
        data = _make_data(points)
        result = CELEvaluator(data).evaluate(
            "duration_where(precipitation_mm > 0.2, weekend()) >= minutes(60)",
            data=data,
        )
        assert result.error is None
        assert result.result is True

    def test_count_where(self) -> None:
        points = [
            _make_point(_NOW - timedelta(hours=2), cloud_cover_pct=80.0),
            _make_point(_NOW - timedelta(hours=1), cloud_cover_pct=30.0),
            _make_point(_NOW, cloud_cover_pct=90.0),
        ]
        data = _make_data(points)
        result = CELEvaluator(data).evaluate(
            "count_where(cloud_cover_pct > 50, weekend()) >= 2",
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
        result = CELEvaluator(data).evaluate(
            "any(wind_gusts_10m_ms >= 12, weekend())",
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
        result = CELEvaluator(data).evaluate(
            "all(relative_humidity_2m_pct > 70, weekend())",
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
        result = CELEvaluator(data).evaluate(
            "all(relative_humidity_2m_pct > 70, weekend())",
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
        result = CELEvaluator(data).evaluate(
            "any(wind_gusts_10m_ms >= 12, weekend())",
            data=data,
        )
        assert result.error is None
        assert result.result is False


class TestCELEvaluatorErrorCases:
    def test_unknown_function(self) -> None:
        result = CELEvaluator().evaluate('foobar("temperature_2m_c") > 0')
        assert result.error is not None
        assert "Unknown function" in result.error or "foobar" in result.error

    def test_unknown_metric_in_expression(self) -> None:
        result = CELEvaluator().evaluate("unknown_metric > 20")
        assert result.error is not None
        assert "unknown_metric" in result.error

    def test_syntax_error(self) -> None:
        result = CELEvaluator().evaluate("temperature_2m_c >>> 20")
        assert result.error is not None
        assert "Syntax error" in result.error

    def test_evaluator_does_not_call_llm(self) -> None:
        evaluator = CELEvaluator()
        result = evaluator.validate("temperature_2m_c > 20")
        assert result.valid

    def test_validation_result_without_data(self) -> None:
        result = CELEvaluator().validate("temperature_2m_c > 20")
        assert result.valid

    def test_validation_captures_metrics_and_functions(self) -> None:
        result = CELEvaluator().validate('max("wind_gusts_10m_ms", weekend()) >= 12')
        assert result.valid
        assert "wind_gusts_10m_ms" in result.evaluated_metrics
        assert "max" in result.evaluated_functions

    def test_empty_expression_error(self) -> None:
        result = CELEvaluator().evaluate("")
        assert result.error is not None

    def test_no_data_points_for_metric(self) -> None:
        data = _make_data([_make_point(_NOW, precipitation_mm=1.0)])
        result = CELEvaluator(data).evaluate(
            'avg("temperature_2m_c", weekend()) > 0',
            data=data,
        )
        assert result.error is not None
        assert "No data points" in result.error


class TestCELEvaluationResultModel:
    def test_result_fields(self) -> None:
        result = CELEvaluationResult(
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
        result = CELEvaluationResult(
            expression="bad_expr",
            error="Unknown function: bad_func",
        )
        assert result.result is None
        assert result.error is not None

    def test_valid_property(self) -> None:
        result_ok = CELEvaluationResult(
            expression="1 + 1",
            result=2,
        )
        assert result_ok.valid

        result_err = CELEvaluationResult(
            expression="bad_expr",
            error="Some error",
        )
        assert not result_err.valid
