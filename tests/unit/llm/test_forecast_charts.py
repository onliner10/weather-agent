from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from weather_agent.domain.weather import (
    ForecastPoint,
    TimeRange,
    WeatherVariable,
)
from weather_agent.llm.tools.forecast_charts import (
    ForecastChartError,
    default_forecast_chart_spec,
    forecast_points_to_records,
    prepare_vega_lite_spec,
    render_forecast_chart_png,
)


def _time_range(days: int = 1) -> TimeRange:
    start = datetime(2026, 5, 4, tzinfo=UTC)
    return TimeRange(start=start, end=start + timedelta(days=days))


def _hour_range(hours: int) -> TimeRange:
    start = datetime(2026, 5, 4, tzinfo=UTC)
    return TimeRange(start=start, end=start + timedelta(hours=hours))


def _points() -> list[ForecastPoint]:
    fetched_at = datetime(2026, 5, 3, tzinfo=UTC)
    return [
        ForecastPoint(
            target_time=datetime(2026, 5, 4, hour, tzinfo=UTC),
            fetched_at=fetched_at,
            provider="fake",
            location_id="loc",
            wind_speed_10m_ms=4.0 + hour,
            wind_gusts_10m_ms=7.0 + hour,
            raw_payload={},
        )
        for hour in range(3)
    ]


def _wind_spec() -> dict[str, object]:
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
        "title": "Wiatr w czasie",
        "data": {"name": "forecast"},
        "layer": [
            {
                "mark": {"type": "line"},
                "encoding": {
                    "x": {"field": "time", "type": "temporal", "title": "Godzina"},
                    "y": {
                        "field": "wind_speed_10m_ms",
                        "type": "quantitative",
                        "title": "m/s",
                    },
                    "color": {"datum": "Prędkość wiatru"},
                },
            },
            {
                "mark": {"type": "line", "strokeDash": [4, 3]},
                "encoding": {
                    "x": {"field": "time", "type": "temporal"},
                    "y": {"field": "wind_gusts_10m_ms", "type": "quantitative"},
                    "color": {"datum": "Porywy"},
                },
            },
        ],
    }


def test_prepare_vega_lite_spec_injects_forecast_dataset() -> None:
    records = forecast_points_to_records(_points())

    spec = prepare_vega_lite_spec(
        spec=_wind_spec(),
        records=records,
        variables=[WeatherVariable.wind_speed_10m_ms, WeatherVariable.wind_gusts_10m_ms],
        time_range=_time_range(),
    )

    assert spec["data"] == {"name": "forecast"}
    assert spec["datasets"] == {"forecast": records}
    assert spec["width"] == 720
    assert spec["height"] == 360


def test_forecast_points_to_records_uses_warsaw_wall_clock_time() -> None:
    records = forecast_points_to_records(_points())

    assert records[0]["time"] == "2026-05-04T02:00:00"


def test_default_forecast_chart_spec_builds_valid_layered_wind_spec() -> None:
    spec = default_forecast_chart_spec(
        [WeatherVariable.wind_speed_10m_ms, WeatherVariable.wind_gusts_10m_ms]
    )
    records = forecast_points_to_records(_points())

    prepared = prepare_vega_lite_spec(
        spec=spec,
        records=records,
        variables=[WeatherVariable.wind_speed_10m_ms, WeatherVariable.wind_gusts_10m_ms],
        time_range=_time_range(),
    )

    assert prepared["data"] == {"name": "forecast"}
    assert prepared["datasets"] == {"forecast": records}
    assert prepared["title"] == "Wiatr w czasie"


def test_prepare_vega_lite_spec_adds_readable_hourly_time_axis() -> None:
    spec = default_forecast_chart_spec([WeatherVariable.wind_speed_10m_ms])

    prepared = prepare_vega_lite_spec(
        spec=spec,
        records=forecast_points_to_records(_points()),
        variables=[WeatherVariable.wind_speed_10m_ms],
        time_range=_hour_range(hours=12),
    )

    encoding = cast(dict[str, object], prepared["encoding"])
    x = cast(dict[str, object], encoding["x"])
    axis = cast(dict[str, object], x["axis"])
    assert axis["format"] == "%H:%M"
    assert axis["labelAngle"] == 0
    assert axis["labelOverlap"] == "greedy"
    assert axis["labelBound"] is True
    assert axis["labelFlush"] is True
    assert axis["labelPadding"] == 4
    assert axis["values"] == [
        "2026-05-04T02:00:00",
        "2026-05-04T03:00:00",
        "2026-05-04T04:00:00",
        "2026-05-04T05:00:00",
        "2026-05-04T06:00:00",
        "2026-05-04T07:00:00",
        "2026-05-04T08:00:00",
        "2026-05-04T09:00:00",
        "2026-05-04T10:00:00",
        "2026-05-04T11:00:00",
        "2026-05-04T12:00:00",
        "2026-05-04T13:00:00",
        "2026-05-04T14:00:00",
    ]


def test_prepare_vega_lite_spec_uses_coarser_one_day_time_axis() -> None:
    spec = default_forecast_chart_spec([WeatherVariable.wind_speed_10m_ms])

    prepared = prepare_vega_lite_spec(
        spec=spec,
        records=forecast_points_to_records(_points()),
        variables=[WeatherVariable.wind_speed_10m_ms],
        time_range=_hour_range(hours=24),
    )

    encoding = cast(dict[str, object], prepared["encoding"])
    x = cast(dict[str, object], encoding["x"])
    axis = cast(dict[str, object], x["axis"])
    assert axis["format"] == "%H:%M"
    assert axis["labelAngle"] == 0
    assert axis["values"] == [
        "2026-05-04T02:00:00",
        "2026-05-04T04:00:00",
        "2026-05-04T06:00:00",
        "2026-05-04T08:00:00",
        "2026-05-04T10:00:00",
        "2026-05-04T12:00:00",
        "2026-05-04T14:00:00",
        "2026-05-04T16:00:00",
        "2026-05-04T18:00:00",
        "2026-05-04T20:00:00",
        "2026-05-04T22:00:00",
        "2026-05-05T00:00:00",
        "2026-05-05T02:00:00",
    ]


def test_prepare_vega_lite_spec_angles_multi_day_time_axis_labels() -> None:
    spec = default_forecast_chart_spec([WeatherVariable.wind_speed_10m_ms])

    prepared = prepare_vega_lite_spec(
        spec=spec,
        records=forecast_points_to_records(_points()),
        variables=[WeatherVariable.wind_speed_10m_ms],
        time_range=_time_range(days=2),
    )

    encoding = cast(dict[str, object], prepared["encoding"])
    x = cast(dict[str, object], encoding["x"])
    axis = cast(dict[str, object], x["axis"])
    assert axis["format"] == "%d.%m %H:%M"
    assert axis["labelAngle"] == -35
    values = cast(list[str], axis["values"])
    assert values[:3] == [
        "2026-05-04T02:00:00",
        "2026-05-04T05:00:00",
        "2026-05-04T08:00:00",
    ]
    assert values[-1] == "2026-05-06T02:00:00"


def test_prepare_vega_lite_spec_adds_time_axes_to_layered_custom_specs() -> None:
    prepared = prepare_vega_lite_spec(
        spec=_wind_spec(),
        records=forecast_points_to_records(_points()),
        variables=[WeatherVariable.wind_speed_10m_ms, WeatherVariable.wind_gusts_10m_ms],
        time_range=_time_range(days=3),
    )

    layers = cast(list[dict[str, object]], prepared["layer"])
    for layer in layers:
        encoding = cast(dict[str, object], layer["encoding"])
        x = cast(dict[str, object], encoding["x"])
        axis = cast(dict[str, object], x["axis"])
        assert axis["format"] == "%d.%m %H:%M"
        assert axis["labelAngle"] == -35
        assert axis["labelOverlap"] == "greedy"
        values = cast(list[str], axis["values"])
        assert values[:3] == [
            "2026-05-04T02:00:00",
            "2026-05-04T08:00:00",
            "2026-05-04T14:00:00",
        ]
        assert values[-1] == "2026-05-07T02:00:00"


def test_prepare_vega_lite_spec_preserves_explicit_time_axis_values() -> None:
    spec = default_forecast_chart_spec([WeatherVariable.wind_speed_10m_ms])
    encoding = cast(dict[str, object], spec["encoding"])
    x = cast(dict[str, object], encoding["x"])
    x["axis"] = {"labelAngle": -90, "format": "%H"}

    prepared = prepare_vega_lite_spec(
        spec=spec,
        records=forecast_points_to_records(_points()),
        variables=[WeatherVariable.wind_speed_10m_ms],
        time_range=_hour_range(hours=24),
    )

    prepared_encoding = cast(dict[str, object], prepared["encoding"])
    prepared_x = cast(dict[str, object], prepared_encoding["x"])
    axis = cast(dict[str, object], prepared_x["axis"])
    assert axis["format"] == "%H"
    assert axis["labelAngle"] == -90
    assert axis["labelOverlap"] == "greedy"
    assert axis["values"] == [
        "2026-05-04T02:00:00",
        "2026-05-04T04:00:00",
        "2026-05-04T06:00:00",
        "2026-05-04T08:00:00",
        "2026-05-04T10:00:00",
        "2026-05-04T12:00:00",
        "2026-05-04T14:00:00",
        "2026-05-04T16:00:00",
        "2026-05-04T18:00:00",
        "2026-05-04T20:00:00",
        "2026-05-04T22:00:00",
        "2026-05-05T00:00:00",
        "2026-05-05T02:00:00",
    ]


def test_render_forecast_chart_png_returns_png_bytes() -> None:
    png = render_forecast_chart_png(
        spec=_wind_spec(),
        records=forecast_points_to_records(_points()),
        variables=[WeatherVariable.wind_speed_10m_ms, WeatherVariable.wind_gusts_10m_ms],
        time_range=_time_range(),
    )

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 1000


@pytest.mark.parametrize(
    ("mutated_spec", "message"),
    [
        ({**_wind_spec(), "data": {"url": "https://example.test/data.json"}}, "data"),
        ({**_wind_spec(), "data": {"values": [{"time": "x"}]}}, "data"),
        ({**_wind_spec(), "datasets": {"forecast": []}}, "datasets"),
        (
            {
                **_wind_spec(),
                "layer": [
                    {
                        "mark": "line",
                        "encoding": {
                            "x": {"field": "time", "type": "temporal"},
                            "y": {"field": "unknown", "type": "quantitative"},
                        },
                    }
                ],
            },
            "Nieznane pole",
        ),
    ],
)
def test_prepare_vega_lite_spec_rejects_unsafe_or_unknown_fields(
    mutated_spec: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ForecastChartError, match=message):
        prepare_vega_lite_spec(
            spec=mutated_spec,
            records=forecast_points_to_records(_points()),
            variables=[WeatherVariable.wind_speed_10m_ms],
            time_range=_time_range(),
        )


def test_prepare_vega_lite_spec_rejects_ranges_over_seven_days() -> None:
    with pytest.raises(ForecastChartError, match="maksymalnie 7 dni"):
        prepare_vega_lite_spec(
            spec=_wind_spec(),
            records=forecast_points_to_records(_points()),
            variables=[WeatherVariable.wind_speed_10m_ms],
            time_range=_time_range(days=8),
        )
