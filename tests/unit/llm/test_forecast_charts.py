from __future__ import annotations

from datetime import UTC, datetime, timedelta

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
