from __future__ import annotations

from datetime import UTC, datetime

import pytest

from weather_agent.adapters.open_meteo.forecast_provider import OpenMeteoDwdIconProvider
from weather_agent.domain.weather import (
    ForecastResolution,
    LocationRef,
    TimeRange,
    WeatherVariable,
)
from weather_agent.settings import OpenMeteoSettings

pytestmark = pytest.mark.smoke

WARSAW = LocationRef(id="warsaw", name="Warsaw", latitude=52.2297, longitude=21.0122)

ALL_VARIABLES = list(WeatherVariable)


@pytest.mark.asyncio
async def test_open_meteo_forecast_today() -> None:
    settings = OpenMeteoSettings()
    provider = OpenMeteoDwdIconProvider(settings)

    today = datetime.now(tz=UTC).date()
    time_range = TimeRange(
        start=datetime(today.year, today.month, today.day, tzinfo=UTC),
        end=datetime(today.year, today.month, today.day, 23, 59, 59, tzinfo=UTC),
    )

    result = await provider.get_forecast(
        location=WARSAW,
        time_range=time_range,
        variables=ALL_VARIABLES,
        resolution=ForecastResolution.hourly,
    )

    assert result.provider == "open-meteo", f"Unexpected provider: {result.provider}"
    assert result.model == "dwd-icon", f"Unexpected model: {result.model}"
    assert result.raw_payload is not None, "raw_payload should be present in ForecastResult"

    assert len(result.points) > 0, (
        f"Expected at least 1 forecast point for Warsaw today, got {len(result.points)}"
    )

    non_none_count = sum(1 for p in result.points if p.temperature_2m_c is not None)
    assert non_none_count > 0, (
        f"Expected some points with non-None temperature_2m_c, "
        f"but all {len(result.points)} points had None. "
        f"First point: {result.points[0]}"
    )

    first = result.points[0]
    assert first.target_time is not None, "First point should have a target_time"
    assert first.location_id == "warsaw", f"Expected location_id='warsaw', got {first.location_id}"
