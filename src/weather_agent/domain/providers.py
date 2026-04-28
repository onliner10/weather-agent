from __future__ import annotations

from typing import Protocol, runtime_checkable

from weather_agent.domain.weather import (
    ForecastResolution,
    ForecastResult,
    LocationRef,
    ObservationResult,
    TimeRange,
    WeatherVariable,
    WeatherWarning,
)


@runtime_checkable
class ForecastProvider(Protocol):
    async def get_forecast(
        self,
        location: LocationRef,
        time_range: TimeRange,
        variables: list[WeatherVariable],
        resolution: ForecastResolution,
    ) -> ForecastResult: ...


@runtime_checkable
class ObservationProvider(Protocol):
    async def get_observations(
        self,
        location: LocationRef,
        radius_km: float,
        variables: list[WeatherVariable],
    ) -> ObservationResult: ...


@runtime_checkable
class WarningProvider(Protocol):
    async def get_warnings(
        self,
        location: LocationRef,
        time_range: TimeRange,
    ) -> list[WeatherWarning]: ...