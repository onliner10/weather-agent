from __future__ import annotations

from datetime import UTC, datetime

from weather_agent.domain.providers import (
    ForecastProvider,
    ObservationProvider,
    WarningProvider,
)
from weather_agent.domain.weather import (
    ForecastResolution,
    ForecastResult,
    LocationRef,
    ObservationResult,
    TimeRange,
    WeatherVariable,
    WeatherWarning,
)

_WARSZAWA = LocationRef(id="warszawa-1", name="Warszawa", latitude=52.2297, longitude=21.0122)
_NOW = datetime(2026, 4, 28, 12, 0, 0, tzinfo=UTC)
_HOUR = datetime(2026, 4, 28, 13, 0, 0, tzinfo=UTC)
_TIME_RANGE = TimeRange(start=_NOW, end=_HOUR)


class _FakeForecastProvider:
    async def get_forecast(
        self,
        location: LocationRef,
        time_range: TimeRange,
        variables: list[WeatherVariable],
        resolution: ForecastResolution,
    ) -> ForecastResult:
        return ForecastResult(
            provider="fake",
            model="test-model",
            location=location,
            fetched_at=_NOW,
            points=[],
            raw_payload={},
        )


class _FakeObservationProvider:
    async def get_observations(
        self,
        location: LocationRef,
        radius_km: float,
        variables: list[WeatherVariable],
    ) -> ObservationResult:
        return ObservationResult(
            provider="fake",
            location=location,
            fetched_at=_NOW,
            points=[],
            raw_payload={},
        )


class _FakeWarningProvider:
    async def get_warnings(
        self,
        location: LocationRef,
        time_range: TimeRange,
    ) -> list[WeatherWarning]:
        return []


class TestForecastProviderProtocol:
    def test_satisfies_protocol(self) -> None:
        provider: ForecastProvider = _FakeForecastProvider()
        assert hasattr(provider, "get_forecast")
        assert callable(provider.get_forecast)

    def test_protocol_signature_matches(self) -> None:
        import inspect

        sig = inspect.signature(_FakeForecastProvider.get_forecast)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "location" in params
        assert "time_range" in params
        assert "variables" in params
        assert "resolution" in params


class TestObservationProviderProtocol:
    def test_satisfies_protocol(self) -> None:
        provider: ObservationProvider = _FakeObservationProvider()
        assert hasattr(provider, "get_observations")
        assert callable(provider.get_observations)

    def test_protocol_signature_matches(self) -> None:
        import inspect

        sig = inspect.signature(_FakeObservationProvider.get_observations)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "location" in params
        assert "radius_km" in params
        assert "variables" in params


class TestWarningProviderProtocol:
    def test_satisfies_protocol(self) -> None:
        provider: WarningProvider = _FakeWarningProvider()
        assert hasattr(provider, "get_warnings")
        assert callable(provider.get_warnings)

    def test_protocol_signature_matches(self) -> None:
        import inspect

        sig = inspect.signature(_FakeWarningProvider.get_warnings)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "location" in params
        assert "time_range" in params


class TestStructuralTyping:
    def test_forecast_provider_is_structural(self) -> None:
        class MyProvider:
            async def get_forecast(
                self,
                location: LocationRef,
                time_range: TimeRange,
                variables: list[WeatherVariable],
                resolution: ForecastResolution,
            ) -> ForecastResult: ...

        provider: ForecastProvider = MyProvider()
        assert callable(provider.get_forecast)

    def test_observation_provider_is_structural(self) -> None:
        class MyProvider:
            async def get_observations(
                self,
                location: LocationRef,
                radius_km: float,
                variables: list[WeatherVariable],
            ) -> ObservationResult: ...

        provider: ObservationProvider = MyProvider()
        assert callable(provider.get_observations)

    def test_warning_provider_is_structural(self) -> None:
        class MyProvider:
            async def get_warnings(
                self,
                location: LocationRef,
                time_range: TimeRange,
            ) -> list[WeatherWarning]: ...

        provider: WarningProvider = MyProvider()
        assert callable(provider.get_warnings)
