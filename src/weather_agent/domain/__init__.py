"""Domain package for core application contracts."""

from weather_agent.domain.errors import (
    WeatherProviderError,
    WeatherProviderResponseError,
    WeatherProviderTimeoutError,
    WeatherProviderUnavailableError,
)
from weather_agent.domain.providers import (
    ForecastProvider,
    ObservationProvider,
    WarningProvider,
)
from weather_agent.domain.weather import (
    ForecastPoint,
    ForecastResolution,
    ForecastResult,
    LocationRef,
    ObservationPoint,
    ObservationResult,
    TimeRange,
    WarningCategory,
    WarningSeverity,
    WeatherVariable,
    WeatherWarning,
)

__all__ = [
    "ForecastPoint",
    "ForecastProvider",
    "ForecastResolution",
    "ForecastResult",
    "LocationRef",
    "ObservationPoint",
    "ObservationProvider",
    "ObservationResult",
    "TimeRange",
    "WarningCategory",
    "WarningSeverity",
    "WeatherProviderError",
    "WeatherProviderResponseError",
    "WeatherProviderTimeoutError",
    "WeatherProviderUnavailableError",
    "WeatherVariable",
    "WeatherWarning",
    "WarningProvider",
]