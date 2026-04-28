"""Domain package for core application contracts."""

from weather_agent.domain.auth import AuthorizationService, AuthorizedUserRepo, UnauthorizedError
from weather_agent.domain.errors import (
    WeatherProviderError,
    WeatherProviderResponseError,
    WeatherProviderTimeoutError,
    WeatherProviderUnavailableError,
)
from weather_agent.domain.global_settings import GlobalSettingsService, GlobalUnits, SettingsRepo
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
    "AuthorizationService",
    "AuthorizedUserRepo",
    "ForecastPoint",
    "ForecastProvider",
    "ForecastResolution",
    "ForecastResult",
    "GlobalSettingsService",
    "GlobalUnits",
    "SettingsRepo",
    "LocationRef",
    "ObservationPoint",
    "ObservationProvider",
    "ObservationResult",
    "TimeRange",
    "UnauthorizedError",
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