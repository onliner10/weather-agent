"""Domain package for core application contracts."""

from weather_agent.domain.auth import AuthorizationService, AuthorizedUserRepo, UnauthorizedError
from weather_agent.domain.cel import (
    ALL_ALLOWED_FUNCTION_NAMES,
    ALLOWED_FUNCTIONS,
    ALLOWED_METRICS,
    CELEvalError,
    CELEvaluationResult,
    CELEvaluator,
    ValidationResult,
    get_allowlist_for_prompt,
    validate_expression,
)
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
    "ALL_ALLOWED_FUNCTION_NAMES",
    "ALLOWED_FUNCTIONS",
    "ALLOWED_METRICS",
    "AuthorizationService",
    "AuthorizedUserRepo",
    "CELEvalError",
    "CELEvaluationResult",
    "CELEvaluator",
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
    "ValidationResult",
    "WarningCategory",
    "WarningSeverity",
    "WeatherProviderError",
    "WeatherProviderResponseError",
    "WeatherProviderTimeoutError",
    "WeatherProviderUnavailableError",
    "WeatherVariable",
    "WeatherWarning",
    "WarningProvider",
    "get_allowlist_for_prompt",
    "validate_expression",
]