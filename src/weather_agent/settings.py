"""Typed application configuration for the weather agent."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class TelegramSettings(BaseModel):
    """Telegram-specific secrets and access controls."""

    model_config = ConfigDict(extra="forbid")

    bot_token: SecretStr
    allowed_user_ids: Annotated[tuple[int, ...], NoDecode] = ()

    @field_validator("allowed_user_ids", mode="before")
    @classmethod
    def parse_allowed_user_ids(cls, value: object) -> object:
        if value in (None, "", ()):
            return ()
        if isinstance(value, str):
            user_ids = [item.strip() for item in value.split(",") if item.strip()]
            return tuple(int(item) for item in user_ids)
        if isinstance(value, list):
            return tuple(int(item) for item in value)
        return value


class LangSmithSettings(BaseModel):
    """Tracing configuration for learning and evaluation workflows."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    api_key: SecretStr | None = None
    project: str = "weather-agent-dev"
    endpoint: str = "https://api.smith.langchain.com"


class ModelSettings(BaseModel):
    """LLM provider defaults used by conversational orchestration."""

    model_config = ConfigDict(extra="forbid")

    provider: str = "openai"
    model_name: str = "gpt-4.1-mini"
    temperature: float = 0.2
    api_key: SecretStr | None = None
    base_url: str | None = None


class OpenMeteoSettings(BaseModel):
    """Forecast provider defaults for Open-Meteo DWD ICON."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = "https://api.open-meteo.com/v1/forecast"
    model: str = "dwd-icon"
    timeout_seconds: int = 15


class ImgwSettings(BaseModel):
    """IMGW source configuration for observations and warnings."""

    model_config = ConfigDict(extra="forbid")

    synop_base_url: str = "https://danepubliczne.imgw.pl/api/data/synop"
    warnings_base_url: str = "https://danepubliczne.imgw.pl/api/data/warningsmeteo"
    timeout_seconds: int = 15


class GlobalUnitsSettings(BaseModel):
    """Global unit preferences kept consistent across user-facing output."""

    model_config = ConfigDict(extra="forbid")

    temperature: str = "celsius"
    wind_speed: str = "ms"
    precipitation: str = "mm"
    pressure: str = "hpa"


class SchedulerSettings(BaseModel):
    """Polling and evaluation cadence for background workers."""

    model_config = ConfigDict(extra="forbid")

    forecast_refresh_minutes: int = 30
    rule_evaluation_minutes: int = 15
    warning_poll_minutes: int = 15


class ObservabilitySettings(BaseModel):
    """HTTP observability surface configuration for bot and worker processes."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    bot_port: int = 8080
    worker_port: int = 8081


class RetentionSettings(BaseModel):
    """Retention windows for user context, weather data, and audit evidence."""

    model_config = ConfigDict(extra="forbid")

    thread_memory_days: int = 14
    raw_forecast_days: int = 60
    aggregated_weather_days: int = 90
    notification_log_days: int = 365
    audit_log_days: int = 365
    trace_days: int = 14

    @field_validator("*")
    @classmethod
    def validate_positive_days(cls, value: int, info: ValidationInfo) -> int:
        if value <= 0:
            field_name = info.field_name or "retention value"
            raise ValueError(f"{field_name} must be a positive integer")
        return value


class AppSettings(BaseSettings):
    """Top-level application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="WEATHER_AGENT_",
        env_nested_delimiter="__",
        env_file="/nonexistent",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    telegram: TelegramSettings
    database_url: str = Field(..., description="PostgreSQL/TimescaleDB connection string.")
    langsmith: LangSmithSettings = LangSmithSettings()
    model: ModelSettings = ModelSettings()
    open_meteo: OpenMeteoSettings = OpenMeteoSettings()
    imgw: ImgwSettings = ImgwSettings()
    units: GlobalUnitsSettings = GlobalUnitsSettings()
    scheduler: SchedulerSettings = SchedulerSettings()
    observability: ObservabilitySettings = ObservabilitySettings()
    retention: RetentionSettings = RetentionSettings()
    default_timezone: str = "Europe/Warsaw"
    default_language: str = "pl-PL"

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("database_url must not be empty")
        return value


@lru_cache(maxsize=1)
def load_settings() -> AppSettings:
    """Load and cache validated application settings.

    Production entry point that explicitly loads from ``.env`` so tests
    calling ``AppSettings()`` directly remain hermetic.
    """

    return AppSettings(_env_file=".env")  # type: ignore[call-arg]
