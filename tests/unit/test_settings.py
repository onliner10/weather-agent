from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from weather_agent.settings import AppSettings


def apply_base_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEATHER_AGENT_TELEGRAM__BOT_TOKEN", "telegram-token")
    monkeypatch.setenv("WEATHER_AGENT_TELEGRAM__ALLOWED_USER_IDS", "1, 2,3")
    monkeypatch.setenv(
        "WEATHER_AGENT_DATABASE_URL",
        "postgresql+psycopg://weather_agent:weather_agent@localhost:5432/weather_agent",
    )


def test_settings_parse_allowed_user_ids_from_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)

    settings = AppSettings()

    assert settings.telegram.allowed_user_ids == (1, 2, 3)


def test_settings_apply_model_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)

    settings = AppSettings()

    assert settings.model.provider == "openrouter"
    assert settings.model.model_name == "qwen/qwen3.5-flash-02-23"
    assert settings.model.routing_sort is None
    assert settings.model.require_supported_parameters is True
    assert settings.model.timeout_seconds == 45.0
    assert settings.model.fallback_provider is None
    assert settings.model.fallback_model_name is None
    assert settings.telegram.bot_token == SecretStr("telegram-token")


def test_settings_allow_model_routing_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("WEATHER_AGENT_MODEL__ROUTING_SORT", "latency")
    monkeypatch.setenv("WEATHER_AGENT_MODEL__REQUIRE_SUPPORTED_PARAMETERS", "false")

    settings = AppSettings()

    assert settings.model.routing_sort == "latency"
    assert settings.model.require_supported_parameters is False


def test_settings_allow_unit_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("WEATHER_AGENT_UNITS__WIND_SPEED", "kmh")
    monkeypatch.setenv("WEATHER_AGENT_UNITS__PRESSURE", "pa")

    settings = AppSettings()

    assert settings.units.wind_speed == "kmh"
    assert settings.units.pressure == "pa"


def test_settings_validate_retention_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("WEATHER_AGENT_RETENTION__TRACE_DAYS", "-1")

    with pytest.raises(ValidationError, match="trace_days must be a positive integer"):
        AppSettings()


def test_settings_fail_fast_when_required_values_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WEATHER_AGENT_TELEGRAM__BOT_TOKEN", raising=False)
    monkeypatch.delenv("WEATHER_AGENT_DATABASE_URL", raising=False)

    with pytest.raises(ValidationError) as exc_info:
        AppSettings()

    message = str(exc_info.value)
    assert "telegram" in message
    assert "database_url" in message


def test_settings_observability_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)

    settings = AppSettings()

    assert settings.observability.enabled is True
    assert settings.observability.bot_port == 8080
    assert settings.observability.worker_port == 8081
    assert settings.health.worker_stale_after_minutes == 45
    assert settings.health.forecast_stale_after_minutes == 120
    assert settings.telegram.max_concurrent_updates == 4


def test_settings_observability_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("WEATHER_AGENT_OBSERVABILITY__ENABLED", "false")
    monkeypatch.setenv("WEATHER_AGENT_OBSERVABILITY__BOT_PORT", "9090")
    monkeypatch.setenv("WEATHER_AGENT_OBSERVABILITY__WORKER_PORT", "9091")

    settings = AppSettings()

    assert settings.observability.enabled is False
    assert settings.observability.bot_port == 9090
    assert settings.observability.worker_port == 9091
