from __future__ import annotations

import os
from dataclasses import dataclass

from weather_agent.settings import LangSmithSettings


@dataclass(frozen=True)
class LangSmithStatus:
    configured: bool
    env_tracing_enabled: bool
    has_api_key: bool
    client_tracing_enabled: bool
    upload_ready: bool
    project: str | None
    endpoint: str | None


_TRACING_KEYS = (
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING_V2",
)

_API_KEY_KEYS = (
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
)

_PROJECT_KEYS = (
    "LANGSMITH_PROJECT",
    "LANGCHAIN_PROJECT",
)

_ENDPOINT_KEYS = (
    "LANGSMITH_ENDPOINT",
    "LANGCHAIN_ENDPOINT",
)

_ALL_CLEARABLE_KEYS = (*_TRACING_KEYS, *_API_KEY_KEYS, *_PROJECT_KEYS, *_ENDPOINT_KEYS)


def _is_env_tracing_enabled() -> bool:
    return any(os.environ.get(k) == "true" for k in _TRACING_KEYS)


def _has_api_key() -> bool:
    return any((v := os.environ.get(k)) is not None and v.strip() != "" for k in _API_KEY_KEYS)


def _get_first_env_value(keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value is not None and value.strip() != "":
            return value
    return None


def _client_tracing_enabled() -> bool:
    """Return whether the langsmith client considers tracing enabled."""
    try:
        from langsmith.utils import get_env_var, tracing_is_enabled

        get_env_var.cache_clear()  # type: ignore[attr-defined]
        return bool(tracing_is_enabled())
    except Exception:
        return _is_env_tracing_enabled() and _has_api_key()


class LangSmithTracing:
    def __init__(self, settings: LangSmithSettings | None = None) -> None:
        self._settings = settings

    def configure_tracing(self, settings: LangSmithSettings | None = None) -> None:
        effective = settings or self._settings
        if effective is None:
            return

        if effective.enabled:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGSMITH_TRACING_V2"] = "true"
            if effective.api_key is not None:
                secret = effective.api_key.get_secret_value()
                os.environ["LANGCHAIN_API_KEY"] = secret
                os.environ["LANGSMITH_API_KEY"] = secret
            if effective.project:
                os.environ["LANGCHAIN_PROJECT"] = effective.project
                os.environ["LANGSMITH_PROJECT"] = effective.project
            if effective.endpoint:
                os.environ["LANGCHAIN_ENDPOINT"] = effective.endpoint
                os.environ["LANGSMITH_ENDPOINT"] = effective.endpoint
        else:
            for key in _ALL_CLEARABLE_KEYS:
                os.environ.pop(key, None)

    @staticmethod
    def is_enabled() -> bool:
        return _client_tracing_enabled()

    @staticmethod
    def get_status(settings: LangSmithSettings | None = None) -> LangSmithStatus:
        env_tracing = _is_env_tracing_enabled()
        has_key = _has_api_key()
        client_enabled = _client_tracing_enabled()
        return LangSmithStatus(
            configured=settings.enabled if settings is not None else False,
            env_tracing_enabled=env_tracing,
            has_api_key=has_key,
            client_tracing_enabled=client_enabled,
            upload_ready=client_enabled,
            project=_get_first_env_value(_PROJECT_KEYS),
            endpoint=_get_first_env_value(_ENDPOINT_KEYS),
        )


def configure_tracing(settings: LangSmithSettings) -> None:
    LangSmithTracing(settings).configure_tracing()
