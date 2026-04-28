from __future__ import annotations

import os

from weather_agent.settings import LangSmithSettings


class LangSmithTracing:
    def __init__(self, settings: LangSmithSettings | None = None) -> None:
        self._settings = settings

    def configure_tracing(self, settings: LangSmithSettings | None = None) -> None:
        effective = settings or self._settings
        if effective is None:
            return
        if effective.enabled:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            if effective.api_key is not None:
                os.environ["LANGCHAIN_API_KEY"] = effective.api_key.get_secret_value()
            if effective.project:
                os.environ["LANGCHAIN_PROJECT"] = effective.project
            if effective.endpoint:
                os.environ["LANGCHAIN_ENDPOINT"] = effective.endpoint
        else:
            os.environ.pop("LANGCHAIN_TRACING_V2", None)
            os.environ.pop("LANGCHAIN_API_KEY", None)
            os.environ.pop("LANGCHAIN_PROJECT", None)
            os.environ.pop("LANGCHAIN_ENDPOINT", None)

    @staticmethod
    def is_enabled() -> bool:
        return os.environ.get("LANGCHAIN_TRACING_V2") == "true"


def configure_tracing(settings: LangSmithSettings) -> None:
    LangSmithTracing(settings).configure_tracing()