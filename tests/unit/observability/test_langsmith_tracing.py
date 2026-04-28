from __future__ import annotations

import os

from pydantic import SecretStr

from weather_agent.observability.langsmith_tracing import LangSmithTracing, configure_tracing
from weather_agent.settings import LangSmithSettings


class TestLangSmithTracing:
    def setup_method(self) -> None:
        for key in (
            "LANGCHAIN_TRACING_V2",
            "LANGCHAIN_API_KEY",
            "LANGCHAIN_PROJECT",
            "LANGCHAIN_ENDPOINT",
        ):
            os.environ.pop(key, None)

    def test_env_vars_set_when_enabled(self) -> None:
        settings = LangSmithSettings(
            enabled=True,
            api_key=SecretStr("ls-test-key"),
            project="test-project",
            endpoint="https://api.smith.langchain.com",
        )
        tracing = LangSmithTracing(settings)
        tracing.configure_tracing()
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGCHAIN_API_KEY"] == "ls-test-key"
        assert os.environ["LANGCHAIN_PROJECT"] == "test-project"
        assert os.environ["LANGCHAIN_ENDPOINT"] == "https://api.smith.langchain.com"

    def test_no_env_vars_when_disabled(self) -> None:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = "old-key"
        settings = LangSmithSettings(enabled=False)
        tracing = LangSmithTracing(settings)
        tracing.configure_tracing()
        assert "LANGCHAIN_TRACING_V2" not in os.environ
        assert "LANGCHAIN_API_KEY" not in os.environ

    def test_is_enabled_returns_true_when_set(self) -> None:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        assert LangSmithTracing.is_enabled() is True

    def test_is_enabled_returns_false_when_not_set(self) -> None:
        assert LangSmithTracing.is_enabled() is False

    def test_app_works_without_langsmith_configured(self) -> None:
        settings = LangSmithSettings(enabled=False)
        tracing = LangSmithTracing(settings)
        tracing.configure_tracing()
        assert not LangSmithTracing.is_enabled()

    def test_enabled_without_api_key_sets_tracing_flag(self) -> None:
        settings = LangSmithSettings(
            enabled=True,
            api_key=None,
            project="test-project",
            endpoint="https://api.smith.langchain.com",
        )
        tracing = LangSmithTracing(settings)
        tracing.configure_tracing()
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert "LANGCHAIN_API_KEY" not in os.environ

    def test_configure_tracing_with_explicit_settings(self) -> None:
        settings = LangSmithSettings(
            enabled=True,
            api_key=SecretStr("explicit-key"),
            project="explicit-project",
        )
        tracing = LangSmithTracing()
        tracing.configure_tracing(settings)
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGCHAIN_API_KEY"] == "explicit-key"

    def test_default_project_name(self) -> None:
        settings = LangSmithSettings(enabled=False)
        assert settings.project == "weather-agent-dev"

    def test_configure_tracing_module_function(self) -> None:
        settings = LangSmithSettings(
            enabled=True,
            api_key=SecretStr("func-key"),
            project="func-project",
        )
        configure_tracing(settings)
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGCHAIN_API_KEY"] == "func-key"

    def teardown_method(self) -> None:
        for key in (
            "LANGCHAIN_TRACING_V2",
            "LANGCHAIN_API_KEY",
            "LANGCHAIN_PROJECT",
            "LANGCHAIN_ENDPOINT",
        ):
            os.environ.pop(key, None)