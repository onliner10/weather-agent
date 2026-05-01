from __future__ import annotations

from typing import Any

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from weather_agent.llm.model_factory import ModelFactory, ModelProvider
from weather_agent.settings import ModelSettings


def _make_settings(**overrides: Any) -> ModelSettings:
    return ModelSettings(**overrides)


class TestModelProvider:
    def test_provider_enum_values(self) -> None:
        assert ModelProvider.openai == "openai"
        assert ModelProvider.anthropic == "anthropic"
        assert ModelProvider.deepseek == "deepseek"
        assert ModelProvider.glm == "glm"
        assert ModelProvider.openrouter == "openrouter"

    def test_provider_from_string(self) -> None:
        assert ModelProvider("openai") is ModelProvider.openai
        assert ModelProvider("anthropic") is ModelProvider.anthropic
        assert ModelProvider("deepseek") is ModelProvider.deepseek
        assert ModelProvider("glm") is ModelProvider.glm
        assert ModelProvider("openrouter") is ModelProvider.openrouter

    def test_invalid_provider_raises(self) -> None:
        with pytest.raises(ValueError):
            ModelProvider("unknown")


class TestModelFactoryDefaults:
    def test_default_provider_is_openrouter(self) -> None:
        factory = ModelFactory()
        assert factory._provider is ModelProvider.openrouter

    def test_default_model_name(self) -> None:
        factory = ModelFactory()
        assert factory._settings.model_name == "qwen/qwen3.5-flash-02-23"

    def test_default_temperature(self) -> None:
        factory = ModelFactory()
        assert factory._settings.temperature == 0.2

    def test_custom_settings(self) -> None:
        settings = _make_settings(
            provider="anthropic",
            model_name="claude-3-5-sonnet-latest",
            temperature=0.5,
        )
        factory = ModelFactory(settings=settings)
        assert factory._provider is ModelProvider.anthropic
        assert factory._settings.model_name == "claude-3-5-sonnet-latest"
        assert factory._settings.temperature == 0.5


class TestCreateChatModel:
    def test_openai_creates_chat_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings = _make_settings(provider="openai", model_name="gpt-5-mini")
        factory = ModelFactory(settings=settings)
        model = factory.create_chat_model()
        assert isinstance(model, ChatOpenAI)

    def test_openai_passes_model_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings = _make_settings(provider="openai", model_name="gpt-5-mini")
        factory = ModelFactory(settings=settings)
        model = factory.create_chat_model()
        assert model.model_name == "gpt-5-mini"

    def test_openai_passes_temperature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings = _make_settings(provider="openai", model_name="gpt-4o-mini", temperature=0.7)
        factory = ModelFactory(settings=settings)
        model = factory.create_chat_model()
        assert model.temperature == 0.7

    def test_anthropic_creates_chat_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
        settings = _make_settings(provider="anthropic", model_name="claude-3-5-sonnet-latest")
        factory = ModelFactory(settings=settings)
        model = factory.create_chat_model()
        assert isinstance(model, ChatAnthropic)

    def test_anthropic_passes_model_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
        settings = _make_settings(provider="anthropic", model_name="claude-3-5-sonnet-latest")
        factory = ModelFactory(settings=settings)
        model = factory.create_chat_model()
        assert model.model == "claude-3-5-sonnet-latest"

    def test_deepseek_creates_chat_openai_with_deepseek_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings = _make_settings(provider="deepseek", model_name="deepseek-chat")
        factory = ModelFactory(settings=settings)
        model = factory.create_chat_model()
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "deepseek-chat"
        assert "deepseek" in model.openai_api_base

    def test_glm_creates_chat_openai_with_glm_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings = _make_settings(provider="glm", model_name="glm-4-flash")
        factory = ModelFactory(settings=settings)
        model = factory.create_chat_model()
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "glm-4-flash"
        assert "bigmodel" in model.openai_api_base

    def test_openrouter_creates_chat_openai_with_openrouter_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings = _make_settings(provider="openrouter", model_name="openai/gpt-4.1-nano")
        factory = ModelFactory(settings=settings)
        model = factory.create_chat_model()
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "openai/gpt-4.1-nano"
        assert model.openai_api_base == "https://openrouter.ai/api/v1"


class TestCreateChatModelCustomBaseUrl:
    def test_openai_custom_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings = _make_settings(
            provider="openai",
            model_name="gpt-5-mini",
            base_url="https://custom-openai.example.com/v1",
        )
        factory = ModelFactory(settings=settings)
        model = factory.create_chat_model()
        assert isinstance(model, ChatOpenAI)
        assert model.openai_api_base == "https://custom-openai.example.com/v1"

    def test_deepseek_custom_base_url_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings = _make_settings(
            provider="deepseek",
            model_name="deepseek-chat",
            base_url="https://my-proxy.example.com/v1",
        )
        factory = ModelFactory(settings=settings)
        model = factory.create_chat_model()
        assert model.openai_api_base == "https://my-proxy.example.com/v1"

    def test_glm_custom_base_url_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings = _make_settings(
            provider="glm",
            model_name="glm-4-flash",
            base_url="https://my-glm-proxy.example.com/v1",
        )
        factory = ModelFactory(settings=settings)
        model = factory.create_chat_model()
        assert model.openai_api_base == "https://my-glm-proxy.example.com/v1"

    def test_anthropic_custom_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
        settings = _make_settings(
            provider="anthropic",
            model_name="claude-3-5-sonnet-latest",
            base_url="https://my-anthropic-proxy.example.com",
        )
        factory = ModelFactory(settings=settings)
        model = factory.create_chat_model()
        assert isinstance(model, ChatAnthropic)
        assert model.anthropic_api_url == "https://my-anthropic-proxy.example.com"

    def test_openrouter_custom_base_url_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings = _make_settings(
            provider="openrouter",
            model_name="openai/gpt-4.1-nano",
            base_url="https://my-openrouter-proxy.example.com/v1",
        )
        factory = ModelFactory(settings=settings)
        model = factory.create_chat_model()
        assert isinstance(model, ChatOpenAI)
        assert model.openai_api_base == "https://my-openrouter-proxy.example.com/v1"


class TestRoutingConfiguration:
    def test_openrouter_maps_routing_config_to_extra_body(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings = _make_settings(
            provider="openrouter",
            model_name="openai/gpt-4.1-nano",
            routing_sort="latency",
            require_supported_parameters=True,
        )
        factory = ModelFactory(settings=settings)
        model = factory.create_chat_model()
        assert isinstance(model, ChatOpenAI)
        assert model.extra_body == {
            "provider": {
                "require_parameters": True,
                "sort": "latency",
            }
        }

    def test_openrouter_can_disable_required_supported_parameters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings = _make_settings(
            provider="openrouter",
            model_name="openai/gpt-4.1-nano",
            require_supported_parameters=False,
        )
        factory = ModelFactory(settings=settings)
        model = factory.create_chat_model()
        assert isinstance(model, ChatOpenAI)
        assert model.extra_body == {
            "provider": {
                "require_parameters": False,
            }
        }

    def test_routing_sort_on_direct_provider_raises(self) -> None:
        settings = _make_settings(
            provider="openai",
            model_name="gpt-4.1-nano",
            routing_sort="price",
        )
        with pytest.raises(ValueError, match="routing_sort is only supported"):
            ModelFactory(settings=settings)

    def test_disabling_required_parameters_on_direct_provider_raises(self) -> None:
        settings = _make_settings(
            provider="openai",
            model_name="gpt-4.1-nano",
            require_supported_parameters=False,
        )
        with pytest.raises(ValueError, match="require_supported_parameters"):
            ModelFactory(settings=settings)


class TestApiKeyHandling:
    def test_openai_api_key_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
        settings = _make_settings(
            provider="openai",
            model_name="gpt-5-mini",
            api_key=SecretStr("sk-explicit-key"),
        )
        factory = ModelFactory(settings=settings)
        model = factory.create_chat_model()
        assert isinstance(model, ChatOpenAI)
        assert model.openai_api_key.get_secret_value() == "sk-explicit-key"

    def test_anthropic_api_key_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-env-key")
        settings = _make_settings(
            provider="anthropic",
            model_name="claude-3-5-sonnet-latest",
            api_key=SecretStr("ant-explicit-key"),
        )
        factory = ModelFactory(settings=settings)
        model = factory.create_chat_model()
        assert isinstance(model, ChatAnthropic)
        assert model.anthropic_api_key.get_secret_value() == "ant-explicit-key"

    def test_no_api_key_uses_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
        settings = _make_settings(provider="openai", model_name="gpt-5-mini")
        factory = ModelFactory(settings=settings)
        model = factory.create_chat_model()
        assert isinstance(model, ChatOpenAI)
        assert model.openai_api_key.get_secret_value() == "sk-env-key"


class TestCreateStructuredOutput:
    def test_structured_output_returns_runnable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pydantic import BaseModel as PydanticModel

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        class TestSchema(PydanticModel):
            answer: str
            confidence: float

        settings = _make_settings(provider="openai", model_name="gpt-5-mini")
        factory = ModelFactory(settings=settings)
        runnable = factory.create_structured_output(TestSchema)
        assert runnable is not None

    def test_invalid_provider_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="not a valid ModelProvider"):
            ModelFactory(_make_settings(provider="invalid_provider"))
