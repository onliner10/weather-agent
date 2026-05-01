from __future__ import annotations

from enum import StrEnum
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from weather_agent.settings import ModelSettings


class ModelProvider(StrEnum):
    openai = "openai"
    anthropic = "anthropic"
    deepseek = "deepseek"
    glm = "glm"
    openrouter = "openrouter"


_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
_GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _base_chat_kwargs(settings: ModelSettings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "temperature": settings.temperature,
    }
    if settings.api_key is not None:
        kwargs["api_key"] = settings.api_key.get_secret_value()
    return kwargs


def _create_openai_model(settings: ModelSettings) -> BaseChatModel:
    kwargs = _base_chat_kwargs(settings)
    if settings.base_url is not None:
        kwargs["base_url"] = settings.base_url
    return ChatOpenAI(
        model=settings.model_name,
        **kwargs,
    )


def _create_anthropic_model(settings: ModelSettings) -> BaseChatModel:
    kwargs = _base_chat_kwargs(settings)
    if settings.base_url is not None:
        kwargs["base_url"] = settings.base_url
    return ChatAnthropic(
        model_name=settings.model_name,
        **kwargs,
    )


def _create_deepseek_model(settings: ModelSettings) -> BaseChatModel:
    kwargs = _base_chat_kwargs(settings)
    kwargs["base_url"] = settings.base_url or _DEEPSEEK_BASE_URL
    return ChatOpenAI(
        model=settings.model_name,
        **kwargs,
    )


def _create_glm_model(settings: ModelSettings) -> BaseChatModel:
    kwargs = _base_chat_kwargs(settings)
    kwargs["base_url"] = settings.base_url or _GLM_BASE_URL
    return ChatOpenAI(
        model=settings.model_name,
        **kwargs,
    )


def _openrouter_extra_body(settings: ModelSettings) -> dict[str, object]:
    provider: dict[str, object] = {
        "require_parameters": settings.require_supported_parameters,
    }
    if settings.routing_sort is not None:
        provider["sort"] = settings.routing_sort
    return {"provider": provider}


def _create_openrouter_model(settings: ModelSettings) -> BaseChatModel:
    kwargs = _base_chat_kwargs(settings)
    kwargs["base_url"] = settings.base_url or _OPENROUTER_BASE_URL
    kwargs["extra_body"] = _openrouter_extra_body(settings)
    return ChatOpenAI(
        model=settings.model_name,
        **kwargs,
    )


_ROUTING_PROVIDERS: frozenset[ModelProvider] = frozenset({ModelProvider.openrouter})


class ModelFactory:
    def __init__(self, settings: ModelSettings | None = None) -> None:
        self._settings = settings or ModelSettings()
        self._provider = ModelProvider(self._settings.provider)
        self._validate_routing_settings()

    def create_chat_model(self) -> BaseChatModel:
        if self._provider is ModelProvider.openai:
            return _create_openai_model(self._settings)
        if self._provider is ModelProvider.anthropic:
            return _create_anthropic_model(self._settings)
        if self._provider is ModelProvider.deepseek:
            return _create_deepseek_model(self._settings)
        if self._provider is ModelProvider.glm:
            return _create_glm_model(self._settings)
        if self._provider is ModelProvider.openrouter:
            return _create_openrouter_model(self._settings)

        raise ValueError(f"Unsupported model provider: {self._provider}")

    def _validate_routing_settings(self) -> None:
        if self._provider in _ROUTING_PROVIDERS:
            return
        if self._settings.routing_sort is not None:
            raise ValueError(
                f"routing_sort is only supported for routing providers; "
                f"{self._provider.value} does not support it"
            )
        if not self._settings.require_supported_parameters:
            raise ValueError(
                "require_supported_parameters can only be disabled for routing providers; "
                f"{self._provider.value} does not support provider routing"
            )

    @property
    def provider(self) -> str:
        return str(self._provider)

    @property
    def model_name(self) -> str:
        return self._settings.model_name

    def create_structured_output(self, schema: type) -> Runnable[Any, Any]:
        return self.create_chat_model().with_structured_output(schema)
