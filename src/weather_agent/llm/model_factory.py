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


_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
_GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


class ModelFactory:
    def __init__(self, settings: ModelSettings | None = None) -> None:
        self._settings = settings or ModelSettings()
        self._provider = ModelProvider(self._settings.provider)

    def create_chat_model(self) -> BaseChatModel:
        kwargs: dict[str, Any] = {
            "temperature": self._settings.temperature,
        }

        api_key = self._settings.api_key
        if api_key is not None:
            kwargs["api_key"] = api_key.get_secret_value()

        base_url = self._settings.base_url

        if self._provider is ModelProvider.openai:
            if base_url is not None:
                kwargs["base_url"] = base_url
            return ChatOpenAI(
                model=self._settings.model_name,
                **kwargs,
            )

        if self._provider is ModelProvider.anthropic:
            if base_url is not None:
                kwargs["base_url"] = base_url
            return ChatAnthropic(
                model_name=self._settings.model_name,
                **kwargs,
            )

        if self._provider is ModelProvider.deepseek:
            effective_base_url = base_url or _DEEPSEEK_BASE_URL
            kwargs["base_url"] = effective_base_url
            return ChatOpenAI(
                model=self._settings.model_name,
                **kwargs,
            )

        if self._provider is ModelProvider.glm:
            effective_base_url = base_url or _GLM_BASE_URL
            kwargs["base_url"] = effective_base_url
            return ChatOpenAI(
                model=self._settings.model_name,
                **kwargs,
            )

        raise ValueError(f"Unsupported model provider: {self._provider}")

    def create_structured_output(self, schema: type) -> Runnable[Any, Any]:
        return self.create_chat_model().with_structured_output(schema)