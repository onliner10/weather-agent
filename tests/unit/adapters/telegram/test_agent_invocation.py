from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from weather_agent.adapters.telegram import agent_invocation


class _FakeAgent:
    def __init__(self, result: str | Exception) -> None:
        self._result = result

    async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        if isinstance(self._result, Exception):
            raise self._result
        return {"messages": [AIMessage(content=self._result)]}


class _FakeModelFactory:
    def __init__(self, primary: object, fallback: object | None) -> None:
        self._primary = primary
        self._fallback = fallback

    def create_chat_model(self) -> object:
        return self._primary

    def create_fallback_chat_model(self) -> object | None:
        return self._fallback


class _Logger:
    def warning(self, event: str, **kwargs: object) -> None:
        pass

    def exception(self, event: str, **kwargs: object) -> None:
        pass


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("temporarily unavailable"),
        RuntimeError("rate limit exceeded"),
        RuntimeError("503 service unavailable"),
    ],
)
async def test_transient_provider_error_uses_fallback(
    monkeypatch: pytest.MonkeyPatch,
    exc: Exception,
) -> None:
    primary = object()
    fallback = object()

    def fake_create_weather_agent(
        model: object,
        tools: list[Any],
        system_prompt_suffix: str,
    ) -> _FakeAgent:
        if model is primary:
            return _FakeAgent(exc)
        return _FakeAgent("fallback answer")

    monkeypatch.setattr(agent_invocation, "create_weather_agent", fake_create_weather_agent)

    answer, failed = await agent_invocation.invoke_agent_with_timeout(
        model_factory=_FakeModelFactory(primary=primary, fallback=fallback),  # type: ignore[arg-type]
        tools=[],
        messages=[HumanMessage(content="pogoda")],
        config={},
        system_prompt_suffix="",
        timeout_seconds=1,
        logger=_Logger(),
    )

    assert answer == "fallback answer"
    assert failed is False


async def test_non_transient_provider_error_does_not_use_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = object()
    fallback = object()
    fallback_called = False

    def fake_create_weather_agent(
        model: object,
        tools: list[Any],
        system_prompt_suffix: str,
    ) -> _FakeAgent:
        nonlocal fallback_called
        if model is primary:
            return _FakeAgent(ValueError("bad prompt"))
        fallback_called = True
        return _FakeAgent("fallback answer")

    monkeypatch.setattr(agent_invocation, "create_weather_agent", fake_create_weather_agent)

    answer, failed = await agent_invocation.invoke_agent_with_timeout(
        model_factory=_FakeModelFactory(primary=primary, fallback=fallback),  # type: ignore[arg-type]
        tools=[],
        messages=[HumanMessage(content="pogoda")],
        config={},
        system_prompt_suffix="",
        timeout_seconds=1,
        logger=_Logger(),
    )

    assert answer == "Przepraszam, wystąpił błąd. Spróbuj ponownie za chwilę."
    assert failed is True
    assert fallback_called is False
