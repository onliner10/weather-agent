from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from weather_agent.agent_factory import create_weather_agent
from weather_agent.llm.model_factory import ModelFactory

_TIMEOUT_ANSWER = "Przepraszam, odpowiedź trwała zbyt długo. Spróbuj ponownie za chwilę."
_GENERIC_FAILURE_ANSWER = "Przepraszam, wystąpił błąd. Spróbuj ponownie za chwilę."
_EMPTY_ANSWER = "Przepraszam, nie udało się przetworzyć zapytania."
_TRANSIENT_ERROR_MARKERS = (
    "timeout",
    "timed out",
    "rate limit",
    "rate_limit",
    "temporar",
    "unavailable",
    "overload",
    "connection",
    "connect",
    "readtimeout",
    "503",
    "502",
    "504",
)


class Logger(Protocol):
    def warning(self, event: str, **kwargs: object) -> None: ...
    def exception(self, event: str, **kwargs: object) -> None: ...


def _extract_answer(result: Any) -> str:
    final = result["messages"][-1]
    answer = final.content if hasattr(final, "content") else str(final)
    return answer or _EMPTY_ANSWER


def _is_transient_model_error(exc: Exception) -> bool:
    error_text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in error_text for marker in _TRANSIENT_ERROR_MARKERS)


async def _invoke_agent(
    model: Any,
    tools: list[Any],
    messages: Sequence[BaseMessage],
    config: RunnableConfig,
    system_prompt_suffix: str,
    timeout_seconds: float,
) -> str:
    agent = create_weather_agent(
        model=model,
        tools=tools,
        system_prompt_suffix=system_prompt_suffix,
    )
    result = await asyncio.wait_for(
        agent.ainvoke({"messages": list(messages)}, config=config),
        timeout=timeout_seconds,
    )
    return _extract_answer(result)


async def _invoke_fallback(
    model_factory: ModelFactory,
    tools: list[Any],
    messages: Sequence[BaseMessage],
    config: RunnableConfig,
    system_prompt_suffix: str,
    timeout_seconds: float,
    logger: Logger,
    no_fallback_answer: str,
) -> tuple[str, bool]:
    fallback_model = model_factory.create_fallback_chat_model()
    if fallback_model is None:
        return no_fallback_answer, True
    try:
        answer = await _invoke_agent(
            model=fallback_model,
            tools=tools,
            messages=messages,
            config=config,
            system_prompt_suffix=system_prompt_suffix,
            timeout_seconds=timeout_seconds,
        )
        return answer, False
    except Exception as exc:
        logger.exception(
            "fallback_agent_invocation_failed",
            error_class=type(exc).__name__,
            outcome="failure",
        )
        return _GENERIC_FAILURE_ANSWER, True


async def invoke_agent_with_timeout(
    model_factory: ModelFactory,
    tools: list[Any],
    messages: Sequence[BaseMessage],
    config: RunnableConfig,
    system_prompt_suffix: str,
    timeout_seconds: float,
    logger: Logger,
) -> tuple[str, bool]:
    try:
        answer = await _invoke_agent(
            model=model_factory.create_chat_model(),
            tools=tools,
            messages=messages,
            config=config,
            system_prompt_suffix=system_prompt_suffix,
            timeout_seconds=timeout_seconds,
        )
        return answer, False
    except TimeoutError:
        answer, failed = await _invoke_fallback(
            model_factory=model_factory,
            tools=tools,
            messages=messages,
            config=config,
            system_prompt_suffix=system_prompt_suffix,
            timeout_seconds=timeout_seconds,
            logger=logger,
            no_fallback_answer=_TIMEOUT_ANSWER,
        )
        if failed and answer == _TIMEOUT_ANSWER:
            logger.warning("agent_invocation_timed_out", outcome="failure")
        return answer, failed
    except Exception as exc:
        if _is_transient_model_error(exc):
            return await _invoke_fallback(
                model_factory=model_factory,
                tools=tools,
                messages=messages,
                config=config,
                system_prompt_suffix=system_prompt_suffix,
                timeout_seconds=timeout_seconds,
                logger=logger,
                no_fallback_answer=_GENERIC_FAILURE_ANSWER,
            )
        logger.exception(
            "agent_invocation_failed",
            error_class=type(exc).__name__,
            outcome="failure",
        )
        return _GENERIC_FAILURE_ANSWER, True
