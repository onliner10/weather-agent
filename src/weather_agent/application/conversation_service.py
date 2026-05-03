from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from typing import Any, Protocol, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from weather_agent.agent_factory import build_current_time_prompt_suffix
from weather_agent.application.agent_invocation import invoke_agent_with_timeout
from weather_agent.application.conversation_models import UserMessage
from weather_agent.domain.locations import LocationService
from weather_agent.domain.rules.service import NotificationRuleService
from weather_agent.infrastructure.repositories.auth_repository import AuthRepository
from weather_agent.llm.model_factory import ModelFactory
from weather_agent.llm.tools.rules_tools import RulesToolbox
from weather_agent.llm.tools.weather_tools import WeatherToolbox
from weather_agent.observability.logging import get_logger
from weather_agent.observability.metrics import (
    CONVERSATION_FAILURES_TOTAL,
    CONVERSATION_TURN_DURATION_SECONDS,
    CONVERSATION_TURNS_TOTAL,
)
from weather_agent.observability.tracing import build_graph_config

logger = get_logger(__name__)

_DIRECT_CONFIRMATIONS: frozenset[str] = frozenset(
    {
        "tak",
        "Tak",
        "TAK",
        "ok",
        "OK",
        "Ok",
        "potwierdzam",
        "Potwierdzam",
        "yes",
        "Yes",
        "YES",
    }
)
_DIRECT_CANCELLATIONS: frozenset[str] = frozenset(
    {
        "nie",
        "Nie",
        "NIE",
        "anuluj",
        "Anuluj",
        "no",
        "No",
        "NO",
    }
)
_GENERIC_FAILURE_ANSWER = "Przepraszam, wystąpił błąd. Spróbuj ponownie za chwilę."


class MemoryService(Protocol):
    async def get_pending_confirmation(self, context_key: str) -> dict[str, Any] | None: ...
    async def save_turn(self, context_key: str, turn: dict[str, Any]) -> None: ...
    async def load_turns(self, context_key: str) -> list[dict[str, Any]]: ...


class ToolProvider(Protocol):
    def to_langchain_tools(self) -> list[Any]: ...


class RulesToolProvider(ToolProvider, Protocol):
    async def confirm_pending_action(self) -> Any: ...
    async def cancel_pending_action(self) -> Any: ...


class Logger(Protocol):
    def warning(self, event: str, **kwargs: object) -> None: ...
    def exception(self, event: str, **kwargs: object) -> None: ...


type AuthResolver = Callable[[AsyncSession, int], Awaitable[int]]
type MemoryFactory = Callable[[AsyncSession], MemoryService]
type WeatherToolboxFactory = Callable[..., ToolProvider]
type RulesToolboxFactory = Callable[..., RulesToolProvider]
type AgentInvoker = Callable[
    [ModelFactory, list[Any], Sequence[BaseMessage], RunnableConfig, str, float, Logger],
    Awaitable[tuple[str, bool]],
]


async def _resolve_authorized_user_id(session: AsyncSession, telegram_user_id: int) -> int:
    auth_repo = AuthRepository(session)
    return await auth_repo.get_or_create_authorized_user_id(telegram_user_id)


class ConversationService:
    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        forecast_provider: Any,
        observation_provider: Any,
        geocoder: Any,
        model_factory: ModelFactory,
        rule_expression_evaluator: Any,
        timeout_seconds: float,
        memory_factory: MemoryFactory,
        auth_resolver: AuthResolver = _resolve_authorized_user_id,
        weather_toolbox_factory: WeatherToolboxFactory = WeatherToolbox,
        rules_toolbox_factory: RulesToolboxFactory = RulesToolbox,
        agent_invoker: AgentInvoker = invoke_agent_with_timeout,
    ) -> None:
        self._session_factory = session_factory
        self._forecast_provider = forecast_provider
        self._observation_provider = observation_provider
        self._geocoder = geocoder
        self._model_factory = model_factory
        self._rule_expression_evaluator = rule_expression_evaluator
        self._timeout_seconds = timeout_seconds
        self._memory_factory = memory_factory
        self._auth_resolver = auth_resolver
        self._weather_toolbox_factory = weather_toolbox_factory
        self._rules_toolbox_factory = rules_toolbox_factory
        self._agent_invoker = agent_invoker

    async def handle(self, request: UserMessage) -> str:
        async with cast(
            AbstractAsyncContextManager[AsyncSession], self._session_factory()
        ) as session:
            answer = await self._handle_with_session(session, request)
            await session.commit()
            return answer

    async def _handle_with_session(self, session: AsyncSession, request: UserMessage) -> str:
        authorized_user_id = await self._auth_resolver(session, request.telegram_user_id)
        location_service = LocationService(session)
        rule_service = NotificationRuleService(
            session=session,
            rule_expression_evaluator=self._rule_expression_evaluator,
        )
        memory_service = self._memory_factory(session)
        tool_session_lock = asyncio.Lock()

        weather_toolbox = self._weather_toolbox_factory(
            forecast_provider=self._forecast_provider,
            observation_provider=self._observation_provider,
            geocoder=self._geocoder,
            location_service=location_service,
            user_id=authorized_user_id,
            session_lock=tool_session_lock,
        )
        rules_toolbox = self._rules_toolbox_factory(
            rule_service=rule_service,
            location_service=location_service,
            rule_expression_evaluator=self._rule_expression_evaluator,
            geocoder=self._geocoder,
            memory_service=memory_service,
            context_key=request.context_key,
            user_id=authorized_user_id,
            chat_id=request.chat_id,
            message_thread_id=request.message_thread_id,
            current_user_message=request.text,
            session_lock=tool_session_lock,
        )

        direct_answer = await self._handle_direct_confirmation(
            request=request,
            memory_service=memory_service,
            rules_toolbox=rules_toolbox,
        )
        if direct_answer is not None:
            await save_turn(memory_service, request.context_key, request.text, direct_answer)
            return direct_answer

        messages = build_conversation_messages(
            await memory_service.load_turns(request.context_key),
            request.text,
        )
        context_suffix = build_current_time_prompt_suffix()
        graph_config = build_graph_config(
            {
                "authorized_user_id": request.telegram_user_id,
                "chat_id": request.chat_id,
                "message_thread_id": request.message_thread_id,
                "message_id": request.message_id,
                "reply_to_message_id": request.reply_to_message_id,
                "context_key": request.context_key,
                "user_message": request.text,
            },
        )
        graph_config["metadata"]["model_provider"] = self._model_factory.provider
        graph_config["metadata"]["model_name"] = self._model_factory.model_name
        runtime_config = cast(
            RunnableConfig,
            {
                "configurable": {"thread_id": request.context_key},
                **graph_config,
            },
        )

        CONVERSATION_TURNS_TOTAL.inc()
        turn_start = time.perf_counter()
        answer, failed = await self._agent_invoker(
            self._model_factory,
            weather_toolbox.to_langchain_tools() + rules_toolbox.to_langchain_tools(),
            messages,
            runtime_config,
            context_suffix,
            self._timeout_seconds,
            logger,
        )
        if failed:
            CONVERSATION_FAILURES_TOTAL.inc()
        CONVERSATION_TURN_DURATION_SECONDS.observe(time.perf_counter() - turn_start)

        await save_turn(memory_service, request.context_key, request.text, answer)
        return answer

    async def _handle_direct_confirmation(
        self,
        *,
        request: UserMessage,
        memory_service: MemoryService,
        rules_toolbox: RulesToolProvider,
    ) -> str | None:
        trimmed = request.text.strip()
        if trimmed not in _DIRECT_CONFIRMATIONS and trimmed not in _DIRECT_CANCELLATIONS:
            return None

        pending_dict = await memory_service.get_pending_confirmation(request.context_key)
        if pending_dict is None:
            return None

        if trimmed in _DIRECT_CONFIRMATIONS:
            result = await rules_toolbox.confirm_pending_action()
        else:
            result = await rules_toolbox.cancel_pending_action()

        answer: object | None
        error: object | None = None
        if isinstance(result, Mapping):
            answer = result.get("answer")
            error = result.get("error")
        else:
            answer = getattr(result, "answer", None)
            error = getattr(result, "error", None)
        if isinstance(answer, str) and answer:
            return answer
        if isinstance(error, str) and error:
            return error
        return _GENERIC_FAILURE_ANSWER


async def save_turn(
    memory_service: MemoryService,
    context_key: str,
    user_message: str,
    bot_message: str,
) -> None:
    try:
        await memory_service.save_turn(
            context_key,
            {
                "role": "user",
                "text": user_message,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        if bot_message:
            await memory_service.save_turn(
                context_key,
                {
                    "role": "bot",
                    "text": bot_message[:200] if len(bot_message) > 200 else bot_message,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
    except Exception:
        logger.warning("save_turn_failed", context_key=context_key, exc_info=True)


def build_conversation_messages(
    conversation_turns: Sequence[Mapping[str, object]],
    current_user_text: str,
) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for turn in conversation_turns:
        role = turn.get("role")
        content = turn.get("text") or turn.get("answer_summary")
        if isinstance(content, str) and role == "user":
            messages.append(HumanMessage(content=content))
        elif isinstance(content, str) and role == "bot":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=current_user_text))
    return messages
