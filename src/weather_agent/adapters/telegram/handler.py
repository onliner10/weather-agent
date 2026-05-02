from __future__ import annotations

import asyncio
import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from telegram import Update
from telegram.ext import ContextTypes

from weather_agent.adapters.telegram.agent_invocation import invoke_agent_with_timeout
from weather_agent.agent_factory import build_current_time_prompt_suffix
from weather_agent.domain.locations import LocationService
from weather_agent.domain.rules.service import NotificationRuleService
from weather_agent.infrastructure.app_container import AppContainer
from weather_agent.infrastructure.memory.thread_memory import ThreadMemoryService
from weather_agent.llm.tools.rules_tools import RulesToolbox
from weather_agent.llm.tools.weather_tools import WeatherToolbox
from weather_agent.observability.logging import (
    bound_telegram_context,
    generate_correlation_id,
    get_logger,
)
from weather_agent.observability.metrics import (
    CONVERSATION_FAILURES_TOTAL,
    CONVERSATION_TURN_DURATION_SECONDS,
    CONVERSATION_TURNS_TOTAL,
    REPLY_SEND_DURATION_SECONDS,
    REPLY_SEND_TOTAL,
)
from weather_agent.observability.tracing import build_graph_config

logger = get_logger(__name__)


async def make_message_handler(container: AppContainer) -> Any:
    logger.info("Application services ready")

    async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None or update.effective_user is None:
            return

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id if update.effective_chat else 0
        thread_id = (
            update.message.message_thread_id
            if hasattr(update.message, "message_thread_id")
            else None
        )
        text = update.message.text

        if text is None:
            return

        message_id = update.message.message_id
        reply_to_message_id = None
        if update.message.reply_to_message is not None:
            reply_to_message_id = update.message.reply_to_message.message_id

        context_key = f"{chat_id}:{thread_id}" if thread_id else str(chat_id)

        with bound_telegram_context(
            correlation_id=generate_correlation_id(),
            chat_id=chat_id,
            message_thread_id=thread_id,
            telegram_user_id=user_id,
            message_id=message_id,
            reply_to_message_id=reply_to_message_id,
            context_key=context_key,
        ):
            from weather_agent.adapters.telegram.context import TelegramContextService
            from weather_agent.infrastructure.repositories.auth_repository import AuthRepository

            async with container.session_factory() as session:
                auth_repo = AuthRepository(session)
                authorized_user_id = await auth_repo.get_or_create_authorized_user_id(user_id)

                location_service = LocationService(session)
                assert container.cel_evaluator is not None
                rule_service = NotificationRuleService(
                    session=session,
                    cel_evaluator=container.cel_evaluator,
                )
                context_service = TelegramContextService(session)
                memory_service = ThreadMemoryService(context_service)

                assert container.forecast_provider is not None
                assert container.geocoder is not None
                assert container.model_factory is not None
                tool_session_lock = asyncio.Lock()

                weather_toolbox = WeatherToolbox(
                    forecast_provider=container.forecast_provider,
                    observation_provider=container.observation_provider,
                    geocoder=container.geocoder,
                    location_service=location_service,
                    user_id=authorized_user_id,
                    session_lock=tool_session_lock,
                )

                rules_toolbox = RulesToolbox(
                    rule_service=rule_service,
                    location_service=location_service,
                    cel_evaluator=container.cel_evaluator,
                    geocoder=container.geocoder,
                    memory_service=memory_service,
                    context_key=context_key,
                    user_id=authorized_user_id,
                    chat_id=chat_id,
                    message_thread_id=thread_id,
                    session_lock=tool_session_lock,
                )

                all_tools = (
                    weather_toolbox.to_langchain_tools() + rules_toolbox.to_langchain_tools()
                )

                context_suffix = build_current_time_prompt_suffix()

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

                direct_answer: str | None = None
                trimmed = text.strip()
                if trimmed in _DIRECT_CONFIRMATIONS or trimmed in _DIRECT_CANCELLATIONS:
                    pending_dict = await memory_service.get_pending_confirmation(context_key)
                    if pending_dict is not None:
                        if trimmed in _DIRECT_CONFIRMATIONS:
                            direct_answer = (await rules_toolbox.confirm_pending_action()).answer
                        else:
                            direct_answer = (await rules_toolbox.cancel_pending_action()).answer
                        if not direct_answer:
                            direct_answer = (
                                "Przepraszam, wystąpił błąd. Spróbuj ponownie za chwilę."
                            )

                if direct_answer is not None:
                    answer = direct_answer
                    await _save_turn(memory_service, context_key, text, answer)
                    await session.commit()
                else:
                    messages: list[BaseMessage] = []
                    conversation_turns = await memory_service.load_turns(context_key)
                    if conversation_turns:
                        for turn in conversation_turns:
                            role = turn.get("role")
                            content = turn.get("text") or turn.get("answer_summary")
                            if content and role == "user":
                                messages.append(HumanMessage(content=content))
                            elif content and role == "bot":
                                messages.append(AIMessage(content=content))

                    messages.append(HumanMessage(content=text))

                    graph_config = build_graph_config(
                        {
                            "authorized_user_id": user_id,
                            "chat_id": chat_id,
                            "context_key": context_key,
                        },
                    )
                    if container.model_factory is not None:
                        graph_config["metadata"]["model_provider"] = (
                            container.model_factory.provider
                        )
                        graph_config["metadata"]["model_name"] = container.model_factory.model_name

                    CONVERSATION_TURNS_TOTAL.inc()
                    turn_start = time.perf_counter()
                    answer, failed = await invoke_agent_with_timeout(
                        model_factory=container.model_factory,
                        tools=all_tools,
                        messages=messages,
                        config={
                            "configurable": {"thread_id": context_key},
                            **graph_config,
                        },
                        system_prompt_suffix=context_suffix,
                        timeout_seconds=container.settings.model.timeout_seconds,
                        logger=logger,
                    )
                    if failed:
                        CONVERSATION_FAILURES_TOTAL.inc()
                    CONVERSATION_TURN_DURATION_SECONDS.observe(time.perf_counter() - turn_start)

                    await _save_turn(
                        memory_service,
                        context_key,
                        text,
                        answer,
                    )
                    await session.commit()

            reply_start = time.perf_counter()
            try:
                await update.message.reply_text(answer)
                REPLY_SEND_TOTAL.labels(outcome="success").inc()
            except Exception:
                REPLY_SEND_TOTAL.labels(outcome="failure").inc()
            finally:
                REPLY_SEND_DURATION_SECONDS.observe(time.perf_counter() - reply_start)

    logger.info("message_handler_ready")
    return message_handler


async def _save_turn(
    memory_service: ThreadMemoryService,
    context_key: str,
    user_message: str,
    bot_message: str,
) -> None:
    from datetime import UTC, datetime

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
