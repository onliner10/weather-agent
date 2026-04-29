from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from telegram import Update
from telegram.ext import ContextTypes

from weather_agent.agent_factory import build_context_suffix, create_weather_agent
from weather_agent.domain.locations import LocationService
from weather_agent.domain.rules.service import NotificationRuleService
from weather_agent.infrastructure.memory.thread_memory import ThreadMemoryService
from weather_agent.infrastructure.services import BotServices
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


async def make_message_handler(services: BotServices) -> Any:
    services.init_services()
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

            async with services.session_factory() as session:
                auth_repo = AuthRepository(session)
                authorized_user_id = await auth_repo.get_or_create_authorized_user_id(user_id)

                location_service = LocationService(session)
                assert services.cel_evaluator is not None
                rule_service = NotificationRuleService(
                    session=session,
                    cel_evaluator=services.cel_evaluator,
                )
                context_service = TelegramContextService(session)
                memory_service = ThreadMemoryService(context_service)

                assert services.forecast_provider is not None
                assert services.geocoder is not None
                assert services.model_factory is not None

                weather_toolbox = WeatherToolbox(
                    forecast_provider=services.forecast_provider,
                    observation_provider=services.observation_provider,
                    geocoder=services.geocoder,
                    location_service=location_service,
                    user_id=authorized_user_id,
                )

                rules_toolbox = RulesToolbox(
                    rule_service=rule_service,
                    location_service=location_service,
                    cel_evaluator=services.cel_evaluator,
                    geocoder=services.geocoder,
                    memory_service=memory_service,
                    context_key=context_key,
                    user_id=authorized_user_id,
                    chat_id=chat_id,
                    message_thread_id=thread_id,
                )

                all_tools = (
                    weather_toolbox.to_langchain_tools() + rules_toolbox.to_langchain_tools()
                )

                last_forecast = await memory_service.load_last_forecast(context_key)
                pending_confirmation = await memory_service.get_pending_confirmation(context_key)
                context_suffix = build_context_suffix(
                    pending_confirmation,
                    last_forecast_context=last_forecast,
                )

                model = services.model_factory.create_chat_model()

                agent = create_weather_agent(
                    model=model,
                    tools=all_tools,
                    system_prompt_suffix=context_suffix,
                )

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
                    {"authorized_user_id": user_id, "chat_id": chat_id, "context_key": context_key},
                )
                if services.model_factory is not None:
                    graph_config["metadata"]["model_provider"] = services.model_factory.provider
                    graph_config["metadata"]["model_name"] = services.model_factory.model_name

                CONVERSATION_TURNS_TOTAL.inc()
                turn_start = time.perf_counter()
                forecast_context = None
                try:
                    result = await agent.ainvoke(
                        {"messages": messages},
                        config={
                            "configurable": {"thread_id": context_key},
                            **graph_config,
                        },
                    )
                    final = result["messages"][-1]
                    answer = final.content if hasattr(final, "content") else str(final)
                    if not answer:
                        answer = "Przepraszam, nie udało się przetworzyć zapytania."
                    forecast_context = _extract_forecast_context(result["messages"])
                except Exception as exc:
                    CONVERSATION_FAILURES_TOTAL.inc()
                    logger.exception(
                        "agent_invocation_failed",
                        error_class=type(exc).__name__,
                        outcome="failure",
                    )
                    answer = "Przepraszam, wystąpił błąd. Spróbuj ponownie za chwilę."
                finally:
                    CONVERSATION_TURN_DURATION_SECONDS.observe(time.perf_counter() - turn_start)

                if forecast_context:
                    await memory_service.store_last_forecast(context_key, forecast_context)

                from weather_agent.application.conversation_models import PendingConfirmation as _PC

                new_pending = await memory_service.get_pending_confirmation(context_key)
                pending_for_save = None
                if new_pending and isinstance(new_pending, dict):
                    pending_for_save = _PC.from_dict(new_pending)

                await _save_turn(
                    memory_service,
                    context_key,
                    text,
                    answer,
                    message_id,
                    pending_for_save,
                )
                await session.commit()

            reply_start = time.perf_counter()
            try:
                sent_message = await update.message.reply_text(answer)
                bot_message_id = sent_message.message_id if sent_message else None
                REPLY_SEND_TOTAL.labels(outcome="success").inc()
            except Exception as exc:
                REPLY_SEND_TOTAL.labels(outcome="failure").inc()
                logger.exception(
                    "reply_send_failed",
                    error_class=type(exc).__name__,
                    outcome="failure",
                )
                bot_message_id = None
            finally:
                REPLY_SEND_DURATION_SECONDS.observe(time.perf_counter() - reply_start)

            if bot_message_id is not None:
                try:
                    async with services.session_factory() as session:
                        context_service2 = TelegramContextService(session)
                        memory_service2 = ThreadMemoryService(context_service2)
                        await memory_service2.update_last_bot_turn_message_id(
                            context_key,
                            bot_message_id,
                        )
                        await session.commit()
                except Exception:
                    logger.warning("bot_message_id_persist_failed", exc_info=True)

    logger.info("message_handler_ready")
    return message_handler


def _extract_forecast_context(messages: list[Any]) -> dict[str, Any] | None:
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            if getattr(tc, "name", None) == "get_forecast":
                args = getattr(tc, "args", {}) or {}
                return {
                    "location_name": args.get("location_name", ""),
                    "start_date": args.get("start_date", ""),
                    "end_date": args.get("end_date", ""),
                    "variables": args.get("variables", []),
                }
    return None


async def _save_turn(
    memory_service: ThreadMemoryService,
    context_key: str,
    user_message: str,
    answer: str,
    message_id: int | None,
    pending_confirmation: Any | None,
) -> None:
    from datetime import UTC, datetime

    try:
        user_turn = {
            "message_id": message_id,
            "role": "user",
            "text": user_message,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await memory_service.save_turn(context_key, user_turn)

        if answer:
            summary = answer[:200] if len(answer) > 200 else answer
            bot_turn = {
                "role": "bot",
                "answer_summary": summary,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            await memory_service.save_turn(context_key, bot_turn)

        if pending_confirmation is not None:
            await memory_service.store_pending_confirmation(
                context_key,
                pending_confirmation.to_dict(),
            )
        else:
            stored = await memory_service.get_pending_confirmation(context_key)
            if stored is not None:
                propag = _PC_from_dict_if_needed(stored)
                if propag and propag.cel_expression == "" and propag.action == "create_rule":
                    await memory_service.clear_pending_confirmation(context_key)
    except Exception:
        logger.warning("save_turn_failed", context_key=context_key, exc_info=True)


def _PC_from_dict_if_needed(data: dict[str, Any]) -> Any:
    from weather_agent.application.conversation_models import PendingConfirmation

    try:
        return PendingConfirmation.from_dict(data)
    except Exception:
        return None
