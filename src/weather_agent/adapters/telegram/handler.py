from __future__ import annotations

import time
from typing import Any

from telegram import Update
from telegram.ext import ContextTypes

from weather_agent.adapters.telegram.home_location import handle_home_location_save_message
from weather_agent.domain.locations import LocationService
from weather_agent.infrastructure.repositories.auth_repository import AuthRepository
from weather_agent.infrastructure.services import BotServices
from weather_agent.observability.logging import (
    bound_telegram_context,
    generate_correlation_id,
    get_logger,
)
from weather_agent.observability.metrics import (
    CONVERSATION_FAILURES_TOTAL,
    CONVERSATION_TURN_DURATION_SECONDS,
    CONVERSATION_TURNS_TOTAL,
    REPLY_CONTEXT_HITS_TOTAL,
    REPLY_SEND_DURATION_SECONDS,
    REPLY_SEND_TOTAL,
)
from weather_agent.observability.tracing import build_graph_config

logger = get_logger(__name__)


async def make_message_handler(services: BotServices) -> Any:
    services.init_services()
    logger.info("Application services ready, compiling base graph...")
    services.compile_graph()
    logger.info("Base graph compiled")

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
            from weather_agent.domain.rules.service import NotificationRuleService
            from weather_agent.graphs.state import ConversationState
            from weather_agent.infrastructure.memory.thread_memory import ThreadMemoryService

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

                try:
                    home_location_answer = await handle_home_location_save_message(
                        text,
                        authorized_user_id,
                        location_service,
                        services.geocoder,
                    )
                except Exception as exc:
                    await session.rollback()
                    logger.exception(
                        "home_location_save_failed",
                        error_class=type(exc).__name__,
                        outcome="failure",
                    )
                    home_location_answer = "Nie udało się zapisać lokalizacji. Spróbuj ponownie."
                if home_location_answer is not None:
                    answer = home_location_answer
                    await session.commit()
                else:
                    graph = services.compile_graph(
                        location_service=location_service,
                        rule_service=rule_service,
                        user_id=authorized_user_id,
                        memory_service=memory_service,
                    )

                    state: ConversationState = {
                        "authorized_user_id": user_id,
                        "chat_id": chat_id,
                        "message_thread_id": thread_id,
                        "context_key": context_key,
                        "user_message": text,
                        "message_id": message_id,
                        "reply_to_message_id": reply_to_message_id,
                    }

                    graph_config = build_graph_config(state)
                    if services.model_factory is not None:
                        graph_config["metadata"]["model_provider"] = services.model_factory.provider
                        graph_config["metadata"]["model_name"] = services.model_factory.model_name

                    CONVERSATION_TURNS_TOTAL.inc()
                    turn_start = time.perf_counter()
                    try:
                        result_state = await graph.ainvoke(state, config=graph_config)
                        answer = result_state.get(
                            "answer",
                            "Przepraszam, nie udało się przetworzyć zapytania.",
                        )
                    except Exception as exc:
                        CONVERSATION_FAILURES_TOTAL.inc()
                        logger.exception(
                            "conversation_graph_failed",
                            error_class=type(exc).__name__,
                            outcome="failure",
                        )
                        answer = "Przepraszam, wystąpił błąd. Spróbuj ponownie za chwilę."
                    finally:
                        CONVERSATION_TURN_DURATION_SECONDS.observe(time.perf_counter() - turn_start)
                    await session.commit()

            if reply_to_message_id is not None:
                REPLY_CONTEXT_HITS_TOTAL.labels(source="reply_to").inc()

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
                            context_key, bot_message_id
                        )
                        await session.commit()
                except Exception:
                    logger.warning(
                        "bot_message_id_persist_failed",
                        exc_info=True,
                    )

    logger.info("message_handler_ready")
    return message_handler
