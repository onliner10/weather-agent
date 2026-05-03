from __future__ import annotations

import time
from io import BytesIO
from typing import Any

from telegram import InputFile, Update
from telegram.ext import ContextTypes

from weather_agent.adapters.telegram.context import TelegramContextService
from weather_agent.application.conversation_models import UserMessage
from weather_agent.application.conversation_service import ConversationService
from weather_agent.infrastructure.app_container import AppContainer
from weather_agent.infrastructure.memory.thread_memory import ThreadMemoryService
from weather_agent.observability.logging import (
    bound_telegram_context,
    generate_correlation_id,
    get_logger,
)
from weather_agent.observability.metrics import (
    REPLY_SEND_DURATION_SECONDS,
    REPLY_SEND_TOTAL,
)

logger = get_logger(__name__)


async def make_message_handler(container: AppContainer) -> Any:
    logger.info("Application services ready")
    conversation_service = ConversationService(
        session_factory=container.session_factory,
        forecast_provider=container.forecast_provider,
        observation_provider=container.observation_provider,
        geocoder=container.geocoder,
        model_factory=container.model_factory,
        rule_expression_evaluator=container.rule_expression_evaluator,
        timeout_seconds=container.settings.model.timeout_seconds,
        memory_factory=lambda session: ThreadMemoryService(TelegramContextService(session)),
    )

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
            reply = await conversation_service.handle_reply(
                UserMessage(
                    telegram_user_id=user_id,
                    chat_id=chat_id,
                    message_thread_id=thread_id,
                    text=text,
                    message_id=message_id,
                    reply_to_message_id=reply_to_message_id,
                )
            )

            reply_start = time.perf_counter()
            try:
                await update.message.reply_text(reply.text)
                for attachment in reply.attachments:
                    if attachment.media_type == "image/png":
                        await update.message.reply_photo(
                            photo=InputFile(
                                BytesIO(attachment.data),
                                filename=attachment.filename,
                            ),
                            caption=attachment.caption,
                        )
                REPLY_SEND_TOTAL.labels(outcome="success").inc()
            except Exception:
                REPLY_SEND_TOTAL.labels(outcome="failure").inc()
            finally:
                REPLY_SEND_DURATION_SECONDS.observe(time.perf_counter() - reply_start)

    logger.info("message_handler_ready")
    return message_handler
