from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from weather_agent.domain.auth import AuthorizationService
from weather_agent.observability.logging import get_logger
from weather_agent.observability.metrics import (
    AUTHORIZATION_FAILURES_TOTAL,
    TELEGRAM_MESSAGES_TOTAL,
)
from weather_agent.settings import TelegramSettings

logger = get_logger(__name__)

MessageHandlerCallback = Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[Any]]
_AppType = Application[Any, Any, Any, Any, Any, Any]


class TelegramBot:
    """Telegram bot runtime with a narrow command surface.

    Wired commands: ``/start``, ``/help``, ``/status``.
    All text messages are routed through ``_auth_check`` → ``message_handler``.
    """

    def __init__(
        self,
        settings: TelegramSettings,
        auth_service: AuthorizationService,
        message_handler: MessageHandlerCallback | None = None,
    ) -> None:
        self._settings = settings
        self._auth_service = auth_service
        self._message_handler = message_handler or _default_message_handler
        self._app: _AppType | None = None

    def setup(self) -> None:
        token = self._settings.bot_token.get_secret_value()
        self._app = Application.builder().token(token).post_init(_post_init).build()
        self._app.add_handler(CommandHandler("start", self._start_command))
        self._app.add_handler(CommandHandler("help", self._help_command))
        self._app.add_handler(CommandHandler("status", self._status_command))
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._auth_check))

    async def start(self) -> None:
        if self._app is None:
            raise RuntimeError("TelegramBot.setup() must be called before start()")
        await self._app.initialize()
        await self._app.start()
        assert self._app.updater is not None
        await self._app.updater.start_polling()

    async def stop(self) -> None:
        if self._app is None:
            return
        assert self._app.updater is not None
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()

    def run(self) -> None:
        if self._app is None:
            raise RuntimeError("TelegramBot.setup() must be called before run()")
        self._app.run_polling(drop_pending_updates=True)

    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        if user is None:
            return
        TELEGRAM_MESSAGES_TOTAL.inc()
        if not self._auth_service.is_authorized(user.id):
            chat_id = update.effective_chat.id if update.effective_chat else None
            logger.info(
                "Unauthorized /start from user_id=%s chat_id=%s",
                user.id,
                chat_id,
            )
            AUTHORIZATION_FAILURES_TOTAL.inc()
            await _send_denial(update, context)
            return
        if update.message is not None:
            await update.message.reply_text(
                "Cześć! Jestem botem pogodowym. \U0001f326\n"
                'Zapytaj mnie o pogodę – np. "jaka pogoda w Warszawie jutro?".'
            )

    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        if user is None:
            return
        TELEGRAM_MESSAGES_TOTAL.inc()
        if not self._auth_service.is_authorized(user.id):
            chat_id = update.effective_chat.id if update.effective_chat else None
            logger.info(
                "Unauthorized /help from user_id=%s chat_id=%s",
                user.id,
                chat_id,
            )
            AUTHORIZATION_FAILURES_TOTAL.inc()
            await _send_denial(update, context)
            return
        if update.message is not None:
            await update.message.reply_text(
                "Dostępne komendy:\n"
                "/start — przywitanie\n"
                "/help — ta pomoc\n"
                "/status — status bota\n\n"
                "Możesz też po prostu napisać pytanie o pogodę."
            )

    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        if user is None:
            return
        TELEGRAM_MESSAGES_TOTAL.inc()
        if not self._auth_service.is_authorized(user.id):
            chat_id = update.effective_chat.id if update.effective_chat else None
            logger.info(
                "Unauthorized /status from user_id=%s chat_id=%s",
                user.id,
                chat_id,
            )
            AUTHORIZATION_FAILURES_TOTAL.inc()
            await _send_denial(update, context)
            return
        if update.message is not None:
            await update.message.reply_text("✅ Bot działa poprawnie.")

    async def _auth_check(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        if user is None:
            return
        TELEGRAM_MESSAGES_TOTAL.inc()
        if not self._auth_service.is_authorized(user.id):
            logger.info(
                "Unauthorized message from user_id=%s chat_id=%s",
                user.id,
                update.effective_chat.id if update.effective_chat else None,
            )
            AUTHORIZATION_FAILURES_TOTAL.inc()
            await _send_denial(update, context)
            return
        await self._message_handler(update, context)


async def _default_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message and update.message.text:
        await update.message.reply_text(
            f"Otrzymałem: {update.message.text}\nObsługa pytań pogodowych będzie dostępna wkrótce."
        )


async def _send_denial(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text("Brak uprawnień do korzystania z tego bota.")


async def _post_init(application: _AppType) -> None:
    logger.info("Telegram bot application initialized")
