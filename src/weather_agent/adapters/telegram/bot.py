from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from weather_agent.domain.auth import AuthorizationService
from weather_agent.domain.locations import LocationCreate, LocationService
from weather_agent.infrastructure.db.base import AuthorizedUser
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

    Wired commands: ``/start``, ``/help``, ``/status``, ``/dodaj_lok``.
    All text messages are routed through ``_auth_check`` → ``message_handler``.
    """

    def __init__(
        self,
        settings: TelegramSettings,
        auth_service: AuthorizationService,
message_handler: MessageHandlerCallback,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._settings = settings
        self._auth_service = auth_service
        self._message_handler = message_handler
        self._session_factory = session_factory
        self._app: _AppType | None = None

    def setup(self) -> None:
        token = self._settings.bot_token.get_secret_value()
        self._app = Application.builder().token(token).post_init(_post_init).build()
        self._app.add_handler(CommandHandler("start", self._start_command))
        self._app.add_handler(CommandHandler("help", self._help_command))
        self._app.add_handler(CommandHandler("status", self._status_command))
        self._app.add_handler(CommandHandler("dodaj_lok", self._dodaj_lok_command))
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
                "/status — status bota\n"
                "/dodaj_lok <nazwa> <lat> <lon> — zapisz lokalizację domową\n\n"
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

    async def _dodaj_lok_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle ``/dodaj_lok <name> <lat> <lon>`` — persist a home location.

        Parses the last two tokens as latitude/longitude and everything before
        as the location name. Persists via :class:`LocationService` and replies
        with a Polish success message only after the database write succeeds.
        """
        user = update.effective_user
        if user is None:
            return
        TELEGRAM_MESSAGES_TOTAL.inc()
        if not self._auth_service.is_authorized(user.id):
            chat_id = update.effective_chat.id if update.effective_chat else None
            logger.info(
                "Unauthorized /dodaj_lok from user_id=%s chat_id=%s",
                user.id,
                chat_id,
            )
            AUTHORIZATION_FAILURES_TOTAL.inc()
            await _send_denial(update, context)
            return

        if update.message is None:
            return

        # context.args is split by whitespace with the command stripped
        parts = context.args or []
        if len(parts) < 3:
            await update.message.reply_text(
                "Użycie: /dodaj_lok <nazwa> <szerokość> <długość>\n"
                "Np. /dodaj_lok Dom 52.2297 21.0122"
            )
            return

        try:
            lat = float(parts[-2])
            lon = float(parts[-1])
        except (ValueError, IndexError):
            await update.message.reply_text(
                "Nieprawidłowe współrzędne. Użyj formatu: /dodaj_lok <nazwa> <lat> <lon>"
            )
            return

        name = " ".join(parts[:-2]).strip()
        if not name:
            await update.message.reply_text(
                "Nieprawidłowa nazwa. Użyj formatu: /dodaj_lok <nazwa> <lat> <lon>"
            )
            return

        if self._session_factory is None:
            logger.error("session_factory not configured — cannot persist location")
            await update.message.reply_text(
                "Przepraszam, wystąpił błąd wewnętrzny. Spróbuj ponownie za chwilę."
            )
            return

        try:
            async with self._session_factory() as session:
                # Ensure an AuthorizedUser row exists for this Telegram user
                # so the FK on locations.user_id → authorized_users.id is satisfied.
                result = await session.execute(
                    select(AuthorizedUser).where(AuthorizedUser.telegram_user_id == user.id)
                )
                authorized_user = result.scalar_one_or_none()
                if authorized_user is None:
                    authorized_user = AuthorizedUser(telegram_user_id=user.id)
                    session.add(authorized_user)
                    await session.flush()

                location_service = LocationService(session)
                data = LocationCreate(
                    name=name,
                    aliases=["dom"],
                    latitude=lat,
                    longitude=lon,
                )
                await location_service.create_location(authorized_user.id, data)
                await session.commit()

            await update.message.reply_text(f"Zapamiętałem Twoją lokalizację domową jako {name}.")
            logger.info(
                "dodaj_lok_success",
                user_id=user.id,
                location_name=name,
                latitude=lat,
                longitude=lon,
            )
        except Exception:
            logger.exception(
                "dodaj_lok_failed",
                user_id=user.id,
                location_name=name,
            )
            await update.message.reply_text("Nie udało się zapisać lokalizacji. Spróbuj ponownie.")

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


async def _send_denial(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text("Brak uprawnień do korzystania z tego bota.")


async def _post_init(application: _AppType) -> None:
    logger.info("Telegram bot application initialized")
