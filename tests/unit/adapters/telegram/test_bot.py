from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import Chat, Message, Update, User
from telegram.ext import ContextTypes

from weather_agent.adapters.telegram.bot import TelegramBot
from weather_agent.domain.auth import AuthorizationService
from weather_agent.settings import TelegramSettings


def _make_settings(user_ids: list[int] | None = None) -> TelegramSettings:
    from pydantic import SecretStr

    return TelegramSettings(
        bot_token=SecretStr("123456:ABC-DEF"),
        allowed_user_ids=tuple(user_ids) if user_ids else (),
    )


def _make_update(
    user_id: int = 42,
    chat_id: int = 100,
    text: str = "hello",
    message_thread_id: int | None = None,
) -> Update:
    user = User(id=user_id, first_name="Test", is_bot=False, username="testuser")
    chat = Chat(id=chat_id, type=Chat.PRIVATE)
    message = MagicMock(spec=Message)
    message.reply_text = AsyncMock()
    message.text = text
    message.from_user = user
    message.chat = chat
    message.message_thread_id = message_thread_id
    update = MagicMock(spec=Update)
    update.effective_user = user
    update.effective_chat = chat
    update.message = message
    return update


def _make_context() -> ContextTypes.DEFAULT_TYPE:
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    return context


def _make_command_update(
    user_id: int = 42,
    chat_id: int = 100,
    command_text: str = "/start",
) -> Update:
    user = User(id=user_id, first_name="Test", is_bot=False, username="testuser")
    chat = Chat(id=chat_id, type=Chat.PRIVATE)
    message = MagicMock(spec=Message)
    message.reply_text = AsyncMock()
    message.text = command_text
    message.from_user = user
    message.chat = chat
    update = MagicMock(spec=Update)
    update.effective_user = user
    update.effective_chat = chat
    update.message = message
    return update


class TestTelegramBotCreation:
    def test_bot_created_from_settings(self) -> None:
        settings = _make_settings(user_ids=[42])
        auth = AuthorizationService(allowed_user_ids=[42])
        bot = TelegramBot(settings=settings, auth_service=auth)
        assert bot is not None

    def test_bot_setup_creates_application(self) -> None:
        settings = _make_settings(user_ids=[42])
        auth = AuthorizationService(allowed_user_ids=[42])
        bot = TelegramBot(settings=settings, auth_service=auth)
        bot.setup()
        assert bot._app is not None

    def test_setup_without_call_raises_on_start(self) -> None:
        settings = _make_settings(user_ids=[42])
        auth = AuthorizationService(allowed_user_ids=[42])
        bot = TelegramBot(settings=settings, auth_service=auth)
        with pytest.raises(RuntimeError, match="setup"):
            import asyncio

            asyncio.get_event_loop().run_until_complete(bot.start())

    def test_setup_without_call_raises_on_run(self) -> None:
        settings = _make_settings(user_ids=[42])
        auth = AuthorizationService(allowed_user_ids=[42])
        bot = TelegramBot(settings=settings, auth_service=auth)
        with pytest.raises(RuntimeError, match="setup"):
            bot.run()

    def test_custom_message_handler(self) -> None:
        custom_handler = AsyncMock()
        settings = _make_settings(user_ids=[42])
        auth = AuthorizationService(allowed_user_ids=[42])
        bot = TelegramBot(settings=settings, auth_service=auth, message_handler=custom_handler)
        assert bot._message_handler is custom_handler


class TestUnauthorizedMessageHandling:
    @pytest.mark.asyncio
    async def test_unauthorized_user_gets_denial(self) -> None:
        settings = _make_settings(user_ids=[42])
        auth = AuthorizationService(allowed_user_ids=[42])
        bot = TelegramBot(settings=settings, auth_service=auth)
        bot.setup()
        update = _make_update(user_id=999)
        context = _make_context()
        await bot._auth_check(update, context)
        update.message.reply_text.assert_awaited_once()
        denial_text = update.message.reply_text.call_args[0][0]
        assert "Brak uprawnień" in denial_text

    @pytest.mark.asyncio
    async def test_unauthorized_user_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        settings = _make_settings(user_ids=[42])
        auth = AuthorizationService(allowed_user_ids=[42])
        bot = TelegramBot(settings=settings, auth_service=auth)
        bot.setup()
        update = _make_update(user_id=999, chat_id=200)
        context = _make_context()
        with caplog.at_level(logging.INFO, logger="weather_agent.adapters.telegram.bot"):
            await bot._auth_check(update, context)
        assert "999" in caplog.text
        assert "200" in caplog.text

    @pytest.mark.asyncio
    async def test_no_secrets_in_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        settings = _make_settings(user_ids=[42])
        auth = AuthorizationService(allowed_user_ids=[42])
        bot = TelegramBot(settings=settings, auth_service=auth)
        bot.setup()
        update = _make_update(user_id=999)
        context = _make_context()
        with caplog.at_level(logging.INFO, logger="weather_agent.adapters.telegram.bot"):
            await bot._auth_check(update, context)
        for record in caplog.records:
            assert "123456:ABC-DEF" not in record.getMessage()
            assert "bot_token" not in record.getMessage()


class TestAuthorizedMessageHandling:
    @pytest.mark.asyncio
    async def test_authorized_message_passes_to_handler(self) -> None:
        custom_handler = AsyncMock()
        settings = _make_settings(user_ids=[42])
        auth = AuthorizationService(allowed_user_ids=[42])
        bot = TelegramBot(settings=settings, auth_service=auth, message_handler=custom_handler)
        bot.setup()
        update = _make_update(user_id=42, text="pogoda warszawa")
        context = _make_context()
        await bot._auth_check(update, context)
        custom_handler.assert_awaited_once_with(update, context)

    @pytest.mark.asyncio
    async def test_default_handler_echoes(self) -> None:
        settings = _make_settings(user_ids=[42])
        auth = AuthorizationService(allowed_user_ids=[42])
        bot = TelegramBot(settings=settings, auth_service=auth)
        bot.setup()
        update = _make_update(user_id=42, text="pogoda")
        context = _make_context()
        await bot._auth_check(update, context)
        update.message.reply_text.assert_awaited_once()
        response = update.message.reply_text.call_args[0][0]
        assert "pogoda" in response

    @pytest.mark.asyncio
    async def test_authorized_user_not_denied(self) -> None:
        settings = _make_settings(user_ids=[42])
        auth = AuthorizationService(allowed_user_ids=[42])
        bot = TelegramBot(settings=settings, auth_service=auth)
        bot.setup()
        update = _make_update(user_id=42)
        context = _make_context()
        await bot._auth_check(update, context)
        denial_call_args = [
            call for call in update.message.reply_text.call_args_list
            if "Brak uprawnień" in str(call)
        ]
        assert len(denial_call_args) == 0


class TestStartCommand:
    @pytest.mark.asyncio
    async def test_start_for_authorized_user(self) -> None:
        settings = _make_settings(user_ids=[42])
        auth = AuthorizationService(allowed_user_ids=[42])
        bot = TelegramBot(settings=settings, auth_service=auth)
        bot.setup()
        update = _make_command_update(user_id=42, command_text="/start")
        context = _make_context()
        await bot._start_command(update, context)
        update.message.reply_text.assert_awaited_once()
        response = update.message.reply_text.call_args[0][0]
        assert "Cześć" in response

    @pytest.mark.asyncio
    async def test_start_for_unauthorized_user(self) -> None:
        settings = _make_settings(user_ids=[42])
        auth = AuthorizationService(allowed_user_ids=[42])
        bot = TelegramBot(settings=settings, auth_service=auth)
        bot.setup()
        update = _make_command_update(user_id=999, command_text="/start")
        context = _make_context()
        await bot._start_command(update, context)
        update.message.reply_text.assert_awaited_once()
        response = update.message.reply_text.call_args[0][0]
        assert "Brak uprawnień" in response


class TestHelpCommand:
    @pytest.mark.asyncio
    async def test_help_for_authorized_user(self) -> None:
        settings = _make_settings(user_ids=[42])
        auth = AuthorizationService(allowed_user_ids=[42])
        bot = TelegramBot(settings=settings, auth_service=auth)
        bot.setup()
        update = _make_command_update(user_id=42, command_text="/help")
        context = _make_context()
        await bot._help_command(update, context)
        update.message.reply_text.assert_awaited_once()
        response = update.message.reply_text.call_args[0][0]
        assert "/start" in response
        assert "/help" in response
        assert "/status" in response

    @pytest.mark.asyncio
    async def test_help_for_unauthorized_user(self) -> None:
        settings = _make_settings(user_ids=[42])
        auth = AuthorizationService(allowed_user_ids=[42])
        bot = TelegramBot(settings=settings, auth_service=auth)
        bot.setup()
        update = _make_command_update(user_id=999, command_text="/help")
        context = _make_context()
        await bot._help_command(update, context)
        update.message.reply_text.assert_awaited_once()
        response = update.message.reply_text.call_args[0][0]
        assert "Brak uprawnień" in response


class TestStatusCommand:
    @pytest.mark.asyncio
    async def test_status_for_authorized_user(self) -> None:
        settings = _make_settings(user_ids=[42])
        auth = AuthorizationService(allowed_user_ids=[42])
        bot = TelegramBot(settings=settings, auth_service=auth)
        bot.setup()
        update = _make_command_update(user_id=42, command_text="/status")
        context = _make_context()
        await bot._status_command(update, context)
        update.message.reply_text.assert_awaited_once()
        response = update.message.reply_text.call_args[0][0]
        assert "działa" in response

    @pytest.mark.asyncio
    async def test_status_for_unauthorized_user(self) -> None:
        settings = _make_settings(user_ids=[42])
        auth = AuthorizationService(allowed_user_ids=[42])
        bot = TelegramBot(settings=settings, auth_service=auth)
        bot.setup()
        update = _make_command_update(user_id=999, command_text="/status")
        context = _make_context()
        await bot._status_command(update, context)
        update.message.reply_text.assert_awaited_once()
        response = update.message.reply_text.call_args[0][0]
        assert "Brak uprawnień" in response


class TestNoUserInUpdate:
    @pytest.mark.asyncio
    async def test_auth_check_with_no_user(self) -> None:
        settings = _make_settings(user_ids=[42])
        auth = AuthorizationService(allowed_user_ids=[42])
        bot = TelegramBot(settings=settings, auth_service=auth)
        bot.setup()
        update = MagicMock(spec=Update)
        update.effective_user = None
        update.effective_chat = Chat(id=100, type=Chat.PRIVATE)
        context = _make_context()
        await bot._auth_check(update, context)
        assert True

    @pytest.mark.asyncio
    async def test_start_command_with_no_user(self) -> None:
        settings = _make_settings(user_ids=[42])
        auth = AuthorizationService(allowed_user_ids=[42])
        bot = TelegramBot(settings=settings, auth_service=auth)
        bot.setup()
        update = MagicMock(spec=Update)
        update.effective_user = None
        update.effective_chat = Chat(id=100, type=Chat.PRIVATE)
        context = _make_context()
        await bot._start_command(update, context)
        assert True