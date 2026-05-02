from __future__ import annotations

import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from weather_agent.adapters.telegram.sender import (
    TelegramNotificationSender,
    format_notification_message,
)
from weather_agent.domain.rules.models import NotificationEvent, NotificationRule


def _make_rule(
    short_id: str = "R7K2",
    telegram_chat_id: int = 100,
    telegram_message_thread_id: int | None = None,
    expression: str = 'max_metric("wind_gusts_10m_ms", weekend()) >= 12',
    description: str | None = None,
    dry_run: bool = False,
) -> NotificationRule:
    return NotificationRule(
        id=1,
        short_id=short_id,
        user_id=42,
        telegram_chat_id=telegram_chat_id,
        telegram_message_thread_id=telegram_message_thread_id,
        location_id=5,
        expression_language="cel",
        expression=expression,
        cooldown_minutes=60,
        enabled=True,
        dry_run=dry_run,
        description=description,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _make_event(
    short_id: str = "E9M4",
    telegram_chat_id: int = 100,
    telegram_message_thread_id: int | None = None,
) -> NotificationEvent:
    return NotificationEvent(
        id=10,
        short_id=short_id,
        rule_id=1,
        evaluation_run_id=None,
        telegram_chat_id=telegram_chat_id,
        telegram_message_thread_id=telegram_message_thread_id,
        sent_at=None,
        suppressed=False,
        suppress_reason=None,
        payload_hash=None,
        message_text=None,
        created_at=datetime.now(UTC),
    )


def _make_bot_app() -> MagicMock:
    app = MagicMock()
    app.bot = MagicMock()
    app.bot.send_message = AsyncMock()
    return app


class TestFormatNotificationMessage:
    def test_contains_rule_short_id(self) -> None:
        rule = _make_rule(short_id="R7K2")
        event = _make_event(short_id="E9M4")
        msg = format_notification_message(rule, event, "Porywy wiatru powyżej 12 m/s")
        assert "#R7K2" in msg

    def test_contains_event_short_id(self) -> None:
        rule = _make_rule(short_id="R7K2")
        event = _make_event(short_id="E9M4")
        msg = format_notification_message(rule, event, "Porywy wiatru powyżej 12 m/s")
        assert "#E9M4" in msg

    def test_contains_expression(self) -> None:
        rule = _make_rule(expression='max_metric("wind_gusts_10m_ms", weekend()) >= 12')
        event = _make_event()
        msg = format_notification_message(rule, event, "Porywy wiatru powyżej 12 m/s")
        assert 'max_metric("wind_gusts_10m_ms", weekend()) >= 12' in msg

    def test_contains_explanation(self) -> None:
        rule = _make_rule()
        event = _make_event()
        explanation = "Porywy wiatru osiągną 15 m/s w weekend"
        msg = format_notification_message(rule, event, explanation)
        assert explanation in msg

    def test_contains_description_when_present(self) -> None:
        rule = _make_rule(description="Alert na silny wiatr")
        event = _make_event()
        msg = format_notification_message(rule, event, "explanation")
        assert "Opis: Alert na silny wiatr" in msg

    def test_no_description_line_when_absent(self) -> None:
        rule = _make_rule(description=None)
        event = _make_event()
        msg = format_notification_message(rule, event, "explanation")
        assert "Opis:" not in msg

    def test_no_secrets_in_message(self) -> None:
        rule = _make_rule()
        event = _make_event()
        explanation = "token=abc123 api_key=secret"
        msg = format_notification_message(rule, event, explanation)
        assert "bot_token" not in msg
        assert "123456:ABC-DEF" not in msg

    def test_hash_prefix_stripped_from_short_ids(self) -> None:
        rule = _make_rule(short_id="R3A1")
        event = _make_event(short_id="E5B2")
        msg = format_notification_message(rule, event, "explanation")
        assert "#R3A1" in msg
        assert "#E5B2" in msg
        assert "##R" not in msg
        assert "##E" not in msg


class TestSendNotification:
    @pytest.mark.asyncio
    async def test_sends_to_correct_chat(self) -> None:
        app = _make_bot_app()
        sender = TelegramNotificationSender(bot=app)
        rule = _make_rule(telegram_chat_id=200)
        event = _make_event()
        result = await sender.send_notification(rule, event, "explanation")
        assert result is True
        app.bot.send_message.assert_awaited_once()
        call_kwargs = app.bot.send_message.call_args[1]
        assert call_kwargs["chat_id"] == 200

    @pytest.mark.asyncio
    async def test_sends_to_thread_when_present(self) -> None:
        app = _make_bot_app()
        sender = TelegramNotificationSender(bot=app)
        rule = _make_rule(telegram_chat_id=200, telegram_message_thread_id=42)
        event = _make_event()
        result = await sender.send_notification(rule, event, "explanation")
        assert result is True
        call_kwargs = app.bot.send_message.call_args[1]
        assert call_kwargs["chat_id"] == 200
        assert call_kwargs["message_thread_id"] == 42

    @pytest.mark.asyncio
    async def test_sends_without_thread_when_absent(self) -> None:
        app = _make_bot_app()
        sender = TelegramNotificationSender(bot=app)
        rule = _make_rule(telegram_chat_id=200, telegram_message_thread_id=None)
        event = _make_event()
        result = await sender.send_notification(rule, event, "explanation")
        assert result is True
        call_kwargs = app.bot.send_message.call_args[1]
        assert call_kwargs["message_thread_id"] is None

    @pytest.mark.asyncio
    async def test_message_contains_short_ids(self) -> None:
        app = _make_bot_app()
        sender = TelegramNotificationSender(bot=app)
        rule = _make_rule(short_id="R7K2")
        event = _make_event(short_id="E9M4")
        result = await sender.send_notification(rule, event, "Porywy wiatru powyżej 12 m/s")
        assert result is True
        sent_text = app.bot.send_message.call_args[1]["text"]
        assert "#R7K2" in sent_text
        assert "#E9M4" in sent_text

    @pytest.mark.asyncio
    async def test_no_secrets_in_sent_message(self) -> None:
        app = _make_bot_app()
        sender = TelegramNotificationSender(bot=app)
        rule = _make_rule()
        event = _make_event()
        result = await sender.send_notification(rule, event, "explanation")
        assert result is True
        sent_text = app.bot.send_message.call_args[1]["text"]
        assert "token" not in sent_text.lower()
        assert "api_key" not in sent_text.lower()
        assert "secret" not in sent_text.lower()

    @pytest.mark.asyncio
    async def test_returns_false_on_send_failure(self) -> None:
        app = _make_bot_app()
        app.bot.send_message.side_effect = Exception("Telegram API error")
        sender = TelegramNotificationSender(bot=app)
        rule = _make_rule()
        event = _make_event()
        result = await sender.send_notification(rule, event, "explanation")
        assert result is False

    @pytest.mark.asyncio
    async def test_failure_does_not_crash(self) -> None:
        app = _make_bot_app()
        app.bot.send_message.side_effect = Exception("Network error")
        sender = TelegramNotificationSender(bot=app)
        rule = _make_rule()
        event = _make_event()
        result = await sender.send_notification(rule, event, "explanation")
        assert result is False


class TestSendNotificationDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_prefix(self) -> None:
        app = _make_bot_app()
        sender = TelegramNotificationSender(bot=app)
        rule = _make_rule()
        event = _make_event()
        result = await sender.send_notification_dry_run(rule, event, "Porywy wiatru powyżej 12 m/s")
        assert result is True
        sent_text = app.bot.send_message.call_args[1]["text"]
        assert sent_text.startswith("[DRY-RUN]")

    @pytest.mark.asyncio
    async def test_dry_run_still_contains_short_ids(self) -> None:
        app = _make_bot_app()
        sender = TelegramNotificationSender(bot=app)
        rule = _make_rule(short_id="R7K2")
        event = _make_event(short_id="E9M4")
        result = await sender.send_notification_dry_run(rule, event, "Porywy wiatru powyżej 12 m/s")
        assert result is True
        sent_text = app.bot.send_message.call_args[1]["text"]
        assert "#R7K2" in sent_text
        assert "#E9M4" in sent_text

    @pytest.mark.asyncio
    async def test_dry_run_sends_to_correct_chat(self) -> None:
        app = _make_bot_app()
        sender = TelegramNotificationSender(bot=app)
        rule = _make_rule(telegram_chat_id=300, telegram_message_thread_id=55)
        event = _make_event()
        result = await sender.send_notification_dry_run(rule, event, "explanation")
        assert result is True
        call_kwargs = app.bot.send_message.call_args[1]
        assert call_kwargs["chat_id"] == 300
        assert call_kwargs["message_thread_id"] == 55

    @pytest.mark.asyncio
    async def test_dry_run_returns_false_on_send_failure(self) -> None:
        app = _make_bot_app()
        app.bot.send_message.side_effect = Exception("Telegram API error")
        sender = TelegramNotificationSender(bot=app)
        rule = _make_rule()
        event = _make_event()
        result = await sender.send_notification_dry_run(rule, event, "explanation")
        assert result is False

    @pytest.mark.asyncio
    async def test_dry_run_no_secrets_in_output(self) -> None:
        app = _make_bot_app()
        sender = TelegramNotificationSender(bot=app)
        rule = _make_rule()
        event = _make_event()
        result = await sender.send_notification_dry_run(rule, event, "explanation")
        assert result is True
        sent_text = app.bot.send_message.call_args[1]["text"]
        assert "token" not in sent_text.lower()
        assert "api_key" not in sent_text.lower()


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_send_failure_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        app = _make_bot_app()
        app.bot.send_message.side_effect = Exception("Telegram API error")
        sender = TelegramNotificationSender(bot=app)
        rule = _make_rule(telegram_chat_id=900)
        event = _make_event()
        with caplog.at_level(logging.ERROR, logger="weather_agent.adapters.telegram.sender"):
            await sender.send_notification(rule, event, "explanation")
        assert len(caplog.records) > 0

    @pytest.mark.asyncio
    async def test_no_internal_error_details_leaked(self) -> None:
        app = _make_bot_app()
        app.bot.send_message.side_effect = RuntimeError("Secret internal token=abc")
        sender = TelegramNotificationSender(bot=app)
        rule = _make_rule()
        event = _make_event()
        result = await sender.send_notification(rule, event, "explanation")
        assert result is False

    @pytest.mark.asyncio
    async def test_network_failure_returns_false(self) -> None:
        app = _make_bot_app()
        app.bot.send_message.side_effect = ConnectionError("Network unreachable")
        sender = TelegramNotificationSender(bot=app)
        rule = _make_rule()
        event = _make_event()
        result = await sender.send_notification(rule, event, "explanation")
        assert result is False
