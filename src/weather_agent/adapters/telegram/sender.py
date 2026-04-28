from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from weather_agent.domain.rules.models import NotificationEvent, NotificationRule
from weather_agent.domain.rules.short_id_generator import strip_hash_prefix

if TYPE_CHECKING:
    from telegram.ext import Application

    _AppType = Application[Any, Any, Any, Any, Any, Any]

logger = logging.getLogger(__name__)


def format_notification_message(
    rule: NotificationRule,
    event: NotificationEvent,
    explanation: str,
) -> str:
    rule_tag = f"#{strip_hash_prefix(rule.short_id)}"
    event_tag = f"#{strip_hash_prefix(event.short_id)}"

    lines: list[str] = [
        f"⚡ Powiadomienie {event_tag}",
        f"Reguła: {rule_tag} — {rule.expression}",
        explanation,
    ]

    desc = rule.description
    if desc:
        lines.append(f"Opis: {desc}")

    return "\n".join(lines)


class TelegramNotificationSender:
    def __init__(self, bot: _AppType) -> None:
        self._bot = bot

    async def send_notification(
        self,
        rule: NotificationRule,
        event: NotificationEvent,
        explanation: str,
    ) -> bool:
        message_text = format_notification_message(rule, event, explanation)
        return await self._send_telegram_message(
            chat_id=rule.telegram_chat_id,
            thread_id=rule.telegram_message_thread_id,
            text=message_text,
        )

    async def send_notification_dry_run(
        self,
        rule: NotificationRule,
        event: NotificationEvent,
        explanation: str,
    ) -> bool:
        message_text = format_notification_message(rule, event, explanation)
        dry_run_text = f"[DRY-RUN] {message_text}"
        logger.info(
            "dry-run notification: chat_id=%s thread_id=%s rule=%s event=%s",
            rule.telegram_chat_id,
            rule.telegram_message_thread_id,
            rule.short_id,
            event.short_id,
        )
        return await self._send_telegram_message(
            chat_id=rule.telegram_chat_id,
            thread_id=rule.telegram_message_thread_id,
            text=dry_run_text,
        )

    async def _send_telegram_message(
        self,
        chat_id: int,
        thread_id: int | None,
        text: str,
    ) -> bool:
        try:
            await self._bot.bot.send_message(
                chat_id=chat_id,
                message_thread_id=thread_id,
                text=text,
            )
            logger.info(
                "notification sent: chat_id=%s thread_id=%s",
                chat_id,
                thread_id,
            )
            return True
        except Exception:
            logger.exception(
                "failed to send notification: chat_id=%s thread_id=%s",
                chat_id,
                thread_id,
            )
            return False