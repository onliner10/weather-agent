from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from langsmith import trace

from weather_agent.domain.rules.models import NotificationEvent, NotificationRule
from weather_agent.domain.rules.short_id_generator import strip_hash_prefix
from weather_agent.observability.logging import get_logger
from weather_agent.observability.metrics import (
    NOTIFICATION_FAILURES_TOTAL,
    NOTIFICATION_SEND_DURATION_SECONDS,
    NOTIFICATIONS_TOTAL,
)

if TYPE_CHECKING:
    from telegram.ext import Application

    _AppType = Application[Any, Any, Any, Any, Any, Any]

logger = get_logger(__name__)


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
            dry_run=False,
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
            "dry_run_notification",
            chat_id=rule.telegram_chat_id,
            thread_id=rule.telegram_message_thread_id,
            rule_short_id=rule.short_id,
            event_short_id=event.short_id,
        )
        return await self._send_telegram_message(
            chat_id=rule.telegram_chat_id,
            thread_id=rule.telegram_message_thread_id,
            text=dry_run_text,
            dry_run=True,
        )

    async def _send_telegram_message(
        self,
        chat_id: int,
        thread_id: int | None,
        text: str,
        dry_run: bool = False,
    ) -> bool:
        notification_type = "dry_run" if dry_run else "normal"
        async with trace(
            "send_notification",
            run_type="tool",
            metadata={
                "chat_id": chat_id,
                "thread_id": thread_id,
                "dry_run": dry_run,
            },
        ):
            try:
                send_start = time.perf_counter()
                await self._bot.bot.send_message(
                    chat_id=chat_id,
                    message_thread_id=thread_id,
                    text=text,
                )
                NOTIFICATIONS_TOTAL.labels(type=notification_type).inc()
                NOTIFICATION_SEND_DURATION_SECONDS.labels(type=notification_type).observe(
                    time.perf_counter() - send_start
                )
                logger.info(
                    "notification_sent",
                    chat_id=chat_id,
                    thread_id=thread_id,
                    dry_run=dry_run,
                )
                return True
            except Exception:
                NOTIFICATION_FAILURES_TOTAL.labels(type=notification_type).inc()
                logger.exception(
                    "telegram_send_failed",
                    chat_id=chat_id,
                    thread_id=thread_id,
                    dry_run=dry_run,
                )
                return False
