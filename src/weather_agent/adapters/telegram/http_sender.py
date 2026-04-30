from __future__ import annotations

import time
from typing import TYPE_CHECKING

import httpx

from weather_agent.observability.logging import get_logger
from weather_agent.observability.metrics import (
    NOTIFICATION_FAILURES_TOTAL,
    NOTIFICATION_SEND_DURATION_SECONDS,
    NOTIFICATIONS_TOTAL,
)

if TYPE_CHECKING:
    from pydantic import SecretStr

logger = get_logger(__name__)


class TelegramHttpNotificationSender:
    def __init__(
        self,
        bot_token: str | SecretStr,
        httpx_client: httpx.AsyncClient,
    ) -> None:
        if hasattr(bot_token, "get_secret_value"):
            bot_token = bot_token.get_secret_value()
        self._token: str = bot_token
        self._client = httpx_client
        self._base_url = f"https://api.telegram.org/bot{self._token}"

    async def send(
        self,
        chat_id: int,
        thread_id: int | None,
        text: str,
    ) -> bool:
        url = f"{self._base_url}/sendMessage"
        payload: dict[str, object] = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
        }
        if thread_id is not None:
            payload["message_thread_id"] = thread_id

        send_start = time.perf_counter()
        try:
            response = await self._client.post(url, json=payload, timeout=15)
            response.raise_for_status()
            NOTIFICATIONS_TOTAL.labels(type="normal").inc()
            NOTIFICATION_SEND_DURATION_SECONDS.labels(type="normal").observe(
                time.perf_counter() - send_start
            )
            logger.info(
                "notification_sent",
                chat_id=chat_id,
                thread_id=thread_id,
            )
            return True
        except Exception:
            NOTIFICATION_FAILURES_TOTAL.labels(type="normal").inc()
            logger.exception(
                "telegram_send_failed",
                chat_id=chat_id,
                thread_id=thread_id,
            )
            return False
