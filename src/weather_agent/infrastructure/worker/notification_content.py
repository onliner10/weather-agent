from __future__ import annotations

import json
from datetime import datetime
from typing import Any, cast

from langchain_core.runnables import RunnableConfig

from weather_agent.agent_factory import build_current_time_prompt_suffix
from weather_agent.application.agent_invocation import invoke_agent_with_timeout
from weather_agent.application.conversation_service import build_conversation_messages
from weather_agent.domain.rules.models import NotificationRule
from weather_agent.domain.time import WARSAW_TZ
from weather_agent.llm.model_factory import ModelFactory
from weather_agent.observability.logging import get_logger
from weather_agent.observability.tracing import build_graph_config

logger = get_logger(__name__)


class LlmNotificationContentGenerator:
    def __init__(
        self,
        model_factory: ModelFactory,
        timeout_seconds: float,
    ) -> None:
        self._model_factory = model_factory
        self._timeout_seconds = timeout_seconds

    async def generate(
        self,
        rule: NotificationRule,
        evaluation_detail: dict[str, Any],
    ) -> str | None:
        context = rule.notification_context
        if context is None:
            return None

        prompt = _build_prompt(rule, evaluation_detail)
        turns = [{"role": turn.role, "text": turn.text} for turn in context.prior_turns]
        messages = build_conversation_messages(turns, prompt)
        graph_config = build_graph_config(
            {
                "context_key": f"scheduled-notification:{rule.id}",
                "chat_id": rule.telegram_chat_id,
                "message_thread_id": rule.telegram_message_thread_id,
                "rule_id": rule.id,
                "rule_short_id": rule.short_id,
                "location_id": rule.location_id,
            },
        )
        graph_config["metadata"]["model_provider"] = self._model_factory.provider
        graph_config["metadata"]["model_name"] = self._model_factory.model_name
        graph_config["metadata"]["run_source"] = "scheduled_notification"
        runtime_config = cast(
            RunnableConfig,
            {
                "configurable": {"thread_id": f"scheduled-notification:{rule.id}"},
                **graph_config,
            },
        )
        answer, failed = await invoke_agent_with_timeout(
            self._model_factory,
            [],
            messages,
            runtime_config,
            build_current_time_prompt_suffix(),
            self._timeout_seconds,
            logger,
        )
        if failed:
            logger.warning(
                "scheduled_notification_content_generation_failed",
                rule_id=rule.id,
                rule_short_id=rule.short_id,
            )
            return None

        message = answer.strip()
        if not _is_valid_message(message):
            logger.warning(
                "scheduled_notification_content_invalid",
                rule_id=rule.id,
                rule_short_id=rule.short_id,
            )
            return None
        return message


def _build_prompt(rule: NotificationRule, evaluation_detail: dict[str, Any]) -> str:
    context = rule.notification_context
    assert context is not None

    forecast_points = evaluation_detail.get("forecast_points", [])
    forecast_json = json.dumps(forecast_points, ensure_ascii=False, default=str)
    triggered_at = datetime.now(WARSAW_TZ).strftime("%Y-%m-%d %H:%M")

    return (
        "Przygotuj treść zaplanowanego powiadomienia pogodowego po polsku.\n"
        "Dane prognozy zostały już pobrane i zweryfikowane przez worker; "
        "nie próbuj pobierać danych ani używać narzędzi. Nie oceniaj, czy "
        "powiadomienie powinno zostać wysłane; to zostało już rozstrzygnięte "
        "deterministycznie.\n"
        "Odpowiedz wyłącznie gotową wiadomością dla użytkownika, bez komentarzy technicznych.\n"
        "Wiadomość ma odpowiadać temu, o co użytkownik poprosił przy planowaniu. "
        "Nie pokazuj JSON, surowych dat ISO ani zakresów UTC. Używaj lokalnego czasu "
        "Europe/Warsaw i zwięzłego, naturalnego stylu.\n\n"
        f"Pierwotna prośba użytkownika: {context.scheduling_message}\n"
        f"Opis zaplanowanego powiadomienia: {context.human_request}\n"
        f"Lokalizacja: {context.location_name or rule.location_id}\n"
        f"Harmonogram: {context.schedule}\n"
        f"Czas uruchomienia: {triggered_at} Europe/Warsaw\n\n"
        f"Zweryfikowane dane prognozy:\n{forecast_json}"
    )


def _is_valid_message(message: str) -> bool:
    if len(message) < 10 or len(message) > 1500:
        return False
    forbidden = ("{", "}", "+00:00", "T00:", "T01:", "T02:", "raw_payload")
    return not any(marker in message for marker in forbidden)
