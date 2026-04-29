from __future__ import annotations

from typing import Any

from langsmith import trace

from weather_agent.application.context_service import (
    load_thread_context,
    save_thread_context,
)
from weather_agent.application.intent_classifier import classify_intent
from weather_agent.application.rules.rule_handler import (
    cancel_rule,
    format_rule_confirmation,
    handle_rule_confirmation,
    propose_rule,
)
from weather_agent.application.weather.weather_handler import (
    handle_weather,
    resolve_location,
)
from weather_agent.domain.cel.evaluator import CELEvaluator
from weather_agent.domain.date_resolver import DateResolver
from weather_agent.domain.locations import LocationService
from weather_agent.domain.providers import ForecastProvider, ObservationProvider
from weather_agent.domain.rules.service import NotificationRuleService
from weather_agent.infrastructure.geocoder import Geocoder
from weather_agent.infrastructure.memory.thread_memory import ThreadMemoryService
from weather_agent.llm.model_factory import ModelFactory
from weather_agent.observability.logging import get_logger
from weather_agent.observability.tracing import build_graph_config

logger = get_logger(__name__)


class ConversationDeps:
    def __init__(
        self,
        location_service: LocationService | None = None,
        date_resolver: DateResolver | None = None,
        forecast_provider: ForecastProvider | None = None,
        observation_provider: ObservationProvider | None = None,
        model_factory: ModelFactory | None = None,
        cel_evaluator: CELEvaluator | None = None,
        rule_service: NotificationRuleService | None = None,
        geocoder: Geocoder | None = None,
        user_id: int = 0,
        memory_service: ThreadMemoryService | None = None,
    ) -> None:
        self.location_service = location_service
        self.date_resolver = date_resolver
        self.forecast_provider = forecast_provider
        self.observation_provider = observation_provider
        self.model_factory = model_factory
        self.cel_evaluator = cel_evaluator
        self.rule_service = rule_service
        self.geocoder = geocoder
        self.user_id = user_id
        self.memory_service = memory_service

    @property
    def has_weather_deps(self) -> bool:
        return all(
            v is not None
            for v in (
                self.date_resolver,
                self.forecast_provider,
                self.model_factory,
                self.geocoder,
            )
        )

    @property
    def has_rule_deps(self) -> bool:
        return all(
            v is not None
            for v in (self.model_factory, self.cel_evaluator, self.rule_service)
        )


class ConversationService:
    def __init__(self, deps: ConversationDeps | None = None) -> None:
        self.deps = deps or ConversationDeps()

    async def handle_turn(self, state: dict[str, Any]) -> dict[str, Any]:
        config = build_graph_config(state)
        with trace(
            config.get("run_name", "conversation_service"),
            run_type="chain",
            tags=config.get("tags"),
            metadata=config.get("metadata"),
        ):
            context_key = state.get("context_key", "")
            user_message = state.get("user_message") or ""
            pending_confirmation_dict = state.get("pending_confirmation")

            loaded = await load_thread_context(
                self.deps.memory_service,
                context_key,
                state.get("reply_to_message_id"),
            )

            if loaded.pending_confirmation is not None:
                pending_confirmation_dict = loaded.pending_confirmation
            resolved_location = state.get("resolved_location") or loaded.resolved_location
            resolved_time_range = state.get("resolved_time_range") or loaded.resolved_time_range
            user_focus = state.get("user_focus") or loaded.user_focus
            reply_context_turns = state.get("reply_context_turns") or loaded.reply_context_turns

            intent_result = await classify_intent(
                state,
                self.deps.model_factory,
            )
            if state.get("resolved_intent") is not None:
                intent = state["resolved_intent"]
            else:
                intent = intent_result.get("resolved_intent") or "weather"

            result: dict[str, Any] = {"resolved_intent": intent}

            if intent == "confirm_rule" and pending_confirmation_dict:
                from weather_agent.application.conversation_models import PendingConfirmation
                pending = PendingConfirmation.from_dict(pending_confirmation_dict)
                rule_result = await handle_rule_confirmation(
                    user_message=user_message,
                    pending=pending,
                    user_id=state.get("authorized_user_id"),
                    rule_service=self.deps.rule_service,
                    location_service=self.deps.location_service,
                    resolved_location=resolved_location,
                    chat_id=state.get("chat_id"),
                    message_thread_id=state.get("message_thread_id"),
                )
                result["answer"] = rule_result.answer
                pc = rule_result.pending_confirmation
                if pc is not None and pc.cel_expression == "" and pc.action == "create_rule":
                    result["pending_confirmation"] = None
                else:
                    result["pending_confirmation"] = rule_result.pending_confirmation
                result["cel_expression"] = rule_result.cel_expression
                result["error"] = rule_result.error

            elif intent == "cancel_rule" and pending_confirmation_dict:
                from weather_agent.application.conversation_models import PendingConfirmation
                pending = PendingConfirmation.from_dict(pending_confirmation_dict)
                rule_result = await cancel_rule(pending)
                result["answer"] = rule_result.answer
                result["pending_confirmation"] = None
                result["cel_expression"] = None

            elif intent in ("command", "help"):
                result["answer"] = "Pomoc: wpisz /start lub /help aby uzyskać informacje."

            elif intent == "rule":
                loc_result = await resolve_location(
                    user_message,
                    self.deps.location_service,
                    self.deps.user_id,
                    geocoder=self.deps.geocoder,
                    model_factory=self.deps.model_factory,
                    existing_location=resolved_location,
                    reply_context_turns=(
                        reply_context_turns if isinstance(reply_context_turns, list) else None
                    ),
                )
                result.update(loc_result)
                if resolved_location is None and result.get("resolved_location") is not None:
                    resolved_location = result["resolved_location"]

                propose_result = await propose_rule(
                    user_message, self.deps.model_factory, self.deps.cel_evaluator
                )
                result.update(propose_result)
                if result.get("cel_expression") and result.get("pending_confirmation") is not None:
                    from weather_agent.application.conversation_models import PendingConfirmation
                    raw = result["pending_confirmation"]
                    pending = PendingConfirmation.from_dict(raw) if isinstance(raw, dict) else raw
                    result["answer"] = format_rule_confirmation(pending)
                elif result.get("error") and not result.get("answer"):
                    result["answer"] = result["error"]

            else:
                weather_state = {
                    **state,
                    "resolved_location": resolved_location,
                    "user_focus": user_focus,
                    "reply_context_turns": (
                        reply_context_turns if isinstance(reply_context_turns, list) else None
                    ),
                }
                weather_result = await handle_weather(
                    weather_state,
                    model_factory=self.deps.model_factory,
                    forecast_provider=self.deps.forecast_provider,
                    observation_provider=self.deps.observation_provider,
                    geocoder=self.deps.geocoder,
                    date_resolver=self.deps.date_resolver,
                    location_service=self.deps.location_service,
                    user_id=self.deps.user_id,
                )
                result.update(weather_result)

            answer = result.get("answer")
            pending_conf = result.get("pending_confirmation")
            pending_dict = (
                pending_conf.to_dict() if hasattr(pending_conf, "to_dict") else pending_conf
            )

            await save_thread_context(
                self.deps.memory_service,
                context_key,
                user_message,
                answer or "",
                state.get("message_id"),
                resolved_location,
                resolved_time_range,
                user_focus,
                (
                    PendingConfirmation.from_dict(pending_dict)
                    if isinstance(pending_dict, dict) and pending_dict
                    else None
                ),
            )

            for key in ("authorized_user_id", "chat_id", "message_thread_id", "context_key"):
                if key in state and key not in result:
                    result[key] = state[key]

            return result

    def compile(self) -> CompiledConversationService:
        return CompiledConversationService(self)


class CompiledConversationService:
    def __init__(self, service: ConversationService) -> None:
        self._service = service

    async def ainvoke(
        self, state: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return await self._service.handle_turn(state)


def build_conversation_service(deps: ConversationDeps | None = None) -> ConversationService:
    return ConversationService(deps)


def compile_conversation_service(
    deps: ConversationDeps | None = None,
) -> CompiledConversationService:
    return ConversationService(deps).compile()