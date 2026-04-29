from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from langsmith import trace

from weather_agent.domain.cel.evaluator import CELEvaluator
from weather_agent.domain.date_resolver import DateResolver, ResolvedTimeRange
from weather_agent.domain.locations import LocationService
from weather_agent.domain.providers import ForecastProvider, ObservationProvider
from weather_agent.domain.rules.service import NotificationRuleService
from weather_agent.domain.weather import LocationRef
from weather_agent.graphs.nodes.rule_management import (
    cancel_rule_node,
    confirm_rule_node,
    is_confirmation_no,
    is_confirmation_yes,
    propose_cel_rule_node,
    require_user_confirmation_node,
)
from weather_agent.graphs.nodes.weather_qa import (
    resolve_location_node,
    resolve_time_range_node,
    weather_agent_node,
)
from weather_agent.graphs.state import ConversationState, TurnRecord
from weather_agent.infrastructure.geocoder import Geocoder
from weather_agent.infrastructure.memory.thread_memory import ThreadMemoryService
from weather_agent.llm.model_factory import ModelFactory
from weather_agent.observability.logging import get_logger
from weather_agent.observability.tracing import (
    build_graph_config,
    build_node_metadata,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------


def _classify_with_pending_confirmation(
    user_message: str,
    pending_confirmation: dict[str, Any] | None,
) -> str | None:
    """Route to confirm/cancel when a pending confirmation exists."""
    if pending_confirmation is None:
        return None
    if is_confirmation_yes(user_message):
        return "confirm_rule"
    if is_confirmation_no(user_message):
        return "cancel_rule"
    return None


async def classify_intent(state: ConversationState) -> dict[str, Any]:
    """Classify user intent from message and conversation context."""
    async with trace(
        "classify_intent",
        run_type="chain",
        metadata=build_node_metadata(state, "classify_intent"),
    ):
        if state.get("resolved_intent") is not None:
            return {"resolved_intent": state["resolved_intent"]}

        msg = (state.get("user_message") or "").lower()
        pending = state.get("pending_confirmation")

        confirmation_route = _classify_with_pending_confirmation(msg, pending)
        if confirmation_route is not None:
            return {"resolved_intent": confirmation_route}

        if any(kw in msg for kw in ("/start", "/help", "/pomoc")):
            intent = "command"
        elif any(kw in msg for kw in ("reguł", "zasad", "powiadom", "notyfik", "cel")):
            intent = "rule"
        else:
            intent = "weather"
        return {"resolved_intent": intent}


# ---------------------------------------------------------------------------
# Command / help handler (backward-compatible standalone function)
# ---------------------------------------------------------------------------


async def route_to_command_or_help(state: ConversationState) -> dict[str, Any]:
    """Return a help message for command intents."""
    async with trace(
        "route_to_command_or_help",
        run_type="chain",
        metadata=build_node_metadata(state, "route_to_command_or_help"),
    ):
        return {"answer": "Pomoc: wpisz /start lub /help aby uzyskać informacje."}


# ---------------------------------------------------------------------------
# Thread context helpers (backward-compatible standalone functions)
# ---------------------------------------------------------------------------


def _make_load_thread_context(
    memory_service: ThreadMemoryService | None,
) -> Callable[[ConversationState], Awaitable[dict[str, Any]]]:
    async def _load_thread_context(state: ConversationState) -> dict[str, Any]:
        async with trace(
            "load_thread_context",
            run_type="chain",
            metadata=build_node_metadata(state, "load_thread_context"),
        ):
            context_key = state.get("context_key", "")
            if memory_service is None:
                return {"context_key": context_key}

            updates: dict[str, Any] = {"context_key": context_key}
            reply_to_message_id = state.get("reply_to_message_id")

            try:
                pending = await memory_service.get_pending_confirmation(context_key)
                if pending is not None:
                    updates["pending_confirmation"] = pending

                if reply_to_message_id is not None:
                    turns = await memory_service.load_turns(context_key)
                    anchor = await memory_service.find_turn_by_message_id(
                        context_key, reply_to_message_id
                    )
                    if anchor is not None:
                        loc = anchor.get("resolved_location")
                        if loc and isinstance(loc, dict):
                            updates["resolved_location"] = LocationRef(**loc)
                        tr = anchor.get("resolved_time_range")
                        if tr and isinstance(tr, dict):
                            updates["resolved_time_range"] = ResolvedTimeRange(**tr)
                        if anchor.get("user_focus"):
                            updates["user_focus"] = anchor["user_focus"]

                        ctx_turns = [anchor]
                        if turns:
                            aid = anchor.get("message_id")
                            anchor_idx = next(
                                (
                                    i
                                    for i, t in enumerate(turns)
                                    if t.get("message_id") is not None
                                    and t.get("message_id") == aid
                                ),
                                -1,
                            )
                            if anchor_idx > 0:
                                prev = turns[anchor_idx - 1]
                                if prev.get("role") == "user":
                                    ctx_turns.insert(0, prev)
                        updates["reply_context_turns"] = ctx_turns
            except Exception:
                logger.warning(
                    "thread_context_load_failed",
                    context_key=context_key,
                    exc_info=True,
                )

            return updates

    return _load_thread_context


def _make_save_thread_context(
    memory_service: ThreadMemoryService | None,
) -> Callable[[ConversationState], Awaitable[dict[str, Any]]]:
    async def _save_thread_context(state: ConversationState) -> dict[str, Any]:
        async with trace(
            "save_thread_context",
            run_type="chain",
            metadata=build_node_metadata(state, "save_thread_context"),
        ):
            context_key = state.get("context_key", "")
            if memory_service is None:
                return {}

            user_message = state.get("user_message") or ""
            answer = state.get("answer") or ""
            user_message_id = state.get("message_id")
            pending_confirmation = state.get("pending_confirmation")

            try:
                if pending_confirmation is not None:
                    await memory_service.store_pending_confirmation(
                        context_key, pending_confirmation
                    )
                else:
                    await memory_service.clear_pending_confirmation(context_key)

                resolved_tr = state.get("resolved_time_range")
                user_turn: TurnRecord = {
                    "message_id": user_message_id,
                    "role": "user",
                    "text": user_message,
                    "resolved_location": (
                        _serialize_location(state.get("resolved_location"))
                        if state.get("resolved_location")
                        else None
                    ),
                    "resolved_time_range": (
                        _json_safe(resolved_tr.model_dump()) if resolved_tr is not None else None
                    ),
                    "user_focus": state.get("user_focus"),
                    "timestamp": None,
                }
                await memory_service.save_turn(context_key, dict(user_turn))

                if answer:
                    summary = answer[:200] if len(answer) > 200 else answer
                    bot_turn: TurnRecord = {
                        "message_id": None,
                        "role": "bot",
                        "text": None,
                        "answer_summary": summary,
                        "resolved_location": (
                            _serialize_location(state.get("resolved_location"))
                            if state.get("resolved_location")
                            else None
                        ),
                        "resolved_time_range": (
                            _json_safe(resolved_tr.model_dump())
                            if resolved_tr is not None
                            else None
                        ),
                        "user_focus": state.get("user_focus"),
                        "timestamp": None,
                    }
                    await memory_service.save_turn(context_key, dict(bot_turn))
            except Exception:
                logger.warning(
                    "thread_context_save_failed",
                    context_key=context_key,
                    exc_info=True,
                )

            return {}

    return _save_thread_context


def _serialize_location(location: LocationRef | None) -> dict[str, Any] | None:
    if location is None:
        return None
    return {
        "id": location.id,
        "name": location.name,
        "latitude": location.latitude,
        "longitude": location.longitude,
    }


def _json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    return json.loads(json.dumps(obj, default=str))


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


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
            for v in (
                self.model_factory,
                self.cel_evaluator,
                self.rule_service,
            )
        )


# ---------------------------------------------------------------------------
# Legacy router (kept for backward compatibility with direct imports)
# ---------------------------------------------------------------------------


def _intent_router(state: ConversationState) -> str:
    intent = state.get("resolved_intent", "weather")
    if intent == "weather":
        return "weather_path"
    if intent == "rule":
        return "rule_path"
    if intent in ("command", "help"):
        return "command_path"
    if intent == "confirm_rule":
        return "confirm_path"
    if intent == "cancel_rule":
        return "cancel_path"
    return "weather_path"


def _merge_state(
    state: ConversationState,
    updates: dict[str, Any],
) -> ConversationState:
    """Merge updates into a conversation state dict.

    Mypy is strict about ``**`` expansion with TypedDicts, so we use
    an explicit helper with a single ignore instead of sprinkling
    ignores throughout the orchestrator.
    """
    merged = dict(state)
    merged.update(updates)
    return merged  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Conversation orchestrator — explicit dispatcher replacing LangGraph
# ---------------------------------------------------------------------------


class ConversationOrchestrator:
    """Handles a single user turn with explicit sequential dispatch.

    Replaces the previous LangGraph-based conversation flow with a simpler,
    linear state machine that matches real Telegram turn boundaries:
    load context → classify intent → run handler → save context.
    """

    def __init__(self, deps: ConversationDeps | None = None) -> None:
        self.deps = deps or ConversationDeps()

    # -- public API ---------------------------------------------------------

    def compile(self) -> CompiledConversationGraph:
        """Return a compiled wrapper compatible with the old graph API."""
        return CompiledConversationGraph(self)

    async def handle_turn(self, state: ConversationState) -> ConversationState:
        """Process one user message end-to-end."""
        graph_config = build_graph_config(state)
        with trace(
            graph_config.get("run_name", "conversation_orchestrator"),
            run_type="chain",
            tags=graph_config.get("tags"),
            metadata=graph_config.get("metadata"),
        ):
            # 1. Load persisted context
            state = await self._load_context(state)

            # 2. Classify intent
            intent_updates = await classify_intent(state)
            state = _merge_state(state, intent_updates)
            intent: str = state.get("resolved_intent") or "weather"

            # 3. Dispatch to handler
            handler_result = await self._dispatch(state, intent)
            state = _merge_state(state, handler_result)

            # 4. Save context
            await self._save_context(state)

            return state

    # -- internal helpers ---------------------------------------------------

    async def _load_context(self, state: ConversationState) -> ConversationState:
        loader = _make_load_thread_context(self.deps.memory_service)
        updates = await loader(state)
        return _merge_state(state, updates)

    async def _save_context(self, state: ConversationState) -> None:
        saver = _make_save_thread_context(self.deps.memory_service)
        await saver(state)

    async def _dispatch(
        self,
        state: ConversationState,
        intent: str,
    ) -> dict[str, Any]:
        if intent == "confirm_rule":
            return await self._handle_confirm(state)
        if intent == "cancel_rule":
            return await self._handle_cancel(state)
        if intent in ("command", "help"):
            return await self._handle_command(state)
        if intent == "rule":
            return await self._handle_rule(state)
        return await self._handle_weather(state)

    # -- intent handlers ----------------------------------------------------

    async def _handle_weather(self, state: ConversationState) -> dict[str, Any]:
        async with trace(
            "handle_weather",
            run_type="chain",
            metadata=build_node_metadata(state, "handle_weather"),
        ):
            result: dict[str, Any] = {}

            # Resolve location
            loc_result = await resolve_location_node(
                state,
                self.deps.location_service,
                self.deps.user_id,
                geocoder=self.deps.geocoder,
                model_factory=self.deps.model_factory,
            )
            result.update(loc_result)
            state = _merge_state(state, loc_result)

            # Resolve time range
            time_result = await resolve_time_range_node(state, self.deps.date_resolver)
            result.update(time_result)
            state = _merge_state(state, time_result)

            # Generate weather answer
            agent_result = await weather_agent_node(
                state,
                model_factory=self.deps.model_factory,
                forecast_provider=self.deps.forecast_provider,
                observation_provider=self.deps.observation_provider,
                geocoder=self.deps.geocoder,
                date_resolver=self.deps.date_resolver,
                location_service=self.deps.location_service,
                user_id=self.deps.user_id,
            )
            result.update(agent_result)
            return result

    async def _handle_rule(self, state: ConversationState) -> dict[str, Any]:
        async with trace(
            "handle_rule",
            run_type="chain",
            metadata=build_node_metadata(state, "handle_rule"),
        ):
            result: dict[str, Any] = {}

            # Rules need location context (for the rule's target location)
            loc_result = await resolve_location_node(
                state,
                self.deps.location_service,
                self.deps.user_id,
                geocoder=self.deps.geocoder,
                model_factory=self.deps.model_factory,
            )
            result.update(loc_result)
            state = _merge_state(state, loc_result)

            # Propose CEL expression
            propose_result = await propose_cel_rule_node(
                state,
                self.deps.model_factory,
                self.deps.cel_evaluator,
            )
            result.update(propose_result)
            state = _merge_state(state, propose_result)

            if state.get("error"):
                result["answer"] = state["error"]
                return result

            # Ask user for confirmation
            confirm_result = await require_user_confirmation_node(state)
            result.update(confirm_result)
            return result

    async def _handle_command(self, state: ConversationState) -> dict[str, Any]:
        async with trace(
            "handle_command",
            run_type="chain",
            metadata=build_node_metadata(state, "handle_command"),
        ):
            return await route_to_command_or_help(state)

    async def _handle_confirm(self, state: ConversationState) -> dict[str, Any]:
        async with trace(
            "handle_confirm",
            run_type="chain",
            metadata=build_node_metadata(state, "handle_confirm"),
        ):
            return await confirm_rule_node(
                state,
                self.deps.rule_service,
                self.deps.location_service,
            )

    async def _handle_cancel(self, state: ConversationState) -> dict[str, Any]:
        async with trace(
            "handle_cancel",
            run_type="chain",
            metadata=build_node_metadata(state, "handle_cancel"),
        ):
            return await cancel_rule_node(state)


# ---------------------------------------------------------------------------
# Backward-compatible compiled wrapper
# ---------------------------------------------------------------------------


class CompiledConversationGraph:
    """Thin adapter so callers can still use ``await graph.ainvoke(state)``."""

    def __init__(self, orchestrator: ConversationOrchestrator) -> None:
        self._orchestrator = orchestrator

    async def ainvoke(
        self,
        state: ConversationState,
        config: dict[str, Any] | None = None,
    ) -> ConversationState:
        """Process a single turn.  *config* is accepted for compatibility but ignored."""
        return await self._orchestrator.handle_turn(state)


# ---------------------------------------------------------------------------
# Factory functions (backward-compatible signatures)
# ---------------------------------------------------------------------------


def build_conversation_graph(
    deps: ConversationDeps | None = None,
) -> ConversationOrchestrator:
    """Build a conversation orchestrator.

    Previously returned a ``StateGraph``; now returns an explicit
    ``ConversationOrchestrator`` that exposes the same ``.compile()`` method.
    """
    return ConversationOrchestrator(deps)


def compile_conversation_graph(
    deps: ConversationDeps | None = None,
) -> CompiledConversationGraph:
    """Compile the conversation orchestrator into a runnable graph."""
    return ConversationOrchestrator(deps).compile()
