from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from weather_agent.domain.cel.evaluator import CELEvaluator
from weather_agent.domain.date_resolver import DateResolver
from weather_agent.domain.locations import LocationService
from weather_agent.domain.providers import ForecastProvider, ObservationProvider
from weather_agent.domain.rules.service import NotificationRuleService
from weather_agent.graphs.nodes.rule_management import (
    persist_rule_change_node,
    propose_cel_rule_node,
    require_user_confirmation_node,
)
from weather_agent.graphs.nodes.weather_qa import (
    resolve_location_node,
    resolve_time_range_node,
    weather_agent_node,
)
from weather_agent.graphs.state import ConversationState
from weather_agent.infrastructure.geocoder import Geocoder
from weather_agent.llm.model_factory import ModelFactory


async def authorize_user(state: ConversationState) -> dict[str, Any]:
    return {"authorized_user_id": state.get("authorized_user_id")}


async def load_thread_context(state: ConversationState) -> dict[str, Any]:
    return {"context_key": state.get("context_key", "")}


async def classify_intent(state: ConversationState) -> dict[str, Any]:
    if state.get("resolved_intent") is not None:
        return {"resolved_intent": state["resolved_intent"]}
    msg = (state.get("user_message") or "").lower()
    if any(kw in msg for kw in ("/start", "/help", "/pomoc")):
        intent = "command"
    elif any(kw in msg for kw in ("reguł", "zasad", "powiadom", "notyfik", "cel")):
        intent = "rule"
    else:
        intent = "weather"
    return {"resolved_intent": intent}


async def route_to_command_or_help(state: ConversationState) -> dict[str, Any]:
    return {"answer": "Pomoc: wpisz /start lub /help aby uzyskać informacje."}


async def save_thread_context(state: ConversationState) -> dict[str, Any]:
    return {}


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


def _intent_router(state: ConversationState) -> str:
    intent = state.get("resolved_intent", "weather")
    if intent == "weather":
        return "weather_path"
    if intent == "rule":
        return "rule_path"
    if intent in ("command", "help"):
        return "command_path"
    return "weather_path"


def build_conversation_graph(
    deps: ConversationDeps | None = None,
) -> StateGraph[ConversationState]:
    graph = StateGraph(ConversationState)

    graph.add_node("authorize_user", authorize_user)
    graph.add_node("load_thread_context", load_thread_context)
    graph.add_node("classify_intent", classify_intent)

    ls = deps.location_service if deps else None
    dr = deps.date_resolver if deps else None
    fp = deps.forecast_provider if deps else None
    op = deps.observation_provider if deps else None
    mf = deps.model_factory if deps else None
    gc = deps.geocoder if deps else None
    uid = deps.user_id if deps else 0

    async def _resolve_location(state: ConversationState) -> dict[str, Any]:
        return await resolve_location_node(state, ls, uid, geocoder=gc, model_factory=mf)

    async def _resolve_time_range(state: ConversationState) -> dict[str, Any]:
        return await resolve_time_range_node(state, dr)

    async def _weather_agent(state: ConversationState) -> dict[str, Any]:
        return await weather_agent_node(
            state,
            model_factory=mf,
            forecast_provider=fp,
            observation_provider=op,
            geocoder=gc,
            date_resolver=dr,
            location_service=ls,
            user_id=uid,
        )

    graph.add_node("resolve_location", _resolve_location)
    graph.add_node("resolve_time_range", _resolve_time_range)
    graph.add_node("weather_agent", _weather_agent)

    cel = deps.cel_evaluator if deps else None
    rs = deps.rule_service if deps else None
    rls = deps.location_service if deps else None

    async def _propose_cel_rule(state: ConversationState) -> dict[str, Any]:
        return await propose_cel_rule_node(state, mf, cel)

    async def _require_user_confirmation(state: ConversationState) -> dict[str, Any]:
        return await require_user_confirmation_node(state)

    async def _persist_rule_change(state: ConversationState) -> dict[str, Any]:
        return await persist_rule_change_node(state, rs, rls)

    graph.add_node("propose_cel_rule", _propose_cel_rule)
    graph.add_node("require_user_confirmation", _require_user_confirmation)
    graph.add_node("persist_rule_change", _persist_rule_change)

    graph.add_node("route_to_command_or_help", route_to_command_or_help)
    graph.add_node("save_thread_context", save_thread_context)

    graph.set_entry_point("authorize_user")

    graph.add_edge("authorize_user", "load_thread_context")
    graph.add_edge("load_thread_context", "classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        _intent_router,
        {
            "weather_path": "resolve_location",
            "rule_path": "resolve_location",
            "command_path": "route_to_command_or_help",
        },
    )

    graph.add_edge("resolve_location", "resolve_time_range")

    graph.add_conditional_edges(
        "resolve_time_range",
        _intent_router,
        {
            "weather_path": "weather_agent",
            "rule_path": "propose_cel_rule",
        },
    )

    graph.add_edge("weather_agent", "save_thread_context")
    graph.add_edge("route_to_command_or_help", "save_thread_context")
    graph.add_edge("propose_cel_rule", "require_user_confirmation")
    graph.add_edge("require_user_confirmation", "persist_rule_change")
    graph.add_edge("persist_rule_change", "save_thread_context")
    graph.add_edge("save_thread_context", END)

    return graph


def compile_conversation_graph(
    deps: ConversationDeps | None = None,
) -> Any:
    return build_conversation_graph(deps).compile()