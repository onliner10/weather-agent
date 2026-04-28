from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from weather_agent.domain.cel.evaluator import CELEvaluator
from weather_agent.domain.date_resolver import DateResolver
from weather_agent.domain.locations import LocationService
from weather_agent.domain.rules.service import NotificationRuleService
from weather_agent.graphs.nodes.rule_management import (
    persist_rule_change_node,
    propose_cel_rule_node,
    require_user_confirmation_node,
)
from weather_agent.graphs.nodes.weather_qa import (
    ForecastProvider,
    ObservationProvider,
    answer_weather_question_node,
    call_weather_tools_node,
    resolve_location_node,
    resolve_time_range_node,
)
from weather_agent.graphs.state import ConversationState
from weather_agent.llm.model_factory import ModelFactory
from weather_agent.observability.langsmith_tracing import configure_tracing
from weather_agent.settings import LangSmithSettings


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
    elif any(
        kw in msg
        for kw in ("pogod", "temperatur", "wiatr", "deszcz", "opad", "śnieg", "chmury")
    ):
        intent = "weather"
    else:
        intent = "weather"
    return {"resolved_intent": intent}


async def resolve_location_stub(state: ConversationState) -> dict[str, Any]:
    return {"resolved_location": state.get("resolved_location")}


async def resolve_time_range_stub(state: ConversationState) -> dict[str, Any]:
    return {"resolved_time_range": state.get("resolved_time_range")}


async def route_to_weather_question(state: ConversationState) -> dict[str, Any]:
    return {}


async def route_to_rule_management(state: ConversationState) -> dict[str, Any]:
    return {}


async def route_to_command_or_help(state: ConversationState) -> dict[str, Any]:
    return {}


async def call_weather_tools_stub(state: ConversationState) -> dict[str, Any]:
    return {
        "forecast_result": state.get("forecast_result"),
        "observation_result": state.get("observation_result"),
    }


async def propose_cel_rule_stub(state: ConversationState) -> dict[str, Any]:
    return {"cel_expression": state.get("cel_expression")}


async def require_user_confirmation_stub(state: ConversationState) -> dict[str, Any]:
    return {"pending_confirmation": state.get("pending_confirmation")}


async def persist_rule_change_stub(state: ConversationState) -> dict[str, Any]:
    return {}


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
        user_id: int = 0,
    ) -> None:
        self.location_service = location_service
        self.date_resolver = date_resolver
        self.forecast_provider = forecast_provider
        self.observation_provider = observation_provider
        self.model_factory = model_factory
        self.cel_evaluator = cel_evaluator
        self.rule_service = rule_service
        self.user_id = user_id

    @property
    def has_weather_deps(self) -> bool:
        return all(
            v is not None
            for v in (
                self.location_service,
                self.date_resolver,
                self.forecast_provider,
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
                self.location_service,
            )
        )

    def __or__(self, other: ConversationDeps) -> ConversationDeps:
        return ConversationDeps(
            location_service=other.location_service or self.location_service,
            date_resolver=other.date_resolver or self.date_resolver,
            forecast_provider=other.forecast_provider or self.forecast_provider,
            observation_provider=other.observation_provider or self.observation_provider,
            model_factory=other.model_factory or self.model_factory,
            cel_evaluator=other.cel_evaluator or self.cel_evaluator,
            rule_service=other.rule_service or self.rule_service,
            user_id=other.user_id or self.user_id,
        )


class WeatherQADependencies(ConversationDeps):
    def __init__(
        self,
        location_service: LocationService | None = None,
        date_resolver: DateResolver | None = None,
        forecast_provider: ForecastProvider | None = None,
        observation_provider: ObservationProvider | None = None,
        model_factory: ModelFactory | None = None,
        user_id: int = 0,
    ) -> None:
        super().__init__(
            location_service=location_service,
            date_resolver=date_resolver,
            forecast_provider=forecast_provider,
            observation_provider=observation_provider,
            model_factory=model_factory,
            user_id=user_id,
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


def _stub_answer_user(state: ConversationState) -> dict[str, Any]:
    if state.get("error"):
        return {"answer": f"Przepraszam, wystąpił błąd: {state['error']}"}
    if state.get("resolved_intent") == "weather" and state.get("forecast_result"):
        return {"answer": "Prognoza pogody jest dostępna."}
    if state.get("resolved_intent") == "rule" and state.get("cel_expression"):
        return {"answer": "Reguła została zapisana."}
    if state.get("resolved_intent") in ("command", "help"):
        return {"answer": "Pomoc: wpisze /start lub /help aby uzyskać informacje."}
    return {"answer": "Przepraszam, nie mogłem przetworzyć zapytania."}


def build_conversation_graph(
    deps: ConversationDeps | None = None,
) -> StateGraph[ConversationState]:
    graph = StateGraph(ConversationState)

    graph.add_node("authorize_user", authorize_user)
    graph.add_node("load_thread_context", load_thread_context)
    graph.add_node("classify_intent", classify_intent)

    if deps is not None and deps.has_weather_deps:
        ls = deps.location_service
        assert ls is not None
        dr = deps.date_resolver
        assert dr is not None
        fp = deps.forecast_provider
        assert fp is not None
        uid = deps.user_id
        op = deps.observation_provider
        mf = deps.model_factory

        async def _resolve_location(state: ConversationState) -> dict[str, Any]:
            return await resolve_location_node(state, ls, uid)

        async def _resolve_time_range(state: ConversationState) -> dict[str, Any]:
            return await resolve_time_range_node(state, dr)

        async def _call_weather_tools(state: ConversationState) -> dict[str, Any]:
            return await call_weather_tools_node(state, fp, op)

        async def _answer_user(state: ConversationState) -> dict[str, Any]:
            return await answer_weather_question_node(state, mf)

        graph.add_node("resolve_location", _resolve_location)
        graph.add_node("resolve_time_range", _resolve_time_range)
        graph.add_node("call_weather_tools", _call_weather_tools)
        graph.add_node("answer_user", _answer_user)
    else:
        graph.add_node("resolve_location", resolve_location_stub)
        graph.add_node("resolve_time_range", resolve_time_range_stub)
        graph.add_node("call_weather_tools", call_weather_tools_stub)
        graph.add_node("answer_user", _stub_answer_user)

    if deps is not None and deps.has_rule_deps:
        mf = deps.model_factory
        assert mf is not None
        cel = deps.cel_evaluator
        assert cel is not None
        rs = deps.rule_service
        assert rs is not None
        rls = deps.location_service
        assert rls is not None

        async def _propose_cel_rule(state: ConversationState) -> dict[str, Any]:
            return await propose_cel_rule_node(state, mf, cel)

        async def _require_user_confirmation(state: ConversationState) -> dict[str, Any]:
            return await require_user_confirmation_node(state)

        async def _persist_rule_change(state: ConversationState) -> dict[str, Any]:
            return await persist_rule_change_node(state, rs, rls)

        graph.add_node("propose_cel_rule", _propose_cel_rule)
        graph.add_node("require_user_confirmation", _require_user_confirmation)
        graph.add_node("persist_rule_change", _persist_rule_change)
    else:
        graph.add_node("propose_cel_rule", propose_cel_rule_stub)
        graph.add_node("require_user_confirmation", require_user_confirmation_stub)
        graph.add_node("persist_rule_change", persist_rule_change_stub)

    graph.add_node("route_to_weather_question", route_to_weather_question)
    graph.add_node("route_to_rule_management", route_to_rule_management)
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
            "weather_path": "call_weather_tools",
            "rule_path": "propose_cel_rule",
        },
    )

    graph.add_edge("call_weather_tools", "answer_user")
    graph.add_edge("route_to_command_or_help", "answer_user")
    graph.add_edge("propose_cel_rule", "require_user_confirmation")
    graph.add_edge("require_user_confirmation", "persist_rule_change")
    graph.add_edge("persist_rule_change", "answer_user")
    graph.add_edge("answer_user", "save_thread_context")
    graph.add_edge("save_thread_context", END)

    return graph


def compile_conversation_graph(
    deps: ConversationDeps | None = None,
) -> Any:
    return build_conversation_graph(deps).compile()


def init_observability(langsmith_settings: LangSmithSettings) -> None:
    configure_tracing(langsmith_settings)