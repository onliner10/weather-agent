from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from weather_agent.graphs.state import ConversationState


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


async def resolve_location(state: ConversationState) -> dict[str, Any]:
    return {"resolved_location": state.get("resolved_location")}


async def resolve_time_range(state: ConversationState) -> dict[str, Any]:
    return {"resolved_time_range": state.get("resolved_time_range")}


async def route_to_weather_question(state: ConversationState) -> dict[str, Any]:
    return {}


async def route_to_rule_management(state: ConversationState) -> dict[str, Any]:
    return {}


async def route_to_command_or_help(state: ConversationState) -> dict[str, Any]:
    return {}


async def call_weather_tools(state: ConversationState) -> dict[str, Any]:
    return {
        "forecast_result": state.get("forecast_result"),
        "observation_result": state.get("observation_result"),
    }


async def propose_cel_rule(state: ConversationState) -> dict[str, Any]:
    return {"cel_expression": state.get("cel_expression")}


async def require_user_confirmation(state: ConversationState) -> dict[str, Any]:
    return {"pending_confirmation": state.get("pending_confirmation")}


async def persist_rule_change(state: ConversationState) -> dict[str, Any]:
    return {}


async def answer_user(state: ConversationState) -> dict[str, Any]:
    if state.get("error"):
        return {"answer": f"Przepraszam, wystąpił błąd: {state['error']}"}
    if state.get("resolved_intent") == "weather" and state.get("forecast_result"):
        return {"answer": "Prognoza pogody jest dostępna."}
    if state.get("resolved_intent") == "rule" and state.get("cel_expression"):
        return {"answer": "Reguła została zapisana."}
    if state.get("resolved_intent") in ("command", "help"):
        return {"answer": "Pomoc: wpisze /start lub /help aby uzyskać informacje."}
    return {"answer": "Przepraszam, nie mogłem przetworzyć zapytania."}


async def save_thread_context(state: ConversationState) -> dict[str, Any]:
    return {}


def _intent_router(state: ConversationState) -> str:
    intent = state.get("resolved_intent", "weather")
    if intent == "weather":
        return "weather_path"
    if intent == "rule":
        return "rule_path"
    if intent in ("command", "help"):
        return "command_path"
    return "weather_path"


def build_conversation_graph() -> StateGraph[ConversationState]:
    graph = StateGraph(ConversationState)

    graph.add_node("authorize_user", authorize_user)
    graph.add_node("load_thread_context", load_thread_context)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("resolve_location", resolve_location)
    graph.add_node("resolve_time_range", resolve_time_range)
    graph.add_node("route_to_weather_question", route_to_weather_question)
    graph.add_node("route_to_rule_management", route_to_rule_management)
    graph.add_node("route_to_command_or_help", route_to_command_or_help)
    graph.add_node("call_weather_tools", call_weather_tools)
    graph.add_node("propose_cel_rule", propose_cel_rule)
    graph.add_node("require_user_confirmation", require_user_confirmation)
    graph.add_node("persist_rule_change", persist_rule_change)
    graph.add_node("answer_user", answer_user)
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


def compile_conversation_graph() -> Any:
    return build_conversation_graph().compile()