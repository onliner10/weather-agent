from __future__ import annotations

import warnings

warnings.warn(
    "graphs/conversation.py is deprecated. Import from "
    "weather_agent.application.conversation_service directly.",
    DeprecationWarning,
    stacklevel=2,
)


from weather_agent.application.context_service import (
    load_thread_context,
    save_thread_context,
)
from weather_agent.application.conversation_models import (
    LoadedContext,
    PendingConfirmation,
    TurnRecord,
    TurnRequest,
    TurnResult,
)
from weather_agent.application.conversation_service import (
    CompiledConversationService,
    ConversationDeps,
    ConversationService,
    build_conversation_service,
    compile_conversation_service,
)
from weather_agent.application.intent_classifier import (
    IntentExtraction,
    classify_intent,
    classify_with_pending_confirmation,
    is_confirmation_no,
    is_confirmation_yes,
)
from weather_agent.graphs.state import ConversationState

_classify_with_pending_confirmation = classify_with_pending_confirmation


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


async def route_to_command_or_help(state: ConversationState) -> dict:
    return {"answer": "Pomoc: wpisz /start lub /help aby uzyskać informacje."}


def build_conversation_graph(deps: ConversationDeps | None = None) -> ConversationService:
    return build_conversation_service(deps)


CompiledConversationGraph = CompiledConversationService
ConversationOrchestrator = ConversationService


def compile_conversation_graph(deps: ConversationDeps | None = None) -> CompiledConversationService:
    return compile_conversation_service(deps)


__all__ = [
    "ConversationDeps",
    "ConversationState",
    "CompiledConversationService",
    "ConversationService",
    "IntentExtraction",
    "LoadedContext",
    "PendingConfirmation",
    "TurnRecord",
    "TurnRequest",
    "TurnResult",
    "build_conversation_graph",
    "build_conversation_service",
    "classify_intent",
    "classify_with_pending_confirmation",
    "compile_conversation_graph",
    "compile_conversation_service",
    "is_confirmation_no",
    "is_confirmation_yes",
    "load_thread_context",
    "route_to_command_or_help",
    "save_thread_context",
]