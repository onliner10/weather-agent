from __future__ import annotations

from weather_agent.application.context_service import (
    load_thread_context,
    save_thread_context,
)
from weather_agent.application.conversation_models import (
    LoadedContext,
    PendingConfirmation,
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
from weather_agent.graphs.state import ConversationState, TurnRecord

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


def _make_load_thread_context(memory_service):

    async def _load(state: ConversationState) -> dict:
        context_key = state.get("context_key", "")
        loaded = await load_thread_context(memory_service, context_key, state.get("reply_to_message_id"))
        result = {"context_key": context_key}
        if loaded.pending_confirmation is not None:
            result["pending_confirmation"] = loaded.pending_confirmation
        if loaded.resolved_location is not None:
            result["resolved_location"] = loaded.resolved_location
        if loaded.resolved_time_range is not None:
            result["resolved_time_range"] = loaded.resolved_time_range
        if loaded.user_focus is not None:
            result["user_focus"] = loaded.user_focus
        if loaded.reply_context_turns is not None:
            result["reply_context_turns"] = loaded.reply_context_turns
        return result

    return _load


def _make_save_thread_context(memory_service):

    async def _save(state: ConversationState) -> dict:
        context_key = state.get("context_key", "")
        user_message = state.get("user_message") or ""
        answer = state.get("answer") or ""
        user_message_id = state.get("message_id")
        resolved_location = state.get("resolved_location")
        resolved_time_range = state.get("resolved_time_range")
        user_focus = state.get("user_focus")
        pending_dict = state.get("pending_confirmation")
        pending = PendingConfirmation.from_dict(pending_dict) if isinstance(pending_dict, dict) else None

        await save_thread_context(
            memory_service, context_key,
            user_message, answer, user_message_id,
            resolved_location, resolved_time_range,
            user_focus, pending,
        )
        return {}

    return _save


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