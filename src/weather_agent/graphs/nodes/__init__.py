from __future__ import annotations

from weather_agent.application.intent_classifier import (
    is_confirmation_no,
    is_confirmation_yes,
)
from weather_agent.application.rules.rule_handler import (
    RuleProposalExtraction,
    cancel_rule,
    confirm_rule,
    format_rule_confirmation,
    handle_rule_confirmation,
    propose_rule,
)
from weather_agent.application.weather.weather_handler import (
    extract_location_and_focus,
    handle_weather,
    resolve_location,
)


async def propose_cel_rule_node(state, model_factory=None, cel_evaluator=None, **kwargs):
    result = await propose_rule(state.get("user_message") or "", model_factory, cel_evaluator)
    if isinstance(result.get("pending_confirmation"), dict):
        pass
    return result

async def confirm_rule_node(state, rule_service=None, location_service=None, **kwargs):
    from weather_agent.application.conversation_models import PendingConfirmation
    pending_dict = state.get("pending_confirmation")
    if pending_dict is None:
        return {"error": "Brak danych do zapisania reguły"}
    pending = PendingConfirmation.from_dict(pending_dict) if isinstance(pending_dict, dict) else pending_dict
    result = await confirm_rule(
        pending, state.get("authorized_user_id"),
        rule_service, location_service,
        state.get("resolved_location"),
        state.get("chat_id"), state.get("message_thread_id"),
    )
    return {
        "answer": result.answer,
        "pending_confirmation": result.pending_confirmation.to_dict() if result.pending_confirmation else None,
        "cel_expression": result.cel_expression,
        "error": result.error,
    }

async def cancel_rule_node(state, **kwargs):
    return {
        "answer": "Reguła została anulowana.",
        "pending_confirmation": None,
        "cel_expression": None,
        "error": None,
    }

async def require_user_confirmation_node(state, **kwargs):
    from weather_agent.application.conversation_models import PendingConfirmation
    pending_dict = state.get("pending_confirmation")
    if pending_dict is None:
        return {"answer": "Nie ma oczekującej reguły do potwierdzenia."}
    pending = PendingConfirmation.from_dict(pending_dict) if isinstance(pending_dict, dict) else pending_dict
    return {"answer": format_rule_confirmation(pending)}

async def persist_rule_change_node(state, rule_service=None, location_service=None, **kwargs):
    from weather_agent.application.conversation_models import PendingConfirmation
    pending_dict = state.get("pending_confirmation")
    if pending_dict is None:
        return {"error": "Brak danych do zapisania reguły"}
    pending = PendingConfirmation.from_dict(pending_dict) if isinstance(pending_dict, dict) else pending_dict
    user_message = (state.get("user_message") or "").strip().lower()
    result = await handle_rule_confirmation(
        user_message, pending,
        state.get("authorized_user_id"),
        rule_service, location_service,
        state.get("resolved_location"),
        state.get("chat_id"), state.get("message_thread_id"),
    )
    return {
        "answer": result.answer,
        "pending_confirmation": result.pending_confirmation,
        "cel_expression": result.cel_expression,
        "error": result.error,
    }

async def weather_agent_node(state, **kwargs):
    return await handle_weather(state, **kwargs)

async def resolve_location_node(state, location_service=None, user_id=0, geocoder=None, model_factory=None, **kwargs):
    return await resolve_location(
        state.get("user_message") or "",
        location_service, user_id,
        geocoder=geocoder, model_factory=model_factory,
        existing_location=state.get("resolved_location"),
        reply_context_turns=state.get("reply_context_turns"),
    )

__all__ = [
    "cancel_rule_node",
    "confirm_rule_node",
    "is_confirmation_no",
    "is_confirmation_yes",
    "persist_rule_change_node",
    "propose_cel_rule_node",
    "require_user_confirmation_node",
    "resolve_location_node",
    "weather_agent_node",
]