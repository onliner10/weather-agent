from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import trace
from pydantic import BaseModel, Field

from weather_agent.application.intent_classifier import is_confirmation_no, is_confirmation_yes
from weather_agent.domain.cel.allowlist import get_allowlist_for_prompt
from weather_agent.domain.cel.evaluator import CELEvaluationResult, CELEvaluator
from weather_agent.domain.locations import LocationService
from weather_agent.domain.rules.models import RuleCreate, RuleUpdate
from weather_agent.domain.rules.service import NotificationRuleService
from weather_agent.domain.rules.short_id_generator import strip_hash_prefix
from weather_agent.graphs.state import ConversationState
from weather_agent.llm.contracts.rules import RuleProposalExtraction
from weather_agent.llm.prompts.rule_prompts import RULE_PROPOSAL_PROMPT
from weather_agent.observability.logging import get_logger

logger = get_logger(__name__)

_GENERIC_RULE_ERROR = (
    "Przepraszam, wystąpił błąd podczas przygotowywania reguły. Spróbuj ponownie za chwilę."
)

async def propose_cel_rule_node(
    state: ConversationState,
    model_factory: Any,
    cel_evaluator: CELEvaluator | None,
) -> dict[str, Any]:
    if model_factory is None or cel_evaluator is None:
        return {"error": "Reguły powiadomień są niedostępne bez pełnej konfiguracji."}

    user_message = state.get("user_message") or ""
    if not user_message:
        return {"error": "Brak wiadomości użytkownika do przetworzenia"}

    chat_model = model_factory.create_chat_model()
    structured = chat_model.with_structured_output(RuleProposalExtraction)

    allowlist = get_allowlist_for_prompt()
    functions_json = json.dumps(allowlist["functions"], ensure_ascii=False, indent=2)
    metrics_json = json.dumps(allowlist["metrics"], ensure_ascii=False, indent=2)

    try:
        async with trace("propose_cel_rule_llm", run_type="llm"):
            chain = RULE_PROPOSAL_PROMPT | structured
            parsed = await chain.ainvoke({
                "cel_functions": functions_json,
                "cel_metrics": metrics_json,
                "user_message": user_message
            })
    except Exception:
        logger.exception("llm_rule_proposal_failed")
        return {"error": _GENERIC_RULE_ERROR}

    cel_expression = parsed.cel_expression
    explanation = parsed.explanation
    short_id = parsed.short_id

    if cel_expression is None:
        logger.warning(
            "llm_rule_no_cel_expression",
            explanation=explanation,
        )
        return {
            "error": _GENERIC_RULE_ERROR,
            "cel_expression": None,
            "pending_confirmation": None,
        }

    validation: CELEvaluationResult = cel_evaluator.validate(cel_expression)

    if not validation.valid:
        logger.error(
            "cel_validation_failed",
            cel_expression=cel_expression,
            validation_error=validation.error,
        )
        return {
            "error": _GENERIC_RULE_ERROR,
            "cel_expression": None,
            "pending_confirmation": None,
        }

    if short_id:
        short_id = strip_hash_prefix(short_id.upper().replace("#", ""))

    resolved_location = state.get("resolved_location")
    location_id: int | None = None
    if resolved_location is not None:
        try:
            location_id = int(resolved_location.id)
        except (ValueError, TypeError):
            location_id = None

    pending: dict[str, Any] = {
        "action": "edit_rule" if short_id else "create_rule",
        "cel_expression": validation.expression,
        "explanation": explanation,
        "validated": True,
        "location_id": location_id,
        "chat_id": state.get("chat_id"),
        "message_thread_id": state.get("message_thread_id"),
        "stored_at": datetime.now(UTC).isoformat(),
    }
    if short_id:
        pending["edit_short_id"] = short_id

    return {
        "cel_expression": validation.expression,
        "cel_validation_result": validation,
        "pending_confirmation": pending,
        "error": None,
    }


async def require_user_confirmation_node(state: ConversationState) -> dict[str, Any]:
    pending = state.get("pending_confirmation")
    if pending is None:
        return {"answer": "Nie ma oczekującej reguły do potwierdzenia."}

    cel_expression = pending.get("cel_expression", "")
    explanation = pending.get("explanation", "")
    action = pending.get("action", "create_rule")

    if action == "edit_rule":
        short_id = pending.get("edit_short_id", "")
        header = f"Propozycja edycji reguły #{short_id}:"
    else:
        header = "Propozycja nowej reguły:"

    lines = [
        header,
        "",
        f"\U0001f4dd Wyrażenie CEL: `{cel_expression}`",
        f"\U0001f4d6 Opis: {explanation}",
        "",
        "Czy chcesz potwierdzić? (tak/nie)",
    ]
    answer = "\n".join(lines)

    return {"answer": answer}


async def confirm_rule_node(
    state: ConversationState,
    rule_service: NotificationRuleService | None,
    location_service: LocationService | None,
) -> dict[str, Any]:
    if rule_service is None or location_service is None:
        return {"error": "Reguły powiadomień są niedostępne bez pełnej konfiguracji."}

    pending = state.get("pending_confirmation")
    if pending is None:
        return {"error": "Brak danych do zapisania reguły"}

    cel_expression = pending.get("cel_expression", "")
    explanation = pending.get("explanation", "")
    action = pending.get("action", "create_rule")

    user_id = state.get("authorized_user_id")
    if user_id is None:
        return {
            "error": "Użytkownik nie jest autoryzowany",
            "pending_confirmation": None,
        }

    chat_id = pending.get("chat_id") or state.get("chat_id", 0)
    message_thread_id = pending.get("message_thread_id") or state.get("message_thread_id")

    location_id: int | None = pending.get("location_id")
    if location_id is None:
        resolved_location = state.get("resolved_location")
        if resolved_location is not None:
            try:
                location_id = int(resolved_location.id)
            except (ValueError, TypeError):
                location_id = None

    if location_id is None:
        return {
            "error": "Nie udało się rozpoznać lokalizacji",
            "pending_confirmation": None,
        }

    if action == "edit_rule":
        short_id = pending.get("edit_short_id", "")
        existing_rule = await rule_service.get_rule(short_id=short_id)
        if existing_rule is None:
            return {"error": f"Nie znaleziono reguły #{short_id}", "pending_confirmation": None}

        rule = await rule_service.update_rule(
            existing_rule.id,
            RuleUpdate(
                expression=cel_expression,
                description=explanation,
            ),
        )
        answer = (
            f"Reguła #{rule.short_id} została zaktualizowana.\n\U0001f4dd CEL: `{rule.expression}`"
        )
    else:
        rule = await rule_service.create_rule(
            user_id,
            RuleCreate(
                telegram_chat_id=chat_id,
                telegram_message_thread_id=message_thread_id,
                location_id=location_id,
                expression=cel_expression,
                description=explanation,
            ),
        )
        answer = (
            f"Nowa reguła #{rule.short_id} została zapisana.\n\U0001f4dd CEL: `{rule.expression}`"
        )

    return {
        "answer": answer,
        "pending_confirmation": None,
        "cel_expression": None,
        "error": None,
    }


async def cancel_rule_node(state: ConversationState) -> dict[str, Any]:
    pending = state.get("pending_confirmation")
    if pending is None:
        return {"answer": "Nie ma oczekującej reguły do anulowania."}

    action = pending.get("action", "create_rule")
    if action == "edit_rule":
        short_id = pending.get("edit_short_id", "")
        answer = f"Edycja reguły #{short_id} została anulowana."
    else:
        answer = "Reguła została anulowana."

    return {
        "answer": answer,
        "pending_confirmation": None,
        "cel_expression": None,
        "error": None,
    }


async def persist_rule_change_node(
    state: ConversationState,
    rule_service: NotificationRuleService | None,
    location_service: LocationService | None,
) -> dict[str, Any]:
    if rule_service is None or location_service is None:
        return {"error": "Reguły powiadomień są niedostępne bez pełnej konfiguracji."}

    pending = state.get("pending_confirmation")
    if pending is None:
        return {"error": "Brak danych do zapisania reguły"}

    user_message = (state.get("user_message") or "").strip().lower()

    if is_confirmation_no(user_message):
        return await cancel_rule_node(state)

    if not is_confirmation_yes(user_message):
        return {
            "answer": "Oczekuję na potwierdzenie (tak) lub odrzucenie (nie).",
            "error": None,
        }

    return await confirm_rule_node(state, rule_service, location_service)
