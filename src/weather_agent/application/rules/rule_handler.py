from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from langsmith import traceable
from weather_agent.application.conversation_models import PendingConfirmation, TurnResult
from weather_agent.application.intent_classifier import is_confirmation_no, is_confirmation_yes
from weather_agent.domain.cel.allowlist import get_allowlist_for_prompt
from weather_agent.domain.cel.evaluator import CELEvaluator
from weather_agent.domain.locations import LocationService
from weather_agent.domain.rules.models import RuleCreate, RuleUpdate
from weather_agent.domain.rules.service import NotificationRuleService
from weather_agent.domain.rules.short_id_generator import strip_hash_prefix
from weather_agent.llm.contracts.rules import RuleProposalExtraction
from weather_agent.llm.model_factory import ModelFactory
from weather_agent.llm.prompts.rule_prompts import RULE_PROPOSAL_PROMPT
from weather_agent.observability.logging import get_logger

logger = get_logger(__name__)

_GENERIC_RULE_ERROR = (
    "Przepraszam, wystąpił błąd podczas przygotowywania reguły."
    " Spróbuj ponownie za chwilę."
)


@traceable(run_type="chain", name="propose_rule")
async def propose_rule(
    user_message: str,
    model_factory: ModelFactory | None,
    cel_evaluator: CELEvaluator | None,
) -> dict[str, Any]:
    if model_factory is None or cel_evaluator is None:
        return {"error": "Reguły powiadomień są niedostępne bez pełnej konfiguracji."}

    if not user_message:
        return {"error": "Brak wiadomości użytkownika do przetworzenia"}

    chat_model = model_factory.create_chat_model()
    structured = chat_model.with_structured_output(RuleProposalExtraction)

    allowlist = get_allowlist_for_prompt()
    functions_json = json.dumps(allowlist["functions"], ensure_ascii=False, indent=2)
    metrics_json = json.dumps(allowlist["metrics"], ensure_ascii=False, indent=2)

    try:
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
        logger.warning("llm_rule_no_cel_expression", explanation=explanation)
        return {
            "error": _GENERIC_RULE_ERROR,
            "cel_expression": None,
            "pending_confirmation": None,
        }

    validation = cel_evaluator.validate(cel_expression)

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

    pending = PendingConfirmation(
        action="create_rule" if not short_id else "edit_rule",
        cel_expression=validation.expression,
        explanation=explanation,
        validated=True,
        location_id=None,
        chat_id=None,
        message_thread_id=None,
        stored_at=datetime.now(UTC).isoformat(),
        edit_short_id=short_id if short_id else None,
    )

    return {
        "cel_expression": validation.expression,
        "cel_validation_result": validation,
        "pending_confirmation": pending,
        "error": None,
    }


def format_rule_confirmation(pending: PendingConfirmation) -> str:
    cel_expression = pending.cel_expression
    explanation = pending.explanation
    action = pending.action

    if action == "edit_rule":
        short_id = pending.edit_short_id or ""
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
    return "\n".join(lines)


async def confirm_rule(
    pending: PendingConfirmation,
    user_id: int | None,
    rule_service: NotificationRuleService | None,
    location_service: LocationService | None,
    resolved_location: Any | None,
    chat_id: int | None,
    message_thread_id: int | None,
) -> TurnResult:
    if rule_service is None or location_service is None:
        return TurnResult(error="Reguły powiadomień są niedostępne bez pełnej konfiguracji.")

    cel_expression = pending.cel_expression
    explanation = pending.explanation
    action = pending.action

    if user_id is None:
        return TurnResult(error="Użytkownik nie jest autoryzowany", pending_confirmation=None)

    location_id: int | None = pending.location_id
    if location_id is None and resolved_location is not None:
        try:
            location_id = int(resolved_location.id)
        except (ValueError, TypeError):
            location_id = None

    if location_id is None:
        return TurnResult(error="Nie udało się rozpoznać lokalizacji", pending_confirmation=None)

    effective_chat_id = pending.chat_id if pending.chat_id is not None else (chat_id or 0)
    effective_thread_id = (
        pending.message_thread_id if pending.message_thread_id is not None
        else message_thread_id
    )

    if action == "edit_rule":
        short_id = pending.edit_short_id or ""
        existing_rule = await rule_service.get_rule(short_id=short_id)
        if existing_rule is None:
            return TurnResult(error=f"Nie znaleziono reguły #{short_id}", pending_confirmation=None)

        rule = await rule_service.update_rule(
            existing_rule.id,
            RuleUpdate(expression=cel_expression, description=explanation),
        )
        answer = (
            f"Reguła #{rule.short_id} została zaktualizowana.\n"
            f"\U0001f4dd CEL: `{rule.expression}`"
        )
    else:
        rule = await rule_service.create_rule(
            user_id,
            RuleCreate(
                telegram_chat_id=effective_chat_id,
                telegram_message_thread_id=effective_thread_id,
                location_id=location_id,
                expression=cel_expression,
                description=explanation,
            ),
        )
        answer = (
            f"Nowa reguła #{rule.short_id} została zapisana.\n"
            f"\U0001f4dd CEL: `{rule.expression}`"
        )

    return TurnResult(
        answer=answer,
        pending_confirmation=PendingConfirmation(),
        cel_expression=None,
    )


async def cancel_rule(pending: PendingConfirmation) -> TurnResult:
    action = pending.action
    if action == "edit_rule":
        short_id = pending.edit_short_id or ""
        answer = f"Edycja reguły #{short_id} została anulowana."
    else:
        answer = "Reguła została anulowana."

    return TurnResult(
        answer=answer,
        pending_confirmation=PendingConfirmation(),
        cel_expression=None,
    )


async def handle_rule_confirmation(
    user_message: str,
    pending: PendingConfirmation,
    user_id: int | None,
    rule_service: NotificationRuleService | None,
    location_service: LocationService | None,
    resolved_location: Any | None,
    chat_id: int | None,
    message_thread_id: int | None,
) -> TurnResult:
    if is_confirmation_no(user_message):
        return await cancel_rule(pending)

    if not is_confirmation_yes(user_message):
        return TurnResult(answer="Oczekuję na potwierdzenie (tak) lub odrzucenie (nie).")

    return await confirm_rule(
        pending, user_id, rule_service, location_service,
        resolved_location, chat_id, message_thread_id,
    )