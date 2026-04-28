from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from weather_agent.domain.cel.allowlist import get_allowlist_for_prompt
from weather_agent.domain.cel.evaluator import CELEvaluationResult, CELEvaluator
from weather_agent.domain.locations import LocationService
from weather_agent.domain.rules.models import RuleCreate, RuleUpdate
from weather_agent.domain.rules.service import NotificationRuleService
from weather_agent.domain.rules.short_id_generator import strip_hash_prefix
from weather_agent.graphs.state import ConversationState

_RULE_PROPOSAL_SYSTEM_PROMPT = """\
Jesteś asystentem botu pogodowego dla użytkowników mówiących po polsku.
Twoim zadaniem jest przekształcenie naturalnego opisu reguły powiadomień w wyrażenie CEL.

Dostępne funkcje CEL:
{cel_functions}

Dostępne metryki pogodowe:
{cel_metrics}

Zasady tworzenia wyrażeń CEL:
1. Używaj TYLKO wymienionych funkcji i metryk.
2. Metryki podawaj jako string (np. "temperature_2m_c").
3. Funkcje agregujące przyjmują metrykę jako string oraz zakres czasowy.
4. Funkcje czasu (now, today, tomorrow, weekend) zwracają zakresy czasowe.
5. Nie używaj żadnych funkcji ani metryk spoza listy.

Odpowiedź MUSI być w formacie JSON z polami:
- "cel_expression": wyrażenie CEL
- "explanation": opis po polsku, co wyrażenie oznacza

Jeśli nie da się zamienić opisu na wyrażenie CEL, zwróć:
- "cel_expression": null
- "explanation": opis problemu po polsku
"""

_SHORT_ID_PATTERN = re.compile(r"(?:#|\b)R[A-HJKMNP-Z0-9]{3,6}\b")


def _extract_short_id(text: str) -> str | None:
    match = _SHORT_ID_PATTERN.search(text)
    if match is None:
        return None
    return strip_hash_prefix(match.group(0).lstrip("#"))


def _build_system_prompt() -> str:
    allowlist = get_allowlist_for_prompt()
    functions_json = json.dumps(allowlist["functions"], ensure_ascii=False, indent=2)
    metrics_json = json.dumps(allowlist["metrics"], ensure_ascii=False, indent=2)
    return _RULE_PROPOSAL_SYSTEM_PROMPT.format(
        cel_functions=functions_json,
        cel_metrics=metrics_json,
    )


async def propose_cel_rule_node(
    state: ConversationState,
    model_factory: Any,
    cel_evaluator: CELEvaluator,
) -> dict[str, Any]:
    user_message = state.get("user_message") or ""
    if not user_message:
        return {"error": "Brak wiadomości użytkownika do przetworzenia"}

    chat_model: BaseChatModel = model_factory.create_chat_model()
    system_prompt = _build_system_prompt()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    try:
        response = await chat_model.ainvoke(messages)
    except Exception as exc:
        return {"error": f"Błąd modelu LLM: {exc}"}

    response_raw = response.content if hasattr(response, "content") else str(response)
    response_text = str(response_raw)

    try:
        parsed = json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        cleaned = re.sub(r"```json\s*|\s*```", "", str(response_text))
        try:
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            return {"error": "Nie udało się przetworzyć odpowiedzi modelu"}

    cel_expression = parsed.get("cel_expression")
    explanation = parsed.get("explanation", "")

    if cel_expression is None:
        return {
            "error": explanation or "Nie udało się wygenerować wyrażenia CEL",
            "cel_expression": None,
            "pending_confirmation": None,
        }

    validation: CELEvaluationResult = cel_evaluator.validate(cel_expression)

    if not validation.valid:
        return {
            "error": f"Niepoprawne wyrażenie CEL: {validation.error}",
            "cel_expression": None,
            "pending_confirmation": None,
        }

    short_id = _extract_short_id(user_message)

    pending: dict[str, Any] = {
        "action": "edit_rule" if short_id else "create_rule",
        "cel_expression": validation.expression,
        "explanation": explanation,
        "validated": True,
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
        f"📝 Wyrażenie CEL: `{cel_expression}`",
        f"📖 Opis: {explanation}",
        "",
        "Czy chcesz potwierdzić? (tak/nie)",
    ]
    answer = "\n".join(lines)

    return {"answer": answer}


async def persist_rule_change_node(
    state: ConversationState,
    rule_service: NotificationRuleService,
    location_service: LocationService,
) -> dict[str, Any]:
    pending = state.get("pending_confirmation")
    if pending is None:
        return {"error": "Brak danych do zapisania reguły"}

    user_message = (state.get("user_message") or "").strip().lower()
    normalized = user_message

    is_yes = normalized in ("tak", "yes", "t", "y", "potwierdz", "potwierdzam", "ok")
    is_no = normalized in ("nie", "no", "n", "anuluj", "anuluję", "rezygnuj")

    if is_no:
        return {
            "answer": "Reguła została anulowana.",
            "pending_confirmation": None,
            "cel_expression": None,
            "error": None,
        }

    if not is_yes:
        return {
            "answer": "Oczekuję na potwierdzenie (tak) lub odrzucenie (nie).",
            "error": None,
        }

    cel_expression = pending.get("cel_expression", "")
    explanation = pending.get("explanation", "")
    action = pending.get("action", "create_rule")

    user_id = state.get("authorized_user_id")
    if user_id is None:
        return {"error": "Użytkownik nie jest autoryzowany"}

    chat_id = state.get("chat_id", 0)
    message_thread_id = state.get("message_thread_id")

    resolved_location = state.get("resolved_location")
    location_id: int | None = None
    if resolved_location is not None:
        try:
            location_id = int(resolved_location.id)
        except (ValueError, TypeError):
            location_id = None

    if location_id is None:
        return {"error": "Nie udało się rozpoznać lokalizacji"}

    if action == "edit_rule":
        short_id = pending.get("edit_short_id", "")
        existing_rule = await rule_service.get_rule(short_id=short_id)
        if existing_rule is None:
            return {"error": f"Nie znaleziono reguły #{short_id}"}

        rule = await rule_service.update_rule(
            existing_rule.id,
            RuleUpdate(
                expression=cel_expression,
                description=explanation,
            ),
        )
        answer = f"Reguła #{rule.short_id} została zaktualizowana.\n📝 CEL: `{rule.expression}`"
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
        answer = f"Nowa reguła #{rule.short_id} została zapisana.\n📝 CEL: `{rule.expression}`"

    return {
        "answer": answer,
        "pending_confirmation": None,
        "cel_expression": None,
        "error": None,
    }