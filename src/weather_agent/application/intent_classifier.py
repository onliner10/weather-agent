from __future__ import annotations

from typing import Any

from weather_agent.llm.contracts.intent import IntentExtraction
from weather_agent.llm.model_factory import ModelFactory
from weather_agent.llm.prompts.intent_prompts import INTENT_PROMPT, get_optional_intents
from weather_agent.observability.logging import get_logger

logger = get_logger(__name__)

_YES_WORDS = frozenset({"tak", "yes", "t", "y", "potwierdz", "potwierdzam", "ok"})
_NO_WORDS = frozenset({"nie", "no", "n", "anuluj", "anuluję", "rezygnuj"})


def classify_with_pending_confirmation(
    user_message: str,
    pending_confirmation: dict[str, Any] | None,
) -> str | None:
    if pending_confirmation is None:
        return None
    msg = user_message.lower()
    if msg.strip() in _YES_WORDS:
        return "confirm_rule"
    if msg.strip() in _NO_WORDS:
        return "cancel_rule"
    return None


def _deterministic_classify(msg: str) -> str:
    if any(kw in msg for kw in ("/start", "/help", "/pomoc")):
        return "command"
    if any(kw in msg for kw in ("reguł", "zasad", "powiadom", "notyfik", "cel")):
        return "rule"
    return "weather"


async def classify_intent(
    state_or_message: Any = None,
    model_factory: ModelFactory | None = None,
    *,
    pending_confirmation: dict[str, Any] | None = None,
    context_key: str = "",
    chat_id: int = 0,
    message_thread_id: int | None = None,
) -> dict[str, Any]:
    if isinstance(state_or_message, dict):
        user_message = (state_or_message.get("user_message") or "").lower()
        pending_confirmation = state_or_message.get("pending_confirmation")
        if state_or_message.get("resolved_intent") is not None:
            return {"resolved_intent": state_or_message["resolved_intent"]}
    elif isinstance(state_or_message, str):
        user_message = state_or_message.lower()
    else:
        user_message = ""

    confirmation_route = classify_with_pending_confirmation(user_message, pending_confirmation)
    if confirmation_route is not None:
        return {"resolved_intent": confirmation_route}

    if model_factory is None:
        return {"resolved_intent": _deterministic_classify(user_message)}

    chat = model_factory.create_chat_model()
    structured = chat.with_structured_output(IntentExtraction)
    optional_intents = get_optional_intents(pending_confirmation is not None)

    try:
        chain = INTENT_PROMPT | structured
        result = await chain.ainvoke({
            "optional_intents": optional_intents,
            "user_message": user_message
        })
        return {"resolved_intent": result.intent}
    except Exception:
        logger.warning("llm_intent_classification_failed", exc_info=True)
        return {"resolved_intent": _deterministic_classify(user_message)}


def is_confirmation_yes(text: str) -> bool:
    return text.strip().lower() in _YES_WORDS


def is_confirmation_no(text: str) -> bool:
    return text.strip().lower() in _NO_WORDS