from __future__ import annotations

import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver

from weather_agent.observability.logging import get_logger

logger = get_logger(__name__)

_AGENTS_MD_CONTENT: str | None = None


def _load_agents_md() -> str:
    global _AGENTS_MD_CONTENT
    if _AGENTS_MD_CONTENT is not None:
        return _AGENTS_MD_CONTENT

    search_dirs = [
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),
    ]
    for d in search_dirs:
        candidate = os.path.join(d, "AGENTS.md")
        if os.path.isfile(candidate):
            with open(candidate, encoding="utf-8") as f:
                content = f.read()
            _AGENTS_MD_CONTENT = content
            logger.info("agents_md_loaded", path=candidate)
            return content

    logger.warning("agents_md_not_found, using fallback")
    _AGENTS_MD_CONTENT = ""
    return _AGENTS_MD_CONTENT


def create_weather_agent(
    model: BaseChatModel,
    tools: list[Any],
    system_prompt_suffix: str = "",
    checkpointer: Any | None = None,
) -> Any:
    agents_md = _load_agents_md()

    system_prompt = agents_md
    if system_prompt_suffix:
        system_prompt = f"{agents_md}\n\n{system_prompt_suffix}"

    cp = checkpointer if checkpointer is not None else MemorySaver()

    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=cp,
        subagents=[],
    )


_WARSAW = ZoneInfo("Europe/Warsaw")


def build_context_suffix(
    pending_confirmation: dict[str, Any] | None = None,
    last_forecast_context: dict[str, Any] | None = None,
) -> str:
    parts: list[str] = []

    now = datetime.now(_WARSAW)
    parts.append(
        f"Bieżąca data i godzina w strefie Europe/Warsaw: {now.strftime('%Y-%m-%d %H:%M')}."
    )

    if last_forecast_context:
        loc = last_forecast_context.get("location_name", "?")
        sd = last_forecast_context.get("start_date", "?")
        ed = last_forecast_context.get("end_date", "?")
        vars_list = last_forecast_context.get("variables", [])
        vars_str = ", ".join(vars_list) if vars_list else "?"
        parts.append(
            f"OSTATNIA PROGNOZA: {loc}, zakres {sd} – {ed} (zmienne: {vars_str}). "
            "Jeśli użytkownik pyta follow-upowo bez zmiany lokalizacji lub zakresu, "
            "odziedzicz lokalizację i zakres z ostatniej prognozy."
        )

    if pending_confirmation:
        action = pending_confirmation.get("action", "create_rule")
        cel = pending_confirmation.get("cel_expression", "")
        explanation = pending_confirmation.get("explanation", "")
        edit_short_id = pending_confirmation.get("edit_short_id")

        if action == "edit_rule" and edit_short_id:
            parts.append(
                f"OCZEKUJĄCA AKCJA: Edycja reguły #{edit_short_id}. "
                f"Wyrażenie CEL: `{cel}`. Opis: {explanation}. "
                "Jeśli użytkownik potwierdza (tak/ok/potwierdzam), użyj confirm_pending_action. "
                "Jeśli użytkownik odrzuca (nie/anuluj), użyj cancel_pending_action."
            )
        else:
            parts.append(
                f"OCZEKUJĄCA AKCJA: Utworzenie nowej reguły powiadomienia. "
                f"Wyrażenie CEL: `{cel}`. Opis: {explanation}. "
                "Jeśli użytkownik potwierdza (tak/ok/potwierdzam), użyj confirm_pending_action. "
                "Jeśli użytkownik odrzuca (nie/anuluj), użyj cancel_pending_action."
            )

    return "\n\n".join(parts)
