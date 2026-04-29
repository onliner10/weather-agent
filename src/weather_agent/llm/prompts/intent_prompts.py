from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

_INTENT_SYSTEM_PROMPT = """\
Zklasyfikuj intencję użytkownika asystenta pogodowego.

Dostępne intencje:
- 'weather': Pytania o pogodę, prognozę, aktualne warunki, lub prośby o ZAPISANIE/ZAPAMIĘTANIE lokalizacji (np. 'zapisz dom').
- 'rule': Prośby o stworzenie, edycję, usunięcie lub listowanie REGUŁ powiadomień pogodowych (np. 'powiadom mnie gdy spadnie śnieg').
- 'command': Komendy systemowe (/start, /help), prośba o pomoc lub informację o działaniu bota.
{optional_intents}
Wybierz najbardziej pasującą intencję."""

_CONFIRMATION_INTENTS = """\
- 'confirm_rule': Użytkownik POTWIERDZA (mówi tak, zgadza się, jasne, ok) oczekującą akcję.
- 'cancel_rule': Użytkownik ANULUJE (mówi nie, rezygnuje, nie chce) oczekującą akcję.
"""

INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _INTENT_SYSTEM_PROMPT),
    ("user", "{user_message}")
])


def get_optional_intents(has_pending_confirmation: bool) -> str:
    return _CONFIRMATION_INTENTS if has_pending_confirmation else ""