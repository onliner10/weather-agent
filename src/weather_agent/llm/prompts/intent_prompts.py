from __future__ import annotations


def build_intent_prompt(has_pending_confirmation: bool) -> str:
    parts = [
        "Zklasyfikuj intencję użytkownika asystenta pogodowego.\n",
        "Dostępne intencje:\n",
        "- 'weather': Pytania o pogodę, prognozę, aktualne warunki, ",
        "lub prośby o ZAPISANIE/ZAPAMIĘTANIE lokalizacji (np. 'zapisz dom').\n",
        "- 'rule': Prośby o stworzenie, edycję, usunięcie lub listowanie ",
        "REGUŁ powiadomień pogodowych (np. 'powiadom mnie gdy spadnie śnieg').\n",
        "- 'command': Komendy systemowe (/start, /help), prośba o pomoc ",
        "lub informację o działaniu bota.\n",
    ]
    if has_pending_confirmation:
        parts.append(
            "- 'confirm_rule': Użytkownik POTWIERDZA (mówi tak, zgadza się, jasne, ok) oczekującą akcję.\n"
        )
        parts.append(
            "- 'cancel_rule': Użytkownik ANULUJE (mówi nie, rezygnuje, nie chce) oczekującą akcję.\n"
        )

    parts.append("\nWybierz najbardziej pasującą intencję.")
    return "".join(parts)