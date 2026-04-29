from __future__ import annotations

import json

from weather_agent.domain.cel.allowlist import get_allowlist_for_prompt

_RULE_PROPOSAL_SYSTEM_PROMPT_TEMPLATE = """\
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


def build_rule_proposal_prompt() -> str:
    allowlist = get_allowlist_for_prompt()
    functions_json = json.dumps(allowlist["functions"], ensure_ascii=False, indent=2)
    metrics_json = json.dumps(allowlist["metrics"], ensure_ascii=False, indent=2)
    return _RULE_PROPOSAL_SYSTEM_PROMPT_TEMPLATE.format(
        cel_functions=functions_json,
        cel_metrics=metrics_json,
    )