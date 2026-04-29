---
name: rules
description: Zarządzanie regulami powiadomień pogodowych. Użyj gdy użytkownik prosi o powiadomienie, regułę, alert, lub gdy mówi o warunkach które mają wywołać notyfikację.
---

# Rules Skill

## Przepływ pracy

1. **Przekaż prośbę** — Nie twórz reguły samodzielnie. System reguł jest oddzielny od modelu językowego.
2. **Poinformuj użytkownika** — System przygotuje propozycję reguły i poprosi o potwierdzenie.
3. **Potwierdzenie** — Po utworzeniu propozycji, użytkownik musi potwierdzić (tak/nie).

## Zasady

- Reguły używają CEL (Common Expression Language) do definiowania warunków.
- Walidacja i wykonanie reguł są deterministyczne.
- Model językowy może zaproponować wyrażenie CEL, ale nie może go samodzielnie aktywować.
