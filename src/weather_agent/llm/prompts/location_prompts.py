from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

LOCATION_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Wyciągnij z wiadomości użytkownika:\n"
        "1. location_name — nazwa miejscowości w mianowniku (np. „w Gdańsku\" → „Gdańsk\", „w Chwarznie\" → „Chwarzno\"). Zwróć null jeśli nie ma nazwy miejsca.\n"
        "2. focus — o co użytkownik pyta szczegółowo (np. wiatr, porywy, temperatura, opady, ciśnienie). Zwróć null jeśli ogólne pytanie.\n\n"
        "UWAGA: „w weekend\", „w majówkę\", „w poniedziałek\" to CZAS, nie lokalizacja. Nie myl przyimków czasowych z miejscowymi."
    )),
    ("user", "{user_message}")
])