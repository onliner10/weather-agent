from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

GEOCODING_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", (
        "Jesteś ekspertem polskiej geografii. Użytkownik podał nazwę miejsca"
        ' (może być w odmianie, np. miejscownik „w Gdańsku", lub opisową'
        ' np. „Lotnisko Modlin").\n\n'
        "Podaj search_query — mianownik miasta lub frazę do geocoding API."
        " display_name to zawsze poprawna polska nazwa w mianowniku."
    )),
    ("user", "{location_name}")
])
