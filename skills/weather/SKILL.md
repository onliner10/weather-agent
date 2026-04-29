---
name: weather-qa
description: Odpowiadanie na pytania o pogodę — prognoza, obserwacje, zapisywanie lokalizacji. Użyj gdy użytkownik pyta o pogodę, temperaturę, opady, wiatr, ciśnienie lub chce zapisać lokalizację.
---

# Weather Q&A Skill

## Przepływ pracy

1. **Zrozum pytanie** — Czy użytkownik pyta o prognozę, aktualne obserwacje, czy chce zapisać lokalizację?
2. **Wybierz narzędzie** — `get_forecast` dla prognozy, `get_observations` dla aktualnych danych, `save_location` dla zapisywania.
3. **Wybierz zmienne** — Tylko te, o które pyta użytkownik (np. temperatura, wiatr, opady).
4. **Sformatuj odpowiedź** — Zwięźle, po polsku, z lokalizacją i zakresem czasu.

## Obsługa lokalizacji

- Jeśli użytkownik nie podał lokalizacji, spróbuj użyć wcześniej użytej lub domyślnej.
- Jeśli nadal brak, poproś o podanie lokalizacji.
- Jeśli użytkownik prosi o zapamiętanie ("zapisz dom w Gdańsku"), użyj `save_location`.

## Obsługa czasu

- Przeliczaj względne określenia czasu (jutro, dziś, pojutrze, weekend, majówka, przyszły tydzień)
  na konkretne daty w formacie yyyy-mm-dd przed wywołaniem narzędzia pogodowego.
- Bieżąca data i godzina w strefie Europe/Warsaw są podane w kontekście.
- Dla pojedynczego dnia powtórz tę samą datę jako start i end.
- Nie pytaj użytkownika o dokładną datę — samodzielnie wykonaj obliczenia.
