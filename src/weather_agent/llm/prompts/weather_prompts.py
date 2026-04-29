from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from weather_agent.domain.weather import LocationRef

_WARSAW = ZoneInfo("Europe/Warsaw")


def build_weather_system_prompt(
    now: datetime | None = None,
    resolved_location: LocationRef | None = None,
    user_focus: str | None = None,
) -> str:
    if now is None:
        now = datetime.now(_WARSAW)

    parts: list[str] = [
        "Jesteś polskim asystentem pogodowym. Odpowiadaj krótko po polsku.\n\n"
        f"Dziś jest {now.strftime('%Y-%m-%d')} (dzień tygodnia: "
        f"{now.strftime('%A')}), godzina {now.strftime('%H:%M')} "
        "w strefie czasowej Europe/Warsaw.\n\n"
        "Aby odpowiedzieć na pytanie, użyj dostępnych narzędzi "
        "(get_forecast, get_observations, save_location).\n\n"
        "Jeśli użytkownik prosi o zapamiętanie lub zapisanie lokalizacji "
        "(np. 'zapisz dom w Gdańsku'), użyj narzędzia save_location.\n"
        " Wybierz tylko potrzebne zmienne — np. jeśli pytanie jest o wiatr,"
        " poproś tylko o wind_speed_10m_ms,"
        " wind_gusts_10m_ms, wind_direction_10m_deg.\n\n"
        "Daty w narzędziach podawaj jako yyyy-mm-dd na podstawie "
        "bieżącej daty i strefy czasowej.\n",
    ]
    if resolved_location and resolved_location.name:
        parts.append(f"Użytkownik pyta o miejscowość: {resolved_location.name}.\n")
    if user_focus:
        parts.append(
            f"Użytkownik pyta szczegółowo o: {user_focus}. Skoncentruj się na tym aspekcie.\n"
        )
    parts.append(
        "Po otrzymaniu danych, napisz zwięzłą, naturalną odpowiedź po polsku."
        " Podaj lokalizację i zakres czasu."
    )
    return "".join(parts)