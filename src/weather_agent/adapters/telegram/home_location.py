from __future__ import annotations

import re

from weather_agent.domain.locations import (
    LocationAliasConflictError,
    LocationCreate,
    LocationNameConflictError,
    LocationService,
)
from weather_agent.infrastructure.geocoder import Geocoder

_HOME_LOCATION_PATTERNS = (
    re.compile(
        r"(?:zapami[eę]taj|zapisz|ustaw)\s+"
        r"(?:moj[aą]\s+)?lokalizacj[eę]\s+domow[aą]\s*"
        r"(?:jako|to|:)?\s*(?P<location>.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:moja\s+)?lokalizacj[aą]\s+domowa\s+"
        r"(?:to|jest|:)\s*(?P<location>.+)",
        re.IGNORECASE,
    ),
)


def extract_home_location_request(message: str) -> str | None:
    """Return requested home-location text for deterministic save intents."""
    for pattern in _HOME_LOCATION_PATTERNS:
        match = pattern.search(message)
        if match is None:
            continue
        location = match.group("location").strip(" .,!?:;\t\n")
        return location or None
    return None


async def handle_home_location_save_message(
    message: str,
    user_id: int,
    location_service: LocationService,
    geocoder: Geocoder | None,
) -> str | None:
    requested_location = extract_home_location_request(message)
    if requested_location is None:
        return None

    if geocoder is None:
        return "Nie mogę teraz rozpoznać adresu. Użyj /dodaj_lok <nazwa> <lat> <lon>."

    resolved = await geocoder.geocode(requested_location)
    if resolved is None:
        return (
            f"Nie udało się rozpoznać lokalizacji „{requested_location}”. "
            "Podaj współrzędne komendą /dodaj_lok <nazwa> <lat> <lon>."
        )

    try:
        await location_service.create_location(
            user_id,
            LocationCreate(
                name=requested_location,
                aliases=["dom"],
                latitude=resolved.latitude,
                longitude=resolved.longitude,
            ),
        )
    except (LocationAliasConflictError, LocationNameConflictError):
        return "Masz już zapisaną lokalizację domową."

    return f"Zapamiętałem Twoją lokalizację domową jako {requested_location}."
