from __future__ import annotations

import logging

import httpx
from pydantic import BaseModel, Field

from weather_agent.domain.weather import LocationRef
from weather_agent.llm.model_factory import ModelFactory

logger = logging.getLogger(__name__)


class _LocationGuess(BaseModel):
    lat: float | None = Field(default=None, description="Latitude if confident about coordinates")
    lon: float | None = Field(default=None, description="Longitude if confident about coordinates")
    display_name: str = Field(description="Canonical Polish name of the place")
    search_query: str | None = Field(
        default=None,
        description="Search query for geocoding API if coordinates not provided",
    )


class Geocoder:
    def __init__(
        self,
        base_url: str = "https://geocoding-api.open-meteo.com/v1/search",
        timeout_seconds: float = 10.0,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout_seconds
        self._model_factory = model_factory

    async def geocode(self, name: str) -> LocationRef | None:
        if self._model_factory is not None:
            guess = await self._ask_llm(name)
            if guess is not None:
                if guess.lat is not None and guess.lon is not None:
                    logger.info("LLM gave coordinates for '%s': %s, %s", name, guess.lat, guess.lon)
                    return LocationRef(
                        id=f"llm:{guess.lat:.4f}:{guess.lon:.4f}",
                        name=guess.display_name,
                        latitude=guess.lat,
                        longitude=guess.lon,
                    )
                if guess.search_query:
                    result = await self._try_geocode(guess.search_query)
                    if result is not None:
                        return result

        result = await self._try_geocode(name)
        if result is not None:
            return result

        return None

    async def _ask_llm(self, name: str) -> _LocationGuess | None:
        try:
            chat = self._model_factory.create_chat_model()  # type: ignore[union-attr]
            structured = chat.with_structured_output(_LocationGuess)
            response = await structured.ainvoke(
                [
                    {
                        "role": "system",
                        "content": (
                            "Jesteś ekspertem polskiej geografii. Użytkownik podał nazwę miejsca"
                            " (może być w odmianie, np. miejscownik „w Gdańsku\", lub opisową"
                            " np. „Lotnisko Modlin\").\n\n"
                            "Jeśli jesteś pewien współrzędnych — podaj lat i lon."
                            " Jeśli nie jesteś pewien — zostaw lat/lon jako null"
                            " i podaj search_query"
                            " (mianownik miasta lub fraza do geocoding API).\n\n"
                            "display_name to zawsze poprawna polska nazwa w mianowniku."
                        ),
                    },
                    {"role": "user", "content": name},
                ],
            )
            if isinstance(response, _LocationGuess):
                logger.info("LLM location guess for '%s': %s", name, response)
                return response
        except Exception:
            logger.warning("LLM location resolution failed for '%s'", name, exc_info=True)
        return None

    async def _try_geocode(self, name: str) -> LocationRef | None:
        normalized = _normalize_polish(name)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                r = await client.get(
                    self._base_url,
                    params={"name": normalized, "count": 1, "language": "pl"},
                )
                r.raise_for_status()
            except httpx.TimeoutException:
                logger.warning("Geocoding timeout for %s", name)
                return None
            except httpx.HTTPStatusError:
                logger.warning("Geocoding HTTP error for %s: %s", name, r.status_code)
                return None

        data = r.json()
        results = data.get("results")
        if not results:
            logger.info("No geocoding results for '%s'", name)
            return None

        first = results[0]
        return LocationRef(
            id=str(first["id"]),
            name=first["name"],
            latitude=first["latitude"],
            longitude=first["longitude"],
        )


_POLISH_ASCII: dict[str, str] = {
    "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n",
    "ó": "o", "ś": "s", "ź": "z", "ż": "z",
}


def _normalize_polish(text: str) -> str:
    return "".join(_POLISH_ASCII.get(c, c) for c in text.lower())