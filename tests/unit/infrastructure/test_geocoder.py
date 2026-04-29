from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import respx

from weather_agent.infrastructure.geocoder import Geocoder
from weather_agent.llm.contracts.geocoder import LocationGuess

_GEOCODE_BASE = "https://geocoding-api.open-meteo.com/v1/search"


def _geocoder_response(name: str, lat: float, lon: float, geo_id: int = 1) -> dict:
    return {
        "results": [
            {
                "id": geo_id,
                "name": name,
                "latitude": lat,
                "longitude": lon,
            }
        ]
    }


def _make_llm_mocks(guess: LocationGuess) -> MagicMock:
    mock_factory = MagicMock()
    mock_chat = MagicMock()
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(return_value=guess)
    mock_chat.with_structured_output.return_value = mock_structured
    mock_factory.create_chat_model.return_value = mock_chat
    return mock_factory


class TestGeocodeWithoutLlm:
    @respx.mock
    async def test_basic_geocode(self) -> None:
        respx.get(_GEOCODE_BASE).mock(
            return_value=httpx.Response(200, json=_geocoder_response("Gdańsk", 54.3520, 18.6466))
        )
        geocoder = Geocoder()
        result = await geocoder.geocode("Gdańsk")
        assert result is not None
        assert result.name == "Gdańsk"
        assert result.latitude == 54.3520
        assert result.longitude == 18.6466
        assert not result.id.startswith("llm:")

    @respx.mock
    async def test_geocode_returns_none_on_empty(self) -> None:
        respx.get(_GEOCODE_BASE).mock(return_value=httpx.Response(200, json={"results": []}))
        geocoder = Geocoder()
        result = await geocoder.geocode("NonexistentPlace")
        assert result is None

    @respx.mock
    async def test_geocode_uses_normalized_polish_name(self) -> None:
        route = respx.get(_GEOCODE_BASE).mock(
            return_value=httpx.Response(200, json=_geocoder_response("Warszawa", 52.2297, 21.0122))
        )
        geocoder = Geocoder()
        result = await geocoder.geocode("w Warszawie")
        assert result is not None
        assert result.name == "Warszawa"
        request = route.calls[0].request
        assert "name=" in str(request.url)


class TestGeocodeWithLlm:
    @respx.mock
    async def test_llm_search_query_routes_to_deterministic_geocoder(self) -> None:
        guess = LocationGuess(display_name="Gdańsk", search_query="Gdańsk")
        mock_factory = _make_llm_mocks(guess)

        respx.get(_GEOCODE_BASE).mock(
            return_value=httpx.Response(
                200, json=_geocoder_response("Gdańsk", 54.3520, 18.6466, geo_id=42)
            )
        )

        geocoder = Geocoder(model_factory=mock_factory)
        result = await geocoder.geocode("w Gdańsku")
        assert result is not None
        assert result.name == "Gdańsk"
        assert result.latitude == 54.3520
        assert result.longitude == 18.6466
        assert result.id == "42"

    @respx.mock
    async def test_llm_no_search_query_uses_display_name(self) -> None:
        guess = LocationGuess(display_name="Kraków")
        mock_factory = _make_llm_mocks(guess)

        respx.get(_GEOCODE_BASE).mock(
            return_value=httpx.Response(
                200, json=_geocoder_response("Kraków", 50.0647, 19.9450, geo_id=7)
            )
        )

        geocoder = Geocoder(model_factory=mock_factory)
        result = await geocoder.geocode("Kraków")
        assert result is not None
        assert result.name == "Kraków"
        assert not result.id.startswith("llm:")

    @respx.mock
    async def test_llm_returns_no_llm_prefix_ids(self) -> None:
        guess = LocationGuess(display_name="Szczecin", search_query="Szczecin")
        mock_factory = _make_llm_mocks(guess)

        respx.get(_GEOCODE_BASE).mock(
            return_value=httpx.Response(
                200, json=_geocoder_response("Szczecin", 53.4285, 14.5528, geo_id=99)
            )
        )

        geocoder = Geocoder(model_factory=mock_factory)
        result = await geocoder.geocode("Szczecin")
        assert result is not None
        assert ":" not in result.id

    @respx.mock
    async def test_llm_failure_falls_back_to_deterministic(self) -> None:
        mock_factory = MagicMock()
        mock_chat = MagicMock()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
        mock_chat.with_structured_output.return_value = mock_structured
        mock_factory.create_chat_model.return_value = mock_chat

        respx.get(_GEOCODE_BASE).mock(
            return_value=httpx.Response(200, json=_geocoder_response("Poznań", 52.4064, 16.9252))
        )

        geocoder = Geocoder(model_factory=mock_factory)
        result = await geocoder.geocode("Poznań")
        assert result is not None
        assert result.name == "Poznań"
        assert not result.id.startswith("llm:")

    @respx.mock
    async def test_llm_normalizes_inflected_input(self) -> None:
        guess = LocationGuess(display_name="Lotnisko Modlin", search_query="Modlin lotnisko")
        mock_factory = _make_llm_mocks(guess)

        respx.get(_GEOCODE_BASE).mock(
            return_value=httpx.Response(
                200, json=_geocoder_response("Modlin", 52.45, 20.65, geo_id=88)
            )
        )

        geocoder = Geocoder(model_factory=mock_factory)
        result = await geocoder.geocode("na lotnisku Modlin")
        assert result is not None
        assert not result.id.startswith("llm:")


class TestNoLlmCoordinatesTrusted:
    async def test_location_guess_has_no_lat_lon_fields(self) -> None:
        guess = LocationGuess(display_name="Test", search_query="Test")
        assert not hasattr(guess, "lat")
        assert not hasattr(guess, "lon")

    @respx.mock
    async def test_no_llm_id_prefix_in_results(self) -> None:
        respx.get(_GEOCODE_BASE).mock(
            return_value=httpx.Response(
                200, json=_geocoder_response("Wrocław", 51.1079, 17.0385, geo_id=55)
            )
        )
        geocoder = Geocoder()
        result = await geocoder.geocode("Wrocław")
        assert result is not None
        assert not result.id.startswith("llm:")

    @respx.mock
    async def test_geocoder_with_llm_never_produces_llm_location_ids(self) -> None:
        guess = LocationGuess(display_name="Łódź", search_query="Łódź")
        mock_factory = _make_llm_mocks(guess)

        respx.get(_GEOCODE_BASE).mock(
            return_value=httpx.Response(
                200, json=_geocoder_response("Łódź", 51.7594, 19.4572, geo_id=33)
            )
        )

        geocoder = Geocoder(model_factory=mock_factory)
        result = await geocoder.geocode("w Łodzi")
        assert result is not None
        assert ":" not in result.id
        assert result.latitude == 51.7594
        assert result.longitude == 19.4572

    @respx.mock
    async def test_coordinates_always_from_deterministic_geocoder(self) -> None:
        guess = LocationGuess(display_name="Katowice", search_query="Katowice")
        mock_factory = _make_llm_mocks(guess)

        respx.get(_GEOCODE_BASE).mock(
            return_value=httpx.Response(
                200, json=_geocoder_response("Katowice", 50.2649, 19.0238, geo_id=12)
            )
        )

        geocoder = Geocoder(model_factory=mock_factory)
        result = await geocoder.geocode("Katowice")
        assert result is not None
        assert result.latitude == 50.2649
        assert result.longitude == 19.0238
        assert result.id == "12"
