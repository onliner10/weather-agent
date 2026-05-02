from __future__ import annotations

from types import MethodType
from unittest.mock import AsyncMock, MagicMock

import httpx
import respx
from pydantic import SecretStr

from weather_agent.infrastructure.geocoder import Geocoder
from weather_agent.llm.contracts.geocoder import LocationGuess
from weather_agent.settings import GeocodingSettings

_BASE_URL = "https://eu1.locationiq.com/v1"
_SEARCH_URL = f"{_BASE_URL}/search"
_AUTOCOMPLETE_URL = f"{_BASE_URL}/autocomplete"


def _settings(*, use_autocomplete_fallback: bool = False) -> GeocodingSettings:
    return GeocodingSettings(
        api_key=SecretStr("locationiq-token"),
        use_autocomplete_fallback=use_autocomplete_fallback,
    )


def _geocoder_response(
    name: str,
    lat: float,
    lon: float,
    *,
    place_id: int = 1,
    display_name: str | None = None,
    category: str = "place",
    place_type: str = "city",
    address: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "place_id": place_id,
        "osm_type": "node",
        "osm_id": place_id * 10,
        "lat": str(lat),
        "lon": str(lon),
        "class": category,
        "type": place_type,
        "name": name,
        "display_name": display_name or f"{name}, Polska",
        "address": {"country_code": "pl", **(address or {})},
    }


def _make_llm_mocks(guess: LocationGuess) -> MagicMock:
    mock_factory = MagicMock()
    mock_chat = MagicMock()
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(return_value=guess)
    mock_chat.with_structured_output.return_value = mock_structured
    mock_factory.create_chat_model.return_value = mock_chat
    return mock_factory


def _stub_llm_guess(geocoder: Geocoder, guess: LocationGuess) -> None:
    async def ask_llm(_self: Geocoder, _name: str) -> LocationGuess:
        return guess

    geocoder._ask_llm = MethodType(ask_llm, geocoder)  # type: ignore[method-assign]


class TestLocationIqGeocode:
    @respx.mock
    async def test_basic_geocode(self) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=[
                    _geocoder_response(
                        "Gdańsk",
                        54.3520,
                        18.6466,
                        display_name="Gdańsk, województwo pomorskie, Polska",
                        address={"city": "Gdańsk"},
                    )
                ],
            )
        )
        geocoder = Geocoder(settings=_settings())

        result = await geocoder.geocode("Gdańsk")

        assert result is not None
        assert result.name == "Gdańsk"
        assert result.latitude == 54.3520
        assert result.longitude == 18.6466
        assert not result.id.startswith("llm:")

    @respx.mock
    async def test_sends_locationiq_search_parameters(self) -> None:
        route = respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=[
                    _geocoder_response(
                        "Warszawa",
                        52.2297,
                        21.0122,
                        address={"city": "Warszawa"},
                    )
                ],
            )
        )
        geocoder = Geocoder(settings=_settings())

        result = await geocoder.geocode("Warszawa")

        assert result is not None
        request = route.calls[0].request
        params = dict(request.url.params)
        assert params["key"] == "locationiq-token"
        assert params["q"] == "Warszawa"
        assert params["format"] == "json"
        assert params["countrycodes"] == "pl"
        assert params["accept-language"] == "pl"
        assert params["addressdetails"] == "1"
        assert params["normalizeaddress"] == "1"
        assert params["normalizecity"] == "1"
        assert params["limit"] == "5"

    async def test_geocode_returns_none_without_api_key(self) -> None:
        geocoder = Geocoder(settings=GeocodingSettings())

        result = await geocoder.geocode("Gdańsk")

        assert result is None

    @respx.mock
    async def test_gdansk_morena_resolves_to_district_result(self) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=[
                    _geocoder_response(
                        "Piecki-Migowo",
                        54.3578437,
                        18.5802681,
                        place_id=160509286,
                        display_name="Piecki-Migowo, Gdańsk, województwo pomorskie, Polska",
                        category="boundary",
                        place_type="administrative",
                        address={
                            "suburb": "Piecki-Migowo",
                            "city": "Gdańsk",
                            "state": "województwo pomorskie",
                        },
                    )
                ],
            )
        )
        geocoder = Geocoder(settings=_settings())

        result = await geocoder.geocode("Gdańsk Morena")

        assert result is not None
        assert result.name == "Piecki-Migowo"
        assert result.latitude == 54.3578437
        assert result.longitude == 18.5802681

    @respx.mock
    async def test_gdynia_chwarzno_resolves_to_quarter(self) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=[
                    _geocoder_response(
                        "Chwarzno",
                        54.5025321,
                        18.4575663,
                        place_id=173829074,
                        display_name=(
                            "Chwarzno, Chwarzno-Wiczlino, Gdynia, województwo pomorskie, Polska"
                        ),
                        category="place",
                        place_type="quarter",
                        address={
                            "quarter": "Chwarzno",
                            "suburb": "Chwarzno-Wiczlino",
                            "city": "Gdynia",
                        },
                    )
                ],
            )
        )
        geocoder = Geocoder(settings=_settings())

        result = await geocoder.geocode("Gdynia Chwarzno")

        assert result is not None
        assert result.name == "Chwarzno"
        assert result.latitude == 54.5025321
        assert result.longitude == 18.4575663

    @respx.mock
    async def test_poi_query_resolves_when_context_matches(self) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=[
                    _geocoder_response(
                        "Lądowisko modelarskie",
                        54.504,
                        18.421,
                        place_id=300,
                        display_name=(
                            "Lądowisko modelarskie, Chwarzno-Wiczlino, "
                            "Gdynia, województwo pomorskie, Polska"
                        ),
                        category="leisure",
                        place_type="pitch",
                        address={"suburb": "Chwarzno-Wiczlino", "city": "Gdynia"},
                    )
                ],
            )
        )
        geocoder = Geocoder(settings=_settings())

        result = await geocoder.geocode("Lądowisko modelarskie chwarzno")

        assert result is not None
        assert result.name == "Lądowisko modelarskie"
        assert result.latitude == 54.504
        assert result.longitude == 18.421

    @respx.mock
    async def test_same_name_poi_in_wrong_city_is_rejected(self) -> None:
        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=[
                    _geocoder_response(
                        "Lądowisko modelarskie",
                        49.9675887,
                        21.9574884,
                        place_id=301,
                        display_name=(
                            "Lądowisko modelarskie, Rzeszów, województwo podkarpackie, Polska"
                        ),
                        category="leisure",
                        place_type="pitch",
                        address={"city": "Rzeszów"},
                    )
                ],
            )
        )
        geocoder = Geocoder(settings=_settings())

        result = await geocoder.geocode("Lądowisko modelarskie chwarzno")

        assert result is None

    @respx.mock
    async def test_autocomplete_fallback_resolves_poi(self) -> None:
        respx.get(_SEARCH_URL).mock(return_value=httpx.Response(200, json=[]))
        autocomplete = respx.get(_AUTOCOMPLETE_URL).mock(
            return_value=httpx.Response(
                200,
                json=[
                    _geocoder_response(
                        "Lądowisko modelarskie",
                        54.504,
                        18.421,
                        place_id=302,
                        display_name="Lądowisko modelarskie, Chwarzno, Gdynia, Polska",
                        category="leisure",
                        place_type="pitch",
                        address={"quarter": "Chwarzno", "city": "Gdynia"},
                    )
                ],
            )
        )
        geocoder = Geocoder(settings=_settings(use_autocomplete_fallback=True))

        result = await geocoder.geocode("Lądowisko modelarskie chwarzno")

        assert result is not None
        assert autocomplete.called

    @respx.mock
    async def test_geocode_returns_none_on_empty_results(self) -> None:
        respx.get(_SEARCH_URL).mock(return_value=httpx.Response(200, json=[]))
        geocoder = Geocoder(settings=_settings())

        result = await geocoder.geocode("NonexistentPlace")

        assert result is None

    @respx.mock
    async def test_timeout_returns_none(self) -> None:
        respx.get(_SEARCH_URL).mock(side_effect=httpx.TimeoutException("timeout"))
        geocoder = Geocoder(settings=_settings())

        result = await geocoder.geocode("Gdańsk")

        assert result is None

    @respx.mock
    async def test_http_error_returns_none(self) -> None:
        respx.get(_SEARCH_URL).mock(return_value=httpx.Response(429, json={"error": "slow down"}))
        geocoder = Geocoder(settings=_settings())

        result = await geocoder.geocode("Gdańsk")

        assert result is None

    @respx.mock
    async def test_malformed_json_returns_none(self) -> None:
        respx.get(_SEARCH_URL).mock(return_value=httpx.Response(200, content=b"{"))
        geocoder = Geocoder(settings=_settings())

        result = await geocoder.geocode("Gdańsk")

        assert result is None


class TestGeocodeWithLlm:
    @respx.mock
    async def test_llm_search_query_routes_to_deterministic_geocoder(self) -> None:
        guess = LocationGuess(display_name="Gdańsk", search_query="Gdańsk")

        respx.get(_SEARCH_URL).mock(
            side_effect=[
                httpx.Response(200, json=[]),
                httpx.Response(
                    200,
                    json=[
                        _geocoder_response(
                            "Gdańsk",
                            54.3520,
                            18.6466,
                            place_id=42,
                            address={"city": "Gdańsk"},
                        )
                    ],
                ),
            ]
        )

        geocoder = Geocoder(settings=_settings(), model_factory=MagicMock())
        _stub_llm_guess(geocoder, guess)
        result = await geocoder.geocode("w Gdańsku")

        assert result is not None
        assert result.name == "Gdańsk"
        assert result.latitude == 54.3520
        assert result.longitude == 18.6466
        assert result.id == "42"

    @respx.mock
    async def test_llm_no_search_query_uses_display_name(self) -> None:
        guess = LocationGuess(display_name="Kraków")

        respx.get(_SEARCH_URL).mock(
            side_effect=[
                httpx.Response(200, json=[]),
                httpx.Response(
                    200,
                    json=[
                        _geocoder_response(
                            "Kraków",
                            50.0647,
                            19.9450,
                            place_id=7,
                            address={"city": "Kraków"},
                        )
                    ],
                ),
            ]
        )

        geocoder = Geocoder(settings=_settings(), model_factory=MagicMock())
        _stub_llm_guess(geocoder, guess)
        result = await geocoder.geocode("w Krakowie")

        assert result is not None
        assert result.name == "Kraków"
        assert not result.id.startswith("llm:")

    @respx.mock
    async def test_llm_failure_falls_back_to_deterministic(self) -> None:
        mock_factory = MagicMock()
        mock_chat = MagicMock()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
        mock_chat.with_structured_output.return_value = mock_structured
        mock_factory.create_chat_model.return_value = mock_chat

        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=[
                    _geocoder_response(
                        "Poznań",
                        52.4064,
                        16.9252,
                        address={"city": "Poznań"},
                    )
                ],
            )
        )

        geocoder = Geocoder(settings=_settings(), model_factory=mock_factory)
        result = await geocoder.geocode("Poznań")

        assert result is not None
        assert result.name == "Poznań"
        assert not result.id.startswith("llm:")


class TestNoLlmCoordinatesTrusted:
    async def test_location_guess_has_no_lat_lon_fields(self) -> None:
        guess = LocationGuess(display_name="Test", search_query="Test")
        assert not hasattr(guess, "lat")
        assert not hasattr(guess, "lon")

    @respx.mock
    async def test_geocoder_with_llm_never_produces_llm_location_ids(self) -> None:
        guess = LocationGuess(display_name="Łódź", search_query="Łódź")
        mock_factory = _make_llm_mocks(guess)

        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=[
                    _geocoder_response(
                        "Łódź",
                        51.7594,
                        19.4572,
                        place_id=33,
                        address={"city": "Łódź"},
                    )
                ],
            )
        )

        geocoder = Geocoder(settings=_settings(), model_factory=mock_factory)
        result = await geocoder.geocode("Łódź")

        assert result is not None
        assert ":" not in result.id
        assert result.latitude == 51.7594
        assert result.longitude == 19.4572

    @respx.mock
    async def test_coordinates_always_from_deterministic_geocoder(self) -> None:
        guess = LocationGuess(display_name="Katowice", search_query="Katowice")
        mock_factory = _make_llm_mocks(guess)

        respx.get(_SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=[
                    _geocoder_response(
                        "Katowice",
                        50.2649,
                        19.0238,
                        place_id=12,
                        address={"city": "Katowice"},
                    )
                ],
            )
        )

        geocoder = Geocoder(settings=_settings(), model_factory=mock_factory)
        result = await geocoder.geocode("Katowice")

        assert result is not None
        assert result.latitude == 50.2649
        assert result.longitude == 19.0238
        assert result.id == "12"
