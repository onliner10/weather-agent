from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
import respx

from weather_agent.adapters.imgw.geo import haversine_km
from weather_agent.adapters.imgw.synop_provider import (
    ImgwSynopProvider,
    _map_station_to_observation,
    _parse_observed_at,
    _parse_optional_float,
)
from weather_agent.domain.errors import (
    WeatherProviderResponseError,
    WeatherProviderTimeoutError,
    WeatherProviderUnavailableError,
)
from weather_agent.domain.weather import LocationRef, WeatherVariable
from weather_agent.settings import ImgwSettings

_WARSAW = LocationRef(id="warszawa-1", name="Warszawa", latitude=52.2297, longitude=21.0122)

_SYNOP_RESPONSE = [
    {
        "id_stacji": "12375",
        "stacja": "Warszawa",
        "data_pomiaru": "2026-04-28",
        "godzina_pomiaru": "11",
        "temperatura": "9.7",
        "predkosc_wiatru": "5",
        "kierunek_wiatru": "350",
        "wilgotnosc_wzgledna": "29.6",
        "suma_opadu": "0",
        "cisnienie": "1025.2",
    },
    {
        "id_stacji": "12465",
        "stacja": "Łódź",
        "data_pomiaru": "2026-04-28",
        "godzina_pomiaru": "11",
        "temperatura": "9.5",
        "predkosc_wiatru": "4",
        "kierunek_wiatru": "99",
        "wilgotnosc_wzgledna": "29.8",
        "suma_opadu": "0",
        "cisnienie": "1026",
    },
    {
        "id_stacji": "12295",
        "stacja": "Białystok",
        "data_pomiaru": "2026-04-28",
        "godzina_pomiaru": "11",
        "temperatura": "8.1",
        "predkosc_wiatru": "4",
        "kierunek_wiatru": "300",
        "wilgotnosc_wzgledna": "41.5",
        "suma_opadu": "0",
        "cisnienie": "1022.9",
    },
    {
        "id_stacji": "12650",
        "stacja": "Kasprowy Wierch",
        "data_pomiaru": "2026-04-28",
        "godzina_pomiaru": "11",
        "temperatura": "-1.4",
        "predkosc_wiatru": "3",
        "kierunek_wiatru": "360",
        "wilgotnosc_wzgledna": "74.6",
        "suma_opadu": "0",
        "cisnienie": None,
    },
]


class TestParseOptionalFloat:
    def test_valid_number(self) -> None:
        assert _parse_optional_float("9.7") == 9.7

    def test_negative_number(self) -> None:
        assert _parse_optional_float("-1.4") == -1.4

    def test_integer_string(self) -> None:
        assert _parse_optional_float("5") == 5.0

    def test_none_input(self) -> None:
        assert _parse_optional_float(None) is None

    def test_empty_string(self) -> None:
        assert _parse_optional_float("") is None

    def test_invalid_string(self) -> None:
        assert _parse_optional_float("abc") is None


class TestParseObservedAt:
    def test_valid_date_and_hour(self) -> None:
        result = _parse_observed_at("2026-04-28", "11")
        assert result is not None
        assert result.year == 2026
        assert result.month == 4
        assert result.day == 28
        assert result.hour == 11
        assert result.tzinfo == UTC

    def test_missing_date(self) -> None:
        assert _parse_observed_at(None, "11") is None

    def test_missing_hour(self) -> None:
        assert _parse_observed_at("2026-04-28", None) is None

    def test_invalid_format(self) -> None:
        assert _parse_observed_at("not-a-date", "11") is None


class TestMapStationToObservation:
    def test_full_mapping(self) -> None:
        fetched_at = datetime(2026, 4, 28, 12, 0, 0, tzinfo=UTC)
        station = _SYNOP_RESPONSE[0]
        point = _map_station_to_observation(station, fetched_at, 5.0)

        assert point.provider == "imgw_synop"
        assert point.station_id == "12375"
        assert point.station_name == "Warszawa"
        assert point.distance_km == 5.0
        assert point.temperature_c == 9.7
        assert point.wind_speed_ms == 5.0
        assert point.wind_direction_deg == 350.0
        assert point.humidity_pct == 29.6
        assert point.precipitation_mm == 0.0
        assert point.pressure_hpa == 1025.2
        assert point.observed_at == datetime(2026, 4, 28, 11, 0, 0, tzinfo=UTC)
        assert point.fetched_at == fetched_at
        assert point.raw_payload["id_stacji"] == "12375"

    def test_null_pressure(self) -> None:
        fetched_at = datetime(2026, 4, 28, 12, 0, 0, tzinfo=UTC)
        station = _SYNOP_RESPONSE[3]
        point = _map_station_to_observation(station, fetched_at, 300.0)

        assert point.pressure_hpa is None
        assert point.temperature_c == -1.4

    def test_preserves_raw_payload(self) -> None:
        fetched_at = datetime(2026, 4, 28, 12, 0, 0, tzinfo=UTC)
        station = _SYNOP_RESPONSE[0]
        point = _map_station_to_observation(station, fetched_at, 0.0)

        assert "temperatura" in point.raw_payload
        assert "stacja" in point.raw_payload


class TestHaversine:
    def test_same_point(self) -> None:
        assert haversine_km(52.0, 21.0, 52.0, 21.0) == pytest.approx(0.0, abs=1e-6)

    def test_known_distance_warsaw_lodz(self) -> None:
        warsaw_lat, warsaw_lon = 52.2297, 21.0122
        lodz_lat, lodz_lon = 51.7375, 19.4353
        dist = haversine_km(warsaw_lat, warsaw_lon, lodz_lat, lodz_lon)
        assert 110 < dist < 140

    def test_symmetry(self) -> None:
        assert haversine_km(52.0, 21.0, 51.0, 20.0) == pytest.approx(
            haversine_km(51.0, 20.0, 52.0, 21.0), abs=1e-6
        )


class TestImgwSynopProvider:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_and_filter_by_radius(self) -> None:
        respx.get(ImgwSettings().synop_base_url).mock(
            return_value=httpx.Response(200, json=_SYNOP_RESPONSE)
        )
        provider = ImgwSynopProvider()
        result = await provider.get_observations(
            _WARSAW, radius_km=30.0, variables=[WeatherVariable.temperature_2m_c]
        )

        assert result.provider == "imgw_synop"
        assert result.location == _WARSAW
        station_names = [p.station_name for p in result.points]
        assert "Warszawa" in station_names
        assert "Łódź" not in station_names

    @respx.mock
    @pytest.mark.asyncio
    async def test_large_radius_returns_many_stations(self) -> None:
        respx.get(ImgwSettings().synop_base_url).mock(
            return_value=httpx.Response(200, json=_SYNOP_RESPONSE)
        )
        provider = ImgwSynopProvider()
        result = await provider.get_observations(_WARSAW, radius_km=200.0, variables=[])

        assert len(result.points) >= 3
        station_names = [p.station_name for p in result.points]
        assert "Warszawa" in station_names

    @respx.mock
    @pytest.mark.asyncio
    async def test_points_sorted_by_distance(self) -> None:
        respx.get(ImgwSettings().synop_base_url).mock(
            return_value=httpx.Response(200, json=_SYNOP_RESPONSE)
        )
        provider = ImgwSynopProvider()
        result = await provider.get_observations(_WARSAW, radius_km=500.0, variables=[])

        distances = [p.distance_km for p in result.points]
        assert distances == sorted(distances)

    @respx.mock
    @pytest.mark.asyncio
    async def test_raw_payload_preserved(self) -> None:
        respx.get(ImgwSettings().synop_base_url).mock(
            return_value=httpx.Response(200, json=_SYNOP_RESPONSE)
        )
        provider = ImgwSynopProvider()
        result = await provider.get_observations(_WARSAW, radius_km=30.0, variables=[])

        assert "stations" in result.raw_payload
        assert isinstance(result.raw_payload["stations"], list)
        assert len(result.raw_payload["stations"]) == len(_SYNOP_RESPONSE)

    @respx.mock
    @pytest.mark.asyncio
    async def test_observation_point_fields(self) -> None:
        respx.get(ImgwSettings().synop_base_url).mock(
            return_value=httpx.Response(200, json=_SYNOP_RESPONSE)
        )
        provider = ImgwSynopProvider()
        result = await provider.get_observations(_WARSAW, radius_km=15.0, variables=[])

        warsaw = next(p for p in result.points if p.station_name == "Warszawa")
        assert warsaw.temperature_c == 9.7
        assert warsaw.wind_speed_ms == 5.0
        assert warsaw.wind_direction_deg == 350.0
        assert warsaw.pressure_hpa == 1025.2
        assert warsaw.humidity_pct == 29.6
        assert warsaw.precipitation_mm == 0.0
        assert warsaw.distance_km is not None
        assert warsaw.distance_km < 15.0

    @respx.mock
    @pytest.mark.asyncio
    async def test_timeout_raises(self) -> None:
        respx.get(ImgwSettings().synop_base_url).mock(side_effect=httpx.ReadTimeout("timeout"))
        provider = ImgwSynopProvider(ImgwSettings(timeout_seconds=1))

        with pytest.raises(WeatherProviderTimeoutError):
            await provider.get_observations(_WARSAW, radius_km=50.0, variables=[])

    @respx.mock
    @pytest.mark.asyncio
    async def test_connection_error_raises(self) -> None:
        respx.get(ImgwSettings().synop_base_url).mock(
            side_effect=httpx.ConnectError("connection failed")
        )
        provider = ImgwSynopProvider()

        with pytest.raises(WeatherProviderUnavailableError):
            await provider.get_observations(_WARSAW, radius_km=50.0, variables=[])

    @respx.mock
    @pytest.mark.asyncio
    async def test_http_error_status_raises(self) -> None:
        respx.get(ImgwSettings().synop_base_url).mock(return_value=httpx.Response(500))
        provider = ImgwSynopProvider()

        with pytest.raises(WeatherProviderResponseError):
            await provider.get_observations(_WARSAW, radius_km=50.0, variables=[])

    @respx.mock
    @pytest.mark.asyncio
    async def test_invalid_json_raises(self) -> None:
        respx.get(ImgwSettings().synop_base_url).mock(
            return_value=httpx.Response(200, text="not json")
        )
        provider = ImgwSynopProvider()

        with pytest.raises(WeatherProviderResponseError):
            await provider.get_observations(_WARSAW, radius_km=50.0, variables=[])

    @respx.mock
    @pytest.mark.asyncio
    async def test_empty_response_returns_no_points(self) -> None:
        respx.get(ImgwSettings().synop_base_url).mock(return_value=httpx.Response(200, json=[]))
        provider = ImgwSynopProvider()
        result = await provider.get_observations(_WARSAW, radius_km=50.0, variables=[])

        assert result.points == []
        assert result.provider == "imgw_synop"

    @respx.mock
    @pytest.mark.asyncio
    async def test_uses_custom_base_url(self) -> None:
        custom_url = "https://custom.imgw.example.com/api/data/synop"
        respx.get(custom_url).mock(return_value=httpx.Response(200, json=_SYNOP_RESPONSE))
        settings = ImgwSettings(synop_base_url=custom_url)
        provider = ImgwSynopProvider(settings=settings)
        result = await provider.get_observations(_WARSAW, radius_km=15.0, variables=[])

        assert len(result.points) == 1
        assert result.points[0].station_name == "Warszawa"

    @respx.mock
    @pytest.mark.asyncio
    async def test_stations_without_coordinates_are_excluded(self) -> None:
        response_with_unknown = [
            {
                "id_stacji": "99999",
                "stacja": "UnknownStation",
                "data_pomiaru": "2026-04-28",
                "godzina_pomiaru": "11",
                "temperatura": "5.0",
                "predkosc_wiatru": "2",
                "kierunek_wiatru": "180",
                "wilgotnosc_wzgledna": "50.0",
                "suma_opadu": "0",
                "cisnienie": "1013.0",
            },
            _SYNOP_RESPONSE[0],
        ]
        respx.get(ImgwSettings().synop_base_url).mock(
            return_value=httpx.Response(200, json=response_with_unknown)
        )
        provider = ImgwSynopProvider()
        result = await provider.get_observations(_WARSAW, radius_km=500.0, variables=[])

        station_ids = [p.station_id for p in result.points]
        assert "99999" not in station_ids
        assert "12375" in station_ids

    @respx.mock
    @pytest.mark.asyncio
    async def test_non_dict_entries_skipped(self) -> None:
        response_with_bad = ["not_a_dict", _SYNOP_RESPONSE[0]]
        respx.get(ImgwSettings().synop_base_url).mock(
            return_value=httpx.Response(200, json=response_with_bad)
        )
        provider = ImgwSynopProvider()
        result = await provider.get_observations(_WARSAW, radius_km=15.0, variables=[])

        assert len(result.points) == 1
        assert result.points[0].station_name == "Warszawa"

    @respx.mock
    @pytest.mark.asyncio
    async def test_non_list_response_raises(self) -> None:
        respx.get(ImgwSettings().synop_base_url).mock(
            return_value=httpx.Response(200, json={"error": "not a list"})
        )
        provider = ImgwSynopProvider()

        with pytest.raises(WeatherProviderResponseError):
            await provider.get_observations(_WARSAW, radius_km=50.0, variables=[])

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetched_at_is_utc(self) -> None:
        respx.get(ImgwSettings().synop_base_url).mock(
            return_value=httpx.Response(200, json=_SYNOP_RESPONSE)
        )
        provider = ImgwSynopProvider()
        result = await provider.get_observations(_WARSAW, radius_km=15.0, variables=[])

        assert result.fetched_at.tzinfo == UTC
