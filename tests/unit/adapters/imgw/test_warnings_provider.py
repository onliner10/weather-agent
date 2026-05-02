from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
import respx

from weather_agent.adapters.imgw.warnings_provider import (
    _HYDRO_CATEGORY,
    _HYDRO_PROVIDER_NAME,
    _HYDRO_SEVERITY_MAP,
    _METEO_CATEGORY_MAP,
    _METEO_PROVIDER_NAME,
    _METEO_SEVERITY_MAP,
    ImgwHydroWarningsProvider,
    ImgwMeteoWarningsProvider,
    _build_hydro_external_id,
    _build_meteo_external_id,
    _parse_hydro_warnings,
    _parse_imgw_datetime,
    _parse_meteo_warnings,
)
from weather_agent.domain.errors import (
    WeatherProviderResponseError,
    WeatherProviderTimeoutError,
    WeatherProviderUnavailableError,
)
from weather_agent.domain.weather import (
    LocationRef,
    TimeRange,
    WarningCategory,
    WarningSeverity,
)
from weather_agent.settings import ImgwSettings

_WARSZAWA = LocationRef(id="warszawa-1", name="Warszawa", latitude=52.2297, longitude=21.0122)
_NOW = datetime(2026, 4, 28, 12, 0, 0, tzinfo=UTC)
_LATER = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
_TIME_RANGE = TimeRange(start=_NOW, end=_LATER)

_METEO_URL = "https://danepubliczne.imgw.pl/api/data/warningsmeteo"
_HYDRO_URL = "https://danepubliczne.imgw.pl/api/data/warningshydro"


def _make_meteo_warning(
    id: str = "Sk20260428093004650",
    nazwa_zdarzenia: str = "Przymrozki",
    stopien: str = "1",
    tresc: str = "Spadek temperatury do -1°C",
    obowiazuje_od: str = "2026-04-28 23:00:00",
    obowiazuje_do: str = "2026-04-29 07:00:00",
    prawdopodobienstwo: str = "80",
    opublikowano: str = "2026-04-28 11:30:00",
    komentarz: str = "Brak.",
    biuro: str = "Centralne Biuro",
    teryt: list[str] | None = None,
) -> dict[str, object]:
    return {
        "id": id,
        "nazwa_zdarzenia": nazwa_zdarzenia,
        "stopien": stopien,
        "prawdopodobienstwo": prawdopodobienstwo,
        "obowiazuje_do": obowiazuje_do,
        "obowiazuje_od": obowiazuje_od,
        "opublikowano": opublikowano,
        "tresc": tresc,
        "komentarz": komentarz,
        "biuro": biuro,
        "teryt": teryt or ["0201", "0202"],
    }


def _make_hydro_warning(
    numer: str = "21",
    zdarzenie: str = "Susza hydrologiczna",
    stopień: str = "-1",
    przebieg: str = "Niskie przepływy wody",
    data_od: str = "2026-04-17 08:46:51",
    data_do: str = "2026-04-25 23:59:59",
    prawdopodobienstwo: str = "90",
    opublikowano: str = "2026-04-17 08:45:56",
    komentarz: str = "Brak.",
    biuro: str = "Biuro Prognoz",
    obszary: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "numer": numer,
        "zdarzenie": zdarzenie,
        "stopień": stopień,
        "przebieg": przebieg,
        "data_od": data_od,
        "data_do": data_do,
        "prawdopodobienstwo": prawdopodobienstwo,
        "opublikowano": opublikowano,
        "komentarz": komentarz,
        "biuro": biuro,
        "obszary": obszary or [],
    }


class TestParseImgwDatetime:
    def test_space_separated_format(self) -> None:
        result = _parse_imgw_datetime("2026-04-28 23:00:00")
        assert result == datetime(2026, 4, 28, 21, 0, 0, tzinfo=UTC)

    def test_iso_format(self) -> None:
        result = _parse_imgw_datetime("2026-04-28T23:00:00")
        assert result == datetime(2026, 4, 28, 21, 0, 0, tzinfo=UTC)

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_imgw_datetime("not-a-date")

    def test_date_only_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_imgw_datetime("2026-04-28")


class TestMeteoSeverityMapping:
    @pytest.mark.parametrize(
        ("stopien", "expected"),
        [
            ("1", WarningSeverity.low),
            ("2", WarningSeverity.moderate),
            ("3", WarningSeverity.high),
        ],
    )
    def test_known_severity_levels(self, stopien: str, expected: WarningSeverity) -> None:
        assert _METEO_SEVERITY_MAP[stopien] == expected

    def test_unknown_severity_is_none(self) -> None:
        assert _METEO_SEVERITY_MAP.get("0") is None

    def test_extreme_not_in_meteo_map(self) -> None:
        assert "4" not in _METEO_SEVERITY_MAP


class TestHydroSeverityMapping:
    @pytest.mark.parametrize(
        ("stopien", "expected"),
        [
            ("-1", WarningSeverity.low),
            ("0", WarningSeverity.low),
            ("1", WarningSeverity.moderate),
            ("2", WarningSeverity.high),
        ],
    )
    def test_known_severity_levels(self, stopien: str, expected: WarningSeverity) -> None:
        assert _HYDRO_SEVERITY_MAP[stopien] == expected


class TestMeteoCategoryMapping:
    def test_known_categories(self) -> None:
        assert _METEO_CATEGORY_MAP["Upały"] == WarningCategory.meteo
        assert _METEO_CATEGORY_MAP["Przymrozki"] == WarningCategory.meteo
        assert _METEO_CATEGORY_MAP["Silny wiatr"] == WarningCategory.meteo
        assert _METEO_CATEGORY_MAP["Burze"] == WarningCategory.meteo

    def test_unknown_category_defaults_to_meteo(self) -> None:
        result = _METEO_CATEGORY_MAP.get("Nieznany", WarningCategory.meteo)
        assert result == WarningCategory.meteo


class TestBuildMeteoExternalId:
    def test_uses_id_field(self) -> None:
        item = _make_meteo_warning(id="Sk20260428093004650")
        result = _build_meteo_external_id(item)
        assert result == "imgw-meteo:Sk20260428093004650"

    def test_fallback_when_no_id(self) -> None:
        item: dict[str, object] = {
            "nazwa_zdarzenia": "Burze",
            "stopien": "2",
            "obowiazuje_od": "2026-04-28 23:00:00",
        }
        result = _build_meteo_external_id(item)
        assert result == "imgw-meteo:Burze:2:2026-04-28 23:00:00"

    def test_id_takes_precedence(self) -> None:
        item = _make_meteo_warning(id="ABC123")
        result = _build_meteo_external_id(item)
        assert result == "imgw-meteo:ABC123"


class TestBuildHydroExternalId:
    def test_uses_numer_field(self) -> None:
        item = _make_hydro_warning(numer="21")
        result = _build_hydro_external_id(item)
        assert result == "imgw-hydro:21"

    def test_fallback_when_no_numer(self) -> None:
        item: dict[str, object] = {
            "zdarzenie": "Susza",
            "data_od": "2026-04-17 08:46:51",
        }
        result = _build_hydro_external_id(item)
        assert result == "imgw-hydro:Susza:2026-04-17 08:46:51"


class TestParseMeteoWarnings:
    def test_single_warning(self) -> None:
        raw = [_make_meteo_warning()]
        result = _parse_meteo_warnings(raw, _WARSZAWA)
        assert len(result) == 1

        w = result[0]
        assert w.provider == _METEO_PROVIDER_NAME
        assert w.external_id == "imgw-meteo:Sk20260428093004650"
        assert w.location_id == "warszawa-1"
        assert w.severity == WarningSeverity.low
        assert w.category == WarningCategory.meteo
        assert w.headline == "Przymrozki"
        assert w.description == "Spadek temperatury do -1°C"
        assert w.valid_from == datetime(2026, 4, 28, 21, 0, 0, tzinfo=UTC)
        assert w.valid_to == datetime(2026, 4, 29, 5, 0, 0, tzinfo=UTC)

    def test_raw_payload_preserved(self) -> None:
        raw_item = _make_meteo_warning()
        result = _parse_meteo_warnings([raw_item], _WARSZAWA)
        assert result[0].raw_payload == raw_item

    def test_multiple_warnings(self) -> None:
        raw = [
            _make_meteo_warning(id="A", stopien="1"),
            _make_meteo_warning(id="B", stopien="2"),
            _make_meteo_warning(id="C", stopien="3"),
        ]
        result = _parse_meteo_warnings(raw, _WARSZAWA)
        assert len(result) == 3
        assert result[0].severity == WarningSeverity.low
        assert result[1].severity == WarningSeverity.moderate
        assert result[2].severity == WarningSeverity.high

    def test_unknown_severity_becomes_none(self) -> None:
        raw = [_make_meteo_warning(stopien="0")]
        result = _parse_meteo_warnings(raw, _WARSZAWA)
        assert result[0].severity is None

    def test_unknown_category_defaults_to_meteo(self) -> None:
        raw = [_make_meteo_warning(nazwa_zdarzenia="Nieznane zjawisko")]
        result = _parse_meteo_warnings(raw, _WARSZAWA)
        assert result[0].category == WarningCategory.meteo
        assert result[0].headline == "Nieznane zjawisko"

    def test_invalid_date_skips_warning(self) -> None:
        raw = [_make_meteo_warning(obowiazuje_od="bad-date")]
        result = _parse_meteo_warnings(raw, _WARSZAWA)
        assert len(result) == 0

    def test_empty_list(self) -> None:
        result = _parse_meteo_warnings([], _WARSZAWA)
        assert result == []

    def test_severity_high(self) -> None:
        raw = [_make_meteo_warning(stopien="3")]
        result = _parse_meteo_warnings(raw, _WARSZAWA)
        assert result[0].severity == WarningSeverity.high

    def test_meteo_category_for_all_known_events(self) -> None:
        for nazwa in _METEO_CATEGORY_MAP:
            raw_item = _make_meteo_warning(nazwa_zdarzenia=nazwa)
            result = _parse_meteo_warnings([raw_item], _WARSZAWA)
            assert result[0].category == WarningCategory.meteo


class TestParseHydroWarnings:
    def test_single_warning(self) -> None:
        raw = [_make_hydro_warning()]
        result = _parse_hydro_warnings(raw, _WARSZAWA)
        assert len(result) == 1

        w = result[0]
        assert w.provider == _HYDRO_PROVIDER_NAME
        assert w.external_id == "imgw-hydro:21"
        assert w.location_id == "warszawa-1"
        assert w.severity == WarningSeverity.low
        assert w.category == WarningCategory.hydro
        assert w.headline == "Susza hydrologiczna"
        assert w.description == "Niskie przepływy wody"
        assert w.valid_from == datetime(2026, 4, 17, 6, 46, 51, tzinfo=UTC)
        assert w.valid_to == datetime(2026, 4, 25, 21, 59, 59, tzinfo=UTC)

    def test_raw_payload_preserved(self) -> None:
        raw_item = _make_hydro_warning()
        result = _parse_hydro_warnings([raw_item], _WARSZAWA)
        assert result[0].raw_payload == raw_item

    def test_severity_moderate(self) -> None:
        raw = [_make_hydro_warning(stopień="1")]
        result = _parse_hydro_warnings(raw, _WARSZAWA)
        assert result[0].severity == WarningSeverity.moderate

    def test_severity_high(self) -> None:
        raw = [_make_hydro_warning(stopień="2")]
        result = _parse_hydro_warnings(raw, _WARSZAWA)
        assert result[0].severity == WarningSeverity.high

    def test_always_hydro_category(self) -> None:
        raw = [_make_hydro_warning()]
        result = _parse_hydro_warnings(raw, _WARSZAWA)
        assert result[0].category == _HYDRO_CATEGORY

    def test_invalid_date_skips_warning(self) -> None:
        raw = [_make_hydro_warning(data_od="bad-date")]
        result = _parse_hydro_warnings(raw, _WARSZAWA)
        assert len(result) == 0

    def test_empty_list(self) -> None:
        result = _parse_hydro_warnings([], _WARSZAWA)
        assert result == []


class TestMeteoProviderIntegration:
    @respx.mock
    async def test_fetches_and_parses_warnings(self) -> None:
        raw_response = [_make_meteo_warning()]
        respx.get(_METEO_URL).mock(return_value=httpx.Response(200, json=raw_response))

        provider = ImgwMeteoWarningsProvider(ImgwSettings())
        result = await provider.get_warnings(_WARSZAWA, _TIME_RANGE)

        assert len(result) == 1
        assert result[0].provider == _METEO_PROVIDER_NAME
        assert result[0].severity == WarningSeverity.low
        assert result[0].category == WarningCategory.meteo

    @respx.mock
    async def test_multiple_warnings(self) -> None:
        raw_response = [
            _make_meteo_warning(id="A", stopien="1", nazwa_zdarzenia="Przymrozki"),
            _make_meteo_warning(id="B", stopien="3", nazwa_zdarzenia="Upały"),
        ]
        respx.get(_METEO_URL).mock(return_value=httpx.Response(200, json=raw_response))

        provider = ImgwMeteoWarningsProvider(ImgwSettings())
        result = await provider.get_warnings(_WARSZAWA, _TIME_RANGE)

        assert len(result) == 2
        assert result[0].severity == WarningSeverity.low
        assert result[1].severity == WarningSeverity.high

    @respx.mock
    async def test_empty_response(self) -> None:
        respx.get(_METEO_URL).mock(return_value=httpx.Response(200, json=[]))

        provider = ImgwMeteoWarningsProvider(ImgwSettings())
        result = await provider.get_warnings(_WARSZAWA, _TIME_RANGE)

        assert result == []

    @respx.mock
    async def test_raw_payload_preserved_in_full_flow(self) -> None:
        raw_item = _make_meteo_warning()
        respx.get(_METEO_URL).mock(return_value=httpx.Response(200, json=[raw_item]))

        provider = ImgwMeteoWarningsProvider(ImgwSettings())
        result = await provider.get_warnings(_WARSZAWA, _TIME_RANGE)

        assert result[0].raw_payload == raw_item


class TestHydroProviderIntegration:
    @respx.mock
    async def test_fetches_and_parses_warnings(self) -> None:
        raw_response = [_make_hydro_warning()]
        respx.get(_HYDRO_URL).mock(return_value=httpx.Response(200, json=raw_response))

        provider = ImgwHydroWarningsProvider(ImgwSettings())
        result = await provider.get_warnings(_WARSZAWA, _TIME_RANGE)

        assert len(result) == 1
        assert result[0].provider == _HYDRO_PROVIDER_NAME
        assert result[0].category == WarningCategory.hydro

    @respx.mock
    async def test_empty_response(self) -> None:
        respx.get(_HYDRO_URL).mock(return_value=httpx.Response(200, json=[]))

        provider = ImgwHydroWarningsProvider(ImgwSettings())
        result = await provider.get_warnings(_WARSZAWA, _TIME_RANGE)

        assert result == []


class TestMeteoProviderErrors:
    @respx.mock
    async def test_timeout_raises_provider_timeout_error(self) -> None:
        respx.get(_METEO_URL).mock(side_effect=httpx.ReadTimeout("timeout"))

        provider = ImgwMeteoWarningsProvider(ImgwSettings(timeout_seconds=1))
        with pytest.raises(WeatherProviderTimeoutError) as exc_info:
            await provider.get_warnings(_WARSZAWA, _TIME_RANGE)
        assert exc_info.value.provider == _METEO_PROVIDER_NAME

    @respx.mock
    async def test_connect_error_raises_unavailable_error(self) -> None:
        respx.get(_METEO_URL).mock(side_effect=httpx.ConnectError("connection refused"))

        provider = ImgwMeteoWarningsProvider(ImgwSettings())
        with pytest.raises(WeatherProviderUnavailableError) as exc_info:
            await provider.get_warnings(_WARSZAWA, _TIME_RANGE)
        assert exc_info.value.provider == _METEO_PROVIDER_NAME

    @respx.mock
    async def test_server_error_raises_response_error(self) -> None:
        respx.get(_METEO_URL).mock(return_value=httpx.Response(500, text="Internal Server Error"))

        provider = ImgwMeteoWarningsProvider(ImgwSettings())
        with pytest.raises(WeatherProviderResponseError) as exc_info:
            await provider.get_warnings(_WARSZAWA, _TIME_RANGE)
        assert "500" in exc_info.value.message

    @respx.mock
    async def test_client_error_raises_response_error(self) -> None:
        respx.get(_METEO_URL).mock(return_value=httpx.Response(400, text="Bad request"))

        provider = ImgwMeteoWarningsProvider(ImgwSettings())
        with pytest.raises(WeatherProviderResponseError) as exc_info:
            await provider.get_warnings(_WARSZAWA, _TIME_RANGE)
        assert "400" in exc_info.value.message

    @respx.mock
    async def test_invalid_json_raises_response_error(self) -> None:
        respx.get(_METEO_URL).mock(return_value=httpx.Response(200, text="not json"))

        provider = ImgwMeteoWarningsProvider(ImgwSettings())
        with pytest.raises(WeatherProviderResponseError) as exc_info:
            await provider.get_warnings(_WARSZAWA, _TIME_RANGE)
        assert "JSON" in exc_info.value.message

    @respx.mock
    async def test_non_list_response_raises_response_error(self) -> None:
        respx.get(_METEO_URL).mock(return_value=httpx.Response(200, json={"error": "bad"}))

        provider = ImgwMeteoWarningsProvider(ImgwSettings())
        with pytest.raises(WeatherProviderResponseError) as exc_info:
            await provider.get_warnings(_WARSZAWA, _TIME_RANGE)
        assert "array" in exc_info.value.message


class TestHydroProviderErrors:
    @respx.mock
    async def test_timeout_raises_provider_timeout_error(self) -> None:
        respx.get(_HYDRO_URL).mock(side_effect=httpx.ReadTimeout("timeout"))

        provider = ImgwHydroWarningsProvider(ImgwSettings(timeout_seconds=1))
        with pytest.raises(WeatherProviderTimeoutError) as exc_info:
            await provider.get_warnings(_WARSZAWA, _TIME_RANGE)
        assert exc_info.value.provider == _HYDRO_PROVIDER_NAME

    @respx.mock
    async def test_server_error_raises_response_error(self) -> None:
        respx.get(_HYDRO_URL).mock(return_value=httpx.Response(500, text="Error"))

        provider = ImgwHydroWarningsProvider(ImgwSettings())
        with pytest.raises(WeatherProviderResponseError):
            await provider.get_warnings(_WARSZAWA, _TIME_RANGE)


class TestRetryBehaviour:
    @respx.mock
    async def test_retries_on_timeout_then_succeeds(self) -> None:
        raw_response = [_make_meteo_warning()]
        route = respx.get(_METEO_URL)
        route.side_effect = [
            httpx.ReadTimeout("timeout"),
            httpx.Response(200, json=raw_response),
        ]

        provider = ImgwMeteoWarningsProvider(ImgwSettings(timeout_seconds=1))
        result = await provider.get_warnings(_WARSZAWA, _TIME_RANGE)
        assert len(result) == 1
        assert route.call_count == 2

    @respx.mock
    async def test_exhausts_retries_on_persistent_timeout(self) -> None:
        respx.get(_METEO_URL).mock(side_effect=httpx.ReadTimeout("timeout"))

        provider = ImgwMeteoWarningsProvider(ImgwSettings(timeout_seconds=1))
        with pytest.raises(WeatherProviderTimeoutError):
            await provider.get_warnings(_WARSZAWA, _TIME_RANGE)

    @respx.mock
    async def test_no_retry_on_response_error(self) -> None:
        route = respx.get(_METEO_URL)
        route.mock(return_value=httpx.Response(500, text="error"))

        provider = ImgwMeteoWarningsProvider(ImgwSettings())
        with pytest.raises(WeatherProviderResponseError):
            await provider.get_warnings(_WARSZAWA, _TIME_RANGE)
        assert route.call_count == 1


class TestProtocolConformance:
    def test_meteo_provider_satisfies_warning_provider(self) -> None:
        from weather_agent.domain.providers import WarningProvider

        provider: WarningProvider = ImgwMeteoWarningsProvider(ImgwSettings())
        assert hasattr(provider, "get_warnings")
        assert callable(provider.get_warnings)

    def test_hydro_provider_satisfies_warning_provider(self) -> None:
        from weather_agent.domain.providers import WarningProvider

        provider: WarningProvider = ImgwHydroWarningsProvider(ImgwSettings())
        assert hasattr(provider, "get_warnings")
        assert callable(provider.get_warnings)
