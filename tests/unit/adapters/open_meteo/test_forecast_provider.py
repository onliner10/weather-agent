from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
import respx

from weather_agent.adapters.open_meteo.forecast_provider import (
    _PROVIDER_NAME,
    OpenMeteoDwdIconProvider,
)
from weather_agent.adapters.open_meteo.variable_map import OPEN_METEO_VARIABLE_MAP
from weather_agent.domain.errors import (
    WeatherProviderResponseError,
    WeatherProviderTimeoutError,
    WeatherProviderUnavailableError,
)
from weather_agent.domain.weather import (
    ForecastResolution,
    LocationRef,
    TimeRange,
    WeatherVariable,
)
from weather_agent.settings import OpenMeteoSettings

_WARSZAWA = LocationRef(
    id="warszawa-1", name="Warszawa", latitude=52.2297, longitude=21.0122
)
_NOW = datetime(2026, 4, 28, 12, 0, 0, tzinfo=UTC)
_LATER = datetime(2026, 4, 28, 13, 0, 0, tzinfo=UTC)
_TIME_RANGE = TimeRange(start=_NOW, end=_LATER)

_BASE_URL = "https://api.open-meteo.com/v1/forecast"

_ALL_VARIABLES = list(WeatherVariable)


def _make_hourly_response(
    times: list[str] | None = None,
    overrides: dict[str, list[float | int | None]] | None = None,
) -> dict[str, object]:
    if times is None:
        times = ["2026-04-28T12:00", "2026-04-28T13:00"]
    data: dict[str, object] = {
        "hourly": {
            "time": times,
            "temperature_2m": [15.0, 16.5],
            "apparent_temperature": [13.0, 14.5],
            "precipitation": [0.0, 0.5],
            "precipitation_probability": [10, 30],
            "rain": [0.0, 0.3],
            "snowfall": [0.0, 0.0],
            "cloud_cover": [25, 50],
            "wind_speed_10m": [3.5, 4.2],
            "wind_gusts_10m": [7.0, 8.5],
            "wind_direction_10m": [180, 200],
            "pressure_msl": [1013.2, 1012.8],
            "relative_humidity_2m": [65, 70],
            "weather_code": [1, 3],
        }
    }
    if overrides:
        hourly = data["hourly"]
        assert isinstance(hourly, dict)
        hourly.update(overrides)
    return data


class TestVariableMapping:
    def test_all_weather_variables_mapped(self) -> None:
        for var in WeatherVariable:
            assert var in OPEN_METEO_VARIABLE_MAP, f"{var} missing from variable map"

    def test_map_count_matches(self) -> None:
        assert len(OPEN_METEO_VARIABLE_MAP) == len(WeatherVariable)

    def test_only_requested_variables_in_api_params(self) -> None:
        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings())
        variables = [
            WeatherVariable.temperature_2m_c,
            WeatherVariable.precipitation_mm,
        ]
        result = provider._map_variables(variables)
        assert result == ["temperature_2m", "precipitation"]

    def test_unknown_variable_skipped(self) -> None:
        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings())
        result = provider._map_variables([])
        assert result == []


class TestSuccessfulForecastFetch:
    @respx.mock
    async def test_returns_forecast_result(self) -> None:
        raw = _make_hourly_response()
        respx.get(_BASE_URL).mock(return_value=httpx.Response(200, json=raw))

        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings())
        result = await provider.get_forecast(
            _WARSZAWA, _TIME_RANGE, _ALL_VARIABLES, ForecastResolution.hourly
        )

        assert result.provider == _PROVIDER_NAME
        assert result.model == "dwd-icon"
        assert result.location == _WARSZAWA
        assert len(result.points) == 2
        assert result.raw_payload == raw

    @respx.mock
    async def test_point_fields_populated(self) -> None:
        raw = _make_hourly_response()
        respx.get(_BASE_URL).mock(return_value=httpx.Response(200, json=raw))

        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings())
        result = await provider.get_forecast(
            _WARSZAWA, _TIME_RANGE, _ALL_VARIABLES, ForecastResolution.hourly
        )

        point = result.points[0]
        assert point.target_time == datetime(2026, 4, 28, 12, 0, 0, tzinfo=UTC)
        assert point.temperature_2m_c == 15.0
        assert point.apparent_temperature_c == 13.0
        assert point.precipitation_mm == 0.0
        assert point.precipitation_probability_pct == 10.0
        assert point.rain_mm == 0.0
        assert point.snowfall_cm == 0.0
        assert point.cloud_cover_pct == 25.0
        assert point.wind_speed_10m_ms == 3.5
        assert point.wind_gusts_10m_ms == 7.0
        assert point.wind_direction_10m_deg == 180.0
        assert point.pressure_msl_hpa == 1013.2
        assert point.relative_humidity_2m_pct == 65.0
        assert point.weather_code == "1"
        assert point.provider == _PROVIDER_NAME
        assert point.model == "dwd-icon"
        assert point.location_id == "warszawa-1"

    @respx.mock
    async def test_second_point(self) -> None:
        raw = _make_hourly_response()
        respx.get(_BASE_URL).mock(return_value=httpx.Response(200, json=raw))

        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings())
        result = await provider.get_forecast(
            _WARSZAWA, _TIME_RANGE, _ALL_VARIABLES, ForecastResolution.hourly
        )

        point = result.points[1]
        assert point.temperature_2m_c == 16.5
        assert point.wind_direction_10m_deg == 200.0

    @respx.mock
    async def test_raw_payload_preserved(self) -> None:
        raw = _make_hourly_response()
        respx.get(_BASE_URL).mock(return_value=httpx.Response(200, json=raw))

        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings())
        result = await provider.get_forecast(
            _WARSZAWA, _TIME_RANGE, _ALL_VARIABLES, ForecastResolution.hourly
        )

        assert result.raw_payload == raw
        for point in result.points:
            assert isinstance(point.raw_payload, dict)
            assert len(point.raw_payload) > 0


class TestUnitConversionPassthrough:
    @respx.mock
    async def test_temperature_passthrough_celsius(self) -> None:
        raw = _make_hourly_response(overrides={"temperature_2m": [25.3]})
        respx.get(_BASE_URL).mock(return_value=httpx.Response(200, json=raw))

        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings())
        result = await provider.get_forecast(
            _WARSZAWA, _TIME_RANGE, [WeatherVariable.temperature_2m_c], ForecastResolution.hourly
        )

        assert result.points[0].temperature_2m_c == 25.3

    @respx.mock
    async def test_wind_speed_passthrough_ms(self) -> None:
        raw = _make_hourly_response(overrides={"wind_speed_10m": [12.7]})
        respx.get(_BASE_URL).mock(return_value=httpx.Response(200, json=raw))

        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings())
        result = await provider.get_forecast(
            _WARSZAWA, _TIME_RANGE, [WeatherVariable.wind_speed_10m_ms], ForecastResolution.hourly
        )

        assert result.points[0].wind_speed_10m_ms == 12.7

    @respx.mock
    async def test_snowfall_passthrough_cm(self) -> None:
        raw = _make_hourly_response(overrides={"snowfall": [2.5]})
        respx.get(_BASE_URL).mock(return_value=httpx.Response(200, json=raw))

        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings())
        result = await provider.get_forecast(
            _WARSZAWA, _TIME_RANGE, [WeatherVariable.snowfall_cm], ForecastResolution.hourly
        )

        assert result.points[0].snowfall_cm == 2.5

    @respx.mock
    async def test_weather_code_as_string(self) -> None:
        raw = _make_hourly_response(overrides={"weather_code": [61]})
        respx.get(_BASE_URL).mock(return_value=httpx.Response(200, json=raw))

        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings())
        result = await provider.get_forecast(
            _WARSZAWA, _TIME_RANGE, [WeatherVariable.weather_code], ForecastResolution.hourly
        )

        assert result.points[0].weather_code == "61"

    @respx.mock
    async def test_pressure_passthrough_hpa(self) -> None:
        raw = _make_hourly_response(overrides={"pressure_msl": [1020.5]})
        respx.get(_BASE_URL).mock(return_value=httpx.Response(200, json=raw))

        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings())
        result = await provider.get_forecast(
            _WARSZAWA, _TIME_RANGE, [WeatherVariable.pressure_msl_hpa], ForecastResolution.hourly
        )

        assert result.points[0].pressure_msl_hpa == 1020.5


class TestTimeoutHandling:
    @respx.mock
    async def test_timeout_raises_provider_timeout_error(self) -> None:
        respx.get(_BASE_URL).mock(side_effect=httpx.ReadTimeout("timeout"))

        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings(timeout_seconds=1))
        with pytest.raises(WeatherProviderTimeoutError) as exc_info:
            await provider.get_forecast(
                _WARSZAWA, _TIME_RANGE, _ALL_VARIABLES, ForecastResolution.hourly
            )
        assert exc_info.value.provider == _PROVIDER_NAME

    @respx.mock
    async def test_connect_timeout_raises_provider_timeout_error(self) -> None:
        respx.get(_BASE_URL).mock(side_effect=httpx.ConnectTimeout("connect timeout"))

        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings(timeout_seconds=1))
        with pytest.raises(WeatherProviderTimeoutError):
            await provider.get_forecast(
                _WARSZAWA, _TIME_RANGE, _ALL_VARIABLES, ForecastResolution.hourly
            )


class TestHttpErrorHandling:
    @respx.mock
    async def test_server_error_raises_response_error(self) -> None:
        respx.get(_BASE_URL).mock(return_value=httpx.Response(500, text="Internal Server Error"))

        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings())
        with pytest.raises(WeatherProviderResponseError) as exc_info:
            await provider.get_forecast(
                _WARSZAWA, _TIME_RANGE, _ALL_VARIABLES, ForecastResolution.hourly
            )
        assert exc_info.value.provider == _PROVIDER_NAME
        assert "500" in exc_info.value.message

    @respx.mock
    async def test_client_error_raises_response_error(self) -> None:
        respx.get(_BASE_URL).mock(return_value=httpx.Response(400, text="Bad request"))

        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings())
        with pytest.raises(WeatherProviderResponseError) as exc_info:
            await provider.get_forecast(
                _WARSZAWA, _TIME_RANGE, _ALL_VARIABLES, ForecastResolution.hourly
            )
        assert "400" in exc_info.value.message

    @respx.mock
    async def test_invalid_json_raises_response_error(self) -> None:
        respx.get(_BASE_URL).mock(return_value=httpx.Response(200, text="not json"))

        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings())
        with pytest.raises(WeatherProviderResponseError) as exc_info:
            await provider.get_forecast(
                _WARSZAWA, _TIME_RANGE, _ALL_VARIABLES, ForecastResolution.hourly
            )
        assert "JSON" in exc_info.value.message


class TestConnectionError:
    @respx.mock
    async def test_connect_error_raises_unavailable_error(self) -> None:
        respx.get(_BASE_URL).mock(side_effect=httpx.ConnectError("connection refused"))

        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings())
        with pytest.raises(WeatherProviderUnavailableError) as exc_info:
            await provider.get_forecast(
                _WARSZAWA, _TIME_RANGE, _ALL_VARIABLES, ForecastResolution.hourly
            )
        assert exc_info.value.provider == _PROVIDER_NAME


class TestRetryBehaviour:
    @respx.mock
    async def test_retries_on_timeout_then_succeeds(self) -> None:
        raw = _make_hourly_response()
        route = respx.get(_BASE_URL)
        route.side_effect = [
            httpx.ReadTimeout("timeout"),
            httpx.Response(200, json=raw),
        ]

        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings(timeout_seconds=1))
        result = await provider.get_forecast(
            _WARSZAWA, _TIME_RANGE, _ALL_VARIABLES, ForecastResolution.hourly
        )
        assert len(result.points) == 2
        assert route.call_count == 2

    @respx.mock
    async def test_exhausts_retries_on_persistent_timeout(self) -> None:
        respx.get(_BASE_URL).mock(side_effect=httpx.ReadTimeout("timeout"))

        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings(timeout_seconds=1))
        with pytest.raises(WeatherProviderTimeoutError):
            await provider.get_forecast(
                _WARSZAWA, _TIME_RANGE, _ALL_VARIABLES, ForecastResolution.hourly
            )

    @respx.mock
    async def test_no_retry_on_response_error(self) -> None:
        route = respx.get(_BASE_URL)
        route.mock(return_value=httpx.Response(500, text="error"))

        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings())
        with pytest.raises(WeatherProviderResponseError):
            await provider.get_forecast(
                _WARSZAWA, _TIME_RANGE, _ALL_VARIABLES, ForecastResolution.hourly
            )
        assert route.call_count == 1


class TestBuildParams:
    def test_hourly_resolution(self) -> None:
        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings())
        vars_api = provider._map_variables([WeatherVariable.temperature_2m_c])
        params = provider._build_params(_WARSZAWA, _TIME_RANGE, vars_api, ForecastResolution.hourly)
        assert params["hourly"] == "temperature_2m"
        assert params["model"] == "dwd-icon"
        assert params["latitude"] == 52.2297
        assert params["longitude"] == 21.0122
        assert "start_date" in params
        assert "end_date" in params

    def test_fifteen_min_resolution(self) -> None:
        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings())
        vars_api = provider._map_variables([WeatherVariable.temperature_2m_c])
        params = provider._build_params(
            _WARSZAWA, _TIME_RANGE, vars_api, ForecastResolution.fifteen_min
        )
        assert params["minutely_15"] == "temperature_2m"
        assert "hourly" not in params


class TestEmptyResponse:
    @respx.mock
    async def test_no_time_data_returns_empty_points(self) -> None:
        raw: dict[str, object] = {"hourly": {}}
        respx.get(_BASE_URL).mock(return_value=httpx.Response(200, json=raw))

        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings())
        result = await provider.get_forecast(
            _WARSZAWA, _TIME_RANGE, _ALL_VARIABLES, ForecastResolution.hourly
        )
        assert result.points == []

    @respx.mock
    async def test_none_values_skipped(self) -> None:
        raw = _make_hourly_response(overrides={"temperature_2m": [None, 16.5]})
        respx.get(_BASE_URL).mock(return_value=httpx.Response(200, json=raw))

        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings())
        result = await provider.get_forecast(
            _WARSZAWA,
            _TIME_RANGE,
            [WeatherVariable.temperature_2m_c],
            ForecastResolution.hourly,
        )

        assert result.points[0].temperature_2m_c is None
        assert result.points[1].temperature_2m_c == 16.5


class TestFifteenMinResolution:
    @respx.mock
    async def test_minutely_15_parsing(self) -> None:
        raw: dict[str, object] = {
            "minutely_15": {
                "time": ["2026-04-28T12:00", "2026-04-28T12:15"],
                "temperature_2m": [14.0, 14.2],
            }
        }
        respx.get(_BASE_URL).mock(return_value=httpx.Response(200, json=raw))

        provider = OpenMeteoDwdIconProvider(OpenMeteoSettings())
        result = await provider.get_forecast(
            _WARSZAWA,
            _TIME_RANGE,
            [WeatherVariable.temperature_2m_c],
            ForecastResolution.fifteen_min,
        )
        assert len(result.points) == 2
        assert result.points[0].temperature_2m_c == 14.0
        assert result.points[1].temperature_2m_c == 14.2