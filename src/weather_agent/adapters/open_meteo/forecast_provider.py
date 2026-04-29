from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import httpx

from weather_agent.adapters.open_meteo.variable_map import OPEN_METEO_VARIABLE_MAP
from weather_agent.domain.errors import (
    WeatherProviderResponseError,
    WeatherProviderTimeoutError,
    WeatherProviderUnavailableError,
)
from weather_agent.domain.weather import (
    ForecastPoint,
    ForecastResolution,
    ForecastResult,
    LocationRef,
    TimeRange,
    WeatherVariable,
)

_PROVIDER_NAME = "open-meteo"
_DWD_ICON_MODEL = "dwd-icon"
_MAX_RETRIES = 3
_BASE_BACKOFF_SECONDS = 0.5

_FIELD_TO_NORMALIZED: dict[str, str] = {
    "temperature_2m": "temperature_2m_c",
    "apparent_temperature": "apparent_temperature_c",
    "precipitation": "precipitation_mm",
    "precipitation_probability": "precipitation_probability_pct",
    "rain": "rain_mm",
    "snowfall": "snowfall_cm",
    "cloud_cover": "cloud_cover_pct",
    "wind_speed_10m": "wind_speed_10m_ms",
    "wind_gusts_10m": "wind_gusts_10m_ms",
    "wind_direction_10m": "wind_direction_10m_deg",
    "pressure_msl": "pressure_msl_hpa",
    "relative_humidity_2m": "relative_humidity_2m_pct",
    "weather_code": "weather_code",
}


class OpenMeteoDwdIconProvider:
    def __init__(self, settings: object = None) -> None:
        if settings is None:
            from weather_agent.settings import OpenMeteoSettings

            settings = OpenMeteoSettings()
        self._base_url: str = getattr(
            settings, "base_url", "https://api.open-meteo.com/v1/forecast"
        )
        self._model: str = getattr(settings, "model", _DWD_ICON_MODEL)
        self._timeout_seconds: int = getattr(settings, "timeout_seconds", 15)

    async def get_forecast(
        self,
        location: LocationRef,
        time_range: TimeRange,
        variables: list[WeatherVariable],
        resolution: ForecastResolution,
    ) -> ForecastResult:
        api_vars = self._map_variables(variables)
        params: dict[str, str | int | float] = self._build_params(
            location, time_range, api_vars, resolution
        )
        raw_payload = await self._request_with_retry(params)
        fetched_at = datetime.now(UTC)
        points = self._parse_response(raw_payload, location, fetched_at)
        return ForecastResult(
            provider=_PROVIDER_NAME,
            model=self._model,
            location=location,
            fetched_at=fetched_at,
            points=points,
            raw_payload=raw_payload,
        )

    async def _request_with_retry(self, params: dict[str, str | int | float]) -> dict[str, object]:
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return await self._make_request(params)
            except WeatherProviderTimeoutError as exc:
                last_error = exc
            except WeatherProviderUnavailableError as exc:
                last_error = exc
            except WeatherProviderResponseError:
                raise
            if attempt < _MAX_RETRIES - 1:
                backoff = _BASE_BACKOFF_SECONDS * (2**attempt)
                await asyncio.sleep(backoff)
        raise last_error  # type: ignore[misc]

    async def _make_request(self, params: dict[str, str | int | float]) -> dict[str, object]:
        timeout = httpx.Timeout(self._timeout_seconds)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(self._base_url, params=params)
        except httpx.TimeoutException:
            raise WeatherProviderTimeoutError(_PROVIDER_NAME) from None
        except httpx.ConnectError:
            raise WeatherProviderUnavailableError(_PROVIDER_NAME) from None
        except httpx.HTTPError as exc:
            raise WeatherProviderUnavailableError(_PROVIDER_NAME, f"HTTP error: {exc}") from exc

        if response.status_code >= 500:
            raise WeatherProviderResponseError(
                _PROVIDER_NAME, f"Server error: {response.status_code}"
            )
        if response.status_code >= 400:
            raise WeatherProviderResponseError(
                _PROVIDER_NAME,
                f"Client error: {response.status_code} — {response.text[:200]}",
            )

        try:
            data = response.json()
        except Exception as exc:
            raise WeatherProviderResponseError(
                _PROVIDER_NAME, "Response is not valid JSON"
            ) from exc

        if not isinstance(data, dict):
            raise WeatherProviderResponseError(_PROVIDER_NAME, "Response is not a JSON object")

        return data

    @staticmethod
    def _map_variables(variables: list[WeatherVariable]) -> list[str]:
        result: list[str] = []
        for var in variables:
            api_name = OPEN_METEO_VARIABLE_MAP.get(var)
            if api_name is not None:
                result.append(api_name)
        return result

    def _build_params(
        self,
        location: LocationRef,
        time_range: TimeRange,
        api_vars: list[str],
        resolution: ForecastResolution,
    ) -> dict[str, str | int | float]:
        start_date = time_range.start.strftime("%Y-%m-%d")
        end_date = time_range.end.strftime("%Y-%m-%d")
        params: dict[str, str | int | float] = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "model": self._model,
            "wind_speed_unit": "ms",
        }
        if resolution == ForecastResolution.fifteen_min:
            params["minutely_15"] = ",".join(api_vars)
        else:
            params["hourly"] = ",".join(api_vars)
        return params

    @staticmethod
    def _parse_response(
        raw_payload: dict[str, object],
        location: LocationRef,
        fetched_at: datetime,
    ) -> list[ForecastPoint]:
        hourly = raw_payload.get("hourly")
        minutely_15 = raw_payload.get("minutely_15")
        time_data: list[str] | None = None
        data_arrays: dict[str, list[object]] = {}

        if isinstance(hourly, dict):
            time_data = hourly.get("time")
            for api_field, _normalized in _FIELD_TO_NORMALIZED.items():
                values = hourly.get(api_field)
                if isinstance(values, list):
                    data_arrays[api_field] = values
        elif isinstance(minutely_15, dict):
            time_data = minutely_15.get("time")
            for api_field, _normalized in _FIELD_TO_NORMALIZED.items():
                values = minutely_15.get(api_field)
                if isinstance(values, list):
                    data_arrays[api_field] = values

        if not time_data:
            return []

        points: list[ForecastPoint] = []
        for idx, time_str in enumerate(time_data):
            try:
                target_time = datetime.fromisoformat(time_str).replace(tzinfo=UTC)
            except (ValueError, TypeError):
                continue

            point_data: dict[str, object] = {}
            normalized: dict[str, float | str | None] = {}

            for api_field, norm_field in _FIELD_TO_NORMALIZED.items():
                arr = data_arrays.get(api_field)
                if arr is not None and idx < len(arr):
                    raw_val = arr[idx]
                    if raw_val is not None:
                        point_data[api_field] = raw_val
                        if api_field == "weather_code":
                            normalized[norm_field] = str(int(raw_val))  # type: ignore[call-overload]
                        elif api_field == "snowfall":
                            normalized[norm_field] = float(raw_val)  # type: ignore[arg-type]
                        else:
                            normalized[norm_field] = float(raw_val)  # type: ignore[arg-type]

            points.append(
                ForecastPoint(
                    target_time=target_time,
                    fetched_at=fetched_at,
                    provider=_PROVIDER_NAME,
                    model=_DWD_ICON_MODEL,
                    location_id=location.id,
                    temperature_2m_c=normalized.get("temperature_2m_c"),  # type: ignore[arg-type]
                    apparent_temperature_c=normalized.get("apparent_temperature_c"),  # type: ignore[arg-type]
                    precipitation_mm=normalized.get("precipitation_mm"),  # type: ignore[arg-type]
                    precipitation_probability_pct=normalized.get("precipitation_probability_pct"),  # type: ignore[arg-type]
                    rain_mm=normalized.get("rain_mm"),  # type: ignore[arg-type]
                    snowfall_cm=normalized.get("snowfall_cm"),  # type: ignore[arg-type]
                    cloud_cover_pct=normalized.get("cloud_cover_pct"),  # type: ignore[arg-type]
                    wind_speed_10m_ms=normalized.get("wind_speed_10m_ms"),  # type: ignore[arg-type]
                    wind_gusts_10m_ms=normalized.get("wind_gusts_10m_ms"),  # type: ignore[arg-type]
                    wind_direction_10m_deg=normalized.get("wind_direction_10m_deg"),  # type: ignore[arg-type]
                    pressure_msl_hpa=normalized.get("pressure_msl_hpa"),  # type: ignore[arg-type]
                    relative_humidity_2m_pct=normalized.get("relative_humidity_2m_pct"),  # type: ignore[arg-type]
                    weather_code=normalized.get("weather_code"),  # type: ignore[arg-type]
                    raw_payload=point_data,
                )
            )

        return points
