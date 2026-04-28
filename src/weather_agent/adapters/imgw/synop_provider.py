from __future__ import annotations

from datetime import UTC, datetime

import httpx

from weather_agent.adapters.imgw.geo import haversine_km
from weather_agent.adapters.imgw.stations import STATION_COORDINATES
from weather_agent.domain.errors import (
    WeatherProviderResponseError,
    WeatherProviderTimeoutError,
    WeatherProviderUnavailableError,
)
from weather_agent.domain.weather import (
    LocationRef,
    ObservationPoint,
    ObservationResult,
    WeatherVariable,
)
from weather_agent.settings import ImgwSettings

_PROVIDER = "imgw_synop"


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _parse_observed_at(date_str: str | None, hour_str: str | None) -> datetime | None:
    if not date_str or not hour_str:
        return None
    try:
        hour = int(hour_str)
        return datetime.strptime(date_str, "%Y-%m-%d").replace(
            hour=hour, minute=0, second=0, microsecond=0, tzinfo=UTC
        )
    except (ValueError, TypeError):
        return None


def _map_station_to_observation(
    station: dict[str, object],
    fetched_at: datetime,
    distance_km: float | None,
) -> ObservationPoint:
    station_id = str(station.get("id_stacji", ""))
    station_name = station.get("stacja")
    if station_name is not None:
        station_name = str(station_name)

    raw = {k: v for k, v in station.items()}

    temperatura = _parse_optional_float(station.get("temperatura"))  # type: ignore[arg-type]
    predkosc_wiatru_raw = _parse_optional_float(station.get("predkosc_wiatru"))  # type: ignore[arg-type]
    kierunek_wiatru = _parse_optional_float(station.get("kierunek_wiatru"))  # type: ignore[arg-type]
    wilgotnosc = _parse_optional_float(station.get("wilgotnosc_wzgledna"))  # type: ignore[arg-type]
    suma_opadu = _parse_optional_float(station.get("suma_opadu"))  # type: ignore[arg-type]
    cisnienie = _parse_optional_float(station.get("cisnienie"))  # type: ignore[arg-type]
    observed_at = _parse_observed_at(
        station.get("data_pomiaru"),  # type: ignore[arg-type]
        station.get("godzina_pomiaru"),  # type: ignore[arg-type]
    )

    return ObservationPoint(
        observed_at=observed_at or fetched_at,
        fetched_at=fetched_at,
        provider=_PROVIDER,
        station_id=station_id or None,
        station_name=station_name,
        distance_km=distance_km,
        temperature_c=temperatura,
        wind_speed_ms=predkosc_wiatru_raw,
        wind_direction_deg=kierunek_wiatru,
        pressure_hpa=cisnienie,
        humidity_pct=wilgotnosc,
        precipitation_mm=suma_opadu,
        raw_payload=raw,
    )


class ImgwSynopProvider:
    def __init__(self, settings: ImgwSettings | None = None) -> None:
        self._settings = settings or ImgwSettings()
        self._client = httpx.AsyncClient(
            timeout=self._settings.timeout_seconds,
        )

    async def get_observations(
        self,
        location: LocationRef,
        radius_km: float,
        variables: list[WeatherVariable],
    ) -> ObservationResult:
        fetched_at = datetime.now(tz=UTC)
        try:
            response = await self._client.get(self._settings.synop_base_url)
        except httpx.TimeoutException:
            raise WeatherProviderTimeoutError(_PROVIDER) from None
        except httpx.HTTPError:
            raise WeatherProviderUnavailableError(_PROVIDER) from None

        if response.status_code != 200:
            raise WeatherProviderResponseError(
                _PROVIDER, f"HTTP {response.status_code}"
            )

        try:
            stations: list[dict[str, object]] = response.json()
        except Exception as exc:
            raise WeatherProviderResponseError(_PROVIDER, f"Invalid JSON: {exc}") from exc

        if not isinstance(stations, list):
            raise WeatherProviderResponseError(
                _PROVIDER, "Expected a list of stations"
            )

        points: list[ObservationPoint] = []
        for station in stations:
            if not isinstance(station, dict):
                continue
            station_id = str(station.get("id_stacji", ""))
            coords = STATION_COORDINATES.get(station_id)
            distance: float | None = None
            if coords:
                distance = haversine_km(
                    location.latitude, location.longitude, coords[0], coords[1]
                )
                if distance > radius_km:
                    continue
            else:
                continue

            point = _map_station_to_observation(station, fetched_at, distance)
            points.append(point)

        points.sort(key=lambda p: p.distance_km if p.distance_km is not None else float("inf"))

        return ObservationResult(
            provider=_PROVIDER,
            location=location,
            fetched_at=fetched_at,
            points=points,
            raw_payload={"stations": stations},
        )