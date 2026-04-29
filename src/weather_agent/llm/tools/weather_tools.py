from __future__ import annotations

import json
import time as _time
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.tools import StructuredTool
from langsmith import traceable
from pydantic import BaseModel, Field

from weather_agent.domain.date_resolver import DateResolver, ResolvedTimeRange
from weather_agent.domain.errors import WeatherProviderError
from weather_agent.domain.locations import (
    LocationAliasConflictError,
    LocationCreate,
    LocationNameConflictError,
    LocationService,
)
from weather_agent.domain.providers import ForecastProvider, ObservationProvider
from weather_agent.domain.weather import (
    ForecastResolution,
    LocationRef,
    TimeRange,
    WeatherVariable,
)
from weather_agent.infrastructure.geocoder import Geocoder
from weather_agent.observability.logging import get_logger
from weather_agent.observability.metrics import (
    PROVIDER_REQUEST_DURATION_SECONDS,
    PROVIDER_REQUESTS_TOTAL,
    TOOL_CALL_DURATION_SECONDS,
    TOOL_CALLS_TOTAL,
)

logger = get_logger(__name__)

_WARSAW = ZoneInfo("Europe/Warsaw")

_WEATHER_CODE_MAP: dict[int, str] = {
    0: "bezchmurnie", 1: "przeważnie czysto", 2: "częściowe zachmurzenie",
    3: "zachmurzenie", 45: "mgła", 48: "osadzająca mgła",
    51: "delikatna mżawka", 53: "umiarkowana mżawka", 55: "gęsta mżawka",
    56: "zamarzająca mżawka", 57: "gęsta zamarzająca mżawka",
    61: "niewielki deszcz", 63: "umiarkowany deszcz", 65: "silny deszcz",
    66: "zamarzający deszcz", 67: "silny zamarzający deszcz",
    71: "niewielki śnieg", 73: "umiarkowany śnieg", 75: "silny śnieg",
    77: "ziarna śnieżne", 80: "niewielki deszcz nawalny",
    81: "umiarkowany deszcz nawalny", 82: "silny deszcz nawalny",
    85: "niewielki śnieg nawalny", 86: "silny śnieg nawalny",
    95: "burza", 96: "burza z niewielkim gradem", 99: "burza z silnym gradem",
}


def _weather_code_description(code: str | None) -> str | None:
    if code is None:
        return None
    try:
        return _WEATHER_CODE_MAP.get(int(code))
    except (ValueError, TypeError):
        return None


def _format_point(p: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"time": str(p.target_time)}
    for attr in (
        "temperature_2m_c", "apparent_temperature_c", "precipitation_mm",
        "precipitation_probability_pct", "rain_mm", "snowfall_cm",
        "cloud_cover_pct", "wind_speed_10m_ms", "wind_gusts_10m_ms",
        "wind_direction_10m_deg", "pressure_msl_hpa",
        "relative_humidity_2m_pct", "weather_code",
    ):
        v = getattr(p, attr, None)
        if v is not None:
            d[attr] = v
    return d


def _format_observation_point(p: Any) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for attr in (
        "observed_at", "station_name", "distance_km", "temperature_c",
        "wind_speed_ms", "wind_direction_deg", "pressure_hpa",
        "humidity_pct", "precipitation_mm",
    ):
        v = getattr(p, attr, None)
        if v is not None:
            d[attr] = str(v) if hasattr(v, "isoformat") else v
    return d


class GetForecastArgs(BaseModel):
    location_name: str = Field(description="Nazwa miejscowości (np. Gdańsk, Chwarzno)")
    start_date: str = Field(description="Data początkowa w formacie yyyy-mm-dd")
    end_date: str = Field(description="Końcowa data yyyy-mm-dd. Dla jednego dnia powtórz.")
    variables: list[str] = Field(
        default_factory=list,
        description=(
            "Lista zmiennych pogodowych do pobrania. Dostępne: "
            "temperature_2m_c, apparent_temperature_c, precipitation_mm, "
            "precipitation_probability_pct, rain_mm, snowfall_cm, cloud_cover_pct, "
            "wind_speed_10m_ms, wind_gusts_10m_ms, wind_direction_10m_deg, "
            "pressure_msl_hpa, relative_humidity_2m_pct, weather_code"
        ),
    )


class GetObservationsArgs(BaseModel):
    location_name: str = Field(description="Nazwa miejscowości")


class SaveLocationArgs(BaseModel):
    location_name: str = Field(description="Adres lub nazwa miejscowości do zapisania")
    alias: str = Field(default="", description="Opcjonalny alias, np. 'dom', 'praca'")


class WeatherToolbox:
    def __init__(
        self,
        forecast_provider: ForecastProvider,
        observation_provider: ObservationProvider | None,
        geocoder: Geocoder,
        date_resolver: DateResolver,
        location_service: LocationService | None,
        user_id: int,
    ) -> None:
        self.forecast_provider = forecast_provider
        self.observation_provider = observation_provider
        self.geocoder = geocoder
        self.date_resolver = date_resolver
        self.location_service = location_service
        self.user_id = user_id

    async def _resolve_location(self, name: str) -> LocationRef | None:
        try:
            if self.location_service is not None:
                resolved = await self.location_service.resolve_location(name, self.user_id)
                if resolved is not None:
                    return resolved
            return await self.geocoder.geocode(name)
        except Exception:
            logger.exception("resolve_location_failed", location_name=name, user_id=self.user_id)
            return None

    @traceable(run_type="tool")
    async def get_forecast(self, location_name: str, start_date: str, end_date: str, variables: list[str] | None = None) -> str:
        TOOL_CALLS_TOTAL.labels(tool="get_forecast").inc()
        start = _time.perf_counter()
        try:
            return await self._execute_get_forecast(location_name, start_date, end_date, variables or [])
        finally:
            TOOL_CALL_DURATION_SECONDS.labels(tool="get_forecast").observe(_time.perf_counter() - start)

    async def _execute_get_forecast(self, location_name: str, start_date_str: str, end_date_str: str, variable_names: list[str]) -> str:
        location = await self._resolve_location(location_name)
        if location is None:
            return json.dumps({"error": f"Nie znaleziono lokalizacji: {location_name}"}, ensure_ascii=False)

        try:
            s = date.fromisoformat(start_date_str)
            e = date.fromisoformat(end_date_str)
            start_dt = datetime(s.year, s.month, s.day, tzinfo=_WARSAW)
            end_dt = datetime(e.year, e.month, e.day, 23, 59, tzinfo=_WARSAW)
            if start_dt > end_dt:
                return json.dumps({"error": "start_date nie może być późniejsza niż end_date"}, ensure_ascii=False)
            time_range = ResolvedTimeRange(start=start_dt, end=end_dt, explanation=f"{start_date_str} – {end_date_str}")
        except ValueError:
            return json.dumps({"error": "Nieprawidłowy format daty. Użyj yyyy-mm-dd."}, ensure_ascii=False)

        variables = []
        for vn in variable_names:
            try:
                variables.append(WeatherVariable(vn))
            except ValueError:
                pass
        if not variables:
            variables = list(WeatherVariable)

        tr = TimeRange(start=time_range.start, end=time_range.end)
        provider_name = getattr(self.forecast_provider, "provider", "unknown")
        start = _time.perf_counter()
        try:
            forecast = await self.forecast_provider.get_forecast(
                location=location, time_range=tr, variables=variables,
                resolution=ForecastResolution.hourly,
            )
            PROVIDER_REQUESTS_TOTAL.labels(provider=provider_name, outcome="success").inc()
        except WeatherProviderError as exc:
            PROVIDER_REQUESTS_TOTAL.labels(provider=getattr(exc, "provider", provider_name), outcome="failure").inc()
            return json.dumps({"error": f"Błąd dostawcy prognozy ({exc.provider}): {exc.message}"}, ensure_ascii=False)
        except Exception:
            PROVIDER_REQUESTS_TOTAL.labels(provider=provider_name, outcome="failure").inc()
            return json.dumps({"error": "Błąd podczas pobierania prognozy. Spróbuj ponownie."}, ensure_ascii=False)
        finally:
            PROVIDER_REQUEST_DURATION_SECONDS.labels(provider=provider_name).observe(_time.perf_counter() - start)

        points_data = [_format_point(p) for p in forecast.points]
        result = {"location": location.name, "time_range": time_range.explanation, "forecast_points": points_data, "provider": forecast.provider, "model": forecast.model}
        return json.dumps(result, ensure_ascii=False, default=str)

    @traceable(run_type="tool")
    async def get_observations(self, location_name: str) -> str:
        TOOL_CALLS_TOTAL.labels(tool="get_observations").inc()
        start = _time.perf_counter()
        try:
            return await self._execute_get_observations(location_name)
        finally:
            TOOL_CALL_DURATION_SECONDS.labels(tool="get_observations").observe(_time.perf_counter() - start)

    async def _execute_get_observations(self, location_name: str) -> str:
        if self.observation_provider is None:
            return json.dumps({"error": "Obserwacje niedostępne"})

        location = await self._resolve_location(location_name)
        if location is None:
            return json.dumps({"error": f"Nie znaleziono lokalizacji: {location_name}"}, ensure_ascii=False)

        provider_name = getattr(self.observation_provider, "provider", "unknown")
        start = _time.perf_counter()
        try:
            obs = await self.observation_provider.get_observations(
                location=location, radius_km=50.0, variables=list(WeatherVariable),
            )
            PROVIDER_REQUESTS_TOTAL.labels(provider=provider_name, outcome="success").inc()
        except WeatherProviderError as exc:
            PROVIDER_REQUESTS_TOTAL.labels(provider=getattr(exc, "provider", provider_name), outcome="failure").inc()
            return json.dumps({"error": f"Błąd dostawcy obserwacji: {exc.message}"}, ensure_ascii=False)
        except Exception:
            PROVIDER_REQUESTS_TOTAL.labels(provider=provider_name, outcome="failure").inc()
            return json.dumps({"error": "Błąd podczas pobierania obserwacji. Spróbuj ponownie."}, ensure_ascii=False)
        finally:
            PROVIDER_REQUEST_DURATION_SECONDS.labels(provider=provider_name).observe(_time.perf_counter() - start)

        points_data = [_format_observation_point(p) for p in obs.points]
        return json.dumps({"location": location.name, "observations": points_data}, ensure_ascii=False, default=str)

    @traceable(run_type="tool")
    async def save_location(self, location_name: str, alias: str = "") -> str:
        TOOL_CALLS_TOTAL.labels(tool="save_location").inc()
        start = _time.perf_counter()
        try:
            return await self._execute_save_location(location_name, alias)
        finally:
            TOOL_CALL_DURATION_SECONDS.labels(tool="save_location").observe(_time.perf_counter() - start)

    async def _execute_save_location(self, location_name: str, alias: str) -> str:
        if self.location_service is None:
            return json.dumps({"error": "Usługa lokalizacji jest niedostępna."})

        if not location_name.strip():
            return json.dumps({"error": "Podaj nazwę lokalizacji do zapisania."})

        if self.geocoder is None:
            return json.dumps({"error": "Geokoder jest niedostępny."})

        resolved = await self.geocoder.geocode(location_name)
        if resolved is None:
            return json.dumps({"error": f"Nie udało się rozpoznać lokalizacji „{location_name}”."}, ensure_ascii=False)

        try:
            aliases = [alias] if alias else []
            await self.location_service.create_location(
                self.user_id,
                LocationCreate(name=location_name, aliases=aliases, latitude=resolved.latitude, longitude=resolved.longitude),
            )
            msg = f"Zapamiętałem lokalizację: {location_name}"
            if alias:
                msg += f" (alias: {alias})"
            return json.dumps({"success": msg}, ensure_ascii=False)
        except (LocationAliasConflictError, LocationNameConflictError):
            return json.dumps({"error": "Masz już zapisaną lokalizację o tej nazwie lub aliasie."}, ensure_ascii=False)
        except Exception as exc:
            logger.exception("save_location_failed", user_id=self.user_id, location_name=location_name)
            return json.dumps({"error": f"Błąd podczas zapisywania lokalizacji: {exc}"}, ensure_ascii=False)

    def to_langchain_tools(self) -> list:
        return [
            StructuredTool.from_function(
                coroutine=self.get_forecast,
                name="get_forecast",
                description=(
                    "Pobierz prognozę pogody dla lokalizacji i zakresu dat."
                    " Zwraca godzinowe dane: temperatura, opady, wiatr, zachmurzenie itp."
                    " Daty podawaj jako yyyy-mm-dd w strefie Europe/Warsaw."
                ),
                args_schema=GetForecastArgs,
            ),
            StructuredTool.from_function(
                coroutine=self.get_observations,
                name="get_observations",
                description=(
                    "Pobierz aktualne obserwacje ze stacji meteorologicznych"
                    " wokół lokalizacji (ostatni pomiar)."
                ),
                args_schema=GetObservationsArgs,
            ),
            StructuredTool.from_function(
                coroutine=self.save_location,
                name="save_location",
                description=(
                    "Zapisz lokalizację użytkownika (np. dom, praca) pod konkretną nazwą lub adresem."
                    " Jeśli użytkownik prosi o 'zapamiętanie lokalizacji domowej', ustaw alias na 'dom'."
                ),
                args_schema=SaveLocationArgs,
            ),
        ]