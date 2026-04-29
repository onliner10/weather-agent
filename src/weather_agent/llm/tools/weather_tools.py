from __future__ import annotations

import time as _time
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.tools import BaseTool, StructuredTool
from langsmith import traceable
from pydantic import BaseModel, Field

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
    0: "bezchmurnie",
    1: "przeważnie czysto",
    2: "częściowe zachmurzenie",
    3: "zachmurzenie",
    45: "mgła",
    48: "osadzająca mgła",
    51: "delikatna mżawka",
    53: "umiarkowana mżawka",
    55: "gęsta mżawka",
    56: "zamarzająca mżawka",
    57: "gęsta zamarzająca mżawka",
    61: "niewielki deszcz",
    63: "umiarkowany deszcz",
    65: "silny deszcz",
    66: "zamarzający deszcz",
    67: "silny zamarzający deszcz",
    71: "niewielki śnieg",
    73: "umiarkowany śnieg",
    75: "silny śnieg",
    77: "ziarna śnieżne",
    80: "niewielki deszcz nawalny",
    81: "umiarkowany deszcz nawalny",
    82: "silny deszcz nawalny",
    85: "niewielki śnieg nawalny",
    86: "silny śnieg nawalny",
    95: "burza",
    96: "burza z niewielkim gradem",
    99: "burza z silnym gradem",
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
        "temperature_2m_c",
        "apparent_temperature_c",
        "precipitation_mm",
        "precipitation_probability_pct",
        "rain_mm",
        "snowfall_cm",
        "cloud_cover_pct",
        "wind_speed_10m_ms",
        "wind_gusts_10m_ms",
        "wind_direction_10m_deg",
        "pressure_msl_hpa",
        "relative_humidity_2m_pct",
        "weather_code",
    ):
        v = getattr(p, attr, None)
        if v is not None:
            d[attr] = v
    return d


def _format_observation_point(p: Any) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for attr in (
        "observed_at",
        "station_name",
        "distance_km",
        "temperature_c",
        "wind_speed_ms",
        "wind_direction_deg",
        "pressure_hpa",
        "humidity_pct",
        "precipitation_mm",
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


class ForecastToolResult(BaseModel):
    location: str | None = None
    time_range: str | None = None
    forecast_points: list[dict[str, Any]] | None = None
    provider: str | None = None
    model: str | None = None
    error: str | None = None


class GetObservationsArgs(BaseModel):
    location_name: str = Field(description="Nazwa miejscowości")


class ObservationToolResult(BaseModel):
    location: str | None = None
    observations: list[dict[str, Any]] | None = None
    error: str | None = None


class SaveLocationArgs(BaseModel):
    location_name: str = Field(description="Adres lub nazwa miejscowości do zapisania")
    alias: str = Field(default="", description="Opcjonalny alias, np. 'dom', 'praca'")


class SaveLocationToolResult(BaseModel):
    success: str | None = None
    error: str | None = None


class ListLocationsArgs(BaseModel):
    include_disabled: bool = Field(
        default=False,
        description="Czy uwzględnić wyłączone lokalizacje (domyślnie nie)",
    )


class ListLocationsToolResult(BaseModel):
    locations: list[dict[str, Any]] | None = None
    count: int = 0
    error: str | None = None


class WeatherToolbox:
    def __init__(
        self,
        forecast_provider: ForecastProvider,
        observation_provider: ObservationProvider | None,
        geocoder: Geocoder,
        location_service: LocationService | None,
        user_id: int,
    ) -> None:
        self.forecast_provider = forecast_provider
        self.observation_provider = observation_provider
        self.geocoder = geocoder
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
    async def get_forecast(
        self,
        location_name: str,
        start_date: str,
        end_date: str,
        variables: list[str] | None = None,
    ) -> ForecastToolResult:
        TOOL_CALLS_TOTAL.labels(tool="get_forecast").inc()
        start = _time.perf_counter()
        try:
            return await self._execute_get_forecast(
                location_name, start_date, end_date, variables or []
            )
        finally:
            TOOL_CALL_DURATION_SECONDS.labels(tool="get_forecast").observe(
                _time.perf_counter() - start
            )

    async def _execute_get_forecast(
        self,
        location_name: str,
        start_date_str: str,
        end_date_str: str,
        variable_names: list[str],
    ) -> ForecastToolResult:
        location = await self._resolve_location(location_name)
        if location is None:
            return ForecastToolResult(error=f"Nie znaleziono lokalizacji: {location_name}")

        try:
            s = date.fromisoformat(start_date_str)
            e = date.fromisoformat(end_date_str)
            start_dt = datetime(s.year, s.month, s.day, tzinfo=_WARSAW)
            end_dt = datetime(e.year, e.month, e.day, 23, 59, tzinfo=_WARSAW)
            if start_dt > end_dt:
                return ForecastToolResult(error="start_date nie może być późniejsza niż end_date")
            time_range = TimeRange(start=start_dt, end=end_dt)
            time_range_explanation = f"{start_date_str} – {end_date_str}"
        except ValueError:
            return ForecastToolResult(error="Nieprawidłowy format daty. Użyj yyyy-mm-dd.")

        variables = []
        for vn in variable_names:
            try:
                variables.append(WeatherVariable(vn))
            except ValueError:
                pass
        if not variables:
            variables = list(WeatherVariable)

        provider_name = getattr(self.forecast_provider, "provider", "unknown")
        start = _time.perf_counter()
        try:
            forecast = await self.forecast_provider.get_forecast(
                location=location,
                time_range=time_range,
                variables=variables,
                resolution=ForecastResolution.hourly,
            )
            PROVIDER_REQUESTS_TOTAL.labels(provider=provider_name, outcome="success").inc()
        except WeatherProviderError as exc:
            PROVIDER_REQUESTS_TOTAL.labels(
                provider=getattr(exc, "provider", provider_name), outcome="failure"
            ).inc()
            return ForecastToolResult(
                error=f"Błąd dostawcy prognozy ({exc.provider}): {exc.message}"
            )
        except Exception:
            PROVIDER_REQUESTS_TOTAL.labels(provider=provider_name, outcome="failure").inc()
            return ForecastToolResult(error="Błąd podczas pobierania prognozy. Spróbuj ponownie.")
        finally:
            PROVIDER_REQUEST_DURATION_SECONDS.labels(provider=provider_name).observe(
                _time.perf_counter() - start
            )

        points_data = [_format_point(p) for p in forecast.points]
        return ForecastToolResult(
            location=location.name,
            time_range=time_range_explanation,
            forecast_points=points_data,
            provider=forecast.provider,
            model=forecast.model,
        )

    @traceable(run_type="tool")
    async def get_observations(self, location_name: str) -> ObservationToolResult:
        TOOL_CALLS_TOTAL.labels(tool="get_observations").inc()
        start = _time.perf_counter()
        try:
            return await self._execute_get_observations(location_name)
        finally:
            TOOL_CALL_DURATION_SECONDS.labels(tool="get_observations").observe(
                _time.perf_counter() - start
            )

    async def _execute_get_observations(self, location_name: str) -> ObservationToolResult:
        if self.observation_provider is None:
            return ObservationToolResult(error="Obserwacje niedostępne")

        location = await self._resolve_location(location_name)
        if location is None:
            return ObservationToolResult(error=f"Nie znaleziono lokalizacji: {location_name}")

        provider_name = getattr(self.observation_provider, "provider", "unknown")
        start = _time.perf_counter()
        try:
            obs = await self.observation_provider.get_observations(
                location=location,
                radius_km=50.0,
                variables=list(WeatherVariable),
            )
            PROVIDER_REQUESTS_TOTAL.labels(provider=provider_name, outcome="success").inc()
        except WeatherProviderError as exc:
            PROVIDER_REQUESTS_TOTAL.labels(
                provider=getattr(exc, "provider", provider_name), outcome="failure"
            ).inc()
            return ObservationToolResult(error=f"Błąd dostawcy obserwacji: {exc.message}")
        except Exception:
            PROVIDER_REQUESTS_TOTAL.labels(provider=provider_name, outcome="failure").inc()
            return ObservationToolResult(
                error="Błąd podczas pobierania obserwacji. Spróbuj ponownie."
            )
        finally:
            PROVIDER_REQUEST_DURATION_SECONDS.labels(provider=provider_name).observe(
                _time.perf_counter() - start
            )

        points_data = [_format_observation_point(p) for p in obs.points]
        return ObservationToolResult(location=location.name, observations=points_data)

    @traceable(run_type="tool")
    async def save_location(self, location_name: str, alias: str = "") -> SaveLocationToolResult:
        TOOL_CALLS_TOTAL.labels(tool="save_location").inc()
        start = _time.perf_counter()
        try:
            return await self._execute_save_location(location_name, alias)
        finally:
            TOOL_CALL_DURATION_SECONDS.labels(tool="save_location").observe(
                _time.perf_counter() - start
            )

    async def _execute_save_location(
        self, location_name: str, alias: str
    ) -> SaveLocationToolResult:
        if self.location_service is None:
            return SaveLocationToolResult(error="Usługa lokalizacji jest niedostępna.")

        if not location_name.strip():
            return SaveLocationToolResult(error="Podaj nazwę lokalizacji do zapisania.")

        if self.geocoder is None:
            return SaveLocationToolResult(error="Geokoder jest niedostępny.")

        resolved = await self.geocoder.geocode(location_name)
        if resolved is None:
            return SaveLocationToolResult(
                error=f"Nie udało się rozpoznać lokalizacji '{location_name}'."
            )

        try:
            aliases = [alias] if alias else []
            await self.location_service.create_location(
                self.user_id,
                LocationCreate(
                    name=location_name,
                    aliases=aliases,
                    latitude=resolved.latitude,
                    longitude=resolved.longitude,
                ),
            )
            msg = f"Zapamiętałem lokalizację: {location_name}"
            if alias:
                msg += f" (alias: {alias})"
            return SaveLocationToolResult(success=msg)
        except (LocationAliasConflictError, LocationNameConflictError):
            return SaveLocationToolResult(
                error="Masz już zapisaną lokalizację o tej nazwie lub aliasie."
            )
        except Exception as exc:
            logger.exception(
                "save_location_failed", user_id=self.user_id, location_name=location_name
            )
            return SaveLocationToolResult(error=f"Błąd podczas zapisywania lokalizacji: {exc}")

    @traceable(run_type="tool")
    async def list_locations(self, include_disabled: bool = False) -> ListLocationsToolResult:
        TOOL_CALLS_TOTAL.labels(tool="list_locations").inc()
        start = _time.perf_counter()
        try:
            return await self._execute_list_locations(include_disabled)
        finally:
            TOOL_CALL_DURATION_SECONDS.labels(tool="list_locations").observe(
                _time.perf_counter() - start
            )

    async def _execute_list_locations(self, include_disabled: bool) -> ListLocationsToolResult:
        if self.location_service is None:
            return ListLocationsToolResult(error="Usługa lokalizacji jest niedostępna.")

        try:
            locations = await self.location_service.list_locations(
                self.user_id, include_disabled=include_disabled
            )
            locations_data: list[dict[str, Any]] = []
            for loc in locations:
                entry: dict[str, Any] = {
                    "id": loc.id,
                    "name": loc.name,
                    "latitude": loc.latitude,
                    "longitude": loc.longitude,
                    "enabled": loc.enabled,
                }
                if loc.aliases:
                    entry["aliases"] = loc.aliases
                if loc.description:
                    entry["description"] = loc.description
                locations_data.append(entry)
            return ListLocationsToolResult(locations=locations_data, count=len(locations_data))
        except Exception as exc:
            logger.exception("list_locations_failed", user_id=self.user_id)
            return ListLocationsToolResult(error=f"Błąd podczas pobierania lokalizacji: {exc}")

    def to_langchain_tools(self) -> list[BaseTool]:
        return [
            StructuredTool.from_function(
                coroutine=self.get_forecast,
                name="get_forecast",
                description=(
                    "Pobierz prognozę pogody dla lokalizacji i zakresu dat."
                    " Zwraca godzinowe dane: temperatura, opady, wiatr, zachmurzenie itp."
                    " Daty start_date/end_date podaj jako yyyy-mm-dd w strefie Europe/Warsaw."
                    " Samodzielnie przelicz względne okresy (jutro, dziś, weekend, majówka)"
                    " na konkretne daty przed wywołaniem narzędzia."
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
                    "Zapisz lokalizację użytkownika (np. dom, praca)"
                    " pod konkretną nazwą lub adresem."
                    " Jeśli użytkownik prosi o 'zapamiętanie"
                    " lokalizacji domowej', ustaw alias na 'dom'."
                ),
                args_schema=SaveLocationArgs,
            ),
            StructuredTool.from_function(
                coroutine=self.list_locations,
                name="list_locations",
                description=(
                    "Wyświetl zapisane lokalizacje użytkownika. "
                    "Domyślnie pokazuje tylko aktywne lokalizacje."
                ),
                args_schema=ListLocationsArgs,
            ),
        ]
