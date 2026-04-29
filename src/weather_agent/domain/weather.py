from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict


class WeatherVariable(StrEnum):
    temperature_2m_c = "temperature_2m_c"
    apparent_temperature_c = "apparent_temperature_c"
    precipitation_mm = "precipitation_mm"
    precipitation_probability_pct = "precipitation_probability_pct"
    rain_mm = "rain_mm"
    snowfall_cm = "snowfall_cm"
    cloud_cover_pct = "cloud_cover_pct"
    wind_speed_10m_ms = "wind_speed_10m_ms"
    wind_gusts_10m_ms = "wind_gusts_10m_ms"
    wind_direction_10m_deg = "wind_direction_10m_deg"
    pressure_msl_hpa = "pressure_msl_hpa"
    relative_humidity_2m_pct = "relative_humidity_2m_pct"
    weather_code = "weather_code"


class ForecastResolution(StrEnum):
    hourly = "hourly"
    fifteen_min = "fifteen_min"


class LocationRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    latitude: float
    longitude: float


class TimeRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: datetime
    end: datetime
    timezone: Literal["Europe/Warsaw"] = "Europe/Warsaw"


class ForecastPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_time: datetime
    fetched_at: datetime
    provider: str
    model: str | None = None
    location_id: str

    temperature_2m_c: float | None = None
    apparent_temperature_c: float | None = None
    precipitation_mm: float | None = None
    precipitation_probability_pct: float | None = None
    rain_mm: float | None = None
    snowfall_cm: float | None = None
    cloud_cover_pct: float | None = None
    wind_speed_10m_ms: float | None = None
    wind_gusts_10m_ms: float | None = None
    wind_direction_10m_deg: float | None = None
    pressure_msl_hpa: float | None = None
    relative_humidity_2m_pct: float | None = None
    weather_code: str | None = None

    raw_payload: dict[str, object]


class ForecastResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str | None = None
    location: LocationRef
    fetched_at: datetime
    points: list[ForecastPoint]
    raw_payload: dict[str, object]


class ObservationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    location: LocationRef
    fetched_at: datetime
    points: list[ObservationPoint]
    raw_payload: dict[str, object]


class ObservationPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observed_at: datetime
    fetched_at: datetime
    provider: str
    station_id: str | None = None
    station_name: str | None = None
    distance_km: float | None = None

    temperature_c: float | None = None
    wind_speed_ms: float | None = None
    wind_direction_deg: float | None = None
    pressure_hpa: float | None = None
    humidity_pct: float | None = None
    precipitation_mm: float | None = None

    raw_payload: dict[str, object]


class WarningSeverity(StrEnum):
    low = "low"
    moderate = "moderate"
    high = "high"
    extreme = "extreme"


class WarningCategory(StrEnum):
    meteo = "meteo"
    hydro = "hydro"


class WeatherWarning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    external_id: str
    location_id: str
    severity: str | None = None
    category: str
    headline: str
    description: str
    valid_from: datetime
    valid_to: datetime
    raw_payload: dict[str, object]
