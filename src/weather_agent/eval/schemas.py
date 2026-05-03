from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

WeatherAttribute = Literal[
    "precipitation_mm",
    "temperature_c",
    "wind_speed_ms",
    "wind_direction_deg",
    "pressure_hpa",
    "humidity_pct",
]

WEATHER_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "precipitation_mm",
        "temperature_c",
        "wind_speed_ms",
        "wind_direction_deg",
        "pressure_hpa",
        "humidity_pct",
    }
)


class WeatherFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    location: str
    period: str
    temperature_c: float | None = None
    wind_speed_ms: float | None = None
    wind_direction_deg: float | None = None
    pressure_hpa: float | None = None
    humidity_pct: float | None = None
    precipitation_mm: float | None = None


class WeatherGroundingExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    frozen_facts: WeatherFacts
    required_location: bool = True
    requested_attributes: list[WeatherAttribute]
    note: str | None = None


class WeatherAnswerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    example_id: str


class WeatherToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    args: dict[str, object]
    result_error: str | None = None


class WeatherPresentationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    example_id: str
    tool_calls: list[WeatherToolCallRecord]
    attachment_count: int
