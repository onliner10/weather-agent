from __future__ import annotations

import pytest
from pydantic import ValidationError

from weather_agent.eval.schemas import WeatherAnswerOutput, WeatherFacts, WeatherGroundingExample


class TestWeatherFacts:
    def test_full_facts(self) -> None:
        facts = WeatherFacts(
            location="Chwarzno",
            period="teraz",
            temperature_c=12.0,
            wind_speed_ms=5.0,
            wind_direction_deg=270.0,
            pressure_hpa=1012.0,
            humidity_pct=82.0,
            precipitation_mm=1.2,
        )
        assert facts.location == "Chwarzno"
        assert facts.wind_direction_deg == 270.0

    def test_forbid_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            WeatherFacts(location="X", period="jutro", extra=True)  # type: ignore[call-arg]


class TestWeatherGroundingExample:
    def test_minimal(self) -> None:
        ex = WeatherGroundingExample(
            id="grounding-001",
            question="Czy będzie padać?",
            frozen_facts=WeatherFacts(
                location="Chwarzno",
                period="teraz",
                temperature_c=12.0,
            ),
            requested_attributes=["temperature_c"],
        )
        assert ex.id == "grounding-001"
        assert ex.frozen_facts.temperature_c == 12.0
        assert ex.required_location is True
        assert ex.requested_attributes == ["temperature_c"]

    def test_with_note(self) -> None:
        ex = WeatherGroundingExample(
            id="grounding-001",
            question="Czy będzie padać?",
            frozen_facts=WeatherFacts(location="Chwarzno", period="teraz"),
            requested_attributes=["precipitation_mm"],
            note="rain-focused example",
        )
        assert ex.note == "rain-focused example"

    def test_forbids_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            WeatherGroundingExample(
                id="x",
                question="x",
                frozen_facts=WeatherFacts(location="X", period="teraz"),
                requested_attributes=["temperature_c"],
                extra=True,  # type: ignore[call-arg]
            )


class TestWeatherAnswerOutput:
    def test_round_trip(self) -> None:
        out = WeatherAnswerOutput(example_id="grounding-001", answer="W Chwarznie będzie padać.")
        restored = WeatherAnswerOutput.model_validate(out.model_dump())
        assert restored == out
