from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from weather_agent.eval.schemas import WeatherFacts

DATASET_NAME = "weather-agent-weather-presentation-v2"
_WARSAW = ZoneInfo("Europe/Warsaw")
_CURRENT_TIME = datetime(2026, 5, 3, 12, 0, tzinfo=_WARSAW)
_TOMORROW_NOON = datetime(2026, 5, 4, 12, 0, tzinfo=_WARSAW)
_TOMORROW_DATE = _TOMORROW_NOON.date().isoformat()


def _hourly_weather_values() -> dict[int, dict[str, float]]:
    return {
        hour: {
            "wind_speed_ms": 3.0 + (hour / 6.0),
            "wind_direction_deg": 240.0,
            "temperature_c": 8.0 + min(hour, 15) * 0.5 - max(hour - 15, 0) * 0.35,
            "precipitation_mm": 0.0
            if hour < 10 or hour > 17
            else round(0.2 + ((hour - 10) % 4) * 0.35, 2),
        }
        for hour in range(24)
    }


def generate_weather_presentation_cases() -> list[dict[str, object]]:
    facts = WeatherFacts(
        location="Chwarzno",
        period="jutro",
        wind_speed_ms=5.0,
        wind_direction_deg=240.0,
        temperature_c=14.0,
        precipitation_mm=0.6,
    )
    hourly_values = _hourly_weather_values()
    return [
        {
            "id": "weather-presentation-chart-explicit-001",
            "question": "Rozrysuj wiatr w Chwarznie jutro.",
            "current_time": _CURRENT_TIME.isoformat(),
            "expected_target_time": _TOMORROW_NOON.isoformat(),
            "frozen_facts": facts.model_dump(mode="json"),
            "hourly_values": hourly_values,
            "expect_chart": True,
            "expected_chart_variables": ["wind_speed_10m_ms"],
            "expected_chart_start_date": _TOMORROW_DATE,
            "expected_chart_end_date": _TOMORROW_DATE,
        },
        {
            "id": "weather-presentation-chart-trend-001",
            "question": "Jak będzie się zmieniać wiatr w Chwarznie jutro przez cały dzień?",
            "current_time": _CURRENT_TIME.isoformat(),
            "expected_target_time": _TOMORROW_NOON.isoformat(),
            "frozen_facts": facts.model_dump(mode="json"),
            "hourly_values": hourly_values,
            "expect_chart": None,
            "expected_chart_variables": ["wind_speed_10m_ms"],
            "expected_chart_start_date": _TOMORROW_DATE,
            "expected_chart_end_date": _TOMORROW_DATE,
        },
        {
            "id": "weather-presentation-text-point-001",
            "question": "Jaki będzie wiatr w Chwarznie jutro o 12:00?",
            "current_time": _CURRENT_TIME.isoformat(),
            "expected_target_time": _TOMORROW_NOON.isoformat(),
            "frozen_facts": facts.model_dump(mode="json"),
            "hourly_values": hourly_values,
            "expect_chart": False,
        },
        {
            "id": "weather-presentation-chart-temperature-001",
            "question": "Pokaż wykres temperatury w Chwarznie jutro.",
            "current_time": _CURRENT_TIME.isoformat(),
            "expected_target_time": _TOMORROW_NOON.isoformat(),
            "frozen_facts": facts.model_dump(mode="json"),
            "hourly_values": hourly_values,
            "expect_chart": True,
            "expected_chart_variables": ["temperature_2m_c"],
            "expected_chart_start_date": _TOMORROW_DATE,
            "expected_chart_end_date": _TOMORROW_DATE,
        },
        {
            "id": "weather-presentation-chart-precipitation-001",
            "question": "Pokaż godzinowy wykres opadów w Chwarznie jutro.",
            "current_time": _CURRENT_TIME.isoformat(),
            "expected_target_time": _TOMORROW_NOON.isoformat(),
            "frozen_facts": facts.model_dump(mode="json"),
            "hourly_values": hourly_values,
            "expect_chart": True,
            "expected_chart_variables": ["precipitation_mm"],
            "expected_chart_start_date": _TOMORROW_DATE,
            "expected_chart_end_date": _TOMORROW_DATE,
        },
    ]
