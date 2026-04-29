from __future__ import annotations

from weather_agent.agent_factory import build_context_suffix


class TestBuildContextSuffix:
    def test_without_context(self) -> None:
        result = build_context_suffix()
        assert "Bieżąca data i godzina w strefie Europe/Warsaw:" in result

    def test_with_last_forecast_context(self) -> None:
        forecast = {
            "location_name": "Gdańsk",
            "start_date": "2026-04-30",
            "end_date": "2026-04-30",
            "variables": ["temperature_2m_c"],
        }
        result = build_context_suffix(last_forecast_context=forecast)
        assert "OSTATNIA PROGNOZA: Gdańsk" in result
        assert "zakres 2026-04-30 – 2026-04-30" in result
        assert "temperature_2m_c" in result
        assert "odziedzicz lokalizację i zakres" in result
        assert "Bieżąca data i godzina" in result

    def test_last_forecast_with_multiple_variables(self) -> None:
        forecast = {
            "location_name": "Warszawa",
            "start_date": "2026-05-01",
            "end_date": "2026-05-03",
            "variables": ["temperature_2m_c", "wind_speed_10m", "precipitation"],
        }
        result = build_context_suffix(last_forecast_context=forecast)
        assert "Warszawa" in result
        assert "2026-05-01 – 2026-05-03" in result
        assert "temperature_2m_c, wind_speed_10m, precipitation" in result

    def test_both_pending_and_forecast(self) -> None:
        pending = {"action": "create_rule", "cel_expression": "temp > 30", "explanation": "test"}
        forecast = {
            "location_name": "Gdańsk",
            "start_date": "2026-04-30",
            "end_date": "2026-04-30",
            "variables": ["temperature_2m_c"],
        }
        result = build_context_suffix(
            pending_confirmation=pending,
            last_forecast_context=forecast,
        )
        assert "OSTATNIA PROGNOZA: Gdańsk" in result
        assert "OCZEKUJĄCA AKCJA" in result
