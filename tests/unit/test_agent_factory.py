from __future__ import annotations

import pytest

from weather_agent.agent_factory import (
    _load_agents_md,
    _load_weather_agent_prompt,
    build_context_suffix,
)


def test_weather_agent_prompt_is_loaded_from_runtime_prompt_file() -> None:
    result = _load_weather_agent_prompt()

    assert "Pogodowy Asystent" in result
    assert "Jesteś polskim asystentem pogodowym" in result
    assert "Weather Agent Repository Instructions" not in result


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


class TestLoadAgentsMd:
    def test_loads_from_repo_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Loads AGENTS.md content regardless of current working directory."""
        import os

        monkeypatch.setattr("weather_agent.agent_factory._AGENTS_MD", None)

        original_cwd = os.getcwd()
        try:
            os.chdir("/tmp")
            content = _load_agents_md()
            assert content is not None
            assert len(content) > 0
        finally:
            os.chdir(original_cwd)

    def test_caches_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Second call returns cached content without reading the file again."""
        import pathlib

        monkeypatch.setattr("weather_agent.agent_factory._AGENTS_MD", None)

        original_read = pathlib.Path.read_text
        call_count = 0

        def counting_read(path: pathlib.Path, **kwargs: str) -> str:
            nonlocal call_count
            call_count += 1
            return original_read(path, **kwargs)

        monkeypatch.setattr(pathlib.Path, "read_text", counting_read)

        result1 = _load_agents_md()
        result2 = _load_agents_md()

        assert result1 == result2
        assert call_count == 1

    def test_missing_file_raises_file_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Raises FileNotFoundError when AGENTS.md does not exist."""
        from pathlib import Path

        monkeypatch.setattr("weather_agent.agent_factory._AGENTS_MD", None)

        def raise_not_found(*args: object, **kwargs: object) -> str:
            raise FileNotFoundError("No such file")

        monkeypatch.setattr(Path, "read_text", raise_not_found)

        with pytest.raises(FileNotFoundError):
            _load_agents_md()
