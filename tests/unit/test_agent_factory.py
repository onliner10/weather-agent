from __future__ import annotations

from weather_agent.agent_factory import _load_weather_agent_prompt


def test_weather_agent_prompt_is_loaded_from_runtime_prompt_file() -> None:
    result = _load_weather_agent_prompt()

    assert "Pogodowy Asystent" in result
    assert "Jesteś polskim asystentem pogodowym" in result
    assert "Weather Agent Repository Instructions" not in result
