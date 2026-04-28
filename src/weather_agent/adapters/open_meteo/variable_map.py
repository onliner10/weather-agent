from __future__ import annotations

from weather_agent.domain.weather import WeatherVariable

OPEN_METEO_VARIABLE_MAP: dict[WeatherVariable, str] = {
    WeatherVariable.temperature_2m_c: "temperature_2m",
    WeatherVariable.apparent_temperature_c: "apparent_temperature",
    WeatherVariable.precipitation_mm: "precipitation",
    WeatherVariable.precipitation_probability_pct: "precipitation_probability",
    WeatherVariable.rain_mm: "rain",
    WeatherVariable.snowfall_cm: "snowfall",
    WeatherVariable.cloud_cover_pct: "cloud_cover",
    WeatherVariable.wind_speed_10m_ms: "wind_speed_10m",
    WeatherVariable.wind_gusts_10m_ms: "wind_gusts_10m",
    WeatherVariable.wind_direction_10m_deg: "wind_direction_10m",
    WeatherVariable.pressure_msl_hpa: "pressure_msl",
    WeatherVariable.relative_humidity_2m_pct: "relative_humidity_2m",
    WeatherVariable.weather_code: "weather_code",
}