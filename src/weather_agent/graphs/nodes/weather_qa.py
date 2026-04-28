from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

from weather_agent.domain.date_resolver import DateResolver, ResolvedTimeRange
from weather_agent.domain.errors import WeatherProviderError
from weather_agent.domain.locations import LocationService
from weather_agent.domain.weather import (
    ForecastResolution,
    ForecastResult,
    LocationRef,
    ObservationResult,
    WeatherVariable,
)
from weather_agent.graphs.state import ConversationState
from weather_agent.llm.model_factory import ModelFactory

_DEFAULT_VARIABLES = [
    WeatherVariable.temperature_2m_c,
    WeatherVariable.apparent_temperature_c,
    WeatherVariable.precipitation_mm,
    WeatherVariable.precipitation_probability_pct,
    WeatherVariable.wind_speed_10m_ms,
    WeatherVariable.wind_gusts_10m_ms,
    WeatherVariable.cloud_cover_pct,
    WeatherVariable.weather_code,
    WeatherVariable.relative_humidity_2m_pct,
]

_WIND_VARIABLES = [
    WeatherVariable.wind_speed_10m_ms,
    WeatherVariable.wind_gusts_10m_ms,
    WeatherVariable.wind_direction_10m_deg,
]

_FULL_VARIABLES = list(dict.fromkeys(_DEFAULT_VARIABLES + _WIND_VARIABLES + [
    WeatherVariable.apparent_temperature_c,
    WeatherVariable.pressure_msl_hpa,
    WeatherVariable.snowfall_cm,
    WeatherVariable.rain_mm,
]))


@runtime_checkable
class ForecastProvider(Protocol):
    async def get_forecast(
        self,
        location: LocationRef,
        time_range: Any,
        variables: list[WeatherVariable],
        resolution: ForecastResolution,
    ) -> ForecastResult: ...


@runtime_checkable
class ObservationProvider(Protocol):
    async def get_observations(
        self,
        location: LocationRef,
        radius_km: float,
        variables: list[WeatherVariable],
    ) -> ObservationResult: ...


def _extract_location_reference(message: str) -> str | None:
    patterns = [
        r"w\s+([A-Z\u00c0-\u017e][\w\u00c0-\u017e]+)",
        r"w\s+([a-z\u00c0-\u017e][\w\u00c0-\u017e]+)",
        r"w\s+(\w+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, message, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _select_variables(message: str) -> list[WeatherVariable]:
    msg_lower = message.lower()
    if any(kw in msg_lower for kw in ("wiatr", "wietrzn", "wiew", "poryw")):
        return _FULL_VARIABLES
    if any(kw in msg_lower for kw in ("deszcz", "opad", "pada", "padać", "śnieg")):
        return _FULL_VARIABLES
    if any(kw in msg_lower for kw in ("ubr", "ubierz", "odież", "odzież", "ubiór")):
        return _FULL_VARIABLES
    return _DEFAULT_VARIABLES


def _extract_time_reference(message: str) -> str | None:
    time_patterns = [
        r"dziś\s+wiecz(?:orem|ór)",
        r"dzisiaj\s+wiecz(?:orem|ór)",
        r"jutro\s+rano",
        r"jutro\s+po\s+południu",
        r"jutro\s+po\s+poludniu",
        r"jutro\s+wiecz(?:orem|ór)",
        r"następne\s+\d+\s+dni",
        r"nastepne\s+\d+\s+dni",
        r"następny\s+weekend",
        r"nastepny\s+weekend",
        r"ten\s+weekend",
        r"weekend",
        r"majówk[aeę]",
        r"majowk[aeę]",
        r"jutro",
        r"dziś",
        r"dzisiaj",
    ]
    for pattern in time_patterns:
        m = re.search(pattern, message, re.IGNORECASE)
        if m:
            return m.group(0).lower()
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", message)
    if date_match:
        return date_match.group(0)
    return None


_WEAther_CODE_MAP: dict[int, str] = {
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
        return _WEAther_CODE_MAP.get(int(code))
    except (ValueError, TypeError):
        return None


def _format_forecast_summary(
    location: LocationRef,
    time_range: ResolvedTimeRange,
    forecast: ForecastResult,
    observation: ObservationResult | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"Pogoda dla {location.name} ({time_range.explanation}):")

    points = forecast.points
    if not points:
        lines.append("Brak danych prognozy dla wskazanego zakresu.")
        return "\n".join(lines)

    temps = [p for p in points if p.temperature_2m_c is not None]
    if temps:
        t_min = min(p.temperature_2m_c for p in temps)  # type: ignore[type-var]
        t_max = max(p.temperature_2m_c for p in temps)  # type: ignore[type-var]
        t_avg = sum(p.temperature_2m_c for p in temps) / len(temps)  # type: ignore[misc]
        lines.append(f"Temperatura: {t_min:.0f}°C – {t_max:.0f}°C (średnia {t_avg:.1f}°C)")

    feels = [p for p in points if p.apparent_temperature_c is not None]
    if feels:
        f_min = min(p.apparent_temperature_c for p in feels)  # type: ignore[type-var]
        f_max = max(p.apparent_temperature_c for p in feels)  # type: ignore[type-var]
        lines.append(f"Temperatura odczuwalna: {f_min:.0f}°C – {f_max:.0f}°C")

    precip = [p for p in points if p.precipitation_mm is not None]
    if precip:
        total = sum(p.precipitation_mm for p in precip)  # type: ignore[misc]
        max_prob = max(
            (
                p.precipitation_probability_pct
                for p in points
                if p.precipitation_probability_pct is not None
            ),
            default=None,
        )
        prob_str = f", prawdopodobieństwo {max_prob:.0f}%" if max_prob is not None else ""
        lines.append(f"Opady łącznie: {total:.1f} mm{prob_str}")

    wind = [p for p in points if p.wind_speed_10m_ms is not None]
    if wind:
        w_max = max(p.wind_speed_10m_ms for p in wind)  # type: ignore[type-var]
        w_avg = sum(p.wind_speed_10m_ms for p in wind) / len(wind)  # type: ignore[misc]
        gusts = [p for p in points if p.wind_gusts_10m_ms is not None]
        gust_str = ""
        if gusts:
            g_max = max(p.wind_gusts_10m_ms for p in gusts)  # type: ignore[type-var]
            gust_str = f", porywy do {g_max:.0f} m/s"
        lines.append(f"Wiatr: średnio {w_avg:.1f} m/s, maks. {w_max:.0f} m/s{gust_str}")

    cloud = [p for p in points if p.cloud_cover_pct is not None]
    if cloud:
        c_avg = sum(p.cloud_cover_pct for p in cloud) / len(cloud)  # type: ignore[misc]
        lines.append(f"Zachmurzenie: średnio {c_avg:.0f}%")

    humidity = [p for p in points if p.relative_humidity_2m_pct is not None]
    if humidity:
        h_avg = sum(p.relative_humidity_2m_pct for p in humidity) / len(humidity)  # type: ignore[misc]
        lines.append(f"Wilgotność: średnio {h_avg:.0f}%")

    codes = [p for p in points if p.weather_code is not None]
    if codes:
        unique_codes: set[str] = set()
        for p in codes:
            if p.weather_code is not None:
                unique_codes.add(p.weather_code)
        descriptions = []
        for code in sorted(unique_codes, key=lambda c: int(c)):
            desc = _weather_code_description(code)
            if desc:
                descriptions.append(desc)
        if descriptions:
            lines.append("Warunki: " + ", ".join(descriptions))

    if observation and observation.points:
        obs = observation.points[0]
        lines.append(f"\nObecnie ({obs.station_name or 'najbliższa stacja'}):")
        if obs.temperature_c is not None:
            lines.append(f"  Temperatura: {obs.temperature_c:.1f}°C")
        if obs.wind_speed_ms is not None:
            lines.append(f"  Wiatr: {obs.wind_speed_ms:.1f} m/s")
        if obs.humidity_pct is not None:
            lines.append(f"  Wilgotność: {obs.humidity_pct:.0f}%")

    lines.append(f"\n(Prognoza: {forecast.provider}, model: {forecast.model or 'N/A'})")
    return "\n".join(lines)


async def resolve_location_node(
    state: ConversationState,
    location_service: LocationService | None,
    user_id: int,
    geocoder: Any | None = None,
) -> dict[str, Any]:
    message = state.get("user_message") or ""
    location_ref = state.get("resolved_location")

    if location_ref is not None:
        return {"resolved_location": location_ref}

    extracted = _extract_location_reference(message)
    if extracted:
        if location_service is not None:
            resolved = await location_service.resolve_location(extracted, user_id)
            if resolved is not None:
                return {"resolved_location": resolved}

        if geocoder is not None:
            resolved = await geocoder.geocode(extracted)
            if resolved is not None:
                return {"resolved_location": resolved}

        return {
            "error": (
                f"Nie udało się rozpoznać lokalizacji \u201e{extracted}\u201d."
                " Podaj lokalizację jawnie (np. w Gdańsku)."
            ),
            "resolved_location": None,
        }

    if location_service is not None:
        locations = await location_service.list_locations(user_id)
        if len(locations) == 1:
            loc = locations[0]
            return {
                "resolved_location": LocationRef(
                    id=str(loc.id),
                    name=loc.name,
                    latitude=loc.latitude,
                    longitude=loc.longitude,
                ),
            }

    return {
        "error": (
            "Nie podałeś lokalizacji. Napisz np. „jaka pogoda w Gdańsku jutro\""
            " lub ustaw lokalizację domową komendą /dodaj_lok."
        ),
        "resolved_location": None,
    }


async def resolve_time_range_node(
    state: ConversationState,
    date_resolver: DateResolver,
) -> dict[str, Any]:
    existing_time_range = state.get("resolved_time_range")
    if existing_time_range is not None:
        return {"resolved_time_range": existing_time_range}

    message = state.get("user_message") or ""
    time_ref = _extract_time_reference(message)

    if time_ref is not None:
        result = await date_resolver.resolve(time_ref)
        if result is not None:
            return {"resolved_time_range": result}

    result = await date_resolver.resolve("dziś")
    return {"resolved_time_range": result}


async def call_weather_tools_node(
    state: ConversationState,
    forecast_provider: ForecastProvider,
    observation_provider: ObservationProvider | None = None,
) -> dict[str, Any]:
    location = state.get("resolved_location")
    time_range = state.get("resolved_time_range")

    if location is None:
        return {
            "forecast_result": None,
            "observation_result": None,
            "error": "Brak lokalizacji — nie można pobrać prognozy.",
        }

    if time_range is None:
        return {
            "forecast_result": None,
            "observation_result": None,
            "error": "Brak zakresu czasu — nie można pobrać prognozy.",
        }

    from weather_agent.domain.weather import TimeRange

    tr = TimeRange(start=time_range.start, end=time_range.end)
    message = state.get("user_message") or ""
    variables = _select_variables(message)

    try:
        forecast = await forecast_provider.get_forecast(
            location=location,
            time_range=tr,
            variables=variables,
            resolution=ForecastResolution.hourly,
        )
    except WeatherProviderError as exc:
        return {
            "forecast_result": None,
            "observation_result": None,
            "error": f"Błąd dostawcy prognozy ({exc.provider}): {exc.message}",
        }

    observation: ObservationResult | None = None
    if observation_provider is not None:
        try:
            observation = await observation_provider.get_observations(
                location=location,
                radius_km=50.0,
                variables=variables,
            )
        except WeatherProviderError:
            pass
        except Exception:
            pass

    return {
        "forecast_result": forecast,
        "observation_result": observation,
        "error": None,
    }


async def answer_weather_question_node(
    state: ConversationState,
    model_factory: ModelFactory | None = None,
) -> dict[str, Any]:
    if state.get("error"):
        return {"answer": f"Przepraszam, wystąpił błąd: {state['error']}"}

    location = state.get("resolved_location")
    time_range = state.get("resolved_time_range")
    forecast = state.get("forecast_result")
    observation = state.get("observation_result")

    if location is None:
        return {"answer": "Nie mogę udzielić prognozy — brak lokalizacji."}

    if forecast is None:
        return {"answer": "Nie udało się pobrać danych prognozy. Spróbuj ponownie za chwilę."}

    if time_range is None:
        return {"answer": "Nie udało się ustalić zakresu czasu. Spróbuj ponownie."}

    if model_factory is not None:
        try:
            summary = _format_forecast_summary(location, time_range, forecast, observation)
            chat = model_factory.create_chat_model()
            prompt = (
                "Na podstawie poniższego podsumowania danych pogodowych, "
                " Przygotuj zwięzłą, naturalną odpowiedź po polsku "
                f"dla lokalizacji {location.name}.\n\n"
                f"{summary}\n\n"
                "Odpowiedz krótko i naturalnie. Podaj lokalizację i zakres czasu."
            )
            response = await chat.ainvoke(prompt)
            if hasattr(response, "content"):
                answer = str(response.content)
            else:
                answer = _format_forecast_summary(location, time_range, forecast, observation)
            return {"answer": answer}
        except Exception:
            pass

    answer = _format_forecast_summary(location, time_range, forecast, observation)
    return {"answer": answer}