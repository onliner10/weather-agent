from __future__ import annotations

import json
import re
import time
from typing import Any

from langsmith import traceable
from pydantic import BaseModel, Field

from weather_agent.domain.date_resolver import DateResolver
from weather_agent.domain.errors import WeatherProviderError
from weather_agent.domain.locations import LocationService
from weather_agent.domain.providers import ForecastProvider, ObservationProvider
from weather_agent.domain.weather import (
    ForecastResolution,
    LocationRef,
    WeatherVariable,
)
from weather_agent.graphs.state import ConversationState
from weather_agent.infrastructure.geocoder import Geocoder
from weather_agent.llm.model_factory import ModelFactory
from weather_agent.observability.logging import get_logger
from weather_agent.observability.metrics import (
    LLM_REQUEST_DURATION_SECONDS,
    LLM_REQUESTS_TOTAL,
    PROVIDER_REQUEST_DURATION_SECONDS,
    PROVIDER_REQUESTS_TOTAL,
    TOOL_CALL_DURATION_SECONDS,
    TOOL_CALLS_TOTAL,
)

logger = get_logger(__name__)

_GENERIC_USER_ERROR = "Przepraszam, wystąpił błąd. Spróbuj ponownie za chwilę."

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


class _LocationExtraction(BaseModel):
    location_name: str | None = Field(
        default=None,
        description="Place name in nominative case, or null if none",
    )
    focus: str | None = Field(
        default=None,
        description="What user asks about e.g. wiatr, temperatura. Null if general.",
    )


async def _extract_location_and_focus(
    message: str,
    model_factory: ModelFactory | None,
) -> _LocationExtraction:
    if model_factory is None:
        return _LocationExtraction(location_name=None, focus=None)
    start = time.perf_counter()
    try:
        chat = model_factory.create_chat_model()
        structured = chat.with_structured_output(_LocationExtraction)
        result = await structured.ainvoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Wyciągnij z wiadomości użytkownika:\n"
                        "1. location_name — nazwa miejscowości w mianowniku"
                        ' (np. „w Gdańsku" → „Gdańsk",'
                        ' „w Chwarznie" → „Chwarzno").'
                        " Zwróć null jeśli nie ma nazwy miejsca.\n"
                        "2. focus — o co użytkownik pyta szczegółowo"
                        " (np. wiatr, porywy, temperatura, opady, ciśnienie)."
                        " Zwróć null jeśli ogólne pytanie.\n\n"
                        'UWAGA: „w weekend", „w majówkę",'
                        ' „w poniedziałek" to CZAS, nie lokalizacja.'
                        " Nie myl przyimków czasowych z miejscowymi."
                    ),
                },
                {"role": "user", "content": message},
            ],
        )
        LLM_REQUESTS_TOTAL.labels(outcome="success").inc()
        if isinstance(result, _LocationExtraction):
            return result
    except Exception:
        LLM_REQUESTS_TOTAL.labels(outcome="failure").inc()
        logger.warning(
            "llm_location_extraction_failed",
            exc_info=True,
        )
    finally:
        LLM_REQUEST_DURATION_SECONDS.observe(time.perf_counter() - start)
    return _LocationExtraction(location_name=None, focus=None)


def _build_tools(
    forecast_provider: ForecastProvider,
    observation_provider: ObservationProvider | None,
    geocoder: Geocoder,
    date_resolver: DateResolver,
    location_service: LocationService | None,
    user_id: int,
) -> list[dict[str, Any]]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_forecast",
                "description": (
                    "Pobierz prognozę pogody dla lokalizacji i zakresu czasu."
                    " Zwraca godzinowe dane: temperatura, opady, wiatr, zachmurzenie itp."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location_name": {
                            "type": "string",
                            "description": "Nazwa miejscowości (np. Gdańsk, Chwarzno)",
                        },
                        "time_expression": {
                            "type": "string",
                            "description": "Wyrażenie czasowe (np. jutro, weekend, dziś wieczorem)",
                        },
                        "variables": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Lista zmiennych pogodowych do pobrania."
                                " Dostępne: temperature_2m_c, apparent_temperature_c,"
                                " precipitation_mm, precipitation_probability_pct,"
                                " rain_mm, snowfall_cm, cloud_cover_pct,"
                                " wind_speed_10m_ms, wind_gusts_10m_ms,"
                                " wind_direction_10m_deg, pressure_msl_hpa,"
                                " relative_humidity_2m_pct, weather_code"
                            ),
                        },
                    },
                    "required": ["location_name", "time_expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_observations",
                "description": (
                    "Pobierz aktualne obserwacje ze stacji meteorologicznych"
                    " wokół lokalizacji (ostatni pomiar)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location_name": {
                            "type": "string",
                            "description": "Nazwa miejscowości",
                        },
                    },
                    "required": ["location_name"],
                },
            },
        },
    ]
    return tools


@traceable(run_type="tool")
async def _execute_tool_call(
    tool_name: str,
    tool_args: dict[str, Any],
    forecast_provider: ForecastProvider,
    observation_provider: ObservationProvider | None,
    geocoder: Geocoder,
    date_resolver: DateResolver,
    location_service: LocationService | None,
    user_id: int,
) -> str:
    TOOL_CALLS_TOTAL.labels(tool=tool_name).inc()
    start = time.perf_counter()
    try:
        if tool_name == "get_forecast":
            result = await _execute_get_forecast(
                tool_args,
                forecast_provider,
                geocoder,
                date_resolver,
                location_service,
                user_id,
            )
        elif tool_name == "get_observations":
            result = await _execute_get_observations(
                tool_args,
                observation_provider,
                geocoder,
                location_service,
                user_id,
            )
        else:
            result = json.dumps({"error": f"Unknown tool: {tool_name}"})
    finally:
        TOOL_CALL_DURATION_SECONDS.labels(tool=tool_name).observe(time.perf_counter() - start)
    return result


@traceable(run_type="tool")
async def _resolve_location(
    name: str,
    geocoder: Geocoder,
    location_service: LocationService | None,
    user_id: int,
) -> LocationRef | None:
    try:
        if location_service is not None:
            resolved = await location_service.resolve_location(name, user_id)
            if resolved is not None:
                return resolved
        return await geocoder.geocode(name)
    except Exception:
        logger.exception(
            "resolve_location_failed",
            location_name=name,
            user_id=user_id,
        )
        return None


@traceable(run_type="tool")
async def _execute_get_forecast(
    args: dict[str, Any],
    forecast_provider: ForecastProvider,
    geocoder: Geocoder,
    date_resolver: DateResolver,
    location_service: LocationService | None,
    user_id: int,
) -> str:
    location_name = args.get("location_name", "")
    time_expr = args.get("time_expression", "dziś")
    variable_names = args.get("variables", [])

    location = await _resolve_location(location_name, geocoder, location_service, user_id)
    if location is None:
        err_msg = f"Nie znaleziono lokalizacji: {location_name}"
        return json.dumps({"error": err_msg}, ensure_ascii=False)

    time_range = await date_resolver.resolve(time_expr)
    if time_range is None:
        time_range = await date_resolver.resolve("dziś")
    assert time_range is not None

    variables = []
    for vn in variable_names:
        try:
            variables.append(WeatherVariable(vn))
        except ValueError:
            pass
    if not variables:
        variables = list(WeatherVariable)

    from weather_agent.domain.weather import TimeRange

    tr = TimeRange(start=time_range.start, end=time_range.end)
    provider_name = getattr(forecast_provider, "provider", "unknown")
    start = time.perf_counter()
    try:
        forecast = await forecast_provider.get_forecast(
            location=location,
            time_range=tr,
            variables=variables,
            resolution=ForecastResolution.hourly,
        )
        PROVIDER_REQUESTS_TOTAL.labels(provider=provider_name, outcome="success").inc()
    except WeatherProviderError as exc:
        PROVIDER_REQUESTS_TOTAL.labels(
            provider=getattr(exc, "provider", provider_name), outcome="failure"
        ).inc()
        logger.warning(
            "forecast_provider_error",
            provider=exc.provider,
            error_message=exc.message,
            location_name=location_name,
        )
        err_msg = f"Błąd dostawcy prognozy ({exc.provider}): {exc.message}"
        return json.dumps({"error": err_msg}, ensure_ascii=False)
    except Exception:
        PROVIDER_REQUESTS_TOTAL.labels(provider=provider_name, outcome="failure").inc()
        logger.exception(
            "forecast_provider_unexpected_error",
            location_name=location_name,
            time_expression=time_expr,
        )
        return json.dumps(
            {"error": "Błąd podczas pobierania prognozy. Spróbuj ponownie."},
            ensure_ascii=False,
        )
    finally:
        PROVIDER_REQUEST_DURATION_SECONDS.labels(provider=provider_name).observe(
            time.perf_counter() - start
        )

    points_data = [_format_point(p) for p in forecast.points]
    result = {
        "location": location.name,
        "time_range": time_range.explanation,
        "forecast_points": points_data,
        "provider": forecast.provider,
        "model": forecast.model,
    }
    return json.dumps(result, ensure_ascii=False, default=str)


@traceable(run_type="tool")
async def _execute_get_observations(
    args: dict[str, Any],
    observation_provider: ObservationProvider | None,
    geocoder: Geocoder,
    location_service: LocationService | None,
    user_id: int,
) -> str:
    if observation_provider is None:
        return json.dumps({"error": "Obserwacje niedostępne"})

    location_name = args.get("location_name", "")
    location = await _resolve_location(location_name, geocoder, location_service, user_id)
    if location is None:
        err_msg = f"Nie znaleziono lokalizacji: {location_name}"
        return json.dumps({"error": err_msg}, ensure_ascii=False)

    provider_name = getattr(observation_provider, "provider", "unknown")
    start = time.perf_counter()
    try:
        obs = await observation_provider.get_observations(
            location=location,
            radius_km=50.0,
            variables=list(WeatherVariable),
        )
        PROVIDER_REQUESTS_TOTAL.labels(provider=provider_name, outcome="success").inc()
    except WeatherProviderError as exc:
        PROVIDER_REQUESTS_TOTAL.labels(
            provider=getattr(exc, "provider", provider_name), outcome="failure"
        ).inc()
        logger.warning(
            "observation_provider_error",
            provider=exc.provider,
            error_message=exc.message,
            location_name=location_name,
        )
        return json.dumps(
            {"error": f"Błąd dostawcy obserwacji: {exc.message}"},
            ensure_ascii=False,
        )
    except Exception:
        PROVIDER_REQUESTS_TOTAL.labels(provider=provider_name, outcome="failure").inc()
        logger.exception(
            "observation_provider_unexpected_error",
            location_name=location_name,
        )
        return json.dumps(
            {"error": "Błąd podczas pobierania obserwacji. Spróbuj ponownie."},
            ensure_ascii=False,
        )
    finally:
        PROVIDER_REQUEST_DURATION_SECONDS.labels(provider=provider_name).observe(
            time.perf_counter() - start
        )

    points_data = [_format_observation_point(p) for p in obs.points]
    return json.dumps(
        {"location": location.name, "observations": points_data},
        ensure_ascii=False,
        default=str,
    )


async def weather_agent_node(
    state: ConversationState,
    model_factory: ModelFactory | None,
    forecast_provider: ForecastProvider | None,
    observation_provider: ObservationProvider | None,
    geocoder: Geocoder | None,
    date_resolver: DateResolver | None,
    location_service: LocationService | None,
    user_id: int = 0,
) -> dict[str, Any]:
    if state.get("error"):
        logger.error(
            "weather_agent_node_error_in_state",
            state_error=state["error"],
        )
        return {"answer": _GENERIC_USER_ERROR}

    user_message = state.get("user_message") or ""

    if (
        model_factory is None
        or forecast_provider is None
        or geocoder is None
        or date_resolver is None
    ):
        return {"answer": "Przepraszam, usługa pogodowa jest niedostępna."}

    try:
        extraction = await _extract_location_and_focus(user_message, model_factory)
        location_hint = ""
        if extraction.location_name:
            location_hint = f"Użytkownik pyta o miejscowość: {extraction.location_name}.\n"

        focus_hint = ""
        if extraction.focus:
            focus_hint = (
                f"Użytkownik pyta szczegółowo o: {extraction.focus}."
                " Skoncentruj się na tym aspekcie.\n"
            )

        tools = _build_tools(
            forecast_provider,
            observation_provider,
            geocoder,
            date_resolver,
            location_service,
            user_id,
        )

        chat = model_factory.create_chat_model()
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Jesteś polskim asystentem pogodowym. Odpowiadaj krótko po polsku.\n\n"
                    "Aby odpowiedzieć na pytanie, użyj dostępnych narzędzi "
                    "(get_forecast, get_observations)."
                    " Wybierz tylko potrzebne zmienne — np. jeśli pytanie jest o wiatr,"
                    " poproś tylko o wind_speed_10m_ms,"
                    " wind_gusts_10m_ms, wind_direction_10m_deg.\n"
                    f"{location_hint}{focus_hint}"
                    "Po otrzymaniu danych, napisz zwięzłą, naturalną odpowiedź po polsku."
                    " Podaj lokalizację i zakres czasu."
                ),
            },
            {"role": "user", "content": user_message},
        ]

        max_iterations = 5
        for _ in range(max_iterations):
            llm_start = time.perf_counter()
            try:
                response = await chat.ainvoke(messages, tools=tools)
                LLM_REQUESTS_TOTAL.labels(outcome="success").inc()
            except Exception:
                LLM_REQUESTS_TOTAL.labels(outcome="failure").inc()
                raise
            finally:
                LLM_REQUEST_DURATION_SECONDS.observe(time.perf_counter() - llm_start)

            if not response.tool_calls:
                answer = str(response.content) if response.content else ""
                if not answer:
                    answer = "Przepraszam, nie udało się przetworzyć zapytania."
                return {"answer": answer, "resolved_location": state.get("resolved_location")}

            for tc in response.tool_calls:
                tc_dict = tc if isinstance(tc, dict) else dict(tc)
                tool_name: str = tc_dict.get("name", "")
                tool_args: Any = tc_dict.get("args", {})
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}

                logger.info(
                    "tool_call_executed",
                    tool_name=tool_name,
                    location_name=tool_args.get("location_name")
                    if isinstance(tool_args, dict)
                    else None,
                    time_expression=tool_args.get("time_expression")
                    if isinstance(tool_args, dict)
                    else None,
                )
                result = await _execute_tool_call(
                    tool_name,
                    tool_args,
                    forecast_provider,
                    observation_provider,
                    geocoder,
                    date_resolver,
                    location_service,
                    user_id,
                )

                messages.append({"role": "assistant", "content": None, "tool_calls": [tc_dict]})
                tc_id: str = str(tc_dict.get("id", ""))
                messages.append(
                    {
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tc_id,
                    }
                )

                if tool_name == "get_forecast" and '"error"' not in result:
                    try:
                        data = json.loads(result)
                        loc_name = data.get("location")
                        if loc_name:
                            loc_ref = await _resolve_location(
                                loc_name,
                                geocoder,
                                location_service,
                                user_id,
                            )
                            if loc_ref:
                                state = {**state, "resolved_location": loc_ref}
                    except Exception:
                        pass

        return {"answer": "Przepraszam, nie udało się przetworzyć zapytania po zbyt wielu krokach."}
    except Exception:
        logger.exception(
            "weather_agent_node_failed",
            user_id=user_id,
            user_message=user_message,
        )
        return {"answer": _GENERIC_USER_ERROR}


async def resolve_location_node(
    state: ConversationState,
    location_service: LocationService | None,
    user_id: int,
    geocoder: Any | None = None,
    model_factory: ModelFactory | None = None,
) -> dict[str, Any]:
    message = state.get("user_message") or ""
    location_ref = state.get("resolved_location")

    if location_ref is not None:
        return {"resolved_location": location_ref}

    extraction = await _extract_location_and_focus(message, model_factory)
    extracted = extraction.location_name

    updates: dict[str, Any] = {}
    if extraction.focus:
        updates["user_focus"] = extraction.focus

    if extracted:
        if location_service is not None:
            resolved = await location_service.resolve_location(extracted, user_id)
            if resolved is not None:
                return {"resolved_location": resolved, **updates}

        if geocoder is not None:
            resolved = await geocoder.geocode(extracted)
            if resolved is not None:
                return {"resolved_location": resolved, **updates}

        return {
            "error": (
                f"Nie udało się rozpoznać lokalizacji \u201e{extracted}\u201d."
                " Podaj lokalizację jawnie (np. w Gdańsku)."
            ),
            "resolved_location": None,
            **updates,
        }

    return {
        "error": (
            'Nie podałeś lokalizacji. Napisz np. „jaka pogoda w Gdańsku jutro"'
            " lub ustaw lokalizację domową komendą /dodaj_lok."
        ),
        "resolved_location": None,
        **updates,
    }


async def resolve_time_range_node(
    state: ConversationState,
    date_resolver: DateResolver | None,
) -> dict[str, Any]:
    if date_resolver is None:
        return {"resolved_time_range": state.get("resolved_time_range")}

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
