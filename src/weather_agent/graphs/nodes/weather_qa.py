from __future__ import annotations

import json
import time
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from langsmith import traceable
from pydantic import BaseModel, Field

from weather_agent.application.conversation_models import convert_turns_to_messages
from weather_agent.domain.date_resolver import DateResolver, ResolvedTimeRange
from weather_agent.domain.errors import WeatherProviderError
from weather_agent.domain.locations import (
    LocationAliasConflictError,
    LocationCreate,
    LocationNameConflictError,
    LocationService,
)
from weather_agent.domain.providers import ForecastProvider, ObservationProvider
from weather_agent.domain.weather import (
    ForecastResolution,
    LocationRef,
    WeatherVariable,
)
from weather_agent.graphs.state import ConversationState
from weather_agent.infrastructure.geocoder import Geocoder
from weather_agent.llm.contracts.location import LocationExtraction
from weather_agent.llm.model_factory import ModelFactory
from weather_agent.llm.prompts.location_prompts import LOCATION_EXTRACTION_PROMPT
from weather_agent.llm.prompts.weather_prompts import WEATHER_PROMPT
from weather_agent.llm.tools.weather_tools import WeatherToolbox
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

_WARSAW = ZoneInfo("Europe/Warsaw")

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


async def _extract_location_and_focus(
    message: str,
    model_factory: ModelFactory | None,
) -> LocationExtraction:
    if model_factory is None:
        return LocationExtraction(location_name=None, focus=None)
    start = time.perf_counter()
    try:
        chat = model_factory.create_chat_model()
        structured = chat.with_structured_output(LocationExtraction)
        chain = LOCATION_EXTRACTION_PROMPT | structured
        result = await chain.ainvoke({"user_message": message})
        LLM_REQUESTS_TOTAL.labels(outcome="success").inc()
        if isinstance(result, LocationExtraction):
            return result
    except Exception:
        LLM_REQUESTS_TOTAL.labels(outcome="failure").inc()
        logger.warning(
            "llm_location_extraction_failed",
            exc_info=True,
        )
    finally:
        LLM_REQUEST_DURATION_SECONDS.observe(time.perf_counter() - start)
    return LocationExtraction(location_name=None, focus=None)


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
                    "Pobierz prognozę pogody dla lokalizacji i zakresu dat."
                    " Zwraca godzinowe dane: temperatura, opady, wiatr, zachmurzenie itp."
                    " Daty podawaj jako yyyy-mm-dd w strefie Europe/Warsaw."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location_name": {
                            "type": "string",
                            "description": "Nazwa miejscowości (np. Gdańsk, Chwarzno)",
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Data początkowa w formacie yyyy-mm-dd (np. 2026-05-01)",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "Konkowa data yyyy-mm-dd. Dla jednego dnia powtórz.",
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
                    "required": ["location_name", "start_date", "end_date"],
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
        {
            "type": "function",
            "function": {
                "name": "save_location",
                "description": (
                    "Zapisz lokalizację użytkownika (np. dom, praca) pod konkretną nazwą lub adresem."
                    " Jeśli użytkownik prosi o 'zapamiętanie lokalizacji domowej', ustaw alias na 'dom'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location_name": {
                            "type": "string",
                            "description": "Adres lub nazwa miejscowości do zapisania (np. Gdańsk, Chwarzno)",
                        },
                        "alias": {
                            "type": "string",
                            "description": "Opcjonalny alias, np. 'dom', 'praca'",
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
        elif tool_name == "save_location":
            result = await _execute_save_location(
                tool_args,
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
    start_date_str = args.get("start_date")
    end_date_str = args.get("end_date")
    variable_names = args.get("variables", [])

    location = await _resolve_location(location_name, geocoder, location_service, user_id)
    if location is None:
        err_msg = f"Nie znaleziono lokalizacji: {location_name}"
        return json.dumps({"error": err_msg}, ensure_ascii=False)

    if start_date_str and end_date_str:
        try:
            s = date.fromisoformat(start_date_str)
            e = date.fromisoformat(end_date_str)
            start_dt = datetime(s.year, s.month, s.day, tzinfo=_WARSAW)
            end_dt = datetime(e.year, e.month, e.day, 23, 59, tzinfo=_WARSAW)
            if start_dt > end_dt:
                return json.dumps(
                    {"error": "start_date nie może być późniejsza niż end_date"},
                    ensure_ascii=False,
                )
            time_range = ResolvedTimeRange(
                start=start_dt,
                end=end_dt,
                explanation=f"{start_date_str} – {end_date_str}",
            )
        except ValueError:
            return json.dumps(
                {"error": "Nieprawidłowy format daty. Użyj yyyy-mm-dd."},
                ensure_ascii=False,
            )
    else:
        time_expr = args.get("time_expression")
        if time_expr:
            resolved = await date_resolver.resolve(time_expr)
            if resolved is None:
                return json.dumps(
                    {"error": f"Nie udało się rozpoznać wyrażenia czasowego: {time_expr}"},
                    ensure_ascii=False,
                )
            time_range = resolved
        else:
            return json.dumps(
                {"error": "Brak parametrów start_date/end_date lub time_expression"},
                ensure_ascii=False,
            )

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


@traceable(run_type="tool")
async def _execute_save_location(
    args: dict[str, Any],
    geocoder: Geocoder | None,
    location_service: LocationService | None,
    user_id: int,
) -> str:
    if location_service is None:
        return json.dumps({"error": "Usługa lokalizacji jest niedostępna."})

    location_name = args.get("location_name", "").strip()
    alias = args.get("alias", "").strip()

    if not location_name:
        return json.dumps({"error": "Podaj nazwę lokalizacji do zapisania."})

    if geocoder is None:
        return json.dumps({"error": "Geokoder jest niedostępny."})

    resolved = await geocoder.geocode(location_name)
    if resolved is None:
        return json.dumps(
            {"error": f"Nie udało się rozpoznać lokalizacji „{location_name}”."},
            ensure_ascii=False,
        )

    try:
        aliases = [alias] if alias else []
        await location_service.create_location(
            user_id,
            LocationCreate(
                name=location_name,
                aliases=aliases,
                latitude=resolved.latitude,
                longitude=resolved.longitude,
            ),
        )
        msg = f"Zapamiętałem lokalizację: {location_name}"
        if alias:
            msg += f" (alias: {alias})"
        return json.dumps({"success": msg}, ensure_ascii=False)
    except (LocationAliasConflictError, LocationNameConflictError):
        return json.dumps(
            {"error": "Masz już zapisaną lokalizację o tej nazwie lub aliasie."},
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.exception("save_location_failed", user_id=user_id, location_name=location_name)
        return json.dumps(
            {"error": f"Błąd podczas zapisywania lokalizacji: {str(exc)}"},
            ensure_ascii=False,
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
        resolved_loc = state.get("resolved_location")

        tools = _build_tools(
            forecast_provider,
            observation_provider,
            geocoder,
            date_resolver,
            location_service,
            user_id,
        )

        now_warsaw = datetime.now(_WARSAW)
        location_context = ""
        if resolved_loc and resolved_loc.name:
            location_context = f"Użytkownik pyta o miejscowość: {resolved_loc.name}.\n"

        focus_context = ""
        focus = state.get("user_focus")
        if focus:
            focus_context = f"Użytkownik pyta szczegółowo o: {focus}. Skoncentruj się na tym aspekcie.\n"

        system_content = WEATHER_SYSTEM_INSTRUCTIONS.format(
            date=now_warsaw.strftime("%Y-%m-%d"),
            day_of_week=now_warsaw.strftime("%A"),
            time=now_warsaw.strftime("%H:%M"),
            location_context=location_context,
            focus_context=focus_context,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
        ]

        reply_context_turns = state.get("reply_context_turns")
        if reply_context_turns:
            for turn in reply_context_turns:
                role = turn.get("role", "")
                text = turn.get("text") or ""
                answer = turn.get("answer_summary") or ""
                if role == "user" and text:
                    messages.append({"role": "user", "content": text})
                elif role == "bot" and answer:
                    messages.append({"role": "assistant", "content": answer})

        messages.append({"role": "user", "content": user_message})

        chat = model_factory.create_chat_model()

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
                return {"answer": answer, "resolved_location": resolved_loc}

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
                    start_date=tool_args.get("start_date") if isinstance(tool_args, dict) else None,
                    end_date=tool_args.get("end_date") if isinstance(tool_args, dict) else None,
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

    existing = state.get("resolved_location")
    if existing is not None:
        return {"resolved_location": existing, **updates}

    if state.get("reply_context_turns"):
        return {"resolved_location": None, **updates}

    if location_service is not None:
        default = await location_service.get_default_location(user_id)
        if default is not None:
            return {"resolved_location": default, **updates}

    return {
        "error": (
            'Nie podałeś lokalizacji. Napisz np. „jaka pogoda w Gdańsku jutro"'
            " lub ustaw lokalizację domową."
        ),
        "resolved_location": None,
        **updates,
    }


