from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langsmith import traceable
from pydantic import BaseModel, Field

from weather_agent.domain.providers import ForecastProvider, ObservationProvider
from weather_agent.domain.weather import LocationRef
from weather_agent.infrastructure.geocoder import Geocoder
from weather_agent.llm.model_factory import ModelFactory
from weather_agent.llm.prompts.location_prompts import build_location_extraction_prompt
from weather_agent.llm.prompts.weather_prompts import build_weather_system_prompt
from weather_agent.llm.tools.weather_tools import WeatherToolbox
from weather_agent.observability.logging import get_logger
from weather_agent.observability.metrics import (
    LLM_REQUEST_DURATION_SECONDS,
    LLM_REQUESTS_TOTAL,
)

logger = get_logger(__name__)
_WARSAW = ZoneInfo("Europe/Warsaw")
_GENERIC_USER_ERROR = "Przepraszam, wystąpił błąd. Spróbuj ponownie za chwilę."


class _LocationExtraction(BaseModel):
    location_name: str | None = Field(default=None, description="Place name in nominative case, or null if none")
    focus: str | None = Field(default=None, description="What user asks about e.g. wiatr, temperatura. Null if general.")


async def extract_location_and_focus(
    message: str,
    model_factory: ModelFactory | None,
) -> _LocationExtraction:
    if model_factory is None:
        return _LocationExtraction()
    start = time.perf_counter()
    try:
        chat = model_factory.create_chat_model()
        structured = chat.with_structured_output(_LocationExtraction)
        prompt_text = build_location_extraction_prompt()
        result = await structured.ainvoke(
            [
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": message},
            ],
        )
        LLM_REQUESTS_TOTAL.labels(outcome="success").inc()
        if isinstance(result, _LocationExtraction):
            return result
        return _LocationExtraction()
    except Exception:
        LLM_REQUESTS_TOTAL.labels(outcome="failure").inc()
        logger.warning("llm_location_extraction_failed", exc_info=True)
        return _LocationExtraction()
    finally:
        LLM_REQUEST_DURATION_SECONDS.observe(time.perf_counter() - start)


async def resolve_location(
    message: str,
    location_service: Any,
    user_id: int,
    geocoder: Geocoder | None = None,
    model_factory: ModelFactory | None = None,
    existing_location: LocationRef | None = None,
    reply_context_turns: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:

    extraction = await extract_location_and_focus(message, model_factory)
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
            "error": f'Nie udało się rozpoznać lokalizacji \u201e{extracted}\u201d. Podaj lokalizację jawnie (np. w Gdańsku).',
            "resolved_location": None,
            **updates,
        }

    if existing_location is not None:
        return {"resolved_location": existing_location, **updates}

    if reply_context_turns:
        return {"resolved_location": None, **updates}

    if location_service is not None:
        default = await location_service.get_default_location(user_id)
        if default is not None:
            return {"resolved_location": default, **updates}

    return {
        "error": 'Nie podałeś lokalizacji. Napisz np. „jaka pogoda w Gdańsku" lub ustaw lokalizację domową.',
        "resolved_location": None,
        **updates,
    }


@traceable(run_type="chain", name="weather_handler")
async def handle_weather(
    state: dict[str, Any],
    model_factory: ModelFactory | None,
    forecast_provider: ForecastProvider | None,
    observation_provider: ObservationProvider | None,
    geocoder: Geocoder | None,
    date_resolver: Any | None,
    location_service: Any | None,
    user_id: int = 0,
) -> dict[str, Any]:
    if state.get("error"):
        logger.error("weather_handler_error_in_state", state_error=state["error"])
        return {"answer": _GENERIC_USER_ERROR}

    user_message = state.get("user_message") or ""

    if model_factory is None or forecast_provider is None or geocoder is None or date_resolver is None:
        return {"answer": "Przepraszam, usługa pogodowa jest niedostępna."}

    try:
        resolved_loc = state.get("resolved_location")
        tools = _build_tools(forecast_provider, observation_provider, geocoder, date_resolver, location_service, user_id)

        now_warsaw = datetime.now(_WARSAW)
        system_content = build_weather_system_prompt(
            now=now_warsaw,
            resolved_location=resolved_loc,
            user_focus=state.get("user_focus"),
        )

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]

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
                    location_name=tool_args.get("location_name") if isinstance(tool_args, dict) else None,
                    start_date=tool_args.get("start_date") if isinstance(tool_args, dict) else None,
                    end_date=tool_args.get("end_date") if isinstance(tool_args, dict) else None,
                )

                toolbox = WeatherToolbox(
                    forecast_provider=forecast_provider,
                    observation_provider=observation_provider,
                    geocoder=geocoder,
                    date_resolver=date_resolver,
                    location_service=location_service,
                    user_id=user_id,
                )
                result = ""
                if tool_name == "get_forecast":
                    result = await toolbox.get_forecast(
                        location_name=tool_args.get("location_name", ""),
                        start_date=tool_args.get("start_date", ""),
                        end_date=tool_args.get("end_date", ""),
                        variables=tool_args.get("variables", []),
                    )
                elif tool_name == "get_observations":
                    result = await toolbox.get_observations(
                        location_name=tool_args.get("location_name", ""),
                    )
                elif tool_name == "save_location":
                    result = await toolbox.save_location(
                        location_name=tool_args.get("location_name", ""),
                        alias=tool_args.get("alias", ""),
                    )
                else:
                    result = json.dumps({"error": f"Unknown tool: {tool_name}"})

                messages.append({"role": "assistant", "content": None, "tool_calls": [tc_dict]})
                tc_id: str = str(tc_dict.get("id", ""))
                messages.append({"role": "tool", "content": result, "tool_call_id": tc_id})

        return {"answer": "Przepraszam, nie udało się przetworzyć zapytania po zbyt wielu krokach."}
    except Exception:
        logger.exception("weather_handler_failed", user_id=user_id, user_message=user_message)
        return {"answer": _GENERIC_USER_ERROR}


def _build_tools(
    forecast_provider: ForecastProvider,
    observation_provider: ObservationProvider | None,
    geocoder: Geocoder,
    date_resolver: Any | None,
    location_service: Any | None,
    user_id: int,
) -> list:
    toolbox = WeatherToolbox(
        forecast_provider=forecast_provider,
        observation_provider=observation_provider,
        geocoder=geocoder,
        date_resolver=date_resolver,
        location_service=location_service,
        user_id=user_id,
    )
    return toolbox.to_langchain_tools()