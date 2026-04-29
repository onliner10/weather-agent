from __future__ import annotations

import time as _time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langsmith import traceable

from weather_agent.agent_factory import create_weather_agent
from weather_agent.application.conversation_models import convert_turns_to_messages
from weather_agent.domain.providers import ForecastProvider, ObservationProvider
from weather_agent.infrastructure.geocoder import Geocoder
from weather_agent.llm.contracts.location import LocationExtraction
from weather_agent.llm.model_factory import ModelFactory
from weather_agent.llm.prompts.location_prompts import LOCATION_EXTRACTION_PROMPT
from weather_agent.llm.tools.weather_tools import WeatherToolbox
from weather_agent.observability.logging import get_logger
from weather_agent.observability.metrics import (
    LLM_REQUEST_DURATION_SECONDS,
    LLM_REQUESTS_TOTAL,
)

logger = get_logger(__name__)
_WARSAW = ZoneInfo("Europe/Warsaw")
_GENERIC_USER_ERROR = "Przepraszam, wystąpił błąd. Spróbuj ponownie za chwilę."


async def extract_location_and_focus(
    message: str,
    model_factory: ModelFactory | None,
) -> LocationExtraction:
    if model_factory is None:
        return LocationExtraction()
    start = _time.perf_counter()
    try:
        chat = model_factory.create_chat_model()
        structured = chat.with_structured_output(LocationExtraction)
        chain = LOCATION_EXTRACTION_PROMPT | structured
        result = await chain.ainvoke({"user_message": message})
        LLM_REQUESTS_TOTAL.labels(outcome="success").inc()
        if isinstance(result, LocationExtraction):
            return result
        return LocationExtraction()
    except Exception:
        LLM_REQUESTS_TOTAL.labels(outcome="failure").inc()
        logger.warning("llm_location_extraction_failed", exc_info=True)
        return LocationExtraction()
    finally:
        LLM_REQUEST_DURATION_SECONDS.observe(_time.perf_counter() - start)


async def resolve_location(
    message: str,
    location_service: Any,
    user_id: int,
    geocoder: Geocoder | None = None,
    model_factory: ModelFactory | None = None,
    existing_location: Any | None = None,
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
            "error": (
                f'Nie udało się rozpoznać lokalizacji \u201e{extracted}\u201d.'
                ' Podaj lokalizację jawnie (np. w Gdańsku).'
            ),
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
        "error": (
            'Nie podałeś lokalizacji. Napisz np. "jaka pogoda w Gdańsku"'
            ' lub ustaw lokalizację domową.'
        ),
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

    if None in (model_factory, forecast_provider, geocoder, date_resolver):
        return {"answer": "Przepraszam, usługa pogodowa jest niedostępna."}

    try:
        resolved_loc = state.get("resolved_location")

        toolbox = WeatherToolbox(
            forecast_provider=forecast_provider,
            observation_provider=observation_provider,
            geocoder=geocoder,
            date_resolver=date_resolver,
            location_service=location_service,
            user_id=user_id,
        )
        tools = toolbox.to_langchain_tools()

        model = model_factory.create_chat_model()
        agent = create_weather_agent(model=model, tools=tools)

        messages = []
        reply_turns = state.get("reply_context_turns")
        if reply_turns:
            converted = convert_turns_to_messages(reply_turns if isinstance(reply_turns, list) else None)
            messages.extend(converted)
        messages.append(("user", user_message))

        context_key = state.get("context_key", "")

        result = await agent.ainvoke(
            {"messages": messages},
            config={"configurable": {"thread_id": context_key}},
        )

        final = result["messages"][-1]
        answer = final.content if hasattr(final, "content") else str(final)
        if not answer:
            answer = "Przepraszam, nie udało się przetworzyć zapytania."

        return {"answer": answer, "resolved_location": resolved_loc}

    except Exception:
        logger.exception("weather_handler_failed", user_id=user_id, user_message=user_message)
        return {"answer": _GENERIC_USER_ERROR}
