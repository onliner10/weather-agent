from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from weather_agent.agent_factory import create_weather_agent
from weather_agent.domain.providers import ForecastProvider, ObservationProvider
from weather_agent.domain.weather import (
    ForecastPoint,
    ForecastResolution,
    ForecastResult,
    LocationRef,
    ObservationPoint,
    ObservationResult,
    TimeRange,
    WeatherVariable,
)
from weather_agent.eval.schemas import WeatherAnswerOutput, WeatherFacts
from weather_agent.llm.tools.weather_tools import WeatherToolbox
from weather_agent.observability.logging import get_logger

logger = get_logger(__name__)


class _FixtureGeocoder:
    def __init__(self, facts: WeatherFacts) -> None:
        self._facts = facts

    async def geocode(self, query: str) -> LocationRef | None:
        return LocationRef(
            id="eval-location",
            name=self._facts.location,
            latitude=54.0,
            longitude=18.0,
        )


class _FixtureForecastProvider(ForecastProvider):
    provider = "eval-fixture"

    def __init__(self, facts: WeatherFacts) -> None:
        self._facts = facts

    async def get_forecast(
        self,
        location: LocationRef,
        time_range: TimeRange,
        variables: list[WeatherVariable],
        resolution: ForecastResolution,
    ) -> ForecastResult:
        del variables, resolution

        fetched_at = datetime.now(UTC)
        points = [
            ForecastPoint(
                target_time=time_range.start,
                fetched_at=fetched_at,
                provider=self.provider,
                model="eval-fixture",
                location_id=location.id,
                temperature_2m_c=self._facts.temperature_c,
                precipitation_mm=self._facts.precipitation_mm,
                wind_speed_10m_ms=self._facts.wind_speed_ms,
                wind_direction_10m_deg=self._facts.wind_direction_deg,
                pressure_msl_hpa=self._facts.pressure_hpa,
                relative_humidity_2m_pct=self._facts.humidity_pct,
                raw_payload={},
            ),
            ForecastPoint(
                target_time=time_range.end,
                fetched_at=fetched_at,
                provider=self.provider,
                model="eval-fixture",
                location_id=location.id,
                temperature_2m_c=self._facts.temperature_c,
                precipitation_mm=self._facts.precipitation_mm,
                wind_speed_10m_ms=self._facts.wind_speed_ms,
                wind_direction_10m_deg=self._facts.wind_direction_deg,
                pressure_msl_hpa=self._facts.pressure_hpa,
                relative_humidity_2m_pct=self._facts.humidity_pct,
                raw_payload={},
            ),
        ]
        return ForecastResult(
            provider=self.provider,
            model="eval-fixture",
            location=location,
            fetched_at=fetched_at,
            points=points,
            raw_payload={"source": "eval-fixture", "facts": self._facts.model_dump()},
        )


class _FixtureObservationProvider(ObservationProvider):
    provider = "eval-fixture"

    def __init__(self, facts: WeatherFacts) -> None:
        self._facts = facts

    async def get_observations(
        self,
        location: LocationRef,
        radius_km: float,
        variables: list[WeatherVariable],
    ) -> ObservationResult:
        del radius_km, variables

        fetched_at = datetime.now(UTC)
        point = ObservationPoint(
            observed_at=fetched_at,
            fetched_at=fetched_at,
            provider=self.provider,
            station_id="eval-station",
            station_name=f"{location.name} fixture station",
            distance_km=0.0,
            temperature_c=self._facts.temperature_c,
            wind_speed_ms=self._facts.wind_speed_ms,
            wind_direction_deg=self._facts.wind_direction_deg,
            pressure_hpa=self._facts.pressure_hpa,
            humidity_pct=self._facts.humidity_pct,
            precipitation_mm=self._facts.precipitation_mm,
            raw_payload={},
        )
        return ObservationResult(
            provider=self.provider,
            location=location,
            fetched_at=fetched_at,
            points=[point],
            raw_payload={"source": "eval-fixture", "facts": self._facts.model_dump()},
        )


def _build_fixture_weather_tools(facts: WeatherFacts) -> list[Any]:
    toolbox = WeatherToolbox(
        forecast_provider=_FixtureForecastProvider(facts),
        observation_provider=_FixtureObservationProvider(facts),
        geocoder=cast(Any, _FixtureGeocoder(facts)),
        location_service=None,
        user_id=0,
    )
    return toolbox.to_langchain_tools()


def build_weather_answer_target(
    model: BaseChatModel,
) -> Callable[[dict[str, object]], dict[str, Any]]:
    def weather_answer_target(inputs: dict[str, object]) -> dict[str, Any]:
        example_id = str(inputs["id"])
        question = str(inputs["question"])
        facts = WeatherFacts.model_validate(inputs["frozen_facts"])
        logger.debug("weather_grounding_eval_run", id=example_id, question=question[:80])
        agent = create_weather_agent(
            model=model,
            tools=_build_fixture_weather_tools(facts),
        )

        async def _run_agent() -> dict[str, Any]:
            return cast(
                dict[str, Any],
                await agent.ainvoke(
                    {"messages": [HumanMessage(content=question)]},
                    config={"configurable": {"thread_id": example_id}},
                ),
            )

        result = asyncio.run(_run_agent())
        final = result["messages"][-1]
        answer = final.content if hasattr(final, "content") else str(final)
        return WeatherAnswerOutput(example_id=example_id, answer=str(answer)).model_dump()

    return weather_answer_target
