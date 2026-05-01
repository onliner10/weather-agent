from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from weather_agent.agent_factory import build_current_time_prompt_suffix, create_weather_agent
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


def _normalize_hourly_values(raw: object) -> dict[int, dict[str, float]] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    normalized: dict[int, dict[str, float]] = {}
    for raw_hour, raw_values in raw.items():
        if not isinstance(raw_values, dict):
            continue
        try:
            hour = int(raw_hour)
        except (TypeError, ValueError):
            continue
        normalized[hour] = {
            str(attr): float(value)
            for attr, value in raw_values.items()
            if isinstance(value, int | float)
        }
    return normalized


def _parse_datetime(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


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

    def __init__(
        self,
        facts: WeatherFacts,
        hourly_values: dict[int, dict[str, float]] | None = None,
        expected_target_time: datetime | None = None,
    ) -> None:
        self._facts = facts
        self._hourly_values = hourly_values
        self._expected_target_time = expected_target_time

    def _point(
        self,
        *,
        target_time: datetime,
        fetched_at: datetime,
        location: LocationRef,
        values: Mapping[str, float | None],
        raw_payload: dict[str, object],
    ) -> ForecastPoint:
        return ForecastPoint(
            target_time=target_time,
            fetched_at=fetched_at,
            provider=self.provider,
            model="eval-fixture",
            location_id=location.id,
            temperature_2m_c=values.get("temperature_c"),
            precipitation_mm=values.get("precipitation_mm"),
            wind_speed_10m_ms=values.get("wind_speed_ms"),
            wind_direction_10m_deg=values.get("wind_direction_deg"),
            pressure_msl_hpa=values.get("pressure_hpa"),
            relative_humidity_2m_pct=values.get("humidity_pct"),
            raw_payload=raw_payload,
        )

    async def get_forecast(
        self,
        location: LocationRef,
        time_range: TimeRange,
        variables: list[WeatherVariable],
        resolution: ForecastResolution,
    ) -> ForecastResult:
        del variables, resolution

        fetched_at = datetime.now(UTC)
        expected_target_time = self._expected_target_time
        if (
            self._hourly_values is not None
            and expected_target_time is not None
            and not (time_range.start <= expected_target_time <= time_range.end)
        ):
            return ForecastResult(
                provider=self.provider,
                model="eval-fixture",
                location=location,
                fetched_at=fetched_at,
                points=[],
                raw_payload={
                    "source": "eval-fixture",
                    "facts": self._facts.model_dump(),
                    "expected_target_time": expected_target_time.isoformat(),
                    "requested_start": time_range.start.isoformat(),
                    "requested_end": time_range.end.isoformat(),
                    "date_match": False,
                },
            )
        fallback_values = {
            "temperature_c": self._facts.temperature_c,
            "precipitation_mm": self._facts.precipitation_mm,
            "wind_speed_ms": self._facts.wind_speed_ms,
            "wind_direction_deg": self._facts.wind_direction_deg,
            "pressure_hpa": self._facts.pressure_hpa,
            "humidity_pct": self._facts.humidity_pct,
        }
        target_day = expected_target_time or time_range.start
        points = (
            [
                self._point(
                    target_time=target_day.replace(hour=hour, minute=0, second=0, microsecond=0),
                    fetched_at=fetched_at,
                    location=location,
                    values=values,
                    raw_payload={"hour": hour},
                )
                for hour, values in sorted(self._hourly_values.items())
            ]
            if self._hourly_values is not None
            else [
                self._point(
                    target_time=target_time,
                    fetched_at=fetched_at,
                    location=location,
                    values=fallback_values,
                    raw_payload={},
                )
                for target_time in (time_range.start, time_range.end)
            ]
        )

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


def _build_fixture_weather_tools(
    facts: WeatherFacts,
    hourly_values: dict[int, dict[str, float]] | None = None,
    expected_target_time: datetime | None = None,
) -> list[Any]:
    toolbox = WeatherToolbox(
        forecast_provider=_FixtureForecastProvider(facts, hourly_values, expected_target_time),
        observation_provider=_FixtureObservationProvider(facts),
        geocoder=cast(Any, _FixtureGeocoder(facts)),
        location_service=None,
        user_id=0,
    )
    return toolbox.to_langchain_tools()


def build_weather_answer_target(
    model: BaseChatModel,
) -> Callable[[dict[str, object]], dict[str, Any]]:
    return build_weather_answer_target_from_factory(lambda: model)


def build_weather_answer_target_from_factory(
    model_factory: Callable[[], BaseChatModel],
) -> Callable[[dict[str, object]], dict[str, Any]]:
    def weather_answer_target(inputs: dict[str, object]) -> dict[str, Any]:
        example_id = str(inputs["id"])
        question = str(inputs["question"])
        facts = WeatherFacts.model_validate(inputs["frozen_facts"])
        hourly_values = _normalize_hourly_values(inputs.get("hourly_values"))
        expected_target_time = _parse_datetime(inputs.get("expected_target_time"))
        current_time = _parse_datetime(inputs.get("current_time"))
        logger.debug("weather_grounding_eval_run", id=example_id, question=question[:80])
        agent = create_weather_agent(
            model=model_factory(),
            tools=_build_fixture_weather_tools(facts, hourly_values, expected_target_time),
            system_prompt_suffix=build_current_time_prompt_suffix(current_time),
        )

        result = cast(
            dict[str, Any],
            agent.invoke(
                {"messages": [HumanMessage(content=question)]},
                config={"configurable": {"thread_id": example_id}},
            ),
        )
        final = result["messages"][-1]
        answer = final.content if hasattr(final, "content") else str(final)
        return WeatherAnswerOutput(example_id=example_id, answer=str(answer)).model_dump()

    return weather_answer_target


def build_weather_answer_async_target_from_factory(
    model_factory: Callable[[], BaseChatModel],
) -> Callable[[dict[str, object]], Awaitable[dict[str, Any]]]:
    async def weather_answer_target(inputs: dict[str, object]) -> dict[str, Any]:
        example_id = str(inputs["id"])
        question = str(inputs["question"])
        facts = WeatherFacts.model_validate(inputs["frozen_facts"])
        hourly_values = _normalize_hourly_values(inputs.get("hourly_values"))
        expected_target_time = _parse_datetime(inputs.get("expected_target_time"))
        current_time = _parse_datetime(inputs.get("current_time"))
        logger.debug("weather_grounding_eval_run", id=example_id, question=question[:80])
        agent = create_weather_agent(
            model=model_factory(),
            tools=_build_fixture_weather_tools(facts, hourly_values, expected_target_time),
            system_prompt_suffix=build_current_time_prompt_suffix(current_time),
        )

        result = cast(
            dict[str, Any],
            await agent.ainvoke(
                {"messages": [HumanMessage(content=question)]},
                config={"configurable": {"thread_id": example_id}},
            ),
        )
        final = result["messages"][-1]
        answer = final.content if hasattr(final, "content") else str(final)
        return WeatherAnswerOutput(example_id=example_id, answer=str(answer)).model_dump()

    return weather_answer_target
