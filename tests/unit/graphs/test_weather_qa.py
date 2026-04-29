from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from freezegun import freeze_time

from weather_agent.domain.date_resolver import DateResolver, ResolvedTimeRange
from weather_agent.domain.locations import Location, LocationService
from weather_agent.domain.weather import (
    ForecastPoint,
    ForecastResult,
    LocationRef,
    ObservationPoint,
    ObservationResult,
)
from weather_agent.graphs.conversation import compile_conversation_graph
from weather_agent.graphs.nodes.weather_qa import (
    _extract_time_reference,
    resolve_location_node,
    resolve_time_range_node,
    weather_agent_node,
)
from weather_agent.graphs.state import ConversationState


def _loc(name: str = "Warszawa", lat: float = 52.22, lon: float = 21.01) -> LocationRef:
    return LocationRef(id="1", name=name, latitude=lat, longitude=lon)


def _time_range(
    start: datetime | None = None,
    end: datetime | None = None,
    explanation: str = "Jutro",
) -> ResolvedTimeRange:
    if start is None:
        start = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)
    if end is None:
        end = datetime(2026, 5, 1, 23, 59, tzinfo=UTC)
    return ResolvedTimeRange(start=start, end=end, explanation=explanation)


def _forecast(
    location: LocationRef | None = None,
    points: list[ForecastPoint] | None = None,
) -> ForecastResult:
    if location is None:
        location = _loc()
    if points is None:
        points = [
            ForecastPoint(
                target_time=datetime(2026, 5, 1, 12, 0, tzinfo=UTC),
                fetched_at=datetime(2026, 5, 1, 10, 0, tzinfo=UTC),
                provider="open-meteo",
                model="dwd-icon",
                location_id=location.id,
                temperature_2m_c=18.5,
                apparent_temperature_c=17.2,
                precipitation_mm=0.4,
                precipitation_probability_pct=30,
                wind_speed_10m_ms=8.3,
                wind_gusts_10m_ms=14.1,
                cloud_cover_pct=45,
                relative_humidity_2m_pct=65,
                weather_code="3",
                raw_payload={},
            ),
        ]
    return ForecastResult(
        provider="open-meteo",
        model="dwd-icon",
        location=location,
        fetched_at=datetime(2026, 5, 1, 10, 0, tzinfo=UTC),
        points=points,
        raw_payload={},
    )


def _observation(location: LocationRef | None = None) -> ObservationResult:
    if location is None:
        location = _loc()
    return ObservationResult(
        provider="imgw_synop",
        location=location,
        fetched_at=datetime(2026, 5, 1, 10, 0, tzinfo=UTC),
        points=[
            ObservationPoint(
                observed_at=datetime(2026, 5, 1, 9, 0, tzinfo=UTC),
                fetched_at=datetime(2026, 5, 1, 10, 0, tzinfo=UTC),
                provider="imgw_synop",
                station_id="123",
                station_name="Warszawa",
                distance_km=5.3,
                temperature_c=17.5,
                wind_speed_ms=7.1,
                pressure_hpa=1013.2,
                humidity_pct=68,
                raw_payload={},
            )
        ],
        raw_payload={},
    )


def _state(**overrides: Any) -> ConversationState:
    base = ConversationState(
        authorized_user_id=12345,
        chat_id=999,
        message_thread_id=None,
        context_key="999",
        user_message="jaka będzie jutro pogoda?",
        resolved_intent=None,
        resolved_location=None,
        resolved_time_range=None,
        forecast_result=None,
        observation_result=None,
        pending_confirmation=None,
        cel_expression=None,
        cel_validation_result=None,
        answer=None,
        error=None,
    )
    for k, v in overrides.items():
        base[k] = v  # type: ignore[literal-required]
    return base


def _mock_location_service(
    resolved: LocationRef | None = None,
    locations: list[Location] | None = None,
    default_location: LocationRef | None = None,
) -> AsyncMock:
    svc = AsyncMock(spec=LocationService)
    svc.resolve_location = AsyncMock(return_value=resolved)
    svc.get_default_location = AsyncMock(return_value=default_location)
    if locations is not None:
        svc.list_locations = AsyncMock(return_value=locations)
    else:
        svc.list_locations = AsyncMock(return_value=[])
    return svc


def _mock_forecast_provider(forecast: ForecastResult | None = None) -> AsyncMock:
    provider = AsyncMock()
    if forecast is not None:
        provider.get_forecast = AsyncMock(return_value=forecast)
    else:
        provider.get_forecast = AsyncMock(return_value=_forecast())
    return provider


def _mock_observation_provider(obs: ObservationResult | None = None) -> AsyncMock:
    provider = AsyncMock()
    if obs is not None:
        provider.get_observations = AsyncMock(return_value=obs)
    else:
        provider.get_observations = AsyncMock(return_value=_observation())
    return provider


def _mock_geocoder(loc: LocationRef | None = None) -> AsyncMock:
    geocoder = AsyncMock()
    geocoder.geocode = AsyncMock(return_value=loc or _loc())
    return geocoder


def _mock_model_factory_with_answer(
    answer: str = "Jutro w Warszawie 18°C, wiatr 8 m/s.",
) -> MagicMock:
    mf = MagicMock()
    chat = AsyncMock()
    response_no_tools = MagicMock()
    response_no_tools.content = answer
    response_no_tools.tool_calls = []
    chat.ainvoke = AsyncMock(return_value=response_no_tools)
    chat.with_structured_output = MagicMock(return_value=chat)
    mf.create_chat_model = MagicMock(return_value=chat)
    return mf


def _mock_model_factory_with_location(
    location_name: str | None = "Warszawa",
    focus: str | None = None,
) -> MagicMock:
    from weather_agent.graphs.nodes.weather_qa import _LocationExtraction

    mf = MagicMock()
    chat = MagicMock()
    extraction = _LocationExtraction(location_name=location_name, focus=focus)
    structured = AsyncMock()
    structured.ainvoke = AsyncMock(return_value=extraction)
    chat.with_structured_output = MagicMock(return_value=structured)
    mf.create_chat_model = MagicMock(return_value=chat)
    return mf


def _mock_model_factory_with_tool_call(
    tool_name: str = "get_forecast",
    tool_args: dict[str, Any] | None = None,
    answer: str = "Jutro w Warszawie 18°C, wiatr 8 m/s.",
) -> MagicMock:
    mf = MagicMock()
    chat = AsyncMock()

    tool_call_response = MagicMock()
    tool_call_response.content = ""
    tool_call_response.tool_calls = [
        {
            "name": tool_name,
            "args": tool_args or {
                "location_name": "Warszawa",
                "time_expression": "jutro",
            },
            "id": "tc_1",
        },
    ]

    answer_response = MagicMock()
    answer_response.content = answer
    answer_response.tool_calls = []

    chat.ainvoke = AsyncMock(side_effect=[tool_call_response, answer_response])
    chat.with_structured_output = MagicMock(return_value=chat)
    mf.create_chat_model = MagicMock(return_value=chat)
    return mf


class TestExtractTimeReference:
    def test_extracts_jutro(self) -> None:
        result = _extract_time_reference("jaka będzie jutro pogoda?")
        assert result == "jutro"

    def test_extracts_weekend(self) -> None:
        result = _extract_time_reference("jaka będzie pogoda w weekend?")
        assert result == "weekend"

    def test_extracts_majowka(self) -> None:
        result = _extract_time_reference("pogoda na majówkę")
        assert result is not None
        assert "majówk" in result or "majowk" in result

    def test_extracts_dzis_wieczorem(self) -> None:
        result = _extract_time_reference("jak się ubrać na dziś wieczór?")
        assert result is not None
        assert "wieczor" in result or "wieczór" in result

    def test_extracts_nastepne_3_dni(self) -> None:
        result = _extract_time_reference("czy będzie wietrznie przez następne 3 dni?")
        assert result is not None
        assert "następne" in result or "nastepne" in result

    def test_returns_none_for_no_time(self) -> None:
        result = _extract_time_reference("pogoda")
        assert result is None


class TestResolveLocationNode:
    @pytest.mark.asyncio
    async def test_resolves_named_location(self) -> None:
        loc_ref = _loc("Chwarzno", 54.4871, 18.4202)
        svc = _mock_location_service(resolved=loc_ref)
        mf = _mock_model_factory_with_location(location_name="Chwarzno")
        state = _state(user_message="jaka będzie pogoda w Chwarznie jutro?")
        result = await resolve_location_node(state, svc, user_id=1, model_factory=mf)
        assert result["resolved_location"] is not None
        assert result["resolved_location"].name == "Chwarzno"

    @pytest.mark.asyncio
    async def test_no_location_extracted_returns_error(self) -> None:
        svc = _mock_location_service(locations=[])
        mf = _mock_model_factory_with_location(location_name=None)
        state = _state(user_message="jaka będzie jutro pogoda?")
        result = await resolve_location_node(state, svc, user_id=1, model_factory=mf)
        assert result["resolved_location"] is None
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_no_locations_returns_error(self) -> None:
        svc = _mock_location_service(locations=[])
        svc.resolve_location = AsyncMock(return_value=None)
        mf = _mock_model_factory_with_location(location_name=None)
        state = _state(user_message="jaka będzie jutro pogoda?")
        result = await resolve_location_node(state, svc, user_id=1, model_factory=mf)
        assert result["resolved_location"] is None
        assert result["error"] is not None
        assert "lokalizacji" in result["error"].lower() or "lokalizacj" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_ambiguous_location_returns_error(self) -> None:
        locations = [
            Location(
                id=1, name="Dom", aliases=[], latitude=52.22, longitude=21.01,
                description=None, enabled=True,
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                updated_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            Location(
                id=2, name="Chwarzno", aliases=[], latitude=54.4871, longitude=18.4202,
                description=None, enabled=True,
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                updated_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
        ]
        svc = _mock_location_service(locations=locations)
        svc.resolve_location = AsyncMock(return_value=None)
        mf = _mock_model_factory_with_location(location_name=None)
        state = _state(user_message="jaka będzie jutro pogoda?")
        result = await resolve_location_node(state, svc, user_id=1, model_factory=mf)
        assert result["resolved_location"] is None
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_follow_up_without_location_returns_none(self) -> None:
        svc = _mock_location_service()
        mf = _mock_model_factory_with_location(location_name=None)
        state = _state(
            user_message="a będzie mocno wiało?",
            reply_context_turns=[{"role": "user", "text": "jaka pogoda w Gdańsku?"}],
        )
        result = await resolve_location_node(state, svc, user_id=1, model_factory=mf)
        assert result["resolved_location"] is None
        assert "error" not in result


class TestResolveTimeRangeNode:
    @pytest.mark.asyncio
    @freeze_time("2026-05-01 10:00:00", tz_offset=2)
    async def test_resolves_jutro(self) -> None:
        resolver = DateResolver()
        state = _state(user_message="jaka będzie jutro pogoda?")
        result = await resolve_time_range_node(state, resolver)
        assert result["resolved_time_range"] is not None
        assert "Jutro" in result["resolved_time_range"].explanation

    @pytest.mark.asyncio
    @freeze_time("2026-05-01 10:00:00", tz_offset=2)
    async def test_resolves_weekend(self) -> None:
        resolver = DateResolver()
        state = _state(user_message="jaka będzie pogoda w weekend?")
        result = await resolve_time_range_node(state, resolver)
        assert result["resolved_time_range"] is not None
        assert "weekend" in result["resolved_time_range"].explanation.lower()

    @pytest.mark.asyncio
    @freeze_time("2026-05-01 10:00:00", tz_offset=2)
    async def test_defaults_to_today(self) -> None:
        resolver = DateResolver()
        state = _state(user_message="jaka będzie pogoda?")
        result = await resolve_time_range_node(state, resolver)
        assert result["resolved_time_range"] is not None
        assert "Dziś" in result["resolved_time_range"].explanation

    @pytest.mark.asyncio
    async def test_preserves_existing_time_range(self) -> None:
        resolver = DateResolver()
        existing = _time_range(explanation="Custom range")
        state = _state(resolved_time_range=existing)
        result = await resolve_time_range_node(state, resolver)
        assert result["resolved_time_range"] == existing


class TestWeatherAgentNode:
    @pytest.mark.asyncio
    async def test_returns_error_when_state_has_error(self) -> None:
        state = _state(error="Błąd dostawcy prognozy (open-meteo): Server error: 500")
        result = await weather_agent_node(
            state,
            model_factory=None,
            forecast_provider=None,
            observation_provider=None,
            geocoder=None,
            date_resolver=None,
            location_service=None,
            user_id=1,
        )
        assert result["answer"] is not None
        assert "błąd" in result["answer"].lower() or "Błąd" in result["answer"]

    @pytest.mark.asyncio
    async def test_returns_unavailable_when_missing_deps(self) -> None:
        state = _state()
        result = await weather_agent_node(
            state,
            model_factory=None,
            forecast_provider=None,
            observation_provider=None,
            geocoder=None,
            date_resolver=None,
            location_service=None,
            user_id=1,
        )
        assert "niedostępna" in result["answer"].lower()

    @pytest.mark.asyncio
    async def test_llm_returns_answer_directly(self) -> None:
        mf = _mock_model_factory_with_answer("Jutro w Chwarznie 18°C, wiatr 8 m/s.")
        fp = _mock_forecast_provider()
        gc = _mock_geocoder()
        dr = DateResolver()
        ls = _mock_location_service()
        state = _state(user_message="jaka będzie pogoda w Chwarznie jutro?")
        result = await weather_agent_node(
            state,
            model_factory=mf,
            forecast_provider=fp,
            observation_provider=None,
            geocoder=gc,
            date_resolver=dr,
            location_service=ls,
            user_id=1,
        )
        assert result["answer"] is not None
        assert "Chwarzno" in result["answer"] or "Chwarzni" in result["answer"]

    @pytest.mark.asyncio
    async def test_llm_calls_tool_then_answers(self) -> None:
        fp = _mock_forecast_provider()
        gc = _mock_geocoder()
        dr = DateResolver()
        ls = _mock_location_service()

        mf = _mock_model_factory_with_tool_call(
            tool_name="get_forecast",
            tool_args={"location_name": "Warszawa", "time_expression": "jutro"},
            answer="Jutro w Warszawie 18°C, wiatr 8 m/s.",
        )
        state = _state(user_message="jaka będzie jutro pogoda w Warszawie?")
        result = await weather_agent_node(
            state,
            model_factory=mf,
            forecast_provider=fp,
            observation_provider=None,
            geocoder=gc,
            date_resolver=dr,
            location_service=ls,
            user_id=1,
        )
        assert result["answer"] is not None
        assert "Warszaw" in result["answer"]


class TestHomeDefaultFallback:
    """Home/default location fallback in resolve_location_node."""

    @pytest.mark.asyncio
    async def test_uses_default_location_when_none_extracted(self) -> None:
        default = _loc("Gdańsk", 54.35, 18.65)
        svc = _mock_location_service(default_location=default)
        mf = _mock_model_factory_with_location(location_name=None)
        state = _state(user_message="jaka będzie jutro pogoda?")
        result = await resolve_location_node(state, svc, user_id=1, model_factory=mf)
        assert result["resolved_location"] is not None
        assert result["resolved_location"].name == "Gdańsk"
        assert result["resolved_location"].latitude == 54.35
        assert result["resolved_location"].longitude == 18.65
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_default_location_not_used_when_has_explicit(self) -> None:
        default = _loc("Gdańsk", 54.35, 18.65)
        explicit = _loc("Warszawa", 52.22, 21.01)
        svc = _mock_location_service(resolved=explicit, default_location=default)
        mf = _mock_model_factory_with_location(location_name="Warszawa")
        state = _state(user_message="jaka pogoda w Warszawie?")
        result = await resolve_location_node(state, svc, user_id=1, model_factory=mf)
        assert result["resolved_location"] is not None
        assert result["resolved_location"].name == "Warszawa"
        svc.get_default_location.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_default_location_not_used_when_has_state(self) -> None:
        default = _loc("Gdańsk", 54.35, 18.65)
        state_loc = _loc("Kraków", 50.06, 19.94)
        svc = _mock_location_service(default_location=default)
        mf = _mock_model_factory_with_location(location_name=None)
        state = _state(
            user_message="jaka będzie pogoda?",
            resolved_location=state_loc,
        )
        result = await resolve_location_node(state, svc, user_id=1, model_factory=mf)
        assert result["resolved_location"] is not None
        assert result["resolved_location"].name == "Kraków"
        svc.get_default_location.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_default_location_not_used_when_reply_context(self) -> None:
        default = _loc("Gdańsk", 54.35, 18.65)
        svc = _mock_location_service(default_location=default)
        mf = _mock_model_factory_with_location(location_name=None)
        state = _state(
            user_message="a będzie mocno wiało?",
            reply_context_turns=[{"role": "user", "text": "jaka pogoda w Gdańsku?"}],
        )
        result = await resolve_location_node(state, svc, user_id=1, model_factory=mf)
        assert result["resolved_location"] is None
        assert "error" not in result
        svc.get_default_location.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_none_when_no_default(self) -> None:
        svc = _mock_location_service(default_location=None)
        mf = _mock_model_factory_with_location(location_name=None)
        state = _state(user_message="jaka będzie jutro pogoda?")
        result = await resolve_location_node(state, svc, user_id=1, model_factory=mf)
        assert result["resolved_location"] is None
        assert result["error"] is not None
        assert "lokalizac" in result["error"].lower()


class TestStubGraphBackwardsCompat:
    @pytest.mark.asyncio
    async def test_stub_graph_still_works(self) -> None:
        compiled = compile_conversation_graph()
        state = _state(user_message="jaka będzie jutro pogoda?")
        result = await compiled.ainvoke(state)
        assert result["resolved_intent"] == "weather"
        assert result["answer"] is not None