from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from freezegun import freeze_time

from weather_agent.domain.date_resolver import DateResolver, ResolvedTimeRange
from weather_agent.domain.errors import WeatherProviderResponseError
from weather_agent.domain.locations import Location, LocationService
from weather_agent.domain.weather import (
    ForecastPoint,
    ForecastResult,
    LocationRef,
    ObservationPoint,
    ObservationResult,
    WeatherVariable,
)
from weather_agent.graphs.conversation import (
    WeatherQADependencies,
    compile_conversation_graph,
)
from weather_agent.graphs.nodes.weather_qa import (
    _extract_location_reference,
    _extract_time_reference,
    _format_forecast_summary,
    _select_variables,
    answer_weather_question_node,
    call_weather_tools_node,
    resolve_location_node,
    resolve_time_range_node,
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


def _state(**overrides: object) -> ConversationState:
    base: ConversationState = {
        "authorized_user_id": 12345,
        "chat_id": 999,
        "message_thread_id": None,
        "context_key": "999",
        "user_message": "jaka będzie jutro pogoda?",
        "resolved_intent": None,
        "resolved_location": None,
        "resolved_time_range": None,
        "forecast_result": None,
        "observation_result": None,
        "pending_confirmation": None,
        "cel_expression": None,
        "cel_validation_result": None,
        "answer": None,
        "error": None,
    }
    base.update(overrides)
    return base


def _mock_location_service(
    resolved: LocationRef | None = None,
    locations: list[Location] | None = None,
) -> AsyncMock:
    svc = AsyncMock(spec=LocationService)
    svc.resolve_location = AsyncMock(return_value=resolved)
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


class TestExtractLocationReference:
    def test_extracts_location_with_w_prefix(self) -> None:
        result = _extract_location_reference("jaka będzie pogoda w Chwarznie?")
        assert result == "Chwarznie"

    def test_extracts_capitalized_location(self) -> None:
        result = _extract_location_reference("jaka będzie pogoda w Warszawie jutro?")
        assert result == "Warszawie"

    def test_returns_none_for_no_location(self) -> None:
        result = _extract_location_reference("jaka będzie jutro pogoda?")
        assert result is None

    def test_extracts_lowercase_location(self) -> None:
        result = _extract_location_reference("pogoda w krakowie")
        assert result == "krakowie"


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
        assert "wieczor" in result or "wieczór" in result

    def test_extracts_nastepne_3_dni(self) -> None:
        result = _extract_time_reference("czy będzie wietrznie przez następne 3 dni?")
        assert "następne" in result or "nastepne" in result

    def test_returns_none_for_no_time(self) -> None:
        result = _extract_time_reference("pogoda")
        assert result is None


class TestSelectVariables:
    def test_wind_keywords_include_wind_vars(self) -> None:
        vars = _select_variables("czy będzie mocno wietrznie przez następne 3 dni?")
        assert WeatherVariable.wind_speed_10m_ms in vars
        assert WeatherVariable.wind_gusts_10m_ms in vars

    def test_rain_keywords_include_rain_vars(self) -> None:
        vars = _select_variables("czy będzie padać jutro?")
        assert WeatherVariable.rain_mm in vars

    def test_default_variables_include_temperature(self) -> None:
        vars = _select_variables("jaka będzie jutro pogoda?")
        assert WeatherVariable.temperature_2m_c in vars


class TestFormatForecastSummary:
    def test_basic_summary_includes_location(self) -> None:
        loc = _loc("Chwarzno")
        tr = _time_range(explanation="Jutro (2026-05-02)")
        forecast = _forecast(location=loc)
        summary = _format_forecast_summary(loc, tr, forecast)
        assert "Chwarzno" in summary
        assert "Jutro" in summary

    def test_summary_includes_temperature_range(self) -> None:
        loc = _loc()
        tr = _time_range()
        forecast = _forecast(location=loc)
        summary = _format_forecast_summary(loc, tr, forecast)
        assert "°C" in summary

    def test_summary_includes_wind(self) -> None:
        loc = _loc()
        tr = _time_range()
        forecast = _forecast(location=loc)
        summary = _format_forecast_summary(loc, tr, forecast)
        assert "m/s" in summary

    def test_summary_includes_observation(self) -> None:
        loc = _loc()
        tr = _time_range()
        forecast = _forecast(location=loc)
        obs = _observation(location=loc)
        summary = _format_forecast_summary(loc, tr, forecast, obs)
        assert "Obecnie" in summary
        assert "Warszawa" in summary

    def test_empty_forecast_points(self) -> None:
        loc = _loc()
        tr = _time_range(explanation="Dziś")
        forecast = _forecast(location=loc, points=[])
        summary = _format_forecast_summary(loc, tr, forecast)
        assert "Brak danych" in summary


class TestResolveLocationNode:
    @pytest.mark.asyncio
    async def test_resolves_named_location(self) -> None:
        loc_ref = _loc("Chwarzno", 54.4871, 18.4202)
        svc = _mock_location_service(resolved=loc_ref)
        state = _state(user_message="jaka będzie pogoda w Chwarznie jutro?")
        result = await resolve_location_node(state, svc, user_id=1)
        assert result["resolved_location"] is not None
        assert result["resolved_location"].name == "Chwarzno"

    @pytest.mark.asyncio
    async def test_single_default_location(self) -> None:
        locations = [
            Location(
                id=1, name="Dom", aliases=[], latitude=52.22, longitude=21.01,
                description=None, enabled=True,
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                updated_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        ]
        svc = _mock_location_service(locations=locations)
        svc.resolve_location = AsyncMock(return_value=None)
        state = _state(user_message="jaka będzie jutro pogoda?")
        result = await resolve_location_node(state, svc, user_id=1)
        assert result["resolved_location"] is not None
        assert result["resolved_location"].name == "Dom"

    @pytest.mark.asyncio
    async def test_no_locations_returns_error(self) -> None:
        svc = _mock_location_service(locations=[])
        svc.resolve_location = AsyncMock(return_value=None)
        state = _state(user_message="jaka będzie jutro pogoda?")
        result = await resolve_location_node(state, svc, user_id=1)
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
        state = _state(user_message="jaka będzie jutro pogoda?")
        result = await resolve_location_node(state, svc, user_id=1)
        assert result["resolved_location"] is None
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_preresolved_location_passed_through(self) -> None:
        loc_ref = _loc("Chwarzno")
        svc = _mock_location_service()
        state = _state(resolved_location=loc_ref)
        result = await resolve_location_node(state, svc, user_id=1)
        assert result["resolved_location"] == loc_ref


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


class TestCallWeatherToolsNode:
    @pytest.mark.asyncio
    async def test_fetches_forecast_successfully(self) -> None:
        loc = _loc()
        tr = _time_range()
        forecast = _forecast(location=loc)
        fp = _mock_forecast_provider(forecast)
        state = _state(
            resolved_location=loc,
            resolved_time_range=tr,
        )
        result = await call_weather_tools_node(state, fp, observation_provider=None)
        assert result["forecast_result"] is not None
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_fetches_forecast_and_observation(self) -> None:
        loc = _loc()
        tr = _time_range()
        forecast = _forecast(location=loc)
        obs = _observation(location=loc)
        fp = _mock_forecast_provider(forecast)
        op = _mock_observation_provider(obs)
        state = _state(
            resolved_location=loc,
            resolved_time_range=tr,
        )
        result = await call_weather_tools_node(state, fp, observation_provider=op)
        assert result["forecast_result"] is not None
        assert result["observation_result"] is not None

    @pytest.mark.asyncio
    async def test_provider_error_does_not_fabricate(self) -> None:
        loc = _loc()
        tr = _time_range()
        fp = AsyncMock()
        fp.get_forecast = AsyncMock(
            side_effect=WeatherProviderResponseError("open-meteo", "Server error: 500")
        )
        state = _state(
            resolved_location=loc,
            resolved_time_range=tr,
        )
        result = await call_weather_tools_node(state, fp, observation_provider=None)
        assert result["forecast_result"] is None
        assert result["error"] is not None
        assert "open-meteo" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_location_returns_error(self) -> None:
        fp = _mock_forecast_provider()
        state = _state(resolved_location=None, resolved_time_range=_time_range())
        result = await call_weather_tools_node(state, fp, observation_provider=None)
        assert result["forecast_result"] is None
        assert result["error"] is not None
        assert "lokalizacji" in result["error"].lower() or "localizacji" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_missing_time_range_returns_error(self) -> None:
        fp = _mock_forecast_provider()
        state = _state(resolved_location=_loc(), resolved_time_range=None)
        result = await call_weather_tools_node(state, fp, observation_provider=None)
        assert result["forecast_result"] is None
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_observation_failure_does_not_block_forecast(self) -> None:
        loc = _loc()
        tr = _time_range()
        forecast = _forecast(location=loc)
        fp = _mock_forecast_provider(forecast)
        op = AsyncMock()
        op.get_observations = AsyncMock(
            side_effect=WeatherProviderResponseError("imgw", "Unavailable")
        )
        state = _state(
            resolved_location=loc,
            resolved_time_range=tr,
        )
        result = await call_weather_tools_node(state, fp, observation_provider=op)
        assert result["forecast_result"] is not None
        assert result["observation_result"] is None
        assert result["error"] is None


class TestAnswerWeatherQuestionNode:
    @pytest.mark.asyncio
    async def test_generates_polish_answer(self) -> None:
        loc = _loc("Chwarzno")
        tr = _time_range(explanation="Jutro (2026-05-02)")
        forecast = _forecast(location=loc)
        state = _state(
            resolved_location=loc,
            resolved_time_range=tr,
            forecast_result=forecast,
        )
        result = await answer_weather_question_node(state, model_factory=None)
        assert result["answer"] is not None
        assert "Chwarzno" in result["answer"]
        assert "Jutro" in result["answer"]

    @pytest.mark.asyncio
    async def test_error_in_state_produces_transparent_response(self) -> None:
        state = _state(error="Błąd dostawcy prognozy (open-meteo): Server error: 500")
        result = await answer_weather_question_node(state, model_factory=None)
        assert result["answer"] is not None
        assert "błąd" in result["answer"].lower() or "Błąd" in result["answer"]

    @pytest.mark.asyncio
    async def test_missing_location_produces_message(self) -> None:
        state = _state(resolved_location=None)
        result = await answer_weather_question_node(state, model_factory=None)
        assert result["answer"] is not None
        assert (
            "lokalizacji" in result["answer"].lower()
            or "localizacji" in result["answer"].lower()
        )

    @pytest.mark.asyncio
    async def test_missing_forecast_produces_message(self) -> None:
        state = _state(resolved_location=_loc(), forecast_result=None)
        result = await answer_weather_question_node(state, model_factory=None)
        assert result["answer"] is not None
        assert "prognozy" in result["answer"].lower() or "danych" in result["answer"].lower()

    @pytest.mark.asyncio
    async def test_answer_includes_location_and_time_range(self) -> None:
        loc = _loc("Warszawa")
        tr = _time_range(explanation="Jutro (2026-05-02)")
        forecast = _forecast(location=loc)
        state = _state(
            resolved_location=loc,
            resolved_time_range=tr,
            forecast_result=forecast,
        )
        result = await answer_weather_question_node(state, model_factory=None)
        assert "Warszawa" in result["answer"]
        assert "Jutro" in result["answer"]


class TestWeatherQAFullFlow:
    @pytest.mark.asyncio
    @freeze_time("2026-05-01 10:00:00", tz_offset=2)
    async def test_full_weather_qa_flow_with_location(self) -> None:
        loc = _loc("Chwarzno", 54.4871, 18.4202)
        forecast = _forecast(location=loc)

        ls = _mock_location_service(resolved=loc)
        dr = DateResolver()
        fp = _mock_forecast_provider(forecast)

        deps = WeatherQADependencies(
            location_service=ls,
            date_resolver=dr,
            forecast_provider=fp,
            observation_provider=None,
            user_id=1,
        )
        compiled = compile_conversation_graph(deps)

        state = _state(user_message="jaka będzie pogoda w Chwarznie jutro?")
        result = await compiled.ainvoke(state)
        assert result["resolved_intent"] == "weather"
        assert result["answer"] is not None
        assert "Chwarzno" in result["answer"]

    @pytest.mark.asyncio
    @freeze_time("2026-05-01 10:00:00", tz_offset=2)
    async def test_full_weather_qa_flow_weekend(self) -> None:
        loc = _loc("Jeziorak", 53.77, 19.71)
        forecast = _forecast(location=loc)

        ls = _mock_location_service(resolved=loc)
        dr = DateResolver()
        fp = _mock_forecast_provider(forecast)

        deps = WeatherQADependencies(
            location_service=ls,
            date_resolver=dr,
            forecast_provider=fp,
            observation_provider=None,
            user_id=1,
        )
        compiled = compile_conversation_graph(deps)

        state = _state(user_message="jaka będzie pogoda w weekend?")
        result = await compiled.ainvoke(state)
        assert result["resolved_intent"] == "weather"
        assert result["answer"] is not None

    @pytest.mark.asyncio
    @freeze_time("2026-05-01 10:00:00", tz_offset=2)
    async def test_wind_question_extracts_wind_vars(self) -> None:
        loc = _loc()
        forecast = _forecast(location=loc)

        ls = _mock_location_service(resolved=loc)
        dr = DateResolver()
        fp = _mock_forecast_provider(forecast)

        deps = WeatherQADependencies(
            location_service=ls,
            date_resolver=dr,
            forecast_provider=fp,
            observation_provider=None,
            user_id=1,
        )
        compiled = compile_conversation_graph(deps)

        state = _state(user_message="czy będzie mocno wietrznie przez następne 3 dni?")
        result = await compiled.ainvoke(state)
        assert result["resolved_intent"] == "weather"
        assert result["answer"] is not None

    @pytest.mark.asyncio
    async def test_provider_error_flow_transparent(self) -> None:
        loc = _loc("Warszawa")

        ls = _mock_location_service(resolved=loc)
        dr = DateResolver()
        fp = AsyncMock()
        fp.get_forecast = AsyncMock(
            side_effect=WeatherProviderResponseError("open-meteo", "Server error: 500")
        )

        deps = WeatherQADependencies(
            location_service=ls,
            date_resolver=dr,
            forecast_provider=fp,
            observation_provider=None,
            user_id=1,
        )
        compiled = compile_conversation_graph(deps)

        state = _state(user_message="jaka będzie jutro pogoda?")
        result = await compiled.ainvoke(state)
        assert result["answer"] is not None
        assert "błąd" in result["answer"].lower() or "Błąd" in result["answer"]

    @pytest.mark.asyncio
    async def test_majowka_question(self) -> None:
        loc = _loc()
        forecast = _forecast(location=loc)

        ls = _mock_location_service(resolved=loc)
        dr = DateResolver()
        fp = _mock_forecast_provider(forecast)

        deps = WeatherQADependencies(
            location_service=ls,
            date_resolver=dr,
            forecast_provider=fp,
            observation_provider=None,
            user_id=1,
        )
        compiled = compile_conversation_graph(deps)

        state = _state(user_message="jaka będzie pogoda na majówkę?")
        result = await compiled.ainvoke(state)
        assert result["answer"] is not None

    @pytest.mark.asyncio
    @freeze_time("2026-05-01 14:00:00", tz_offset=2)
    async def test_dzis_wieczor_question(self) -> None:
        loc = _loc()
        forecast = _forecast(location=loc)

        ls = _mock_location_service(resolved=loc)
        dr = DateResolver()
        fp = _mock_forecast_provider(forecast)

        deps = WeatherQADependencies(
            location_service=ls,
            date_resolver=dr,
            forecast_provider=fp,
            observation_provider=None,
            user_id=1,
        )
        compiled = compile_conversation_graph(deps)

        state = _state(user_message="jak się ubrać na dziś wieczór?")
        result = await compiled.ainvoke(state)
        assert result["answer"] is not None


class TestStubGraphBackwardsCompat:
    @pytest.mark.asyncio
    async def test_stub_graph_still_works(self) -> None:
        compiled = compile_conversation_graph()
        loc = _loc()
        tr = _time_range()
        fr = _forecast(location=loc)
        state = _state(
            user_message="jaka będzie jutro pogoda?",
            resolved_location=loc,
            resolved_time_range=tr,
            forecast_result=fr,
        )
        result = await compiled.ainvoke(state)
        assert result["resolved_intent"] == "weather"
        assert result["answer"] is not None