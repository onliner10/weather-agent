from __future__ import annotations

from datetime import UTC, datetime

import pytest
from langgraph.graph.state import CompiledStateGraph

from weather_agent.domain.cel.evaluator import CELEvaluationResult
from weather_agent.domain.date_resolver import ResolvedTimeRange
from weather_agent.domain.weather import ForecastResult, LocationRef
from weather_agent.graphs.conversation import (
    build_conversation_graph,
    compile_conversation_graph,
)
from weather_agent.graphs.state import ConversationState


def _default_state(**overrides: object) -> ConversationState:
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


class TestConversationState:
    def test_state_is_typed_dict(self) -> None:
        state = ConversationState(
            authorized_user_id=1,
            chat_id=100,
            message_thread_id=5,
            context_key="100:5",
            user_message="test",
        )
        assert state["authorized_user_id"] == 1
        assert state["chat_id"] == 100

    def test_state_optional_fields_default_none(self) -> None:
        state = ConversationState(
            chat_id=1,
            context_key="1",
        )
        assert state.get("resolved_intent") is None
        assert state.get("error") is None

    def test_state_with_domain_models(self) -> None:
        loc = LocationRef(id="1", name="Warszawa", latitude=52.22, longitude=21.01)
        tr = ResolvedTimeRange(
            start=datetime(2026, 5, 1, tzinfo=UTC),
            end=datetime(2026, 5, 1, 23, 59, tzinfo=UTC),
            explanation="Test",
        )
        state = ConversationState(
            chat_id=1,
            context_key="1",
            resolved_location=loc,
            resolved_time_range=tr,
        )
        assert state["resolved_location"].name == "Warszawa"
        assert state["resolved_time_range"].explanation == "Test"


class TestGraphBuild:
    def test_graph_can_be_built(self) -> None:
        graph = build_conversation_graph()
        assert graph is not None

    def test_graph_can_be_compiled(self) -> None:
        compiled = compile_conversation_graph()
        assert isinstance(compiled, CompiledStateGraph)


class TestWeatherPath:
    @pytest.mark.asyncio
    async def test_weather_question_path(self) -> None:
        compiled = compile_conversation_graph()
        loc = LocationRef(id="1", name="Warszawa", latitude=52.22, longitude=21.01)
        tr = ResolvedTimeRange(
            start=datetime(2026, 5, 1, tzinfo=UTC),
            end=datetime(2026, 5, 1, 23, 59, tzinfo=UTC),
            explanation="Jutro",
        )
        fr = ForecastResult(
            provider="open-meteo",
            location=loc,
            fetched_at=datetime.now(UTC),
            points=[],
            raw_payload={},
        )
        state = _default_state(
            user_message="jaka będzie jutro pogoda?",
            resolved_location=loc,
            resolved_time_range=tr,
            forecast_result=fr,
        )
        result = await compiled.ainvoke(state)
        assert result["resolved_intent"] == "weather"
        assert result["answer"] is not None
        assert result["authorized_user_id"] == 12345

    @pytest.mark.asyncio
    async def test_weather_question_with_location_keyword(self) -> None:
        compiled = compile_conversation_graph()
        state = _default_state(user_message="jaka będzie temperatura jutro?")
        result = await compiled.ainvoke(state)
        assert result["resolved_intent"] == "weather"
        assert result["answer"] is not None


class TestRulePath:
    @pytest.mark.asyncio
    async def test_rule_proposal_path(self) -> None:
        compiled = compile_conversation_graph()
        cel_result = CELEvaluationResult(
            expression="max(temperature_2m_c, today(), data) > 30",
            result=None,
            error=None,
            evaluated_metrics=["temperature_2m_c"],
            evaluated_functions=["max"],
        )
        state = _default_state(
            user_message="dodaj regułę: jeśli temperatura > 30, powiadom mnie",
            resolved_intent="rule",
            cel_expression="max(temperature_2m_c, today(), data) > 30",
            cel_validation_result=cel_result,
            pending_confirmation={"action": "activate_rule"},
        )
        result = await compiled.ainvoke(state)
        assert result["resolved_intent"] == "rule"
        assert result["answer"] is not None

    @pytest.mark.asyncio
    async def test_rule_intent_classification(self) -> None:
        compiled = compile_conversation_graph()
        state = _default_state(user_message="chcę ustawić powiadomienie o deszczu")
        result = await compiled.ainvoke(state)
        assert result["resolved_intent"] == "rule"


class TestCommandPath:
    @pytest.mark.asyncio
    async def test_help_command_path(self) -> None:
        compiled = compile_conversation_graph()
        state = _default_state(user_message="/help")
        result = await compiled.ainvoke(state)
        assert result["resolved_intent"] in ("command", "help")
        assert result["answer"] is not None

    @pytest.mark.asyncio
    async def test_start_command_path(self) -> None:
        compiled = compile_conversation_graph()
        state = _default_state(user_message="/start")
        result = await compiled.ainvoke(state)
        assert result["resolved_intent"] in ("command", "help")


class TestUnauthorizedUser:
    @pytest.mark.asyncio
    async def test_unauthorized_user_has_no_id(self) -> None:
        compiled = compile_conversation_graph()
        state = _default_state(authorized_user_id=None, user_message="pogoda jutro")
        result = await compiled.ainvoke(state)
        assert result["authorized_user_id"] is None
        assert result["answer"] is not None

    @pytest.mark.asyncio
    async def test_authorized_user_preserved(self) -> None:
        compiled = compile_conversation_graph()
        state = _default_state(authorized_user_id=42, user_message="pogoda jutro")
        result = await compiled.ainvoke(state)
        assert result["authorized_user_id"] == 42


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_error_in_state_produces_error_answer(self) -> None:
        compiled = compile_conversation_graph()
        state = _default_state(error="Brak danych prognozy", user_message="pogoda")
        result = await compiled.ainvoke(state)
        assert "błąd" in result["answer"].lower()