from __future__ import annotations

from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from weather_agent.adapters.telegram.context import TelegramContextService
from weather_agent.domain.cel.evaluator import CELEvaluationResult
from weather_agent.domain.date_resolver import ResolvedTimeRange
from weather_agent.domain.weather import ForecastResult, LocationRef
from weather_agent.graphs.conversation import (
    CompiledConversationGraph,
    _classify_with_pending_confirmation,
    _intent_router,
    _make_load_thread_context,
    _make_save_thread_context,
    build_conversation_graph,
    classify_intent,
    compile_conversation_graph,
)
from weather_agent.graphs.state import ConversationState
from weather_agent.infrastructure.db.base import Base
from weather_agent.infrastructure.memory.thread_memory import ThreadMemoryService


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
        assert isinstance(compiled, CompiledConversationGraph)


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


class TestConfirmationRouting:
    def test_intent_router_returns_confirm_path(self) -> None:
        state = _default_state(resolved_intent="confirm_rule")
        assert _intent_router(state) == "confirm_path"

    def test_intent_router_returns_cancel_path(self) -> None:
        state = _default_state(resolved_intent="cancel_rule")
        assert _intent_router(state) == "cancel_path"

    @pytest.mark.asyncio
    async def test_classify_intent_tak_routes_to_confirm(self) -> None:
        state = _default_state(
            user_message="tak",
            pending_confirmation={
                "action": "create_rule",
                "cel_expression": "temp > 30",
                "explanation": "test",
            },
        )
        result = await classify_intent(state)
        assert result["resolved_intent"] == "confirm_rule"

    @pytest.mark.asyncio
    async def test_classify_intent_nie_routes_to_cancel(self) -> None:
        state = _default_state(
            user_message="nie",
            pending_confirmation={
                "action": "create_rule",
                "cel_expression": "temp > 30",
                "explanation": "test",
            },
        )
        result = await classify_intent(state)
        assert result["resolved_intent"] == "cancel_rule"

    @pytest.mark.asyncio
    async def test_classify_intent_no_pending_routes_to_weather(self) -> None:
        state = _default_state(
            user_message="tak",
            pending_confirmation=None,
        )
        result = await classify_intent(state)
        assert result["resolved_intent"] == "weather"

    @pytest.mark.asyncio
    async def test_classify_intent_unrelated_message_with_pending(self) -> None:
        state = _default_state(
            user_message="jaka będzie pogoda jutro?",
            pending_confirmation={
                "action": "create_rule",
                "cel_expression": "temp > 30",
                "explanation": "test",
            },
        )
        result = await classify_intent(state)
        assert result["resolved_intent"] == "weather"

    def test_classify_with_pending_confirmation_yes(self) -> None:
        result = _classify_with_pending_confirmation(
            "tak",
            {"action": "create_rule", "cel_expression": "temp > 30"},
        )
        assert result == "confirm_rule"

    def test_classify_with_pending_confirmation_no(self) -> None:
        result = _classify_with_pending_confirmation(
            "nie",
            {"action": "create_rule", "cel_expression": "temp > 30"},
        )
        assert result == "cancel_rule"

    def test_classify_with_pending_confirmation_none_pending(self) -> None:
        result = _classify_with_pending_confirmation("tak", None)
        assert result is None

    def test_classify_with_pending_confirmation_unrelated_message(self) -> None:
        result = _classify_with_pending_confirmation(
            "jaka będzie pogoda jutro?",
            {"action": "create_rule", "cel_expression": "temp > 30"},
        )
        assert result is None


@pytest_asyncio.fixture()
async def _async_engine() -> AsyncEngine:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture()
async def _async_session(_async_engine: AsyncEngine) -> AsyncSession:
    session_factory = async_sessionmaker(_async_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        async with session.begin():
            yield session


class TestLoadThreadContextWithMemory:
    @pytest.mark.asyncio
    async def test_load_turns_into_recent_context(self, _async_session: AsyncSession) -> None:
        context_service = TelegramContextService(_async_session)
        memory = ThreadMemoryService(context_service)
        _load = _make_load_thread_context(memory)

        await memory.save_turn(
            "999:1",
            {
                "message_id": 10,
                "role": "user",
                "text": "jaka pogoda w Warszawie?",
                "timestamp": None,
            },
        )

        state: ConversationState = {
            "authorized_user_id": 12345,
            "chat_id": 999,
            "message_thread_id": 1,
            "context_key": "999:1",
            "user_message": "a wiatr?",
            "reply_to_message_id": None,
        }
        result = await _load(state)
        assert result["context_key"] == "999:1"
        assert result.get("recent_context") is not None
        assert len(result["recent_context"]) == 1
        assert result["recent_context"][0]["text"] == "jaka pogoda w Warszawie?"

    @pytest.mark.asyncio
    async def test_load_reply_anchor(self, _async_session: AsyncSession) -> None:
        context_service = TelegramContextService(_async_session)
        memory = ThreadMemoryService(context_service)
        _load = _make_load_thread_context(memory)

        await memory.save_turn(
            "999:1",
            {
                "message_id": 42,
                "role": "bot",
                "answer_summary": "W Chwarznie 18°C, wiatr 8 m/s.",
                "timestamp": None,
            },
        )

        state: ConversationState = {
            "authorized_user_id": 12345,
            "chat_id": 999,
            "message_thread_id": 1,
            "context_key": "999:1",
            "user_message": "a opady?",
            "reply_to_message_id": 42,
        }
        result = await _load(state)
        assert result.get("reply_anchor") is not None
        assert result["reply_anchor"]["role"] == "bot"
        assert result["reply_anchor"]["message_id"] == 42

    @pytest.mark.asyncio
    async def test_load_reply_anchor_not_found(self, _async_session: AsyncSession) -> None:
        context_service = TelegramContextService(_async_session)
        memory = ThreadMemoryService(context_service)
        _load = _make_load_thread_context(memory)

        await memory.save_turn(
            "999:1",
            {
                "message_id": 10,
                "role": "user",
                "text": "pogoda?",
                "timestamp": None,
            },
        )

        state: ConversationState = {
            "authorized_user_id": 12345,
            "chat_id": 999,
            "message_thread_id": 1,
            "context_key": "999:1",
            "user_message": "a wiatr?",
            "reply_to_message_id": 999,
        }
        result = await _load(state)
        assert result.get("reply_anchor") is None

    @pytest.mark.asyncio
    async def test_load_without_memory_service(self) -> None:
        _load = _make_load_thread_context(None)
        state: ConversationState = {
            "authorized_user_id": 12345,
            "chat_id": 999,
            "message_thread_id": None,
            "context_key": "999",
            "user_message": "pogoda?",
            "reply_to_message_id": None,
        }
        result = await _load(state)
        assert result["context_key"] == "999"
        assert "recent_context" not in result
        assert "reply_anchor" not in result


class TestSaveThreadContextWithMemory:
    @pytest.mark.asyncio
    async def test_save_user_and_bot_turns(self, _async_session: AsyncSession) -> None:
        context_service = TelegramContextService(_async_session)
        memory = ThreadMemoryService(context_service)
        _save = _make_save_thread_context(memory)

        loc = LocationRef(id="1", name="Chwarzno", latitude=54.4871, longitude=18.4202)
        tr = ResolvedTimeRange(
            start=datetime(2026, 5, 1, tzinfo=UTC),
            end=datetime(2026, 5, 1, 23, 59, tzinfo=UTC),
            explanation="Jutro",
        )

        state: ConversationState = {
            "authorized_user_id": 12345,
            "chat_id": 999,
            "message_thread_id": 1,
            "context_key": "999:1",
            "user_message": "jaka pogoda w Chwarznie?",
            "message_id": 50,
            "answer": "W Chwarznie jutro 18°C, wiatr 8 m/s.",
            "resolved_location": loc,
            "resolved_time_range": tr,
        }
        await _save(state)

        turns = await memory.load_turns("999:1")
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[0]["message_id"] == 50
        assert turns[0]["text"] == "jaka pogoda w Chwarznie?"
        assert turns[1]["role"] == "bot"
        assert turns[1]["answer_summary"] == "W Chwarznie jutro 18°C, wiatr 8 m/s."
        assert turns[1]["resolved_location"]["name"] == "Chwarzno"

    @pytest.mark.asyncio
    async def test_save_without_memory_service(self) -> None:
        _save = _make_save_thread_context(None)
        state: ConversationState = {
            "authorized_user_id": 12345,
            "chat_id": 999,
            "message_thread_id": None,
            "context_key": "999",
            "user_message": "pogoda?",
            "answer": "Słońce.",
        }
        result = await _save(state)
        assert result == {}


class TestConversationStateFields:
    def test_state_has_reply_fields(self) -> None:
        state: ConversationState = {
            "chat_id": 100,
            "context_key": "100:5",
            "user_message": "a wiatr?",
            "message_id": 42,
            "reply_to_message_id": 41,
            "reply_to_message_text": "W Warszawie jutro 18°C.",
        }
        assert state["message_id"] == 42
        assert state["reply_to_message_id"] == 41
        assert state["reply_to_message_text"] == "W Warszawie jutro 18°C."

    def test_state_has_reply_anchor_field(self) -> None:
        state: ConversationState = {
            "chat_id": 100,
            "context_key": "100:5",
            "reply_anchor": {
                "role": "bot",
                "answer_summary": "W Chwarznie 18°C.",
                "message_id": 41,
            },
        }
        assert state["reply_anchor"]["role"] == "bot"

    def test_state_has_recent_context_field(self) -> None:
        state: ConversationState = {
            "chat_id": 100,
            "context_key": "100:5",
            "recent_context": [
                {"role": "user", "text": "pogoda?"},
                {"role": "bot", "answer_summary": "Słońce."},
            ],
        }
        assert len(state["recent_context"]) == 2
