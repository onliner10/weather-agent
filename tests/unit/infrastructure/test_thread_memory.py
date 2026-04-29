from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from weather_agent.adapters.telegram.context import TelegramContextService
from weather_agent.infrastructure.db.base import Base
from weather_agent.infrastructure.memory.thread_memory import ThreadMemoryService


@pytest_asyncio.fixture()
async def async_engine() -> AsyncGenerator[AsyncEngine, None]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture()
async def async_session(async_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    session_factory = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        async with session.begin():
            yield session


@pytest_asyncio.fixture()
async def memory_service(async_session: AsyncSession) -> ThreadMemoryService:
    context_service = TelegramContextService(async_session)
    return ThreadMemoryService(context_service, default_ttl_days=14)


class TestPendingConfirmation:
    @pytest.mark.asyncio()
    async def test_store_and_retrieve_pending_confirmation(
        self, memory_service: ThreadMemoryService
    ) -> None:
        confirmation = {"action": "activate_rule", "rule_id": "R7K2", "expression": "temp > 30"}
        await memory_service.store_pending_confirmation("100:1", confirmation)
        result = await memory_service.get_pending_confirmation("100:1")
        assert result == confirmation

    @pytest.mark.asyncio()
    async def test_different_threads_different_confirmations(
        self, memory_service: ThreadMemoryService
    ) -> None:
        confirmation_a = {"action": "activate_rule", "rule_id": "R7K2", "expression": "temp > 30"}
        confirmation_b = {"action": "activate_rule", "rule_id": "R3M5", "expression": "wind > 20"}
        await memory_service.store_pending_confirmation("100:1", confirmation_a)
        await memory_service.store_pending_confirmation("100:2", confirmation_b)

        result_a = await memory_service.get_pending_confirmation("100:1")
        result_b = await memory_service.get_pending_confirmation("100:2")
        assert result_a == confirmation_a
        assert result_b == confirmation_b
        assert result_a != result_b

    @pytest.mark.asyncio()
    async def test_yes_in_thread_a_does_not_activate_thread_b(
        self, memory_service: ThreadMemoryService
    ) -> None:
        confirmation_a = {"action": "activate_rule", "rule_id": "R7K2"}
        await memory_service.store_pending_confirmation("100:1", confirmation_a)

        result_b = await memory_service.get_pending_confirmation("100:2")
        assert result_b is None

    @pytest.mark.asyncio()
    async def test_clear_pending_confirmation(self, memory_service: ThreadMemoryService) -> None:
        await memory_service.store_pending_confirmation("100:1", {"action": "test"})
        await memory_service.clear_pending_confirmation("100:1")
        result = await memory_service.get_pending_confirmation("100:1")
        assert result is None

    @pytest.mark.asyncio()
    async def test_clear_thread_a_does_not_affect_thread_b(
        self, memory_service: ThreadMemoryService
    ) -> None:
        await memory_service.store_pending_confirmation("100:1", {"action": "a"})
        await memory_service.store_pending_confirmation("100:2", {"action": "b"})
        await memory_service.clear_pending_confirmation("100:1")
        assert await memory_service.get_pending_confirmation("100:1") is None
        assert await memory_service.get_pending_confirmation("100:2") == {"action": "b"}

    @pytest.mark.asyncio()
    async def test_no_thread_fallback_context_key(
        self, memory_service: ThreadMemoryService
    ) -> None:
        confirmation = {"action": "activate_rule", "rule_id": "R1A1"}
        await memory_service.store_pending_confirmation("200", confirmation)
        result = await memory_service.get_pending_confirmation("200")
        assert result == confirmation

    @pytest.mark.asyncio()
    async def test_no_confirmation_returns_none(self, memory_service: ThreadMemoryService) -> None:
        result = await memory_service.get_pending_confirmation("999:1")
        assert result is None


class TestExpiredContext:
    @pytest.mark.asyncio()
    async def test_expired_pending_confirmation_is_ignored(
        self, async_session: AsyncSession
    ) -> None:
        context_service = TelegramContextService(async_session)
        memory = ThreadMemoryService(context_service, default_ttl_days=14)

        confirmation = {"action": "activate_rule", "rule_id": "R7K2"}
        await memory.store_pending_confirmation("100:1", confirmation)

        ctx = await context_service.get_or_create_context(100, 1)
        metadata = dict(ctx.metadata)
        expired_time = (datetime.now(UTC) - timedelta(days=15)).isoformat()
        metadata["pending_confirmation_stored_at"] = expired_time
        await context_service.update_context("100:1", metadata)

        result = await memory.get_pending_confirmation("100:1")
        assert result is None

    @pytest.mark.asyncio()
    async def test_non_expired_confirmation_is_returned(
        self, memory_service: ThreadMemoryService
    ) -> None:
        confirmation = {"action": "activate_rule", "rule_id": "R7K2"}
        await memory_service.store_pending_confirmation("100:1", confirmation)
        result = await memory_service.get_pending_confirmation("100:1")
        assert result == confirmation

    @pytest.mark.asyncio()
    async def test_custom_ttl_overrides_default(self, async_session: AsyncSession) -> None:
        context_service = TelegramContextService(async_session)
        memory = ThreadMemoryService(context_service, default_ttl_days=14)

        confirmation = {"action": "test"}
        await memory.store_pending_confirmation("100:1", confirmation)

        ctx = await context_service.get_or_create_context(100, 1)
        metadata = dict(ctx.metadata)
        slightly_expired = (datetime.now(UTC) - timedelta(days=8)).isoformat()
        metadata["pending_confirmation_stored_at"] = slightly_expired
        await context_service.update_context("100:1", metadata)

        result_default = await memory.get_pending_confirmation("100:1")
        assert result_default == confirmation

        result_short_ttl = await memory.get_pending_confirmation("100:1", ttl_days=7)
        assert result_short_ttl is None


class TestNewThreadCleanState:
    @pytest.mark.asyncio()
    async def test_new_thread_has_no_pending_confirmation(
        self, memory_service: ThreadMemoryService
    ) -> None:
        await memory_service.store_pending_confirmation("100:1", {"action": "a"})
        result = await memory_service.get_pending_confirmation("100:2")
        assert result is None

    @pytest.mark.asyncio()
    async def test_new_thread_same_chat_is_isolated(
        self, memory_service: ThreadMemoryService
    ) -> None:
        confirmation = {"action": "activate_rule", "rule_id": "R7K2"}
        await memory_service.store_pending_confirmation("100:1", confirmation)

        assert await memory_service.get_pending_confirmation("100:5") is None

        assert await memory_service.get_pending_confirmation("100:1") == confirmation


class TestRetentionConfigurable:
    @pytest.mark.asyncio()
    async def test_custom_default_ttl(self, async_session: AsyncSession) -> None:
        context_service = TelegramContextService(async_session)
        memory = ThreadMemoryService(context_service, default_ttl_days=7)

        confirmation = {"action": "test"}
        await memory.store_pending_confirmation("100:1", confirmation)

        ctx = await context_service.get_or_create_context(100, 1)
        metadata = dict(ctx.metadata)
        expired_time = (datetime.now(UTC) - timedelta(days=8)).isoformat()
        metadata["pending_confirmation_stored_at"] = expired_time
        await context_service.update_context("100:1", metadata)

        result = await memory.get_pending_confirmation("100:1")
        assert result is None


class TestTurnPersistence:
    @pytest.mark.asyncio()
    async def test_save_and_load_user_turn(self, memory_service: ThreadMemoryService) -> None:
        turn = {
            "message_id": 42,
            "role": "user",
            "text": "jaka będzie pogoda w Chwarznie?",
            "timestamp": None,
        }
        await memory_service.save_turn("100:1", turn)
        turns = await memory_service.load_turns("100:1")
        assert len(turns) == 1
        assert turns[0]["role"] == "user"
        assert turns[0]["text"] == "jaka będzie pogoda w Chwarznie?"
        assert turns[0]["message_id"] == 42
        assert turns[0]["timestamp"] is not None

    @pytest.mark.asyncio()
    async def test_save_and_load_bot_turn(self, memory_service: ThreadMemoryService) -> None:
        turn = {
            "message_id": 43,
            "role": "bot",
            "text": None,
            "answer_summary": "W Chwarznie 18°C, wiatr 8 m/s.",
            "timestamp": None,
        }
        await memory_service.save_turn("100:1", turn)
        turns = await memory_service.load_turns("100:1")
        assert len(turns) == 1
        assert turns[0]["role"] == "bot"
        assert turns[0]["answer_summary"] == "W Chwarznie 18°C, wiatr 8 m/s."

    @pytest.mark.asyncio()
    async def test_save_multiple_turns_in_order(self, memory_service: ThreadMemoryService) -> None:
        user_turn = {
            "message_id": 1,
            "role": "user",
            "text": "pogoda jutro?",
            "timestamp": None,
        }
        bot_turn = {
            "message_id": 2,
            "role": "bot",
            "answer_summary": "Jutro 20°C, słonecznie.",
            "timestamp": None,
        }
        user_turn2 = {
            "message_id": 3,
            "role": "user",
            "text": "a wiatr?",
            "timestamp": None,
        }
        await memory_service.save_turn("100:1", user_turn)
        await memory_service.save_turn("100:1", bot_turn)
        await memory_service.save_turn("100:1", user_turn2)

        turns = await memory_service.load_turns("100:1")
        assert len(turns) == 3
        assert turns[0]["message_id"] == 1
        assert turns[1]["message_id"] == 2
        assert turns[2]["message_id"] == 3

    @pytest.mark.asyncio()
    async def test_find_turn_by_message_id(self, memory_service: ThreadMemoryService) -> None:
        await memory_service.save_turn(
            "100:1",
            {
                "message_id": 10,
                "role": "user",
                "text": "pogoda?",
                "timestamp": None,
            },
        )
        await memory_service.save_turn(
            "100:1",
            {
                "message_id": 11,
                "role": "bot",
                "answer_summary": "Słońce.",
                "timestamp": None,
            },
        )
        await memory_service.save_turn(
            "100:1",
            {
                "message_id": 12,
                "role": "user",
                "text": "a wiatr?",
                "timestamp": None,
            },
        )

        found = await memory_service.find_turn_by_message_id("100:1", 11)
        assert found is not None
        assert found["role"] == "bot"
        assert found["answer_summary"] == "Słońce."

    @pytest.mark.asyncio()
    async def test_find_turn_by_message_id_not_found(
        self, memory_service: ThreadMemoryService
    ) -> None:
        await memory_service.save_turn(
            "100:1",
            {
                "message_id": 10,
                "role": "user",
                "text": "pogoda?",
                "timestamp": None,
            },
        )
        assert await memory_service.find_turn_by_message_id("100:1", 999) is None

    @pytest.mark.asyncio()
    async def test_find_turn_by_message_id_returns_last_match(
        self, memory_service: ThreadMemoryService
    ) -> None:
        await memory_service.save_turn(
            "100:1",
            {
                "message_id": 10,
                "role": "user",
                "text": "pogoda?",
                "timestamp": None,
            },
        )
        await memory_service.save_turn(
            "100:1",
            {
                "message_id": 10,
                "role": "bot",
                "answer_summary": "Pierwsza odpowiedź",
                "timestamp": None,
            },
        )
        found = await memory_service.find_turn_by_message_id("100:1", 10)
        assert found is not None
        assert found["role"] == "bot"

    @pytest.mark.asyncio()
    async def test_no_turns_returns_empty_list(self, memory_service: ThreadMemoryService) -> None:
        turns = await memory_service.load_turns("999:1")
        assert turns == []

    @pytest.mark.asyncio()
    async def test_turns_are_thread_scoped(self, memory_service: ThreadMemoryService) -> None:
        await memory_service.save_turn(
            "100:1",
            {
                "message_id": 1,
                "role": "user",
                "text": "pogoda w Gdańsku?",
                "timestamp": None,
            },
        )
        await memory_service.save_turn(
            "100:2",
            {
                "message_id": 1,
                "role": "user",
                "text": "pogoda w Warszawie?",
                "timestamp": None,
            },
        )

        turns_1 = await memory_service.load_turns("100:1")
        turns_2 = await memory_service.load_turns("100:2")
        assert len(turns_1) == 1
        assert len(turns_2) == 1
        assert turns_1[0]["text"] == "pogoda w Gdańsku?"
        assert turns_2[0]["text"] == "pogoda w Warszawie?"

    @pytest.mark.asyncio()
    async def test_turns_bounded_at_max(self, async_session: AsyncSession) -> None:
        context_service = TelegramContextService(async_session)
        memory = ThreadMemoryService(context_service, default_ttl_days=14)

        for i in range(25):
            await memory.save_turn(
                "100:1",
                {
                    "message_id": i,
                    "role": "user",
                    "text": f"wiadomość {i}",
                    "timestamp": None,
                },
            )

        turns = await memory.load_turns("100:1")
        assert len(turns) == 20
        assert turns[0]["message_id"] == 5
        assert turns[-1]["message_id"] == 24

    @pytest.mark.asyncio()
    async def test_expired_turns_are_ignored(self, async_session: AsyncSession) -> None:
        context_service = TelegramContextService(async_session)
        memory = ThreadMemoryService(context_service, default_ttl_days=14)

        await memory.save_turn(
            "100:1",
            {
                "message_id": 1,
                "role": "user",
                "text": "pogoda?",
                "timestamp": None,
            },
        )

        ctx = await context_service.get_or_create_context(100, 1)
        metadata = dict(ctx.metadata)
        expired_time = (datetime.now(UTC) - timedelta(days=15)).isoformat()
        metadata["turns_stored_at"] = expired_time
        await context_service.update_context("100:1", metadata)

        result = await memory.load_turns("100:1")
        assert result == []

    @pytest.mark.asyncio()
    async def test_update_last_bot_turn_message_id(
        self, memory_service: ThreadMemoryService
    ) -> None:
        await memory_service.save_turn(
            "100:1",
            {
                "message_id": 50,
                "role": "user",
                "text": "pogoda jutro?",
                "timestamp": None,
            },
        )
        await memory_service.save_turn(
            "100:1",
            {
                "message_id": None,
                "role": "bot",
                "answer_summary": "Jutro 20°C.",
                "timestamp": None,
            },
        )

        await memory_service.update_last_bot_turn_message_id("100:1", 51)

        turns = await memory_service.load_turns("100:1")
        assert len(turns) == 2
        bot_turn = turns[1]
        assert bot_turn["message_id"] == 51
        assert bot_turn["role"] == "bot"
        user_turn = turns[0]
        assert user_turn["message_id"] == 50


class TestLastForecast:
    @pytest.mark.asyncio()
    async def test_store_and_load_last_forecast(self, memory_service: ThreadMemoryService) -> None:
        fc = {
            "location_name": "Gdańsk",
            "start_date": "2026-04-30",
            "end_date": "2026-04-30",
            "variables": ["temperature_2m_c"],
        }
        await memory_service.store_last_forecast("100:1", fc)
        loaded = await memory_service.load_last_forecast("100:1")
        assert loaded == fc

    @pytest.mark.asyncio()
    async def test_no_forecast_returns_none(self, memory_service: ThreadMemoryService) -> None:
        result = await memory_service.load_last_forecast("100:1")
        assert result is None

    @pytest.mark.asyncio()
    async def test_forecast_scoped_to_thread(self, memory_service: ThreadMemoryService) -> None:
        fc = {"location_name": "Gdańsk", "start_date": "2026-04-30", "end_date": "2026-04-30"}
        await memory_service.store_last_forecast("100:1", fc)
        result = await memory_service.load_last_forecast("100:2")
        assert result is None

    @pytest.mark.asyncio()
    async def test_forecast_overwrites_previous(self, memory_service: ThreadMemoryService) -> None:
        fc1 = {"location_name": "Gdańsk", "start_date": "2026-04-29", "end_date": "2026-04-29"}
        fc2 = {"location_name": "Warszawa", "start_date": "2026-05-01", "end_date": "2026-05-01"}
        await memory_service.store_last_forecast("100:1", fc1)
        await memory_service.store_last_forecast("100:1", fc2)
        loaded = await memory_service.load_last_forecast("100:1")
        assert loaded == fc2
        assert loaded["location_name"] == "Warszawa"


class TestReplyAnchorLookup:
    @pytest.mark.asyncio()
    async def test_reply_anchor_found_by_message_id(
        self, memory_service: ThreadMemoryService
    ) -> None:
        await memory_service.save_turn(
            "100:1",
            {
                "message_id": 10,
                "role": "user",
                "text": "pogoda w Chwarznie?",
                "timestamp": None,
            },
        )
        await memory_service.save_turn(
            "100:1",
            {
                "message_id": 11,
                "role": "bot",
                "answer_summary": "W Chwarznie 18°C, wiatr 8 m/s.",
                "timestamp": None,
            },
        )

        anchor = await memory_service.find_turn_by_message_id("100:1", 11)
        assert anchor is not None
        assert anchor["role"] == "bot"
        assert "Chwarzni" in anchor["answer_summary"]

    @pytest.mark.asyncio()
    async def test_reply_anchor_cross_thread_not_found(
        self, memory_service: ThreadMemoryService
    ) -> None:
        await memory_service.save_turn(
            "100:1",
            {
                "message_id": 11,
                "role": "bot",
                "answer_summary": "Odpowiedź w wątku 1",
                "timestamp": None,
            },
        )

        anchor = await memory_service.find_turn_by_message_id("100:2", 11)
        assert anchor is None

    @pytest.mark.asyncio()
    async def test_reply_anchor_cross_chat_not_found(
        self, memory_service: ThreadMemoryService
    ) -> None:
        await memory_service.save_turn(
            "100:1",
            {
                "message_id": 11,
                "role": "bot",
                "answer_summary": "Odpowiedź w czacie 100",
                "timestamp": None,
            },
        )

        anchor = await memory_service.find_turn_by_message_id("200:1", 11)
        assert anchor is None


class TestThreadTopicIsolation:
    @pytest.mark.asyncio()
    async def test_turns_isolated_by_thread(self, memory_service: ThreadMemoryService) -> None:
        await memory_service.save_turn(
            "100:1",
            {
                "message_id": 1,
                "role": "user",
                "text": "wątek 1",
                "timestamp": None,
            },
        )
        await memory_service.save_turn(
            "100:2",
            {
                "message_id": 1,
                "role": "user",
                "text": "wątek 2",
                "timestamp": None,
            },
        )

        turns_1 = await memory_service.load_turns("100:1")
        turns_2 = await memory_service.load_turns("100:2")
        assert len(turns_1) == 1
        assert len(turns_2) == 1
        assert turns_1[0]["text"] == "wątek 1"
        assert turns_2[0]["text"] == "wątek 2"

    @pytest.mark.asyncio()
    async def test_fallback_context_key_no_thread(
        self, memory_service: ThreadMemoryService
    ) -> None:
        await memory_service.save_turn(
            "200",
            {
                "message_id": 1,
                "role": "user",
                "text": "brak wątku",
                "timestamp": None,
            },
        )

        turns = await memory_service.load_turns("200")
        assert len(turns) == 1
        assert turns[0]["text"] == "brak wątku"

    @pytest.mark.asyncio()
    async def test_thread_confirmation_does_not_leak_to_chat_fallback(
        self, memory_service: ThreadMemoryService
    ) -> None:
        await memory_service.store_pending_confirmation("100:1", {"action": "test"})
        assert await memory_service.get_pending_confirmation("100") is None

    @pytest.mark.asyncio()
    async def test_thread_turns_do_not_leak_to_chat_fallback(
        self, memory_service: ThreadMemoryService
    ) -> None:
        await memory_service.save_turn(
            "100:1",
            {
                "message_id": 1,
                "role": "user",
                "text": "wątek",
                "timestamp": None,
            },
        )
        turns = await memory_service.load_turns("100")
        assert turns == []

    @pytest.mark.asyncio()
    async def test_chat_fallback_turns_do_not_leak_to_thread(
        self, memory_service: ThreadMemoryService
    ) -> None:
        await memory_service.save_turn(
            "100",
            {
                "message_id": 1,
                "role": "user",
                "text": "czat",
                "timestamp": None,
            },
        )
        turns = await memory_service.load_turns("100:1")
        assert turns == []
