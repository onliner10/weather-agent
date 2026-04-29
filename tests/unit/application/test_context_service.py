from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from weather_agent.adapters.telegram.context import TelegramContextService
from weather_agent.application.context_service import (
    load_thread_context,
    save_thread_context,
)
from weather_agent.application.conversation_models import PendingConfirmation
from weather_agent.domain.weather import LocationRef
from weather_agent.infrastructure.db.base import Base
from weather_agent.infrastructure.memory.thread_memory import ThreadMemoryService


@pytest_asyncio.fixture()
async def _async_engine() -> AsyncEngine:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture()
async def _async_session(_async_engine: AsyncEngine) -> AsyncSession:
    session_factory = async_sessionmaker(_async_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        async with session.begin():
            yield session


class TestContextService:
    @pytest.mark.asyncio
    async def test_load_reply_context_turns(self, _async_session: AsyncSession) -> None:
        context_service = TelegramContextService(_async_session)
        memory = ThreadMemoryService(context_service)

        await memory.save_turn(
            "999:1",
            {
                "message_id": 10,
                "role": "user",
                "text": "jaka pogoda w Warszawie?",
                "timestamp": None,
            },
        )

        loaded = await load_thread_context(memory, "999:1", 10)
        assert loaded.reply_context_turns is not None
        assert len(loaded.reply_context_turns) == 1
        assert loaded.reply_context_turns[0]["text"] == "jaka pogoda w Warszawie?"

    @pytest.mark.asyncio
    async def test_load_reply_context_with_bot_anchor(self, _async_session: AsyncSession) -> None:
        context_service = TelegramContextService(_async_session)
        memory = ThreadMemoryService(context_service)

        await memory.save_turn(
            "999:1",
            {
                "message_id": 42,
                "role": "bot",
                "answer_summary": "W Chwarznie 18°C, wiatr 8 m/s.",
                "timestamp": None,
            },
        )

        loaded = await load_thread_context(memory, "999:1", 42)
        assert loaded.reply_context_turns is not None
        assert loaded.reply_context_turns[0]["role"] == "bot"
        assert loaded.reply_context_turns[0]["message_id"] == 42

    @pytest.mark.asyncio
    async def test_load_reply_anchor_not_found(self, _async_session: AsyncSession) -> None:
        context_service = TelegramContextService(_async_session)
        memory = ThreadMemoryService(context_service)

        await memory.save_turn(
            "999:1",
            {
                "message_id": 10,
                "role": "user",
                "text": "pogoda?",
                "timestamp": None,
            },
        )

        loaded = await load_thread_context(memory, "999:1", 999)
        assert loaded.reply_context_turns is None

    @pytest.mark.asyncio
    async def test_load_without_memory_service(self) -> None:
        loaded = await load_thread_context(None, "999")
        assert loaded.reply_context_turns is None

    @pytest.mark.asyncio
    async def test_save_user_and_bot_turns(self, _async_session: AsyncSession) -> None:
        context_service = TelegramContextService(_async_session)
        memory = ThreadMemoryService(context_service)

        loc = LocationRef(id="1", name="Warszawa", latitude=52.2297, longitude=21.0122)
        pending = PendingConfirmation(action="create_rule", cel_expression="temp > 20")

        await save_thread_context(
            memory,
            "999:1",
            "jaka pogoda?",
            "W Warszawie jest 22°C.",
            10,
            loc,
            None,
            "weather",
            pending,
        )

        turns = await memory.load_turns("999:1")
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[0]["text"] == "jaka pogoda?"
        assert turns[0]["resolved_location"]["name"] == "Warszawa"
        
        assert turns[1]["role"] == "bot"
        assert turns[1]["answer_summary"] == "W Warszawie jest 22°C."

        stored_pending = await memory.get_pending_confirmation("999:1")
        assert stored_pending is not None
        assert stored_pending["cel_expression"] == "temp > 20"

    @pytest.mark.asyncio
    async def test_save_without_memory_service(self) -> None:
        # Should not raise
        await save_thread_context(
            None, "999", "msg", "ans", 1, None, None, None, None
        )
