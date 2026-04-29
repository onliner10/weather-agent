from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from weather_agent.adapters.telegram.context import (
    ContextKey,
    TelegramContextService,
)
from weather_agent.infrastructure.db.base import Base


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


class TestContextKeyComputation:
    def test_context_key_with_thread_id(self) -> None:
        key = ContextKey(chat_id=100, message_thread_id=5)
        assert key.context_key == "100:5"

    def test_context_key_without_thread_id(self) -> None:
        key = ContextKey(chat_id=100, message_thread_id=None)
        assert key.context_key == "100"

    def test_context_key_with_zero_thread_id(self) -> None:
        key = ContextKey(chat_id=100, message_thread_id=0)
        assert key.context_key == "100:0"

    def test_compute_context_key_via_service(self) -> None:
        key = TelegramContextService.compute_context_key(chat_id=42, message_thread_id=7)
        assert key.context_key == "42:7"

    def test_compute_context_key_fallback(self) -> None:
        key = TelegramContextService.compute_context_key(chat_id=42, message_thread_id=None)
        assert key.context_key == "42"

    def test_different_thread_ids_produce_different_keys(self) -> None:
        key_a = TelegramContextService.compute_context_key(chat_id=100, message_thread_id=1)
        key_b = TelegramContextService.compute_context_key(chat_id=100, message_thread_id=2)
        assert key_a.context_key != key_b.context_key

    def test_no_thread_vs_with_thread_produce_different_keys(self) -> None:
        key_no_thread = TelegramContextService.compute_context_key(
            chat_id=100, message_thread_id=None
        )
        key_with_thread = TelegramContextService.compute_context_key(
            chat_id=100, message_thread_id=1
        )
        assert key_no_thread.context_key != key_with_thread.context_key


class TestGetOrCreateContext:
    @pytest.mark.asyncio()
    async def test_creates_new_context_when_none_exists(self, async_session: AsyncSession) -> None:
        service = TelegramContextService(async_session)
        ctx = await service.get_or_create_context(chat_id=100, message_thread_id=5)
        assert ctx.context_key == "100:5"
        assert ctx.chat_id == 100
        assert ctx.message_thread_id == 5
        assert ctx.metadata == {}

    @pytest.mark.asyncio()
    async def test_returns_existing_context(self, async_session: AsyncSession) -> None:
        service = TelegramContextService(async_session)
        ctx1 = await service.get_or_create_context(chat_id=100, message_thread_id=5)
        ctx2 = await service.get_or_create_context(chat_id=100, message_thread_id=5)
        assert ctx1.context_key == ctx2.context_key

    @pytest.mark.asyncio()
    async def test_creates_context_without_thread_id(self, async_session: AsyncSession) -> None:
        service = TelegramContextService(async_session)
        ctx = await service.get_or_create_context(chat_id=200, message_thread_id=None)
        assert ctx.context_key == "200"
        assert ctx.chat_id == 200
        assert ctx.message_thread_id is None

    @pytest.mark.asyncio()
    async def test_new_thread_gets_fresh_context(self, async_session: AsyncSession) -> None:
        service = TelegramContextService(async_session)
        ctx_main = await service.get_or_create_context(chat_id=100, message_thread_id=None)
        await service.update_context(ctx_main.context_key, {"intent": "weather_query"})

        ctx_thread = await service.get_or_create_context(chat_id=100, message_thread_id=99)
        assert ctx_thread.metadata == {}
        assert ctx_thread.context_key != ctx_main.context_key


class TestUpdateContext:
    @pytest.mark.asyncio()
    async def test_update_preserves_metadata(self, async_session: AsyncSession) -> None:
        service = TelegramContextService(async_session)
        ctx = await service.get_or_create_context(chat_id=100, message_thread_id=1)
        assert ctx.metadata == {}

        updated = await service.update_context(
            ctx.context_key, {"intent": "rule_proposal", "pending_rule_id": "R123"}
        )
        assert updated.metadata == {"intent": "rule_proposal", "pending_rule_id": "R123"}

    @pytest.mark.asyncio()
    async def test_update_overwrites_metadata(self, async_session: AsyncSession) -> None:
        service = TelegramContextService(async_session)
        ctx = await service.get_or_create_context(chat_id=100, message_thread_id=2)
        await service.update_context(ctx.context_key, {"old_key": "old_value"})

        updated = await service.update_context(ctx.context_key, {"new_key": "new_value"})
        assert updated.metadata == {"new_key": "new_value"}
        assert "old_key" not in updated.metadata

    @pytest.mark.asyncio()
    async def test_update_timestamp_advances(self, async_session: AsyncSession) -> None:
        service = TelegramContextService(async_session)
        ctx = await service.get_or_create_context(chat_id=100, message_thread_id=3)
        original_updated_at = ctx.updated_at

        import time

        time.sleep(0.01)

        updated = await service.update_context(ctx.context_key, {"k": "v"})
        assert updated.updated_at >= original_updated_at


class TestClearContext:
    @pytest.mark.asyncio()
    async def test_clear_removes_metadata(self, async_session: AsyncSession) -> None:
        service = TelegramContextService(async_session)
        ctx = await service.get_or_create_context(chat_id=100, message_thread_id=10)
        await service.update_context(ctx.context_key, {"intent": "something"})

        await service.clear_context(ctx.context_key)

        refreshed = await service.get_or_create_context(chat_id=100, message_thread_id=10)
        assert refreshed.metadata == {}

    @pytest.mark.asyncio()
    async def test_clear_nonexistent_context_does_not_error(
        self, async_session: AsyncSession
    ) -> None:
        service = TelegramContextService(async_session)
        await service.clear_context("999:1")


class TestContextIsolation:
    @pytest.mark.asyncio()
    async def test_context_does_not_leak_across_threads(self, async_session: AsyncSession) -> None:
        service = TelegramContextService(async_session)
        ctx_a = await service.get_or_create_context(chat_id=100, message_thread_id=1)
        ctx_b = await service.get_or_create_context(chat_id=100, message_thread_id=2)

        await service.update_context(ctx_a.context_key, {"intent": "weather", "location": "Warsaw"})
        await service.update_context(
            ctx_b.context_key, {"intent": "rule_create", "rule_expr": "temp > 30"}
        )

        refreshed_a = await service.get_or_create_context(chat_id=100, message_thread_id=1)
        refreshed_b = await service.get_or_create_context(chat_id=100, message_thread_id=2)

        assert refreshed_a.metadata == {"intent": "weather", "location": "Warsaw"}
        assert refreshed_b.metadata == {"intent": "rule_create", "rule_expr": "temp > 30"}

    @pytest.mark.asyncio()
    async def test_clear_thread_does_not_affect_other_thread(
        self, async_session: AsyncSession
    ) -> None:
        service = TelegramContextService(async_session)
        ctx_a = await service.get_or_create_context(chat_id=100, message_thread_id=1)
        ctx_b = await service.get_or_create_context(chat_id=100, message_thread_id=2)

        await service.update_context(ctx_a.context_key, {"intent": "weather"})
        await service.update_context(ctx_b.context_key, {"intent": "rule_create"})

        await service.clear_context(ctx_a.context_key)

        refreshed_a = await service.get_or_create_context(chat_id=100, message_thread_id=1)
        refreshed_b = await service.get_or_create_context(chat_id=100, message_thread_id=2)

        assert refreshed_a.metadata == {}
        assert refreshed_b.metadata == {"intent": "rule_create"}

    @pytest.mark.asyncio()
    async def test_main_chat_independent_from_thread(self, async_session: AsyncSession) -> None:
        service = TelegramContextService(async_session)
        ctx_main = await service.get_or_create_context(chat_id=100, message_thread_id=None)
        ctx_thread = await service.get_or_create_context(chat_id=100, message_thread_id=5)

        await service.update_context(ctx_main.context_key, {"state": "active"})
        await service.update_context(ctx_thread.context_key, {"state": "idle"})

        refreshed_main = await service.get_or_create_context(chat_id=100, message_thread_id=None)
        refreshed_thread = await service.get_or_create_context(chat_id=100, message_thread_id=5)

        assert refreshed_main.metadata == {"state": "active"}
        assert refreshed_thread.metadata == {"state": "idle"}
        assert refreshed_main.context_key != refreshed_thread.context_key

    @pytest.mark.asyncio()
    async def test_different_chats_are_independent(self, async_session: AsyncSession) -> None:
        service = TelegramContextService(async_session)
        ctx_1 = await service.get_or_create_context(chat_id=100, message_thread_id=None)
        ctx_2 = await service.get_or_create_context(chat_id=200, message_thread_id=None)

        await service.update_context(ctx_1.context_key, {"chat": "one"})
        await service.update_context(ctx_2.context_key, {"chat": "two"})

        refreshed_1 = await service.get_or_create_context(chat_id=100, message_thread_id=None)
        refreshed_2 = await service.get_or_create_context(chat_id=200, message_thread_id=None)

        assert refreshed_1.metadata == {"chat": "one"}
        assert refreshed_2.metadata == {"chat": "two"}
