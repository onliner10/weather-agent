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
    async def test_clear_pending_confirmation(
        self, memory_service: ThreadMemoryService
    ) -> None:
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
    async def test_no_confirmation_returns_none(
        self, memory_service: ThreadMemoryService
    ) -> None:
        result = await memory_service.get_pending_confirmation("999:1")
        assert result is None


class TestRecentContext:
    @pytest.mark.asyncio()
    async def test_store_and_retrieve_recent_context(
        self, memory_service: ThreadMemoryService
    ) -> None:
        ctx = {"last_intent": "weather", "last_location": "Warszawa"}
        await memory_service.store_recent_context("100:1", ctx)
        result = await memory_service.get_recent_context("100:1")
        assert result == ctx

    @pytest.mark.asyncio()
    async def test_recent_context_is_thread_scoped(
        self, memory_service: ThreadMemoryService
    ) -> None:
        ctx_a = {"last_intent": "weather"}
        ctx_b = {"last_intent": "rules"}
        await memory_service.store_recent_context("100:1", ctx_a)
        await memory_service.store_recent_context("100:2", ctx_b)
        assert await memory_service.get_recent_context("100:1") == ctx_a
        assert await memory_service.get_recent_context("100:2") == ctx_b

    @pytest.mark.asyncio()
    async def test_no_recent_context_returns_none(
        self, memory_service: ThreadMemoryService
    ) -> None:
        result = await memory_service.get_recent_context("999:1")
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
    async def test_expired_recent_context_is_ignored(
        self, async_session: AsyncSession
    ) -> None:
        context_service = TelegramContextService(async_session)
        memory = ThreadMemoryService(context_service, default_ttl_days=14)

        recent = {"last_intent": "weather"}
        await memory.store_recent_context("100:1", recent)

        ctx = await context_service.get_or_create_context(100, 1)
        metadata = dict(ctx.metadata)
        expired_time = (datetime.now(UTC) - timedelta(days=15)).isoformat()
        metadata["recent_context_stored_at"] = expired_time
        await context_service.update_context("100:1", metadata)

        result = await memory.get_recent_context("100:1")
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
    async def test_custom_ttl_overrides_default(
        self, async_session: AsyncSession
    ) -> None:
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
    async def test_new_thread_has_no_recent_context(
        self, memory_service: ThreadMemoryService
    ) -> None:
        await memory_service.store_recent_context("100:1", {"intent": "weather"})
        result = await memory_service.get_recent_context("100:2")
        assert result is None

    @pytest.mark.asyncio()
    async def test_new_thread_same_chat_is_isolated(
        self, memory_service: ThreadMemoryService
    ) -> None:
        confirmation = {"action": "activate_rule", "rule_id": "R7K2"}
        await memory_service.store_pending_confirmation("100:1", confirmation)
        await memory_service.store_recent_context("100:1", {"intent": "weather"})

        assert await memory_service.get_pending_confirmation("100:5") is None
        assert await memory_service.get_recent_context("100:5") is None

        assert await memory_service.get_pending_confirmation("100:1") == confirmation
        assert await memory_service.get_recent_context("100:1") == {"intent": "weather"}


class TestRetentionConfigurable:
    @pytest.mark.asyncio()
    async def test_custom_default_ttl(
        self, async_session: AsyncSession
    ) -> None:
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

    @pytest.mark.asyncio()
    async def test_store_recent_context_with_custom_ttl(
        self, async_session: AsyncSession
    ) -> None:
        context_service = TelegramContextService(async_session)
        memory = ThreadMemoryService(context_service, default_ttl_days=14)

        recent = {"intent": "weather"}
        await memory.store_recent_context("100:1", recent, ttl_days=7)

        ctx = await context_service.get_or_create_context(100, 1)
        metadata = dict(ctx.metadata)
        expired_time = (datetime.now(UTC) - timedelta(days=8)).isoformat()
        metadata["recent_context_stored_at"] = expired_time
        await context_service.update_context("100:1", metadata)

        result = await memory.get_recent_context("100:1")
        assert result is None