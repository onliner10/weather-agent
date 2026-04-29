from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import select as sa_select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from weather_agent.infrastructure.db.base import (
    AuditLog,
    AuthorizedUser,
    Base,
    ForecastPoint,
    ForecastSnapshot,
    Location,
    NotificationEvent,
    NotificationRule,
    Observation,
    RuleEvaluationRun,
    TelegramContext,
)
from weather_agent.infrastructure.jobs.retention import RetentionService
from weather_agent.settings import RetentionSettings


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
    session_factory = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        async with session.begin():
            yield session


@pytest_asyncio.fixture()
def retention_settings() -> RetentionSettings:
    return RetentionSettings(
        thread_memory_days=14,
        raw_forecast_days=60,
        aggregated_weather_days=90,
        notification_log_days=365,
        audit_log_days=365,
        trace_days=14,
    )


@pytest_asyncio.fixture()
def service(async_session: AsyncSession, retention_settings: RetentionSettings) -> RetentionService:
    return RetentionService(async_session, retention_settings)


async def _create_user_and_location(session: AsyncSession) -> tuple[int, int]:
    user = AuthorizedUser(telegram_user_id=111, role="user")
    session.add(user)
    await session.flush()
    location = Location(
        user_id=user.id,
        name="TestLocation",
        latitude=52.0,
        longitude=21.0,
    )
    session.add(location)
    await session.flush()
    return user.id, location.id


class TestCleanupThreadMemory:
    @pytest.mark.asyncio()
    async def test_deletes_old_context(
        self, service: RetentionService, async_session: AsyncSession
    ) -> None:
        old_time = datetime.now(UTC) - timedelta(days=15)
        ctx = TelegramContext(
            chat_id=100,
            message_thread_id=1,
            context_key="100:1",
            metadata_={},
            updated_at=old_time,
            created_at=old_time,
        )
        async_session.add(ctx)
        await async_session.flush()

        deleted = await service.cleanup_thread_memory()
        assert deleted == 1

    @pytest.mark.asyncio()
    async def test_keeps_recent_thread_context(
        self, service: RetentionService, async_session: AsyncSession
    ) -> None:
        recent_time = datetime.now(UTC) - timedelta(days=5)
        ctx = TelegramContext(
            chat_id=100,
            message_thread_id=2,
            context_key="100:2",
            metadata_={},
            updated_at=recent_time,
            created_at=recent_time,
        )
        async_session.add(ctx)
        await async_session.flush()

        deleted = await service.cleanup_thread_memory()
        assert deleted == 0

    @pytest.mark.asyncio()
    async def test_custom_days_override(
        self, service: RetentionService, async_session: AsyncSession
    ) -> None:
        old_time = datetime.now(UTC) - timedelta(days=10)
        ctx = TelegramContext(
            chat_id=100,
            message_thread_id=3,
            context_key="100:3",
            metadata_={},
            updated_at=old_time,
            created_at=old_time,
        )
        async_session.add(ctx)
        await async_session.flush()

        deleted_default = await service.cleanup_thread_memory()
        assert deleted_default == 0

        deleted_custom = await service.cleanup_thread_memory(older_than_days=7)
        assert deleted_custom == 1

    @pytest.mark.asyncio()
    async def test_dry_run_does_not_delete(
        self, service: RetentionService, async_session: AsyncSession
    ) -> None:
        old_time = datetime.now(UTC) - timedelta(days=15)
        ctx = TelegramContext(
            chat_id=100,
            message_thread_id=4,
            context_key="100:4",
            metadata_={},
            updated_at=old_time,
            created_at=old_time,
        )
        async_session.add(ctx)
        await async_session.flush()

        count = await service.cleanup_thread_memory(dry_run=True)
        assert count == 1

        remaining = await async_session.execute(
            sa_select(TelegramContext).where(TelegramContext.context_key == "100:4")
        )
        assert remaining.scalar_one_or_none() is not None


class TestCleanupRawForecasts:
    @pytest.mark.asyncio()
    async def test_deletes_old_snapshots_and_points(
        self, service: RetentionService, async_session: AsyncSession
    ) -> None:
        _, location_id = await _create_user_and_location(async_session)
        old_time = datetime.now(UTC) - timedelta(days=61)

        snapshot = ForecastSnapshot(
            provider="open-meteo",
            model="dwd-icon",
            location_id=location_id,
            fetched_at=old_time,
            raw_payload={},
        )
        async_session.add(snapshot)
        await async_session.flush()

        point = ForecastPoint(
            snapshot_id=snapshot.id,
            target_time=old_time,
            location_id=location_id,
            raw_payload={},
        )
        async_session.add(point)
        await async_session.flush()

        deleted = await service.cleanup_raw_forecasts()
        assert deleted == 1

    @pytest.mark.asyncio()
    async def test_keeps_recent_snapshots(
        self, service: RetentionService, async_session: AsyncSession
    ) -> None:
        _, location_id = await _create_user_and_location(async_session)
        recent_time = datetime.now(UTC) - timedelta(days=30)

        snapshot = ForecastSnapshot(
            provider="open-meteo",
            model="dwd-icon",
            location_id=location_id,
            fetched_at=recent_time,
            raw_payload={},
        )
        async_session.add(snapshot)
        await async_session.flush()

        deleted = await service.cleanup_raw_forecasts()
        assert deleted == 0

    @pytest.mark.asyncio()
    async def test_points_deleted_with_snapshot(
        self, service: RetentionService, async_session: AsyncSession
    ) -> None:
        _, location_id = await _create_user_and_location(async_session)
        old_time = datetime.now(UTC) - timedelta(days=61)

        snapshot = ForecastSnapshot(
            provider="open-meteo",
            model="dwd-icon",
            location_id=location_id,
            fetched_at=old_time,
            raw_payload={},
        )
        async_session.add(snapshot)
        await async_session.flush()

        point = ForecastPoint(
            snapshot_id=snapshot.id,
            target_time=old_time,
            location_id=location_id,
            raw_payload={},
        )
        async_session.add(point)
        await async_session.flush()

        await service.cleanup_raw_forecasts()

        remaining_points = await async_session.execute(
            sa_select(ForecastPoint).where(ForecastPoint.snapshot_id == snapshot.id)
        )
        assert remaining_points.scalar_one_or_none() is None


class TestCleanupAggregatedWeather:
    @pytest.mark.asyncio()
    async def test_deletes_old_observations(
        self, service: RetentionService, async_session: AsyncSession
    ) -> None:
        _, location_id = await _create_user_and_location(async_session)
        old_time = datetime.now(UTC) - timedelta(days=91)

        obs = Observation(
            provider="imgw",
            observed_at=old_time,
            location_id=location_id,
            fetched_at=old_time,
            raw_payload={},
        )
        async_session.add(obs)
        await async_session.flush()

        deleted = await service.cleanup_aggregated_weather()
        assert deleted == 1

    @pytest.mark.asyncio()
    async def test_keeps_recent_observations(
        self, service: RetentionService, async_session: AsyncSession
    ) -> None:
        _, location_id = await _create_user_and_location(async_session)
        recent_time = datetime.now(UTC) - timedelta(days=30)

        obs = Observation(
            provider="imgw",
            observed_at=recent_time,
            location_id=location_id,
            fetched_at=recent_time,
            raw_payload={},
        )
        async_session.add(obs)
        await async_session.flush()

        deleted = await service.cleanup_aggregated_weather()
        assert deleted == 0


class TestCleanupNotificationLog:
    @pytest.mark.asyncio()
    async def test_deletes_old_notification_events(
        self, service: RetentionService, async_session: AsyncSession
    ) -> None:
        old_time = datetime.now(UTC) - timedelta(days=366)

        event = NotificationEvent(
            short_id="NV001",
            telegram_chat_id=100,
            created_at=old_time,
        )
        async_session.add(event)
        await async_session.flush()

        deleted = await service.cleanup_notification_log()
        assert deleted == 1

    @pytest.mark.asyncio()
    async def test_keeps_recent_notification_events(
        self, service: RetentionService, async_session: AsyncSession
    ) -> None:
        recent_time = datetime.now(UTC) - timedelta(days=180)

        event = NotificationEvent(
            short_id="NV002",
            telegram_chat_id=100,
            created_at=recent_time,
        )
        async_session.add(event)
        await async_session.flush()

        deleted = await service.cleanup_notification_log()
        assert deleted == 0


class TestCleanupAuditLog:
    @pytest.mark.asyncio()
    async def test_deletes_old_audit_entries(
        self, service: RetentionService, async_session: AsyncSession
    ) -> None:
        old_time = datetime.now(UTC) - timedelta(days=366)

        entry = AuditLog(
            event_type="notification_sent",
            details={"rule_id": 1},
            created_at=old_time,
        )
        async_session.add(entry)
        await async_session.flush()

        deleted = await service.cleanup_audit_log()
        assert deleted == 1

    @pytest.mark.asyncio()
    async def test_keeps_recent_audit_entries(
        self, service: RetentionService, async_session: AsyncSession
    ) -> None:
        recent_time = datetime.now(UTC) - timedelta(days=180)

        entry = AuditLog(
            event_type="notification_sent",
            details={"rule_id": 2},
            created_at=recent_time,
        )
        async_session.add(entry)
        await async_session.flush()

        deleted = await service.cleanup_audit_log()
        assert deleted == 0


class TestCleanupTraceData:
    @pytest.mark.asyncio()
    async def test_deletes_old_evaluation_runs(
        self, service: RetentionService, async_session: AsyncSession
    ) -> None:
        user_id, location_id = await _create_user_and_location(async_session)
        old_time = datetime.now(UTC) - timedelta(days=15)

        rule = NotificationRule(
            short_id="RA001",
            user_id=user_id,
            telegram_chat_id=100,
            location_id=location_id,
            expression_language="cel",
            expression="temp > 30",
        )
        async_session.add(rule)
        await async_session.flush()

        run = RuleEvaluationRun(
            rule_id=rule.id,
            evaluated_at=old_time,
            result=True,
            evaluation_detail={},
            created_at=old_time,
        )
        async_session.add(run)
        await async_session.flush()

        deleted = await service.cleanup_trace_data()
        assert deleted == 1

    @pytest.mark.asyncio()
    async def test_keeps_recent_evaluation_runs(
        self, service: RetentionService, async_session: AsyncSession
    ) -> None:
        user_id, location_id = await _create_user_and_location(async_session)
        recent_time = datetime.now(UTC) - timedelta(days=5)

        rule = NotificationRule(
            short_id="RA002",
            user_id=user_id,
            telegram_chat_id=100,
            location_id=location_id,
            expression_language="cel",
            expression="temp > 30",
        )
        async_session.add(rule)
        await async_session.flush()

        run = RuleEvaluationRun(
            rule_id=rule.id,
            evaluated_at=recent_time,
            result=True,
            evaluation_detail={},
            created_at=recent_time,
        )
        async_session.add(run)
        await async_session.flush()

        deleted = await service.cleanup_trace_data()
        assert deleted == 0


class TestRunAllCleanup:
    @pytest.mark.asyncio()
    async def test_returns_counts_for_all_jobs(
        self, service: RetentionService, async_session: AsyncSession
    ) -> None:
        old_time = datetime.now(UTC) - timedelta(days=400)

        ctx = TelegramContext(
            chat_id=100,
            message_thread_id=99,
            context_key="100:99",
            metadata_={},
            updated_at=old_time,
            created_at=old_time,
        )
        async_session.add(ctx)

        entry = AuditLog(
            event_type="test",
            details={},
            created_at=old_time,
        )
        async_session.add(entry)

        event = NotificationEvent(
            short_id="NVALL",
            telegram_chat_id=100,
            created_at=old_time,
        )
        async_session.add(event)
        await async_session.flush()

        results = await service.run_all_cleanup()

        assert "thread_memory" in results
        assert "raw_forecasts" in results
        assert "aggregated_weather" in results
        assert "notification_log" in results
        assert "audit_log" in results
        assert "trace_data" in results
        assert results["thread_memory"] == 1
        assert results["audit_log"] == 1
        assert results["notification_log"] == 1
        assert results["raw_forecasts"] == 0
        assert results["aggregated_weather"] == 0
        assert results["trace_data"] == 0

    @pytest.mark.asyncio()
    async def test_dry_run_does_not_delete(
        self, service: RetentionService, async_session: AsyncSession
    ) -> None:
        old_time = datetime.now(UTC) - timedelta(days=400)

        entry = AuditLog(
            event_type="dry_test",
            details={},
            created_at=old_time,
        )
        async_session.add(entry)
        await async_session.flush()

        results = await service.run_all_cleanup(dry_run=True)

        assert results["audit_log"] == 1

        remaining = await async_session.execute(
            sa_select(AuditLog).where(AuditLog.event_type == "dry_test")
        )
        assert remaining.scalar_one_or_none() is not None