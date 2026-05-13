from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from freezegun import freeze_time
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from weather_agent.domain.rule_expression.evaluator import RuleExpressionEvaluator
from weather_agent.domain.rules.models import RuleCreate
from weather_agent.domain.rules.service import NotificationRuleService
from weather_agent.infrastructure.db.base import (
    AuthorizedUser,
    Base,
    ForecastSnapshot,
    Location,
)
from weather_agent.infrastructure.db.base import (
    ForecastPoint as ForecastPointORM,
)
from weather_agent.infrastructure.db.base import (
    NotificationEvent as NotificationEventORM,
)
from weather_agent.infrastructure.db.base import (
    NotificationRule as NotificationRuleORM,
)
from weather_agent.infrastructure.repositories.forecast_repository import ForecastRepository
from weather_agent.infrastructure.worker.rule_evaluator import RuleEvaluationWorker
from weather_agent.settings import SchedulerSettings


def _set_sqlite_foreign_keys(dbapi_connection: object, connection_record: object) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


@pytest.fixture()
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    event.listen(engine.sync_engine, "connect", _set_sqlite_foreign_keys)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture()
def rule_expression_evaluator() -> RuleExpressionEvaluator:
    return RuleExpressionEvaluator()


@pytest.fixture()
def scheduler_settings() -> SchedulerSettings:
    return SchedulerSettings(rule_evaluation_minutes=15)


@pytest.fixture()
def rule_service(
    session: AsyncSession,
    rule_expression_evaluator: RuleExpressionEvaluator,
) -> NotificationRuleService:
    return NotificationRuleService(session, rule_expression_evaluator)


@pytest.fixture()
def forecast_repo(session: AsyncSession) -> ForecastRepository:
    return ForecastRepository(session)


async def _create_user(session: AsyncSession, user_id: int = 1) -> None:
    user = AuthorizedUser(id=user_id, telegram_user_id=user_id * 1000, role="user")
    session.add(user)
    await session.flush()


async def _create_location(
    session: AsyncSession,
    user_id: int = 1,
    loc_id: int = 1,
    name: str = "Test Location",
) -> None:
    loc = Location(
        id=loc_id,
        user_id=user_id,
        name=name,
        aliases=["test"],
        latitude=52.22,
        longitude=21.01,
        enabled=True,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    session.add(loc)
    await session.flush()


async def _create_rule(
    rule_service: NotificationRuleService,
    user_id: int = 1,
    location_id: int = 1,
    expression: str = 'max_metric("wind_gusts_10m_ms", next_hours(24)) >= 12',
    schedule: str | None = None,
    enabled: bool = True,
) -> Any:
    data = RuleCreate(
        telegram_chat_id=12345,
        location_id=location_id,
        expression=expression,
        schedule=schedule,
        enabled=enabled,
    )
    return await rule_service.create_rule(user_id, data)


async def _seed_forecast_data(
    session: AsyncSession,
    location_id: int = 1,
    num_points: int = 3,
    fetched_at: datetime | None = None,
    wind_gusts_base: float = 14.0,
    target_times: list[datetime] | None = None,
) -> int:
    fetched = fetched_at or datetime.now(UTC)
    snapshot = ForecastSnapshot(
        provider="open-meteo",
        model="dwd-icon",
        location_id=location_id,
        fetched_at=fetched,
        raw_payload={"source": "test"},
    )
    session.add(snapshot)
    await session.flush()
    await session.refresh(snapshot)

    for i in range(num_points):
        if target_times is not None:
            tt = target_times[i]
        else:
            tt = fetched + timedelta(hours=i + 1)
        point = ForecastPointORM(
            snapshot_id=snapshot.id,
            target_time=tt,
            location_id=location_id,
            temperature_2m_c=5.0 + i,
            apparent_temperature_c=3.0 + i,
            precipitation_mm=0.1 * i,
            precipitation_probability_pct=10.0 * i,
            rain_mm=0.1 * i,
            snowfall_cm=0.0,
            cloud_cover_pct=50.0 + i * 10,
            wind_speed_10m_ms=3.0 + i,
            wind_gusts_10m_ms=wind_gusts_base + i,
            wind_direction_10m_deg=180.0,
            pressure_msl_hpa=1013.0,
            relative_humidity_2m_pct=70.0,
            weather_code="1",
            raw_payload={"test": True},
        )
        session.add(point)

    await session.flush()
    return snapshot.id


def _make_worker(
    session: AsyncSession,
    forecast_repo: ForecastRepository,
    rule_service: NotificationRuleService,
    rule_expression_evaluator: RuleExpressionEvaluator,
    settings: SchedulerSettings,
    forecast_fetcher: Any = None,
) -> RuleEvaluationWorker:
    return RuleEvaluationWorker(
        session=session,
        forecast_repo=forecast_repo,
        rule_expression_evaluator=rule_expression_evaluator,
        rule_service=rule_service,
        settings=settings,
        forecast_fetcher=forecast_fetcher,
    )


class TestScheduleGating:
    async def test_rule_without_schedule_is_evaluated(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session)
        await _create_rule(rule_service, schedule=None)

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )
        results = await worker.evaluate_rules()

        assert len(results) == 1
        assert results[0].evaluated is True

    async def test_once_before_time_is_skipped(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session)
        future_schedule = "once:2099-12-31T23:59:59"
        await _create_rule(rule_service, schedule=future_schedule)

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )
        results = await worker.evaluate_rules()

        assert results == []

    async def test_once_after_time_is_evaluated(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session)
        past_schedule = "once:2020-01-01T00:00:00"
        rule = await _create_rule(rule_service, schedule=past_schedule)

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )
        results = await worker.evaluate_rules()

        assert len(results) == 1
        assert results[0].evaluated is True
        assert results[0].notification_candidate is True

        stmt = select(NotificationRuleORM).where(NotificationRuleORM.id == rule.id)
        db_result = await session.execute(stmt)
        db_rule = db_result.scalar_one()
        assert db_rule.enabled is False

    async def test_cron_rule_is_evaluated(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session)
        await _create_rule(rule_service, schedule="cron:*/5 * * * *")

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )
        results = await worker.evaluate_rules()

        assert len(results) == 1
        assert results[0].evaluated is True

    async def test_cron_rule_is_evaluated_once_for_false_slot(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        with freeze_time("2026-05-13 10:01:00+00:00"):
            await _create_user(session)
            await _create_location(session)
            await _seed_forecast_data(session)
            await _create_rule(
                rule_service,
                expression='max_metric("temperature_2m_c", next_hours(24)) >= 100',
                schedule="cron:0 12 * * 1-5",
            )

            worker = _make_worker(
                session,
                forecast_repo,
                rule_service,
                rule_expression_evaluator,
                scheduler_settings,
            )
            first_results = await worker.evaluate_rules()
            second_results = await worker.evaluate_rules()

        assert len(first_results) == 1
        assert first_results[0].evaluated is True
        assert first_results[0].result is False
        assert second_results == []

    async def test_cron_rule_outside_evaluation_window_is_skipped(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        with freeze_time("2026-05-13 12:01:00+00:00"):
            await _create_user(session)
            await _create_location(session)
            await _seed_forecast_data(session)
            await _create_rule(rule_service, schedule="cron:0 12 * * 1-5")

            worker = _make_worker(
                session,
                forecast_repo,
                rule_service,
                rule_expression_evaluator,
                scheduler_settings,
            )
            results = await worker.evaluate_rules()

        assert results == []

    async def test_cron_rule_with_recent_event_is_skipped(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session)
        rule = await _create_rule(rule_service, schedule="cron:*/5 * * * *")

        event_orm = NotificationEventORM(
            short_id="NE0001",
            rule_id=rule.id,
            telegram_chat_id=12345,
            sent_at=datetime.now(UTC),
            delivery_status="sent",
            created_at=datetime.now(UTC),
        )
        session.add(event_orm)
        await session.flush()

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )
        results = await worker.evaluate_rules()

        assert results == []

    async def test_cron_rule_with_recent_pending_event_is_retried(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session)
        rule = await _create_rule(rule_service, schedule="cron:*/5 * * * *")

        event_orm = NotificationEventORM(
            short_id="NE0001",
            rule_id=rule.id,
            telegram_chat_id=12345,
            delivery_status="sending",
            created_at=datetime.now(UTC),
        )
        session.add(event_orm)
        await session.flush()

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )
        results = await worker.evaluate_rules()

        assert len(results) == 1
        assert results[0].evaluated is True

    async def test_invalid_schedule_is_skipped(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session)
        await _create_rule(rule_service, schedule="garbage")

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )
        results = await worker.evaluate_rules()

        assert results == []
