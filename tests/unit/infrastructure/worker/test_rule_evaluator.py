from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from weather_agent.domain.notifications.deduplication import NotificationDeduplicator
from weather_agent.domain.notifications.events import NotificationEventService
from weather_agent.domain.rule_expression.evaluator import RuleExpressionEvaluator
from weather_agent.domain.rules.models import RuleCreate, ScheduledNotificationContext
from weather_agent.domain.rules.service import NotificationRuleService
from weather_agent.infrastructure.db.base import (
    AuthorizedUser,
    Base,
    ForecastSnapshot,
    Location,
    RuleEvaluationRun,
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
from weather_agent.infrastructure.worker.rule_evaluator import (
    EvaluationResult,
    RuleEvaluationWorker,
)
from weather_agent.observability.logging import AuditLogger
from weather_agent.settings import SchedulerSettings


def _set_sqlite_foreign_keys(dbapi_connection: object, connection_record: object) -> None:
    cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
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
    expression: str = 'max_metric("wind_gusts_10m_ms", weekend()) >= 12',
    dry_run: bool = False,
    enabled: bool = True,
    schedule: str | None = None,
    description: str | None = None,
    notification_context: ScheduledNotificationContext | None = None,
) -> Any:
    data = RuleCreate(
        telegram_chat_id=12345,
        location_id=location_id,
        expression=expression,
        dry_run=dry_run,
        enabled=enabled,
        schedule=schedule,
        description=description,
        notification_context=notification_context,
    )
    return await rule_service.create_rule(user_id, data)


async def _seed_forecast_data(
    session: AsyncSession,
    location_id: int = 1,
    num_points: int = 3,
    fetched_at: datetime | None = None,
    wind_gusts_base: float = 6.0,
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
    notification_content_generator: Any = None,
) -> RuleEvaluationWorker:
    return RuleEvaluationWorker(
        session=session,
        forecast_repo=forecast_repo,
        rule_expression_evaluator=rule_expression_evaluator,
        rule_service=rule_service,
        settings=settings,
        forecast_fetcher=forecast_fetcher,
        notification_content_generator=notification_content_generator,
    )


class TestEvaluationResult:
    def test_result_fields(self) -> None:
        result = EvaluationResult(
            rule_id=1,
            rule_short_id="R0001",
            expression='max_metric("wind_gusts_10m_ms", weekend()) >= 12',
            evaluated=True,
            result=True,
            notification_candidate=True,
            evaluation_detail={"point_count": 5},
            dry_run=False,
        )
        assert result.rule_id == 1
        assert result.rule_short_id == "R0001"
        assert result.evaluated is True
        assert result.result is True
        assert result.notification_candidate is True
        assert result.error is None
        assert result.evaluation_detail is not None
        assert result.dry_run is False

    def test_error_result(self) -> None:
        result = EvaluationResult(
            rule_id=2,
            rule_short_id="R0002",
            expression="bad(",
            evaluated=False,
            result=None,
            error="Syntax error",
            dry_run=False,
        )
        assert result.evaluated is False
        assert result.result is None
        assert result.error == "Syntax error"
        assert result.notification_candidate is False


class TestRuleEvaluationWorker:
    async def test_evaluate_rules_empty(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )
        results = await worker.evaluate_rules()
        assert results == []

    async def test_evaluate_true_expression_generates_candidate(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session, wind_gusts_base=14.0)
        await _create_rule(rule_service)

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )
        results = await worker.evaluate_rules()

        assert len(results) == 1
        r = results[0]
        assert r.evaluated is True
        assert r.result is True
        assert r.notification_candidate is True
        assert r.error is None
        assert r.evaluation_detail is not None
        assert r.evaluation_detail["point_count"] > 0
        assert "evaluated_metrics" in r.evaluation_detail
        assert "evaluated_functions" in r.evaluation_detail

    async def test_evaluate_false_expression_no_candidate(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session, wind_gusts_base=3.0)

        _ = await _create_rule(rule_service)

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )
        results = await worker.evaluate_rules()

        assert len(results) == 1
        r = results[0]
        assert r.evaluated is True
        assert r.result is False
        assert r.notification_candidate is False

    async def test_evaluate_error_expression_logged(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session, wind_gusts_base=6.0)

        rule_orm = NotificationRuleORM(
            short_id="RERR1",
            user_id=1,
            telegram_chat_id=12345,
            location_id=1,
            expression_language="cel",
            expression='unknown_fn_xyz("temperature_2m_c") >= 5',
            enabled=True,
            dry_run=False,
            cooldown_minutes=60,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        session.add(rule_orm)
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
        r = results[0]
        assert r.evaluated is False
        assert r.error is not None
        assert "unknown_fn_xyz" in r.error or "Unknown" in r.error
        assert r.notification_candidate is False

    async def test_evaluate_no_forecast_data(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _create_rule(rule_service)

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )
        results = await worker.evaluate_rules()

        assert len(results) == 1
        r = results[0]
        assert r.evaluated is False
        assert r.error == "no_forecast_data"
        assert r.notification_candidate is False
        assert r.evaluation_detail is not None
        assert r.evaluation_detail["point_count"] == 0

    async def test_dry_run_mode(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session, wind_gusts_base=14.0)
        await _create_rule(rule_service)

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )
        results = await worker.evaluate_rules(dry_run=True)

        assert len(results) == 1
        r = results[0]
        assert r.dry_run is True
        assert r.notification_candidate is True

    async def test_rule_dry_run_flag(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session, wind_gusts_base=14.0)
        await _create_rule(rule_service, dry_run=True)

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )
        results = await worker.evaluate_rules()

        assert len(results) == 1
        r = results[0]
        assert r.dry_run is True

    async def test_disabled_rules_are_skipped(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session, wind_gusts_base=14.0)
        await _create_rule(rule_service, enabled=False)

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )
        results = await worker.evaluate_rules()

        assert results == []

    async def test_evaluation_is_deterministic(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session, wind_gusts_base=6.0)

        await _create_rule(
            rule_service,
            expression='avg_metric("wind_speed_10m_ms", next_hours(24)) >= 3',
        )

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )
        results1 = await worker.evaluate_rules()
        results2 = await worker.evaluate_rules()

        assert len(results1) == 1
        assert len(results2) == 1
        assert results1[0].result == results2[0].result
        assert results1[0].evaluated == results2[0].evaluated

    async def test_evaluation_detail_contains_explanation_data(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session, wind_gusts_base=14.0)
        await _create_rule(rule_service)

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )
        results = await worker.evaluate_rules()

        assert len(results) == 1
        r = results[0]
        assert r.evaluation_detail is not None
        assert "rule_id" in r.evaluation_detail
        assert "rule_short_id" in r.evaluation_detail
        assert "location_id" in r.evaluation_detail
        assert "point_count" in r.evaluation_detail
        assert r.evaluation_detail["point_count"] > 0
        assert "evaluated_metrics" in r.evaluation_detail
        assert "snapshot_id" in r.evaluation_detail
        assert r.evaluation_detail["snapshot_id"] is not None

    async def test_true_evaluation_detail_has_forecast_window_and_key_metrics(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session, wind_gusts_base=14.0)
        await _create_rule(rule_service)

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )
        results = await worker.evaluate_rules()

        assert len(results) == 1
        r = results[0]
        assert r.notification_candidate is True
        detail = r.evaluation_detail
        assert detail is not None
        assert "forecast_window_start" in detail
        assert "forecast_window_end" in detail
        assert "key_metrics" in detail
        assert "wind_gusts_10m_ms" in detail["key_metrics"]

    async def test_evaluation_run_persisted(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session, wind_gusts_base=14.0)
        rule = await _create_rule(rule_service)

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )
        results = await worker.evaluate_rules()

        assert len(results) == 1
        r = results[0]
        assert r.evaluation_detail is not None
        assert "evaluation_run_id" in r.evaluation_detail

        stmt = select(RuleEvaluationRun).where(RuleEvaluationRun.rule_id == rule.id)
        db_result = await session.execute(stmt)
        eval_run = db_result.scalar_one_or_none()
        assert eval_run is not None
        assert eval_run.rule_id == rule.id
        assert eval_run.result is True
        assert eval_run.evaluation_detail is not None

    async def test_multiple_rules(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session, wind_gusts_base=14.0)

        await _create_rule(
            rule_service,
            expression='max_metric("wind_gusts_10m_ms", weekend()) >= 12',
        )
        await _create_rule(
            rule_service,
            expression='avg_metric("temperature_2m_c", next_hours(24)) >= 100',
        )

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )
        results = await worker.evaluate_rules()

        assert len(results) == 2
        true_results = [r for r in results if r.notification_candidate]
        false_results = [r for r in results if not r.notification_candidate]
        assert len(true_results) == 1
        assert len(false_results) == 1

    async def test_run_once(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session, wind_gusts_base=14.0)
        rule = await _create_rule(rule_service)

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )
        await worker.run_once()

        stmt = select(RuleEvaluationRun).where(RuleEvaluationRun.rule_id == rule.id)
        db_result = await session.execute(stmt)
        eval_run = db_result.scalar_one_or_none()
        assert eval_run is not None

    async def test_forecast_fetcher_called(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        snapshot_id = await _seed_forecast_data(session, wind_gusts_base=14.0)
        rule = await _create_rule(rule_service)

        fetcher = AsyncMock()
        fetcher.fetch_fresh_forecast = AsyncMock(return_value=snapshot_id)

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
            forecast_fetcher=fetcher,
        )
        results = await worker.evaluate_rules()

        fetcher.fetch_fresh_forecast.assert_called_once_with(rule.location_id)
        assert len(results) == 1

    async def test_forecast_fetcher_failure_does_not_crash(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session, wind_gusts_base=14.0)
        rule = await _create_rule(rule_service)

        fetcher = AsyncMock()
        fetcher.fetch_fresh_forecast = AsyncMock(side_effect=Exception("network error"))

        class Sender:
            def __init__(self) -> None:
                self.count = 0

            async def send(self, chat_id: int, thread_id: int | None, text: str) -> bool:
                del chat_id, thread_id, text
                self.count += 1
                return True

        sender = Sender()
        worker = RuleEvaluationWorker(
            session=session,
            forecast_repo=forecast_repo,
            rule_expression_evaluator=rule_expression_evaluator,
            rule_service=rule_service,
            settings=scheduler_settings,
            forecast_fetcher=fetcher,
            notification_sender=sender,
            event_service=NotificationEventService(session, AuditLogger(session)),
            deduplicator=NotificationDeduplicator(session),
        )
        results = await worker.evaluate_rules()

        assert len(results) == 1
        assert results[0].evaluated is False
        assert results[0].notification_candidate is False
        assert results[0].error == "forecast_refresh_failed"
        assert sender.count == 0

        event_result = await session.execute(select(NotificationEventORM))
        assert event_result.scalars().all() == []

        run_result = await session.execute(
            select(RuleEvaluationRun).where(RuleEvaluationRun.rule_id == rule.id)
        )
        eval_run = run_result.scalar_one()
        assert eval_run.evaluation_detail["error"] == "forecast_refresh_failed"

    async def test_forecast_fetcher_none_result_is_refresh_failure(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session, wind_gusts_base=14.0)
        await _create_rule(rule_service)

        fetcher = AsyncMock()
        fetcher.fetch_fresh_forecast = AsyncMock(return_value=None)

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
            forecast_fetcher=fetcher,
        )

        results = await worker.evaluate_rules()

        assert len(results) == 1
        assert results[0].evaluated is False
        assert results[0].notification_candidate is False
        assert results[0].error == "forecast_refresh_failed"

    async def test_fetcher_failure_leaves_once_rule_enabled_and_skips_llm_generation(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session, name="Gdynia Chwarzno")
        await _seed_forecast_data(session, wind_gusts_base=14.0)
        context = ScheduledNotificationContext(
            scheduling_message="Wyślij mi jutro o 9 aktualną prognozę dla Chwarzna",
            human_request="Aktualna prognoza dla Gdyni Chwarzno",
            schedule="once:2026-05-01T09:00:00+02:00",
            location_id=1,
            location_name="Gdynia Chwarzno",
        )
        rule = await _create_rule(
            rule_service,
            expression="true",
            schedule=context.schedule,
            description=context.human_request,
            notification_context=context,
        )

        fetcher = AsyncMock()
        fetcher.fetch_fresh_forecast = AsyncMock(side_effect=Exception("network error"))

        class Sender:
            def __init__(self) -> None:
                self.messages: list[str] = []

            async def send(self, chat_id: int, thread_id: int | None, text: str) -> bool:
                del chat_id, thread_id
                self.messages.append(text)
                return True

        class Generator:
            def __init__(self) -> None:
                self.calls = 0

            async def generate(
                self,
                rule: Any,
                evaluation_detail: dict[str, Any],
            ) -> str:
                del rule, evaluation_detail
                self.calls += 1
                return "Aktualna prognoza dla Chwarzna: ciepło, bez opadów."

        sender = Sender()
        generator = Generator()
        worker = RuleEvaluationWorker(
            session=session,
            forecast_repo=forecast_repo,
            rule_expression_evaluator=rule_expression_evaluator,
            rule_service=rule_service,
            settings=scheduler_settings,
            forecast_fetcher=fetcher,
            notification_sender=sender,
            event_service=NotificationEventService(session, AuditLogger(session)),
            deduplicator=NotificationDeduplicator(session),
            notification_content_generator=generator,
        )

        results = await worker.evaluate_rules()

        assert len(results) == 1
        assert results[0].error == "forecast_refresh_failed"
        assert results[0].notification_candidate is False
        assert sender.messages == []
        assert generator.calls == 0

        db_rule = await session.get(NotificationRuleORM, rule.id)
        assert db_rule is not None
        assert db_rule.enabled is True

    async def test_evaluation_uses_snapshot_returned_by_fetcher(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session, name="Gdynia Chwarzno")
        now = datetime.now(UTC)
        await _seed_forecast_data(
            session,
            fetched_at=now + timedelta(minutes=5),
            wind_gusts_base=20.0,
        )
        fresh_snapshot_id = await _seed_forecast_data(
            session,
            fetched_at=now,
            wind_gusts_base=4.0,
        )
        context = ScheduledNotificationContext(
            scheduling_message="Wyślij mi jutro o 9 aktualną prognozę dla Chwarzna",
            human_request="Aktualna prognoza dla Gdyni Chwarzno",
            schedule="once:2026-05-01T09:00:00+02:00",
            location_id=1,
            location_name="Gdynia Chwarzno",
        )
        await _create_rule(
            rule_service,
            expression="true",
            schedule=context.schedule,
            description=context.human_request,
            notification_context=context,
        )

        fetcher = AsyncMock()
        fetcher.fetch_fresh_forecast = AsyncMock(return_value=fresh_snapshot_id)

        class Sender:
            async def send(self, chat_id: int, thread_id: int | None, text: str) -> bool:
                del chat_id, thread_id, text
                return True

        class Generator:
            def __init__(self) -> None:
                self.details: list[dict[str, Any]] = []

            async def generate(
                self,
                rule: Any,
                evaluation_detail: dict[str, Any],
            ) -> str:
                del rule
                self.details.append(evaluation_detail)
                return "Aktualna prognoza dla Chwarzna: spokojnie i bez opadów."

        generator = Generator()
        worker = RuleEvaluationWorker(
            session=session,
            forecast_repo=forecast_repo,
            rule_expression_evaluator=rule_expression_evaluator,
            rule_service=rule_service,
            settings=scheduler_settings,
            forecast_fetcher=fetcher,
            notification_sender=Sender(),
            event_service=NotificationEventService(session, AuditLogger(session)),
            deduplicator=NotificationDeduplicator(session),
            notification_content_generator=generator,
        )

        results = await worker.evaluate_rules()

        assert len(results) == 1
        assert results[0].evaluation_detail is not None
        assert results[0].evaluation_detail["snapshot_id"] == fresh_snapshot_id
        assert len(generator.details) == 1
        forecast_points = generator.details[0]["forecast_points"]
        assert isinstance(forecast_points, list)
        assert forecast_points[0]["wind_gusts_10m_ms"] == 4.0

    async def test_duplicate_candidate_is_not_sent_twice(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session, wind_gusts_base=14.0)
        await _create_rule(rule_service)

        class Sender:
            def __init__(self) -> None:
                self.count = 0

            async def send(self, chat_id: int, thread_id: int | None, text: str) -> bool:
                if chat_id and thread_id is None and text:
                    self.count += 1
                return True

        sender = Sender()
        event_service = NotificationEventService(session, AuditLogger(session))
        worker = RuleEvaluationWorker(
            session=session,
            forecast_repo=forecast_repo,
            rule_expression_evaluator=rule_expression_evaluator,
            rule_service=rule_service,
            settings=scheduler_settings,
            notification_sender=sender,
            event_service=event_service,
            deduplicator=NotificationDeduplicator(session),
        )

        await worker.evaluate_rules()
        await worker.evaluate_rules()

        assert sender.count == 1

    async def test_duplicate_scheduled_rules_send_once_and_disable_once_rules(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session, name="Gdynia Chwarzno")
        await _seed_forecast_data(session, wind_gusts_base=4.0)
        context = ScheduledNotificationContext(
            scheduling_message="Wyślij mi jutro o 9 aktualną prognozę dla Chwarzna",
            human_request="Aktualna prognoza dla Gdyni Chwarzno",
            schedule="once:2026-05-01T09:00:00+02:00",
            location_id=1,
            location_name="Gdynia Chwarzno",
        )
        rule_a = await _create_rule(
            rule_service,
            expression="true",
            schedule=context.schedule,
            description=context.human_request,
            notification_context=context,
        )
        rule_b = await _create_rule(
            rule_service,
            expression="true",
            schedule=context.schedule,
            description=context.human_request,
            notification_context=context,
        )

        class Sender:
            def __init__(self) -> None:
                self.messages: list[str] = []

            async def send(self, chat_id: int, thread_id: int | None, text: str) -> bool:
                del chat_id, thread_id
                self.messages.append(text)
                return True

        class Generator:
            def __init__(self) -> None:
                self.calls: list[tuple[int, dict[str, Any]]] = []

            async def generate(
                self,
                rule: Any,
                evaluation_detail: dict[str, Any],
            ) -> str:
                self.calls.append((rule.id, evaluation_detail))
                return "Aktualna prognoza dla Chwarzna: ciepło, bez opadów."

        sender = Sender()
        generator = Generator()
        event_service = NotificationEventService(session, AuditLogger(session))
        worker = RuleEvaluationWorker(
            session=session,
            forecast_repo=forecast_repo,
            rule_expression_evaluator=rule_expression_evaluator,
            rule_service=rule_service,
            settings=scheduler_settings,
            notification_sender=sender,
            event_service=event_service,
            deduplicator=NotificationDeduplicator(session),
            notification_content_generator=generator,
        )

        results = await worker.evaluate_rules()

        assert [r.rule_id for r in results] == [rule_a.id, rule_b.id]
        assert len(sender.messages) == 1
        assert sender.messages[0].startswith("Aktualna prognoza")
        assert len(generator.calls) == 1
        assert "forecast_points" in generator.calls[0][1]

        stmt = select(NotificationRuleORM).order_by(NotificationRuleORM.id)
        db_result = await session.execute(stmt)
        rules = db_result.scalars().all()
        assert [rule.enabled for rule in rules] == [False, False]

    async def test_scheduled_notification_falls_back_without_raw_utc_window(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session, name="Gdynia Chwarzno")
        await _seed_forecast_data(session, wind_gusts_base=4.0)
        context = ScheduledNotificationContext(
            scheduling_message="Wyślij mi jutro o 9 aktualną prognozę dla Chwarzna",
            human_request="Aktualna prognoza dla Gdyni Chwarzno",
            schedule="once:2026-05-01T09:00:00+02:00",
            location_id=1,
            location_name="Gdynia Chwarzno",
        )
        await _create_rule(
            rule_service,
            expression="true",
            schedule=context.schedule,
            description=context.human_request,
            notification_context=context,
        )

        class Sender:
            def __init__(self) -> None:
                self.message = ""

            async def send(self, chat_id: int, thread_id: int | None, text: str) -> bool:
                del chat_id, thread_id
                self.message = text
                return True

        class Generator:
            async def generate(
                self,
                rule: Any,
                evaluation_detail: dict[str, Any],
            ) -> None:
                del rule, evaluation_detail
                return None

        sender = Sender()
        worker = RuleEvaluationWorker(
            session=session,
            forecast_repo=forecast_repo,
            rule_expression_evaluator=rule_expression_evaluator,
            rule_service=rule_service,
            settings=scheduler_settings,
            notification_sender=sender,
            event_service=NotificationEventService(session, AuditLogger(session)),
            deduplicator=NotificationDeduplicator(session),
            notification_content_generator=Generator(),
        )

        await worker.evaluate_rules()

        assert "Aktualna prognoza dla Gdyni Chwarzno" in sender.message
        assert "+00:00" not in sender.message
        assert "T00:" not in sender.message

    async def test_scheduled_notification_without_context_uses_polish_fallback(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session, name="Gdynia Chwarzno")
        await _seed_forecast_data(session, wind_gusts_base=4.0)
        await _create_rule(
            rule_service,
            expression="true",
            schedule="once:2026-05-01T09:00:00+02:00",
            description="Aktualna prognoza dla Gdyni Chwarzno",
        )

        class Sender:
            def __init__(self) -> None:
                self.message = ""

            async def send(self, chat_id: int, thread_id: int | None, text: str) -> bool:
                del chat_id, thread_id
                self.message = text
                return True

        sender = Sender()
        worker = RuleEvaluationWorker(
            session=session,
            forecast_repo=forecast_repo,
            rule_expression_evaluator=rule_expression_evaluator,
            rule_service=rule_service,
            settings=scheduler_settings,
            notification_sender=sender,
            event_service=NotificationEventService(session, AuditLogger(session)),
            deduplicator=NotificationDeduplicator(session),
        )

        await worker.evaluate_rules()

        assert "Aktualna prognoza dla Gdyni Chwarzno" in sender.message
        assert "temperatura" in sender.message
        assert "+00:00" not in sender.message
        assert "T00:" not in sender.message

    async def test_run_loop_commits_read_only_cycle_before_sleep(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )

        async def stop_after_first_cycle(_seconds: float) -> None:
            assert not session.in_transaction()
            raise asyncio.CancelledError

        monkeypatch.setattr(
            "weather_agent.infrastructure.worker.rule_evaluator.asyncio.sleep",
            stop_after_first_cycle,
        )

        with pytest.raises(asyncio.CancelledError):
            await worker.run_loop()

    async def test_no_llm_calls_during_evaluation(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session, wind_gusts_base=14.0)
        await _create_rule(rule_service)

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )
        results = await worker.evaluate_rules()

        assert len(results) == 1
        r = results[0]
        assert r.evaluated is True
        assert r.evaluation_detail is not None
        assert "expression_result" in r.evaluation_detail
        assert "evaluated_metrics" in r.evaluation_detail
        assert "evaluated_functions" in r.evaluation_detail

    async def test_individual_rule_failure_does_not_crash_cycle(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session, wind_gusts_base=14.0)

        await _create_rule(
            rule_service,
            expression='max_metric("wind_gusts_10m_ms", weekend()) >= 12',
        )
        await _create_rule(
            rule_service,
            expression='max_metric("temperature_2m_c", next_hours(24)) >= 0',
        )

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )

        call_count = 0
        original_evaluate = worker._evaluate_single_rule

        async def ok_then_failing(rule, dry_run):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("ephemeral failure")
            return await original_evaluate(rule, dry_run)

        await session.commit()

        worker._evaluate_single_rule = ok_then_failing
        results = await worker.evaluate_rules()

        assert len(results) == 2
        assert results[0].evaluated is True
        assert results[1].evaluated is False
        assert results[1].error == "ephemeral failure"

        stmt = select(RuleEvaluationRun)
        db_result = await session.execute(stmt)
        eval_runs = db_result.scalars().all()
        assert len(eval_runs) == 1
        assert eval_runs[0].rule_id == 1

    async def test_rule_evaluation_failure_rolls_back_session(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session, wind_gusts_base=14.0)
        await _create_rule(rule_service)

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )

        await session.commit()

        original = worker._evaluate_single_rule

        async def failing(rule, dry_run):
            raise ValueError("test failure")

        worker._evaluate_single_rule = failing

        results = await worker.evaluate_rules()
        assert len(results) == 1
        assert results[0].evaluated is False
        assert results[0].error is not None

        worker._evaluate_single_rule = original

        results = await worker.evaluate_rules()
        assert len(results) == 1
        assert results[0].evaluated is True

    async def test_forecast_delta_uses_previous_snapshot_values(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)

        now = datetime.now(UTC)
        prev_fetched = now - timedelta(hours=6)
        curr_fetched = now - timedelta(hours=1)

        shared_targets = [now + timedelta(hours=i + 1) for i in range(3)]

        await _seed_forecast_data(
            session,
            fetched_at=prev_fetched,
            wind_gusts_base=10.0,
            num_points=3,
            target_times=shared_targets,
        )

        curr_snapshot_id = await _seed_forecast_data(
            session,
            fetched_at=curr_fetched,
            wind_gusts_base=20.0,
            num_points=3,
            target_times=shared_targets,
        )

        await _create_rule(
            rule_service,
            expression=(
                'forecast_delta_metric("wind_gusts_10m_ms", next_hours(24), '
                "previous_snapshot()) >= 5"
            ),
        )

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
        )
        results = await worker.evaluate_rules()

        assert len(results) == 1
        r = results[0]
        assert r.evaluated is True
        assert r.result is True
        assert r.notification_candidate is True
        assert r.evaluation_detail is not None
        assert r.evaluation_detail["snapshot_id"] == curr_snapshot_id

        data = await worker._build_evaluation_data(1)
        assert len(data["points"]) == 3
        assert len(data["previous_points"]) == 3
        assert data["previous_points"][0]["wind_gusts_10m_ms"] == 10.0
        assert data["points"][0]["wind_gusts_10m_ms"] == 20.0

    async def test_forecast_fetcher_populates_data_when_db_is_empty(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        """Regression test for weather-agent-blw: worker evaluates scheduled rules
        on an empty DB when a forecast fetcher is provided."""

        await _create_user(session)
        await _create_location(session)
        await _create_rule(rule_service)

        fetcher = _FakeForecastFetcher(session, forecast_repo)

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
            forecast_fetcher=fetcher,
        )
        results = await worker.evaluate_rules()

        assert len(results) == 1
        r = results[0]
        assert r.evaluated is True
        assert r.result is True
        assert r.notification_candidate is True
        assert r.error is None
        assert r.evaluation_detail is not None
        assert r.evaluation_detail["point_count"] > 0
        assert r.evaluation_detail["snapshot_id"] is not None

    async def test_forecast_fetcher_called_before_evaluation(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        snapshot_id = await _seed_forecast_data(session, wind_gusts_base=14.0)
        await _create_rule(rule_service)

        fetcher_calls: list[int] = []

        class CountingFetcher:
            async def fetch_fresh_forecast(self, location_id: int) -> int | None:
                fetcher_calls.append(location_id)
                return snapshot_id

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            rule_expression_evaluator,
            scheduler_settings,
            forecast_fetcher=CountingFetcher(),
        )
        results = await worker.evaluate_rules()

        assert len(results) == 1
        assert results[0].evaluated is True
        assert results[0].notification_candidate is True
        assert len(fetcher_calls) == 1
        assert fetcher_calls[0] == 1


class _FakeForecastFetcher:
    def __init__(self, session: AsyncSession, forecast_repo: ForecastRepository) -> None:
        self._session = session
        self._repo = forecast_repo

    async def fetch_fresh_forecast(self, location_id: int) -> int | None:
        return await _seed_forecast_data(
            self._session,
            location_id=location_id if location_id else 1,
            wind_gusts_base=14.0,
            num_points=3,
        )
