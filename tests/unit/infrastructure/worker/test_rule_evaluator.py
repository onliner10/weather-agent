from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from weather_agent.domain.cel.evaluator import CELEvaluator
from weather_agent.domain.rules.models import RuleCreate
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
    NotificationRule as NotificationRuleORM,
)
from weather_agent.infrastructure.repositories.forecast_repository import ForecastRepository
from weather_agent.infrastructure.worker.rule_evaluator import (
    EvaluationResult,
    RuleEvaluationWorker,
)
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
def cel_evaluator() -> CELEvaluator:
    return CELEvaluator()


@pytest.fixture()
def scheduler_settings() -> SchedulerSettings:
    return SchedulerSettings(rule_evaluation_minutes=15)


@pytest.fixture()
def rule_service(session: AsyncSession, cel_evaluator: CELEvaluator) -> NotificationRuleService:
    return NotificationRuleService(session, cel_evaluator)


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
    expression: str = 'max("wind_gusts_10m_ms", weekend()) >= 12',
    dry_run: bool = False,
    enabled: bool = True,
) -> Any:
    data = RuleCreate(
        telegram_chat_id=12345,
        location_id=location_id,
        expression=expression,
        dry_run=dry_run,
        enabled=enabled,
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
    cel_evaluator: CELEvaluator,
    settings: SchedulerSettings,
    forecast_fetcher: Any = None,
) -> RuleEvaluationWorker:
    return RuleEvaluationWorker(
        session=session,
        forecast_repo=forecast_repo,
        cel_evaluator=cel_evaluator,
        rule_service=rule_service,
        settings=settings,
        forecast_fetcher=forecast_fetcher,
    )


class TestEvaluationResult:
    def test_result_fields(self) -> None:
        result = EvaluationResult(
            rule_id=1,
            rule_short_id="R0001",
            expression='max("wind_gusts_10m_ms", weekend()) >= 12',
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
        cel_evaluator: CELEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            cel_evaluator,
            scheduler_settings,
        )
        results = await worker.evaluate_rules()
        assert results == []

    async def test_evaluate_true_expression_generates_candidate(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        cel_evaluator: CELEvaluator,
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
            cel_evaluator,
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
        cel_evaluator: CELEvaluator,
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
            cel_evaluator,
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
        cel_evaluator: CELEvaluator,
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
            cel_evaluator,
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
        cel_evaluator: CELEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _create_rule(rule_service)

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            cel_evaluator,
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
        cel_evaluator: CELEvaluator,
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
            cel_evaluator,
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
        cel_evaluator: CELEvaluator,
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
            cel_evaluator,
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
        cel_evaluator: CELEvaluator,
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
            cel_evaluator,
            scheduler_settings,
        )
        results = await worker.evaluate_rules()

        assert results == []

    async def test_evaluation_is_deterministic(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        cel_evaluator: CELEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session, wind_gusts_base=6.0)

        await _create_rule(
            rule_service,
            expression='avg("wind_speed_10m_ms", next_hours(24)) >= 3',
        )

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            cel_evaluator,
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
        cel_evaluator: CELEvaluator,
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
            cel_evaluator,
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
        cel_evaluator: CELEvaluator,
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
            cel_evaluator,
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
        cel_evaluator: CELEvaluator,
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
            cel_evaluator,
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
        cel_evaluator: CELEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session, wind_gusts_base=14.0)

        await _create_rule(
            rule_service,
            expression='max("wind_gusts_10m_ms", weekend()) >= 12',
        )
        await _create_rule(
            rule_service,
            expression='avg("temperature_2m_c", next_hours(24)) >= 100',
        )

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            cel_evaluator,
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
        cel_evaluator: CELEvaluator,
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
            cel_evaluator,
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
        cel_evaluator: CELEvaluator,
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
            cel_evaluator,
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
        cel_evaluator: CELEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session, wind_gusts_base=14.0)
        await _create_rule(rule_service)

        fetcher = AsyncMock()
        fetcher.fetch_fresh_forecast = AsyncMock(side_effect=Exception("network error"))

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            cel_evaluator,
            scheduler_settings,
            forecast_fetcher=fetcher,
        )
        results = await worker.evaluate_rules()

        assert len(results) == 1
        assert results[0].evaluated is True

    async def test_no_llm_calls_during_evaluation(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        cel_evaluator: CELEvaluator,
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
            cel_evaluator,
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
        cel_evaluator: CELEvaluator,
        scheduler_settings: SchedulerSettings,
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _seed_forecast_data(session, wind_gusts_base=14.0)

        await _create_rule(
            rule_service,
            expression='max("wind_gusts_10m_ms", weekend()) >= 12',
        )
        await _create_rule(
            rule_service,
            expression='max("temperature_2m_c", next_hours(24)) >= 0',
        )

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            cel_evaluator,
            scheduler_settings,
        )

        call_count = 0
        original_evaluate = worker._evaluate_single_rule

        async def failing_then_ok(rule, dry_run):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("ephemeral failure")
            return await original_evaluate(rule, dry_run)

        await session.commit()

        worker._evaluate_single_rule = failing_then_ok
        results = await worker.evaluate_rules()

        assert len(results) == 2
        assert results[0].evaluated is False
        assert results[0].error == "ephemeral failure"
        assert results[1].evaluated is True

        stmt = select(RuleEvaluationRun)
        db_result = await session.execute(stmt)
        eval_runs = db_result.scalars().all()
        assert len(eval_runs) == 1
        assert eval_runs[0].rule_id == 2

    async def test_rule_evaluation_failure_rolls_back_session(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_service: NotificationRuleService,
        cel_evaluator: CELEvaluator,
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
            cel_evaluator,
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
        cel_evaluator: CELEvaluator,
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
                'forecast_delta("wind_gusts_10m_ms", next_hours(24), previous_snapshot()) >= 5'
            ),
        )

        worker = _make_worker(
            session,
            forecast_repo,
            rule_service,
            cel_evaluator,
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
