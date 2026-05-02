from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import event as sa_event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from weather_agent.domain.notifications.events import EventNotFoundError, ExplanationService
from weather_agent.infrastructure.db.base import (
    AuthorizedUser,
    Base,
    Location,
)
from weather_agent.infrastructure.db.base import (
    ForecastSnapshot as ForecastSnapshotORM,
)
from weather_agent.infrastructure.db.base import (
    NotificationEvent as NotificationEventORM,
)
from weather_agent.infrastructure.db.base import (
    NotificationRule as NotificationRuleORM,
)
from weather_agent.infrastructure.db.base import (
    RuleEvaluationRun as RuleEvaluationRunORM,
)


def _set_sqlite_foreign_keys(dbapi_connection: object, connection_record: object) -> None:
    cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


@pytest.fixture()
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sa_event.listen(engine.sync_engine, "connect", _set_sqlite_foreign_keys)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


async def _setup_base_data(session: AsyncSession) -> None:
    user = AuthorizedUser(id=1, telegram_user_id=1000, role="user")
    session.add(user)
    await session.flush()

    loc = Location(
        id=1,
        user_id=1,
        name="Warszawa",
        aliases=["wawa"],
        latitude=52.22,
        longitude=21.01,
        enabled=True,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    session.add(loc)
    await session.flush()

    rule = NotificationRuleORM(
        id=1,
        short_id="R7K2",
        user_id=1,
        telegram_chat_id=12345,
        telegram_message_thread_id=None,
        location_id=1,
        expression_language="cel",
        expression='max_metric("wind_gusts_10m_ms", weekend()) >= 12',
        enabled=True,
        dry_run=False,
        cooldown_minutes=60,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    session.add(rule)
    await session.flush()

    snapshot = ForecastSnapshotORM(
        id=42,
        provider="open-meteo",
        model="icon",
        location_id=1,
        fetched_at=datetime.now(UTC),
        raw_payload={},
    )
    session.add(snapshot)
    await session.flush()


async def _create_event(
    session: AsyncSession,
    event_id: int = 1,
    short_id: str = "E9M4",
    rule_id: int | None = 1,
    evaluation_run_id: int | None = 99,
    suppressed: bool = False,
    suppress_reason: str | None = None,
) -> NotificationEventORM:
    orm = NotificationEventORM(
        id=event_id,
        short_id=short_id,
        rule_id=rule_id,
        evaluation_run_id=evaluation_run_id,
        telegram_chat_id=12345,
        telegram_message_thread_id=None,
        suppressed=suppressed,
        suppress_reason=suppress_reason,
        sent_at=datetime.now(UTC) if not suppressed else None,
        created_at=datetime.now(UTC),
    )
    session.add(orm)
    await session.flush()
    await session.refresh(orm)
    return orm


async def _create_eval_run(
    session: AsyncSession,
    run_id: int = 99,
    rule_id: int = 1,
    snapshot_id: int = 42,
) -> RuleEvaluationRunORM:
    detail = {
        "rule_id": rule_id,
        "rule_short_id": "R7K2",
        "location_id": 1,
        "snapshot_id": snapshot_id,
        "point_count": 48,
        "evaluated_metrics": ["wind_gusts_10m_ms"],
        "evaluated_functions": ["max_metric", "weekend"],
        "expression_result": True,
        "expression_error": None,
        "forecast_window_start": "2025-06-07T00:00:00+02:00",
        "forecast_window_end": "2025-06-08T23:59:00+02:00",
        "key_metrics": {"wind_gusts_10m_ms": 15.3},
        "evaluation_run_id": run_id,
    }
    orm = RuleEvaluationRunORM(
        id=run_id,
        rule_id=rule_id,
        snapshot_id=snapshot_id,
        evaluated_at=datetime.now(UTC),
        result=True,
        evaluation_detail=detail,
    )
    session.add(orm)
    await session.flush()
    await session.refresh(orm)
    return orm


class TestExplainNotification:
    async def test_basic_explanation(self, session: AsyncSession) -> None:
        await _setup_base_data(session)
        await _create_eval_run(session)
        await _create_event(session)

        service = ExplanationService(session)
        explanation = await service.explain_notification("E9M4")

        assert "#E9M4" in explanation
        assert "#R7K2" in explanation
        assert "15.3 m/s" in explanation

    async def test_explanation_with_hash_prefix(self, session: AsyncSession) -> None:
        await _setup_base_data(session)
        await _create_eval_run(session)
        await _create_event(session)

        service = ExplanationService(session)
        explanation = await service.explain_notification("#E9M4")

        assert "#E9M4" in explanation

    async def test_explanation_includes_metrics(self, session: AsyncSession) -> None:
        await _setup_base_data(session)
        await _create_eval_run(session)
        await _create_event(session)

        service = ExplanationService(session)
        explanation = await service.explain_notification("E9M4")

        assert "porywy wiatru" in explanation
        assert "15.3 m/s" in explanation

    async def test_explanation_includes_forecast_window(self, session: AsyncSession) -> None:
        await _setup_base_data(session)
        await _create_eval_run(session)
        await _create_event(session)

        service = ExplanationService(session)
        explanation = await service.explain_notification("E9M4")

        assert "Okno prognozy" in explanation

    async def test_explanation_includes_snapshot(self, session: AsyncSession) -> None:
        await _setup_base_data(session)
        await _create_eval_run(session)
        await _create_event(session)

        service = ExplanationService(session)
        explanation = await service.explain_notification("E9M4")

        assert "Zrzut prognozy" in explanation

    async def test_explanation_includes_expression(self, session: AsyncSession) -> None:
        await _setup_base_data(session)
        await _create_eval_run(session)
        await _create_event(session)

        service = ExplanationService(session)
        explanation = await service.explain_notification("E9M4")

        assert 'max_metric("wind_gusts_10m_ms", weekend()) >= 12' in explanation

    async def test_explanation_suppressed_event(self, session: AsyncSession) -> None:
        await _setup_base_data(session)
        await _create_eval_run(session)
        await _create_event(session, suppressed=True, suppress_reason="cooldown")

        service = ExplanationService(session)
        explanation = await service.explain_notification("E9M4")

        assert "wstrzymane" in explanation
        assert "cooldown" in explanation

    async def test_explanation_event_not_found(self, session: AsyncSession) -> None:
        service = ExplanationService(session)

        with pytest.raises(EventNotFoundError):
            await service.explain_notification("EXXXX")

    async def test_explanation_no_evaluation_run(self, session: AsyncSession) -> None:
        await _setup_base_data(session)
        await _create_event(session, evaluation_run_id=None)

        service = ExplanationService(session)
        explanation = await service.explain_notification("E9M4")

        assert "#E9M4" in explanation
        assert "#R7K2" in explanation

    async def test_explanation_no_rule(self, session: AsyncSession) -> None:
        await _setup_base_data(session)
        await _create_eval_run(session)
        await _create_event(session, rule_id=None)

        service = ExplanationService(session)
        explanation = await service.explain_notification("E9M4")

        assert "(reguła usunięta)" in explanation

    async def test_explanation_multiple_metrics(self, session: AsyncSession) -> None:
        await _setup_base_data(session)
        detail = {
            "rule_id": 1,
            "rule_short_id": "R7K2",
            "location_id": 1,
            "snapshot_id": 42,
            "point_count": 48,
            "evaluated_metrics": ["temperature_2m_c", "wind_gusts_10m_ms"],
            "evaluated_functions": ["max_metric", "weekend"],
            "expression_result": True,
            "expression_error": None,
            "forecast_window_start": "2025-06-07T00:00:00+02:00",
            "forecast_window_end": "2025-06-08T23:59:00+02:00",
            "key_metrics": {"temperature_2m_c": 28.5, "wind_gusts_10m_ms": 15.3},
            "evaluation_run_id": 99,
        }
        orm = RuleEvaluationRunORM(
            id=99,
            rule_id=1,
            snapshot_id=42,
            evaluated_at=datetime.now(UTC),
            result=True,
            evaluation_detail=detail,
        )
        session.add(orm)
        await session.flush()
        await _create_event(session)

        service = ExplanationService(session)
        explanation = await service.explain_notification("E9M4")

        assert "temperatura" in explanation
        assert "porywy wiatru" in explanation
        assert "28.5 °C" in explanation
        assert "15.3 m/s" in explanation

    async def test_explanation_functions_in_polish(self, session: AsyncSession) -> None:
        await _setup_base_data(session)
        await _create_eval_run(session)
        await _create_event(session)

        service = ExplanationService(session)
        explanation = await service.explain_notification("E9M4")

        assert "maksimum" in explanation
        assert "weekend" in explanation

    async def test_explanation_sent_event(self, session: AsyncSession) -> None:
        await _setup_base_data(session)
        await _create_eval_run(session)
        await _create_event(session)

        service = ExplanationService(session)
        explanation = await service.explain_notification("E9M4")

        assert "wysłane" in explanation
