from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event as sa_event
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from weather_agent.domain.notifications.deduplication import compute_dedupe_key
from weather_agent.domain.notifications.events import (
    EventNotFoundError,
    NotificationEventService,
)
from weather_agent.domain.rules.models import NotificationRule
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
from weather_agent.infrastructure.worker.rule_evaluator import EvaluationResult
from weather_agent.observability.logging import AuditLogger


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


def _make_rule(
    id: int = 1,
    short_id: str = "R7K2",
    user_id: int = 1,
    location_id: int = 1,
    cooldown_minutes: int = 60,
    expression: str = 'max_metric("wind_gusts_10m_ms", weekend()) >= 12',
) -> NotificationRule:
    return NotificationRule(
        id=id,
        short_id=short_id,
        user_id=user_id,
        telegram_chat_id=12345,
        telegram_message_thread_id=None,
        location_id=location_id,
        expression_language="cel",
        expression=expression,
        schedule=None,
        lead_time_minutes=None,
        cooldown_minutes=cooldown_minutes,
        enabled=True,
        dry_run=False,
        description=None,
        snooze_until=None,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _make_evaluation_result(
    rule_id: int = 1,
    rule_short_id: str = "R7K2",
    expression: str = 'max_metric("wind_gusts_10m_ms", weekend()) >= 12',
    notification_candidate: bool = True,
    evaluation_detail: dict | None = None,
) -> EvaluationResult:
    if evaluation_detail is None:
        evaluation_detail = {
            "rule_id": rule_id,
            "rule_short_id": rule_short_id,
            "location_id": 1,
            "snapshot_id": None,
            "point_count": 48,
            "evaluated_metrics": ["wind_gusts_10m_ms"],
            "evaluated_functions": ["max", "weekend"],
            "expression_result": True,
            "expression_error": None,
            "forecast_window_start": "2025-06-07T00:00:00+02:00",
            "forecast_window_end": "2025-06-08T23:59:00+02:00",
            "key_metrics": {"wind_gusts_10m_ms": 15.3},
        }
    return EvaluationResult(
        rule_id=rule_id,
        rule_short_id=rule_short_id,
        expression=expression,
        evaluated=True,
        result=True,
        error=None,
        notification_candidate=notification_candidate,
        evaluation_detail=evaluation_detail,
        dry_run=False,
    )


async def _create_user(session: AsyncSession, user_id: int = 1) -> None:
    user = AuthorizedUser(id=user_id, telegram_user_id=user_id * 1000, role="user")
    session.add(user)
    await session.flush()


async def _create_location(session: AsyncSession, user_id: int = 1, loc_id: int = 1) -> None:
    loc = Location(
        id=loc_id,
        user_id=user_id,
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


async def _create_rule_orm(session: AsyncSession, rule_id: int = 1, short_id: str = "R7K2") -> None:
    orm = NotificationRuleORM(
        id=rule_id,
        short_id=short_id,
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
    session.add(orm)
    await session.flush()


async def _create_snapshot_orm(
    session: AsyncSession, snapshot_id: int = 42, location_id: int = 1
) -> None:
    orm = ForecastSnapshotORM(
        id=snapshot_id,
        provider="open-meteo",
        model="icon",
        location_id=location_id,
        fetched_at=datetime.now(UTC),
        raw_payload={},
    )
    session.add(orm)
    await session.flush()


async def _create_eval_run_orm(
    session: AsyncSession,
    run_id: int = 99,
    rule_id: int = 1,
    snapshot_id: int = 42,
    evaluation_detail: dict | None = None,
) -> None:
    if evaluation_detail is None:
        evaluation_detail = {
            "rule_id": rule_id,
            "rule_short_id": "R7K2",
            "location_id": 1,
            "snapshot_id": snapshot_id,
            "point_count": 48,
            "evaluated_metrics": ["wind_gusts_10m_ms"],
            "evaluated_functions": ["max", "weekend"],
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
        evaluation_detail=evaluation_detail,
    )
    session.add(orm)
    await session.flush()


class TestCreateEvent:
    async def test_create_event_generates_short_id_with_e_prefix(
        self, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _create_rule_orm(session)
        audit = AuditLogger(session)
        service = NotificationEventService(session, audit)

        rule = _make_rule()
        evaluation = _make_evaluation_result()
        dedupe_key = compute_dedupe_key(
            rule_id=1,
            location_id=1,
            expression=rule.expression,
        )

        evt = await service.create_event(rule, evaluation, dedupe_key, payload={})

        assert evt.short_id.startswith("E")
        assert len(evt.short_id) == 5
        assert evt.rule_id == 1
        assert evt.evaluation_run_id is None
        assert evt.telegram_chat_id == 12345
        assert evt.sent_at is None
        assert evt.suppressed is False
        assert evt.delivery_status == "sending"
        assert evt.delivery_claimed_at is not None

    async def test_create_event_stores_in_db(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        await _create_rule_orm(session)
        audit = AuditLogger(session)
        service = NotificationEventService(session, audit)

        rule = _make_rule()
        evaluation = _make_evaluation_result()

        evt = await service.create_event(rule, evaluation, None, payload={"temp": 30})

        orm = await session.get(NotificationEventORM, evt.id)
        assert orm is not None
        assert orm.short_id == evt.short_id
        assert orm.payload_hash is not None

    async def test_create_event_with_dedupe_key_stores_hash(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        await _create_rule_orm(session)
        audit = AuditLogger(session)
        service = NotificationEventService(session, audit)

        rule = _make_rule()
        evaluation = _make_evaluation_result()
        dedupe_key = compute_dedupe_key(
            rule_id=1,
            location_id=1,
            expression=rule.expression,
        )

        evt = await service.create_event(rule, evaluation, dedupe_key, payload={})

        assert evt.payload_hash is not None
        assert len(evt.payload_hash) == 64

    async def test_create_event_logs_to_audit(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        await _create_rule_orm(session)
        audit = AuditLogger(session)
        service = NotificationEventService(session, audit)

        rule = _make_rule()
        evaluation = _make_evaluation_result()

        await service.create_event(rule, evaluation, None, payload={})

        from weather_agent.infrastructure.db.base import AuditLog

        stmt = select(AuditLog).where(AuditLog.event_type == "notification_sent")
        result = await session.execute(stmt)
        audit_row = result.scalar_one_or_none()
        assert audit_row is not None
        assert "event_short_id" in audit_row.details

    async def test_create_event_with_evaluation_run_id(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        await _create_rule_orm(session)
        await _create_snapshot_orm(session)
        await _create_eval_run_orm(session)
        audit = AuditLogger(session)
        service = NotificationEventService(session, audit)

        rule = _make_rule()
        evaluation_detail = {
            "rule_id": 1,
            "rule_short_id": "R7K2",
            "location_id": 1,
            "snapshot_id": 42,
            "point_count": 48,
            "evaluated_metrics": ["wind_gusts_10m_ms"],
            "evaluated_functions": ["max", "weekend"],
            "expression_result": True,
            "expression_error": None,
            "forecast_window_start": "2025-06-07T00:00:00+02:00",
            "forecast_window_end": "2025-06-08T23:59:00+02:00",
            "key_metrics": {"wind_gusts_10m_ms": 15.3},
            "evaluation_run_id": 99,
        }
        evaluation = EvaluationResult(
            rule_id=1,
            rule_short_id="R7K2",
            expression='max_metric("wind_gusts_10m_ms", weekend()) >= 12',
            evaluated=True,
            result=True,
            error=None,
            notification_candidate=True,
            evaluation_detail=evaluation_detail,
            dry_run=False,
        )

        evt = await service.create_event(rule, evaluation, None, payload={})
        assert evt.evaluation_run_id == 99

    async def test_create_event_once_reuses_existing_unsuppressed_event(
        self, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _create_rule_orm(session)
        audit = AuditLogger(session)
        service = NotificationEventService(session, audit)

        rule = _make_rule()
        evaluation = _make_evaluation_result()
        dedupe_key = compute_dedupe_key(
            rule_id=rule.id,
            location_id=rule.location_id,
            expression=rule.expression,
        )

        first, first_created = await service.create_event_once(
            rule,
            evaluation,
            dedupe_key,
            payload={},
        )
        second, second_created = await service.create_event_once(
            rule,
            evaluation,
            dedupe_key,
            payload={},
        )

        assert first_created is True
        assert second_created is False
        assert second.id == first.id

        stmt = select(NotificationEventORM).where(NotificationEventORM.rule_id == rule.id)
        result = await session.execute(stmt)
        assert len(result.scalars().all()) == 1

    async def test_create_event_once_reclaims_stale_unsent_event(
        self, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        await _create_rule_orm(session)
        audit = AuditLogger(session)
        service = NotificationEventService(session, audit)

        rule = _make_rule()
        evaluation = _make_evaluation_result()
        dedupe_key = compute_dedupe_key(
            rule_id=rule.id,
            location_id=rule.location_id,
            expression=rule.expression,
        )

        first, first_created = await service.create_event_once(
            rule,
            evaluation,
            dedupe_key,
            payload={},
        )
        orm = await session.get(NotificationEventORM, first.id)
        assert orm is not None
        orm.delivery_claimed_at = datetime.now(UTC) - timedelta(minutes=20)
        await session.flush()

        second, second_created = await service.create_event_once(
            rule,
            evaluation,
            dedupe_key,
            payload={},
            lease_timeout=timedelta(minutes=10),
        )

        assert first_created is True
        assert second_created is True
        assert second.id == first.id
        assert second.delivery_status == "sending"
        assert second.delivery_claimed_at is not None


class TestMarkSent:
    async def test_mark_sent_sets_sent_at(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        await _create_rule_orm(session)
        audit = AuditLogger(session)
        service = NotificationEventService(session, audit)

        rule = _make_rule()
        evaluation = _make_evaluation_result()
        evt = await service.create_event(rule, evaluation, None, payload={})
        assert evt.sent_at is None

        updated = await service.mark_sent(evt.id, message_text="Wind alert!")
        assert updated.sent_at is not None
        assert updated.message_text == "Wind alert!"
        assert updated.delivery_status == "sent"

    async def test_mark_sent_without_message_text(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        await _create_rule_orm(session)
        audit = AuditLogger(session)
        service = NotificationEventService(session, audit)

        rule = _make_rule()
        evaluation = _make_evaluation_result()
        evt = await service.create_event(rule, evaluation, None, payload={})

        updated = await service.mark_sent(evt.id)
        assert updated.sent_at is not None
        assert updated.message_text is None

    async def test_mark_sent_nonexistent_raises(self, session: AsyncSession) -> None:
        audit = AuditLogger(session)
        service = NotificationEventService(session, audit)

        with pytest.raises(EventNotFoundError):
            await service.mark_sent(9999)


class TestMarkSuppressed:
    async def test_mark_suppressed_sets_fields(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        await _create_rule_orm(session)
        audit = AuditLogger(session)
        service = NotificationEventService(session, audit)

        rule = _make_rule()
        evaluation = _make_evaluation_result()
        evt = await service.create_event(rule, evaluation, None, payload={})

        updated = await service.mark_suppressed(evt.id, reason="cooldown")
        assert updated.suppressed is True
        assert updated.suppress_reason == "cooldown"
        assert updated.delivery_status == "suppressed"

    async def test_mark_suppressed_nonexistent_raises(self, session: AsyncSession) -> None:
        audit = AuditLogger(session)
        service = NotificationEventService(session, audit)

        with pytest.raises(EventNotFoundError):
            await service.mark_suppressed(9999, reason="test")


class TestGetEvent:
    async def test_get_event_by_short_id(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        await _create_rule_orm(session)
        audit = AuditLogger(session)
        service = NotificationEventService(session, audit)

        rule = _make_rule()
        evaluation = _make_evaluation_result()
        evt = await service.create_event(rule, evaluation, None, payload={})

        found = await service.get_event(short_id=evt.short_id)
        assert found is not None
        assert found.id == evt.id
        assert found.short_id == evt.short_id

    async def test_get_event_by_short_id_with_hash_prefix(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        await _create_rule_orm(session)
        audit = AuditLogger(session)
        service = NotificationEventService(session, audit)

        rule = _make_rule()
        evaluation = _make_evaluation_result()
        evt = await service.create_event(rule, evaluation, None, payload={})

        found = await service.get_event(short_id=f"#{evt.short_id}")
        assert found is not None
        assert found.id == evt.id

    async def test_get_event_by_id(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        await _create_rule_orm(session)
        audit = AuditLogger(session)
        service = NotificationEventService(session, audit)

        rule = _make_rule()
        evaluation = _make_evaluation_result()
        evt = await service.create_event(rule, evaluation, None, payload={})

        found = await service.get_event(event_id=evt.id)
        assert found is not None
        assert found.short_id == evt.short_id

    async def test_get_event_not_found_returns_none(self, session: AsyncSession) -> None:
        audit = AuditLogger(session)
        service = NotificationEventService(session, audit)

        found = await service.get_event(short_id="EXXXX")
        assert found is None

    async def test_get_event_no_params_returns_none(self, session: AsyncSession) -> None:
        audit = AuditLogger(session)
        service = NotificationEventService(session, audit)

        found = await service.get_event()
        assert found is None


class TestListEventsForRule:
    async def test_list_events_for_rule(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        await _create_rule_orm(session)
        audit = AuditLogger(session)
        service = NotificationEventService(session, audit)

        rule = _make_rule()
        evaluation = _make_evaluation_result()

        e1 = await service.create_event(rule, evaluation, None, payload={})
        e2 = await service.create_event(rule, evaluation, None, payload={})
        e3 = await service.create_event(rule, evaluation, None, payload={})

        events = await service.list_events_for_rule(rule.id)
        assert len(events) == 3
        ids = {e.id for e in events}
        assert e1.id in ids
        assert e2.id in ids
        assert e3.id in ids

    async def test_list_events_respects_limit(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        await _create_rule_orm(session)
        audit = AuditLogger(session)
        service = NotificationEventService(session, audit)

        rule = _make_rule()
        evaluation = _make_evaluation_result()

        for _ in range(5):
            await service.create_event(rule, evaluation, None, payload={})

        events = await service.list_events_for_rule(rule.id, limit=2)
        assert len(events) == 2

    async def test_list_events_ordered_by_created_at_desc(self, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        await _create_rule_orm(session)
        audit = AuditLogger(session)
        service = NotificationEventService(session, audit)

        rule = _make_rule()
        evaluation = _make_evaluation_result()

        e1 = await service.create_event(rule, evaluation, None, payload={})
        e2 = await service.create_event(rule, evaluation, None, payload={})

        events = await service.list_events_for_rule(rule.id)
        assert events[0].id == e2.id
        assert events[1].id == e1.id

    async def test_list_events_empty_for_rule(self, session: AsyncSession) -> None:
        audit = AuditLogger(session)
        service = NotificationEventService(session, audit)

        events = await service.list_events_for_rule(999)
        assert events == []
