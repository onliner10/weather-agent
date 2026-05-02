from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event
from sqlalchemy import select as sa_select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from weather_agent.domain.cel.evaluator import CELEvaluator
from weather_agent.domain.rules.models import (
    CELValidationError,
    RuleCreate,
    RuleNotFoundError,
    RuleUpdate,
)
from weather_agent.domain.rules.service import NotificationRuleService
from weather_agent.infrastructure.db.base import (
    AuthorizedUser,
    Base,
)
from weather_agent.infrastructure.db.base import (
    NotificationEvent as NotificationEventORM,
)
from weather_agent.infrastructure.db.base import (
    NotificationRule as NotificationRuleORM,
)


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
def service(session: AsyncSession, cel_evaluator: CELEvaluator) -> NotificationRuleService:
    return NotificationRuleService(session, cel_evaluator)


async def _create_user(session: AsyncSession, user_id: int = 1) -> None:
    user = AuthorizedUser(id=user_id, telegram_user_id=user_id * 1000, role="user")
    session.add(user)
    await session.flush()


async def _create_location(session: AsyncSession, user_id: int = 1, loc_id: int = 1) -> None:
    from weather_agent.infrastructure.db.base import Location

    loc = Location(
        id=loc_id,
        user_id=user_id,
        name="Test Location",
        aliases=["test"],
        latitude=52.22,
        longitude=21.01,
        enabled=True,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    session.add(loc)
    await session.flush()


async def _create_rule_raw(
    session: AsyncSession,
    user_id: int = 1,
    rule_id: int = 1,
    short_id: str = "R0001",
) -> NotificationRuleORM:
    orm = NotificationRuleORM(
        id=rule_id,
        short_id=short_id,
        user_id=user_id,
        telegram_chat_id=12345,
        telegram_message_thread_id=None,
        location_id=1,
        expression_language="cel",
        expression='max("wind_gusts_10m_ms", weekend()) >= 12',
        enabled=True,
        dry_run=False,
        cooldown_minutes=60,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    session.add(orm)
    await session.flush()
    return orm


async def _create_event_raw(
    session: AsyncSession,
    rule_id: int = 1,
    event_id: int = 1,
    short_id: str = "E0001",
) -> NotificationEventORM:
    orm = NotificationEventORM(
        id=event_id,
        short_id=short_id,
        rule_id=rule_id,
        telegram_chat_id=12345,
        telegram_message_thread_id=None,
        suppressed=False,
        created_at=datetime.now(UTC),
    )
    session.add(orm)
    await session.flush()
    return orm


class TestCreateRule:
    async def test_create_basic(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        data = RuleCreate(
            telegram_chat_id=12345,
            location_id=1,
            expression='max("wind_gusts_10m_ms", weekend()) >= 12',
        )
        rule = await service.create_rule(1, data)
        assert rule.id > 0
        assert rule.short_id.startswith("R")
        assert rule.user_id == 1
        assert rule.telegram_chat_id == 12345
        assert rule.location_id == 1
        assert rule.expression == 'max("wind_gusts_10m_ms", weekend()) >= 12'
        assert rule.expression_language == "cel"
        assert rule.enabled is True
        assert rule.dry_run is False
        assert rule.cooldown_minutes == 60

    async def test_create_with_optional_fields(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        data = RuleCreate(
            telegram_chat_id=12345,
            telegram_message_thread_id=42,
            location_id=1,
            expression='avg("wind_speed_10m_ms", next_hours(24)) >= 7',
            schedule="0 8 * * *",
            lead_time_minutes=30,
            cooldown_minutes=120,
            description="Wind alert",
        )
        rule = await service.create_rule(1, data)
        assert rule.telegram_message_thread_id == 42
        assert rule.schedule == "0 8 * * *"
        assert rule.lead_time_minutes == 30
        assert rule.cooldown_minutes == 120
        assert rule.description == "Wind alert"

    async def test_create_validates_cel(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        data = RuleCreate(
            telegram_chat_id=12345,
            location_id=1,
            expression="invalid_function_xyz(123)",
        )
        with pytest.raises(CELValidationError):
            await service.create_rule(1, data)

    async def test_create_validates_empty_expression(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        data = RuleCreate(
            telegram_chat_id=12345,
            location_id=1,
            expression="",
        )
        with pytest.raises(CELValidationError):
            await service.create_rule(1, data)


class TestListRules:
    async def test_list_empty(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        result = await service.list_rules(1)
        assert result == []

    async def test_list_returns_rules(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        data1 = RuleCreate(
            telegram_chat_id=12345,
            location_id=1,
            expression='max("wind_gusts_10m_ms", weekend()) >= 12',
        )
        data2 = RuleCreate(
            telegram_chat_id=67890,
            location_id=1,
            expression='avg("wind_speed_10m_ms", next_hours(24)) >= 7',
        )
        await service.create_rule(1, data1)
        await service.create_rule(1, data2)
        result = await service.list_rules(1)
        assert len(result) == 2

    async def test_list_excludes_disabled_by_default(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        data = RuleCreate(
            telegram_chat_id=12345,
            location_id=1,
            expression='max("wind_gusts_10m_ms", weekend()) >= 12',
        )
        rule = await service.create_rule(1, data)
        await service.disable_rule(rule.id)
        result = await service.list_rules(1)
        assert result == []

    async def test_list_includes_disabled_when_requested(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        data = RuleCreate(
            telegram_chat_id=12345,
            location_id=1,
            expression='max("wind_gusts_10m_ms", weekend()) >= 12',
        )
        rule = await service.create_rule(1, data)
        await service.disable_rule(rule.id)
        result = await service.list_rules(1, include_disabled=True)
        assert len(result) == 1


class TestGetRule:
    async def test_get_by_id(self, service: NotificationRuleService, session: AsyncSession) -> None:
        await _create_user(session)
        await _create_location(session)
        data = RuleCreate(
            telegram_chat_id=12345,
            location_id=1,
            expression='max("wind_gusts_10m_ms", weekend()) >= 12',
        )
        created = await service.create_rule(1, data)
        fetched = await service.get_rule(rule_id=created.id)
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.expression == created.expression

    async def test_get_by_short_id(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        data = RuleCreate(
            telegram_chat_id=12345,
            location_id=1,
            expression='max("wind_gusts_10m_ms", weekend()) >= 12',
        )
        created = await service.create_rule(1, data)
        fetched = await service.get_rule(short_id=created.short_id)
        assert fetched is not None
        assert fetched.id == created.id

    async def test_get_by_short_id_with_hash(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        data = RuleCreate(
            telegram_chat_id=12345,
            location_id=1,
            expression='max("wind_gusts_10m_ms", weekend()) >= 12',
        )
        created = await service.create_rule(1, data)
        fetched = await service.get_rule(short_id=f"#{created.short_id}")
        assert fetched is not None
        assert fetched.id == created.id

    async def test_get_nonexistent_returns_none(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        result = await service.get_rule(rule_id=9999)
        assert result is None

    async def test_get_no_args_returns_none(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        result = await service.get_rule()
        assert result is None

    async def test_get_rule_for_user_scopes_short_id(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        await _create_user(session, user_id=1)
        await _create_user(session, user_id=2)
        await _create_location(session, user_id=1, loc_id=1)
        await _create_location(session, user_id=2, loc_id=2)
        await _create_rule_raw(session, user_id=1, rule_id=1, short_id="RUSER1")

        fetched = await service.get_rule_for_user(2, short_id="RUSER1")

        assert fetched is None


class TestUpdateRule:
    async def test_update_expression(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        data = RuleCreate(
            telegram_chat_id=12345,
            location_id=1,
            expression='max("wind_gusts_10m_ms", weekend()) >= 12',
        )
        rule = await service.create_rule(1, data)
        updated = await service.update_rule(
            rule.id, RuleUpdate(expression='avg("temperature_2m_c", tomorrow()) <= 0')
        )
        assert updated.expression == 'avg("temperature_2m_c", tomorrow()) <= 0'

    async def test_update_validates_cel(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        data = RuleCreate(
            telegram_chat_id=12345,
            location_id=1,
            expression='max("wind_gusts_10m_ms", weekend()) >= 12',
        )
        rule = await service.create_rule(1, data)
        with pytest.raises(CELValidationError):
            await service.update_rule(rule.id, RuleUpdate(expression="bad_fn(123)"))

    async def test_update_no_cel_validation_when_expression_unchanged(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        data = RuleCreate(
            telegram_chat_id=12345,
            location_id=1,
            expression='max("wind_gusts_10m_ms", weekend()) >= 12',
            description="original",
        )
        rule = await service.create_rule(1, data)
        updated = await service.update_rule(rule.id, RuleUpdate(description="updated desc"))
        assert updated.description == "updated desc"
        assert updated.expression == rule.expression

    async def test_update_nonexistent_raises(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        with pytest.raises(RuleNotFoundError):
            await service.update_rule(9999, RuleUpdate(description="x"))

    async def test_update_cooldown(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        data = RuleCreate(
            telegram_chat_id=12345,
            location_id=1,
            expression='max("wind_gusts_10m_ms", weekend()) >= 12',
        )
        rule = await service.create_rule(1, data)
        updated = await service.update_rule(rule.id, RuleUpdate(cooldown_minutes=30))
        assert updated.cooldown_minutes == 30


class TestEnableDisable:
    async def test_disable_rule(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        data = RuleCreate(
            telegram_chat_id=12345,
            location_id=1,
            expression='max("wind_gusts_10m_ms", weekend()) >= 12',
        )
        rule = await service.create_rule(1, data)
        assert rule.enabled is True
        disabled = await service.disable_rule(rule.id)
        assert disabled.enabled is False

    async def test_enable_rule(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        data = RuleCreate(
            telegram_chat_id=12345,
            location_id=1,
            expression='max("wind_gusts_10m_ms", weekend()) >= 12',
        )
        rule = await service.create_rule(1, data)
        await service.disable_rule(rule.id)
        enabled = await service.enable_rule(rule.id)
        assert enabled.enabled is True

    async def test_enable_nonexistent_raises(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        with pytest.raises(RuleNotFoundError):
            await service.enable_rule(9999)

    async def test_disable_nonexistent_raises(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        with pytest.raises(RuleNotFoundError):
            await service.disable_rule(9999)


class TestDeleteRule:
    async def test_delete_existing(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        data = RuleCreate(
            telegram_chat_id=12345,
            location_id=1,
            expression='max("wind_gusts_10m_ms", weekend()) >= 12',
        )
        rule = await service.create_rule(1, data)
        result = await service.delete_rule(rule.id)
        assert result is True
        fetched = await service.get_rule(rule_id=rule.id)
        assert fetched is None

    async def test_delete_nonexistent(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        result = await service.delete_rule(9999)
        assert result is False

    async def test_delete_keeps_events(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        rule_orm = await _create_rule_raw(session, short_id="RKEEP")
        event_orm = await _create_event_raw(
            session, rule_id=rule_orm.id, event_id=10, short_id="EKEEP"
        )
        result = await service.delete_rule(rule_orm.id)
        assert result is True

        stmt = sa_select(NotificationEventORM).where(NotificationEventORM.id == event_orm.id)
        res = await session.execute(stmt)
        surviving_event = res.scalar_one_or_none()
        assert surviving_event is not None
        assert surviving_event.short_id == "EKEEP"


class TestSnoozeRule:
    async def test_snooze_sets_until(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        data = RuleCreate(
            telegram_chat_id=12345,
            location_id=1,
            expression='max("wind_gusts_10m_ms", weekend()) >= 12',
        )
        rule = await service.create_rule(1, data)
        until = datetime.now(UTC) + timedelta(hours=2)
        snoozed = await service.snooze_rule(rule.id, until)
        assert snoozed.snooze_until is not None

    async def test_snooze_nonexistent_raises(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        until = datetime.now(UTC) + timedelta(hours=2)
        with pytest.raises(RuleNotFoundError):
            await service.snooze_rule(9999, until)


class TestSetDryRun:
    async def test_set_dry_run_on(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        data = RuleCreate(
            telegram_chat_id=12345,
            location_id=1,
            expression='max("wind_gusts_10m_ms", weekend()) >= 12',
        )
        rule = await service.create_rule(1, data)
        assert rule.dry_run is False
        updated = await service.set_dry_run(rule.id, True)
        assert updated.dry_run is True

    async def test_set_dry_run_off(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        await _create_location(session)
        data = RuleCreate(
            telegram_chat_id=12345,
            location_id=1,
            expression='max("wind_gusts_10m_ms", weekend()) >= 12',
            dry_run=True,
        )
        rule = await service.create_rule(1, data)
        assert rule.dry_run is True
        updated = await service.set_dry_run(rule.id, False)
        assert updated.dry_run is False

    async def test_set_dry_run_nonexistent_raises(
        self, service: NotificationRuleService, session: AsyncSession
    ) -> None:
        with pytest.raises(RuleNotFoundError):
            await service.set_dry_run(9999, True)
