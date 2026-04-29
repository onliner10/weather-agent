from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from weather_agent.domain.cel.evaluator import CELEvaluator
from weather_agent.domain.rules.models import (
    CELValidationError,
    NotificationRule,
    RuleCreate,
    RuleNotFoundError,
    RuleUpdate,
    ShortIdCollisionError,
)
from weather_agent.domain.rules.short_id_generator import generate_short_id, strip_hash_prefix
from weather_agent.infrastructure.db.base import NotificationRule as NotificationRuleORM

_MAX_COLLISION_RETRIES = 10


def _orm_to_domain(orm: NotificationRuleORM) -> NotificationRule:
    return NotificationRule(
        id=orm.id,
        short_id=orm.short_id,
        user_id=orm.user_id,
        telegram_chat_id=orm.telegram_chat_id,
        telegram_message_thread_id=orm.telegram_message_thread_id,
        location_id=orm.location_id,
        expression_language=orm.expression_language,
        expression=orm.expression,
        schedule=orm.schedule,
        lead_time_minutes=orm.lead_time_minutes,
        cooldown_minutes=orm.cooldown_minutes or 60,
        enabled=orm.enabled,
        dry_run=orm.dry_run,
        description=orm.description,
        snooze_until=orm.snooze_until,
        created_at=orm.created_at,
        updated_at=orm.updated_at,
    )


class NotificationRuleService:
    def __init__(self, session: AsyncSession, cel_evaluator: CELEvaluator) -> None:
        self._session = session
        self._cel = cel_evaluator

    async def _generate_unique_short_id(self) -> str:
        for _ in range(_MAX_COLLISION_RETRIES):
            short_id = generate_short_id("R")
            stmt = select(NotificationRuleORM).where(NotificationRuleORM.short_id == short_id)
            result = await self._session.execute(stmt)
            existing = result.scalar_one_or_none()
            if existing is None:
                return short_id
        raise ShortIdCollisionError("")

    def _validate_expression(self, expression: str) -> None:
        validation = self._cel.validate(expression)
        if not validation.valid:
            raise CELValidationError(expression, validation.error or "Invalid CEL expression")

    async def create_rule(self, user_id: int, data: RuleCreate) -> NotificationRule:
        self._validate_expression(data.expression)
        short_id = await self._generate_unique_short_id()
        now = datetime.now(UTC)
        orm = NotificationRuleORM(
            short_id=short_id,
            user_id=user_id,
            telegram_chat_id=data.telegram_chat_id,
            telegram_message_thread_id=data.telegram_message_thread_id,
            location_id=data.location_id,
            expression_language=data.expression_language,
            expression=data.expression,
            schedule=data.schedule,
            lead_time_minutes=data.lead_time_minutes,
            cooldown_minutes=data.cooldown_minutes,
            enabled=data.enabled,
            dry_run=data.dry_run,
            description=data.description,
            created_at=now,
            updated_at=now,
        )
        self._session.add(orm)
        await self._session.flush()
        await self._session.refresh(orm)
        return _orm_to_domain(orm)

    async def list_rules(
        self, user_id: int, include_disabled: bool = False
    ) -> list[NotificationRule]:
        stmt = select(NotificationRuleORM).where(NotificationRuleORM.user_id == user_id)
        if not include_disabled:
            stmt = stmt.where(NotificationRuleORM.enabled.is_(True))
        stmt = stmt.order_by(NotificationRuleORM.id)
        result = await self._session.execute(stmt)
        return [_orm_to_domain(r) for r in result.scalars().all()]

    async def list_all_enabled_rules(self) -> list[NotificationRule]:
        stmt = (
            select(NotificationRuleORM)
            .where(NotificationRuleORM.enabled.is_(True))
            .order_by(NotificationRuleORM.id)
        )
        result = await self._session.execute(stmt)
        return [_orm_to_domain(r) for r in result.scalars().all()]

    async def get_rule(
        self, rule_id: int | None = None, short_id: str | None = None
    ) -> NotificationRule | None:
        if rule_id is not None:
            orm = await self._session.get(NotificationRuleORM, rule_id)
            if orm is None:
                return None
            return _orm_to_domain(orm)
        if short_id is not None:
            clean_id = strip_hash_prefix(short_id)
            stmt = select(NotificationRuleORM).where(NotificationRuleORM.short_id == clean_id)
            result = await self._session.execute(stmt)
            orm = result.scalar_one_or_none()
            if orm is None:
                return None
            return _orm_to_domain(orm)
        return None

    async def update_rule(self, rule_id: int, data: RuleUpdate) -> NotificationRule:
        orm = await self._session.get(NotificationRuleORM, rule_id)
        if orm is None:
            raise RuleNotFoundError(rule_id=rule_id)

        if data.expression is not None:
            self._validate_expression(data.expression)

        if data.telegram_chat_id is not None:
            orm.telegram_chat_id = data.telegram_chat_id
        if data.telegram_message_thread_id is not None:
            orm.telegram_message_thread_id = data.telegram_message_thread_id
        if data.location_id is not None:
            orm.location_id = data.location_id
        if data.expression_language is not None:
            orm.expression_language = data.expression_language
        if data.expression is not None:
            orm.expression = data.expression
        if data.schedule is not None:
            orm.schedule = data.schedule
        if data.lead_time_minutes is not None:
            orm.lead_time_minutes = data.lead_time_minutes
        if data.cooldown_minutes is not None:
            orm.cooldown_minutes = data.cooldown_minutes
        if data.enabled is not None:
            orm.enabled = data.enabled
        if data.dry_run is not None:
            orm.dry_run = data.dry_run
        if data.description is not None:
            orm.description = data.description

        orm.updated_at = datetime.now(UTC)
        await self._session.flush()
        await self._session.refresh(orm)
        return _orm_to_domain(orm)

    async def enable_rule(self, rule_id: int) -> NotificationRule:
        orm = await self._session.get(NotificationRuleORM, rule_id)
        if orm is None:
            raise RuleNotFoundError(rule_id=rule_id)
        orm.enabled = True
        orm.updated_at = datetime.now(UTC)
        await self._session.flush()
        await self._session.refresh(orm)
        return _orm_to_domain(orm)

    async def disable_rule(self, rule_id: int) -> NotificationRule:
        orm = await self._session.get(NotificationRuleORM, rule_id)
        if orm is None:
            raise RuleNotFoundError(rule_id=rule_id)
        orm.enabled = False
        orm.updated_at = datetime.now(UTC)
        await self._session.flush()
        await self._session.refresh(orm)
        return _orm_to_domain(orm)

    async def delete_rule(self, rule_id: int) -> bool:
        orm = await self._session.get(NotificationRuleORM, rule_id)
        if orm is None:
            return False
        await self._session.execute(
            delete(NotificationRuleORM).where(NotificationRuleORM.id == rule_id)
        )
        await self._session.flush()
        return True

    async def snooze_rule(self, rule_id: int, until: datetime) -> NotificationRule:
        orm = await self._session.get(NotificationRuleORM, rule_id)
        if orm is None:
            raise RuleNotFoundError(rule_id=rule_id)
        orm.snooze_until = until
        orm.updated_at = datetime.now(UTC)
        await self._session.flush()
        await self._session.refresh(orm)
        return _orm_to_domain(orm)

    async def set_dry_run(self, rule_id: int, dry_run: bool) -> NotificationRule:
        orm = await self._session.get(NotificationRuleORM, rule_id)
        if orm is None:
            raise RuleNotFoundError(rule_id=rule_id)
        orm.dry_run = dry_run
        orm.updated_at = datetime.now(UTC)
        await self._session.flush()
        await self._session.refresh(orm)
        return _orm_to_domain(orm)
