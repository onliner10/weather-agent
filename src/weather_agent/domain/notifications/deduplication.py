from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from weather_agent.domain.rules.models import NotificationRule
from weather_agent.infrastructure.db.base import NotificationEvent as NotificationEventORM
from weather_agent.observability.logging import get_logger

logger = get_logger(__name__)


class NotificationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: int
    location_id: int
    expression: str
    forecast_window_start: datetime | None = None
    forecast_window_end: datetime | None = None
    payload: dict[str, Any] = {}
    dry_run: bool = False


class DedupeKey(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: int
    location_id: int
    expression_hash: str
    forecast_window_start: datetime | None = None
    forecast_window_end: datetime | None = None
    dedupe_key: str = ""

    def model_post_init(self, __context: object) -> None:
        if not self.dedupe_key:
            self.dedupe_key = self._compute()

    def _compute(self) -> str:
        ws = self.forecast_window_start.isoformat() if self.forecast_window_start else "none"
        we = self.forecast_window_end.isoformat() if self.forecast_window_end else "none"
        return f"{self.rule_id}:{self.location_id}:{self.expression_hash}:{ws}:{we}"


def compute_dedupe_key(
    rule_id: int,
    location_id: int,
    expression: str,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> DedupeKey:
    expression_hash = hashlib.sha256(expression.encode()).hexdigest()
    return DedupeKey(
        rule_id=rule_id,
        location_id=location_id,
        expression_hash=expression_hash,
        forecast_window_start=window_start,
        forecast_window_end=window_end,
    )


def compute_payload_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def has_significant_change(
    old_hash: str,
    new_hash: str,
    threshold: float | None = None,
) -> bool:
    if threshold is not None:
        pass
    return old_hash != new_hash


class NotificationDeduplicator:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def should_suppress(
        self,
        rule: NotificationRule,
        event: NotificationCandidate,
    ) -> tuple[bool, str | None]:
        now = datetime.now(UTC)

        if rule.snooze_until is not None and rule.snooze_until > now:
            reason = f"snoozed until {rule.snooze_until.isoformat()}"
            logger.info("suppressing notification for rule %d: %s", rule.id, reason)
            if event.dry_run:
                logger.info("dry-run: would suppress for snooze: %s", reason)
            return True, reason

        cooldown_cutoff = now - timedelta(minutes=rule.cooldown_minutes)
        stmt = (
            select(NotificationEventORM)
            .where(
                NotificationEventORM.rule_id == rule.id,
                NotificationEventORM.suppressed.is_(False),
                NotificationEventORM.sent_at.isnot(None),
                NotificationEventORM.sent_at >= cooldown_cutoff,
            )
            .order_by(NotificationEventORM.sent_at.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        recent_event = result.scalar_one_or_none()
        if recent_event is not None:
            sent_at = recent_event.sent_at
            reason = (
                f"cooldown: last sent at {sent_at.isoformat() if sent_at is not None else 'N/A'}, "
                f"cooldown {rule.cooldown_minutes}min"
            )
            logger.info("suppressing notification for rule %d: %s", rule.id, reason)
            if event.dry_run:
                logger.info("dry-run: would suppress for cooldown: %s", reason)
            return True, reason

        if event.forecast_window_start is not None and event.forecast_window_end is not None:
            dedupe_key = compute_dedupe_key(
                rule_id=rule.id,
                location_id=rule.location_id,
                expression=rule.expression,
                window_start=event.forecast_window_start,
                window_end=event.forecast_window_end,
            )
            dedupe_hash = hashlib.sha256(dedupe_key.dedupe_key.encode()).hexdigest()

            stmt2 = (
                select(NotificationEventORM)
                .where(
                    NotificationEventORM.rule_id == rule.id,
                    NotificationEventORM.payload_hash == dedupe_hash,
                    NotificationEventORM.delivery_status == "sent",
                )
                .limit(1)
            )
            result2 = await self._session.execute(stmt2)
            existing = result2.scalar_one_or_none()
            if existing is not None:
                reason = f"duplicate: dedupe key {dedupe_key.dedupe_key}"
                logger.info("suppressing notification for rule %d: %s", rule.id, reason)
                if event.dry_run:
                    logger.info("dry-run: would suppress for duplicate: %s", reason)
                return True, reason

        if event.payload:
            payload_hash = compute_payload_hash(event.payload)

            stmt3 = (
                select(NotificationEventORM)
                .where(
                    NotificationEventORM.rule_id == rule.id,
                    NotificationEventORM.payload_hash.isnot(None),
                    NotificationEventORM.delivery_status == "sent",
                )
                .order_by(NotificationEventORM.created_at.desc())
                .limit(1)
            )
            result3 = await self._session.execute(stmt3)
            last_event = result3.scalar_one_or_none()
            if last_event is not None and last_event.payload_hash is not None:
                if not has_significant_change(last_event.payload_hash, payload_hash):
                    reason = "payload unchanged"
                    logger.info("suppressing notification for rule %d: %s", rule.id, reason)
                    if event.dry_run:
                        logger.info("dry-run: would suppress: %s", reason)
                    return True, reason

        if event.dry_run:
            logger.info("dry-run: notification for rule %d would NOT be suppressed", rule.id)

        return False, None
