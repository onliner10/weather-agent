"""Notification rule model and CRUD with short IDs."""

from __future__ import annotations

from weather_agent.domain.rules.models import (
    CELValidationError,
    NotificationEvent,
    NotificationRule,
    RuleCreate,
    RuleNotFoundError,
    RuleUpdate,
    ShortIdCollisionError,
)
from weather_agent.domain.rules.schedule import (
    ScheduleParseResult,
    is_rule_due,
    last_cron_slot,
    parse_schedule,
)
from weather_agent.domain.rules.service import NotificationRuleService
from weather_agent.domain.rules.short_id_generator import generate_short_id, strip_hash_prefix

__all__ = [
    "CELValidationError",
    "NotificationEvent",
    "NotificationRule",
    "NotificationRuleService",
    "RuleCreate",
    "RuleNotFoundError",
    "RuleUpdate",
    "ShortIdCollisionError",
    "ScheduleParseResult",
    "generate_short_id",
    "is_rule_due",
    "last_cron_slot",
    "parse_schedule",
    "strip_hash_prefix",
]
