"""Notification rule model and CRUD with short IDs."""

from __future__ import annotations

from weather_agent.domain.rules.models import (
    NotificationEvent,
    NotificationRule,
    RuleCreate,
    RuleUpdate,
)
from weather_agent.domain.rules.service import NotificationRuleService
from weather_agent.domain.rules.short_id_generator import generate_short_id, strip_hash_prefix

__all__ = [
    "NotificationEvent",
    "NotificationRule",
    "NotificationRuleService",
    "RuleCreate",
    "RuleUpdate",
    "generate_short_id",
    "strip_hash_prefix",
]