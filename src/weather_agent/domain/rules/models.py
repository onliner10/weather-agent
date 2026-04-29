from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class NotificationRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    short_id: str
    user_id: int
    telegram_chat_id: int
    telegram_message_thread_id: int | None
    location_id: int
    expression_language: str = "cel"
    expression: str
    schedule: str | None = None
    lead_time_minutes: int | None = None
    cooldown_minutes: int = 60
    enabled: bool = True
    dry_run: bool = False
    description: str | None = None
    snooze_until: datetime | None = None
    created_at: datetime
    updated_at: datetime


class NotificationEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    short_id: str
    rule_id: int | None
    evaluation_run_id: int | None
    telegram_chat_id: int
    telegram_message_thread_id: int | None
    sent_at: datetime | None
    suppressed: bool = False
    suppress_reason: str | None = None
    payload_hash: str | None = None
    message_text: str | None = None
    created_at: datetime


class RuleCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    telegram_chat_id: int
    telegram_message_thread_id: int | None = None
    location_id: int
    expression_language: str = "cel"
    expression: str
    schedule: str | None = None
    lead_time_minutes: int | None = None
    cooldown_minutes: int = 60
    enabled: bool = True
    dry_run: bool = False
    description: str | None = None


class RuleUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    telegram_chat_id: int | None = None
    telegram_message_thread_id: int | None = None
    location_id: int | None = None
    expression_language: str | None = None
    expression: str | None = None
    schedule: str | None = None
    lead_time_minutes: int | None = None
    cooldown_minutes: int | None = None
    enabled: bool | None = None
    dry_run: bool | None = None
    description: str | None = None


class RuleNotFoundError(Exception):
    def __init__(self, rule_id: int | None = None, short_id: str | None = None) -> None:
        if rule_id is not None:
            msg = f"Rule {rule_id} not found"
        elif short_id is not None:
            msg = f"Rule with short_id '{short_id}' not found"
        else:
            msg = "Rule not found"
        self.rule_id = rule_id
        self.short_id = short_id
        super().__init__(msg)


class CELValidationError(Exception):
    def __init__(self, expression: str, error: str) -> None:
        self.expression = expression
        self.error = error
        super().__init__(f"CEL validation error: {error}")


class ShortIdCollisionError(Exception):
    def __init__(self, short_id: str) -> None:
        self.short_id = short_id
        super().__init__(f"Short ID collision: {short_id}")
