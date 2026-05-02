from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel


@dataclass(frozen=True)
class UserMessage:
    telegram_user_id: int
    chat_id: int
    message_thread_id: int | None
    text: str
    message_id: int
    reply_to_message_id: int | None

    @property
    def context_key(self) -> str:
        return (
            f"{self.chat_id}:{self.message_thread_id}"
            if self.message_thread_id is not None
            else str(self.chat_id)
        )


class PendingConfirmation(BaseModel):
    action: Literal["create_rule", "edit_rule", "schedule_notification"] = "create_rule"
    rule_expression: str = ""
    explanation: str = ""
    validated: bool = False
    location_id: int | None = None
    chat_id: int | None = None
    message_thread_id: int | None = None
    stored_at: str | None = None
    edit_short_id: str | None = None
    schedule: str | None = None
    lead_time_minutes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PendingConfirmation:
        return cls.model_validate(data)
