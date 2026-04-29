from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PendingConfirmation:
    action: str = "create_rule"
    cel_expression: str = ""
    explanation: str = ""
    validated: bool = False
    location_id: int | None = None
    chat_id: int | None = None
    message_thread_id: int | None = None
    stored_at: str | None = None
    edit_short_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "action": self.action,
            "cel_expression": self.cel_expression,
            "explanation": self.explanation,
            "validated": self.validated,
            "location_id": self.location_id,
            "chat_id": self.chat_id,
            "message_thread_id": self.message_thread_id,
            "stored_at": self.stored_at,
        }
        if self.edit_short_id is not None:
            d["edit_short_id"] = self.edit_short_id
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PendingConfirmation:
        return cls(
            action=data.get("action", "create_rule"),
            cel_expression=data.get("cel_expression", ""),
            explanation=data.get("explanation", ""),
            validated=data.get("validated", False),
            location_id=data.get("location_id"),
            chat_id=data.get("chat_id"),
            message_thread_id=data.get("message_thread_id"),
            stored_at=data.get("stored_at"),
            edit_short_id=data.get("edit_short_id"),
        )
