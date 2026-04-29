from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from weather_agent.domain.weather import LocationRef


@dataclass(frozen=True)
class TurnRequest:
    authorized_user_id: int | None = None
    chat_id: int = 0
    message_thread_id: int | None = None
    context_key: str = ""
    user_message: str = ""
    message_id: int | None = None
    reply_to_message_id: int | None = None


@dataclass(frozen=True)
class LoadedContext:
    pending_confirmation: dict[str, Any] | None = None
    resolved_location: LocationRef | None = None
    resolved_time_range: Any | None = None
    user_focus: str | None = None
    reply_context_turns: list[dict[str, Any]] | None = None


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


@dataclass
class TurnResult:
    answer: str | None = None
    resolved_intent: str | None = None
    resolved_location: LocationRef | None = None
    resolved_time_range: Any | None = None
    user_focus: str | None = None
    pending_confirmation: PendingConfirmation | None = None
    cel_expression: str | None = None
    error: str | None = None

    def bot_answer(self) -> str:
        if self.answer:
            return self.answer
        if self.error:
            return self.error
        return "Przepraszam, nie udało się przetworzyć zapytania."