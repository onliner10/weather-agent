from __future__ import annotations

from pydantic import BaseModel, Field


class IntentExtraction(BaseModel):
    intent: str = Field(description="Zklasyfikowana intencja użytkownika: 'weather', 'rule', 'command', 'confirm_rule', 'cancel_rule'")
