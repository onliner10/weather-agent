from __future__ import annotations

from pydantic import BaseModel, Field


class RuleProposalExtraction(BaseModel):
    cel_expression: str | None = Field(None, description="Wyrażenie CEL dla reguły powiadomienia")
    explanation: str = Field(description="Czytelne wyjaśnienie reguły w języku polskim")
    short_id: str | None = Field(None, description="Identyfikator reguły do edycji/usunięcia")
