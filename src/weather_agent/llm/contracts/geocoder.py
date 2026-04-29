from __future__ import annotations

from pydantic import BaseModel, Field


class LocationGuess(BaseModel):
    display_name: str = Field(description="Canonical Polish name of the place")
    search_query: str | None = Field(
        default=None,
        description="Search query for geocoding API (mianownik miasta or phrase)",
    )
