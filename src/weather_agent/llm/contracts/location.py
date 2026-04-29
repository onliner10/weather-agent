from __future__ import annotations

from pydantic import BaseModel, Field


class LocationExtraction(BaseModel):
    location_name: str | None = Field(
        None, description="Nazwa miejscowości (np. Gdańsk, Chwarzno). Null jeśli nie podano."
    )
    focus: str | None = Field(
        None, description="Szczegółowy temat pytania (np. wiatr, opady). Null jeśli ogólne."
    )
