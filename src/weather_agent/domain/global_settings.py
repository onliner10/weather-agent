from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from weather_agent.settings import AppSettings, GlobalUnitsSettings


class GlobalUnits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: str = "celsius"
    wind_speed: str = "ms"
    precipitation: str = "mm"
    pressure: str = "hpa"


class SettingsRepo(Protocol):
    async def save_units(self, units: GlobalUnits) -> None: ...
    async def load_units(self) -> GlobalUnits | None: ...


class GlobalSettingsService:
    def __init__(
        self,
        default_units: GlobalUnits | None = None,
        repo: SettingsRepo | None = None,
    ) -> None:
        self._units: GlobalUnits = default_units or GlobalUnits()
        self._repo = repo

    def get_units(self) -> GlobalUnits:
        return self._units.model_copy()

    async def update_units(self, units: GlobalUnits) -> None:
        self._units = units.model_copy()
        if self._repo is not None:
            await self._repo.save_units(units)


def default_units_from_settings(settings: AppSettings) -> GlobalUnits:
    u: GlobalUnitsSettings = settings.units
    return GlobalUnits(
        temperature=u.temperature,
        wind_speed=u.wind_speed,
        precipitation=u.precipitation,
        pressure=u.pressure,
    )