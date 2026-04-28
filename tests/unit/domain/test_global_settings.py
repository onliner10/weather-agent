from __future__ import annotations

import pytest
from pydantic import ValidationError

from weather_agent.domain.global_settings import GlobalSettingsService, GlobalUnits


class FakeRepo:
    def __init__(self) -> None:
        self.saved: list[GlobalUnits] = []

    async def save_units(self, units: GlobalUnits) -> None:
        self.saved.append(units.model_copy())

    async def load_units(self) -> GlobalUnits | None:
        return self.saved[-1] if self.saved else None


class TestGlobalUnitsModel:
    def test_defaults(self) -> None:
        u = GlobalUnits()
        assert u.temperature == "celsius"
        assert u.wind_speed == "ms"
        assert u.precipitation == "mm"
        assert u.pressure == "hpa"

    def test_custom_values(self) -> None:
        u = GlobalUnits(
            temperature="fahrenheit",
            wind_speed="kmh",
            precipitation="in",
            pressure="inhg",
        )
        assert u.temperature == "fahrenheit"
        assert u.wind_speed == "kmh"

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            GlobalUnits(temperature="celsius", unknown_field="x")


class TestGetUnits:
    def test_returns_default_units(self) -> None:
        svc = GlobalSettingsService()
        units = svc.get_units()
        assert units.temperature == "celsius"
        assert units.wind_speed == "ms"

    def test_returns_configured_units(self) -> None:
        initial = GlobalUnits(
            temperature="fahrenheit",
            wind_speed="kmh",
            precipitation="in",
            pressure="inhg",
        )
        svc = GlobalSettingsService(default_units=initial)
        units = svc.get_units()
        assert units.temperature == "fahrenheit"

    def test_returns_copy(self) -> None:
        svc = GlobalSettingsService()
        u1 = svc.get_units()
        u2 = svc.get_units()
        assert u1 == u2
        assert u1 is not u2


class TestUpdateUnits:
    @pytest.mark.asyncio
    async def test_update_changes_units(self) -> None:
        svc = GlobalSettingsService()
        new_units = GlobalUnits(
            temperature="fahrenheit",
            wind_speed="kmh",
            precipitation="in",
            pressure="inhg",
        )
        await svc.update_units(new_units)
        assert svc.get_units().temperature == "fahrenheit"

    @pytest.mark.asyncio
    async def test_update_persists_to_repo(self) -> None:
        repo = FakeRepo()
        svc = GlobalSettingsService(repo=repo)
        new_units = GlobalUnits(
            temperature="fahrenheit",
            wind_speed="kmh",
            precipitation="in",
            pressure="inhg",
        )
        await svc.update_units(new_units)
        assert len(repo.saved) == 1
        assert repo.saved[0].temperature == "fahrenheit"

    @pytest.mark.asyncio
    async def test_update_stores_copy_not_reference(self) -> None:
        repo = FakeRepo()
        svc = GlobalSettingsService(repo=repo)
        new_units = GlobalUnits()
        await svc.update_units(new_units)
        assert repo.saved[0] == new_units
        assert repo.saved[0] is not new_units


class TestNoTelegramDependency:
    def test_service_api_uses_domain_types(self) -> None:
        import inspect

        sig = inspect.signature(GlobalSettingsService.update_units)
        annotation = sig.parameters["units"].annotation
        assert "telegram" not in str(annotation).lower()

    def test_module_has_no_telegram_imports(self) -> None:
        import inspect

        import weather_agent.domain.global_settings as gs_mod

        source = inspect.getsource(gs_mod)
        assert "from telegram" not in source