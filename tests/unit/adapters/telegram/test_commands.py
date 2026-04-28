from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from weather_agent.adapters.telegram.commands import (
    CommandContext,
    handle_dodaj_lok,
    handle_drystart,
    handle_lokalizacje,
    handle_reguly,
    handle_status,
    handle_usun,
    handle_usun_lok,
    handle_wlacz,
    handle_wylacz,
)
from weather_agent.domain.auth import AuthorizationService
from weather_agent.domain.locations import Location, LocationService
from weather_agent.domain.rules import NotificationRuleService
from weather_agent.domain.rules.models import NotificationRule


def _make_location(
    id: int = 1,
    name: str = "Warszawa",
    latitude: float = 52.2297,
    longitude: float = 21.0122,
    enabled: bool = True,
) -> Location:
    return Location(
        id=id,
        name=name,
        aliases=[],
        latitude=latitude,
        longitude=longitude,
        description=None,
        enabled=enabled,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _make_rule(
    id: int = 1,
    short_id: str = "R7K2",
    user_id: int = 42,
    enabled: bool = True,
    dry_run: bool = False,
    expression: str = "temp > 30",
    description: str | None = None,
) -> NotificationRule:
    return NotificationRule(
        id=id,
        short_id=short_id,
        user_id=user_id,
        telegram_chat_id=100,
        telegram_message_thread_id=None,
        location_id=1,
        expression_language="cel",
        expression=expression,
        schedule=None,
        lead_time_minutes=None,
        cooldown_minutes=60,
        enabled=enabled,
        dry_run=dry_run,
        description=description,
        snooze_until=None,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _make_ctx(
    user_id: int = 42,
    location_service: LocationService | None = None,
    rule_service: NotificationRuleService | None = None,
) -> CommandContext:
    loc_svc = location_service or MagicMock(spec=LocationService)
    rule_svc = rule_service or MagicMock(spec=NotificationRuleService)
    auth_svc = MagicMock(spec=AuthorizationService)
    if isinstance(loc_svc, MagicMock):
        loc_svc.list_locations = AsyncMock()
        loc_svc.create_location = AsyncMock()
        loc_svc.get_location = AsyncMock()
        loc_svc.delete_location = AsyncMock()
    if isinstance(rule_svc, MagicMock):
        rule_svc.list_rules = AsyncMock()
        rule_svc.get_rule = AsyncMock()
        rule_svc.enable_rule = AsyncMock()
        rule_svc.disable_rule = AsyncMock()
        rule_svc.delete_rule = AsyncMock()
        rule_svc.set_dry_run = AsyncMock()
    return CommandContext(
        user_id=user_id,
        chat_id=100,
        message_thread_id=None,
        location_service=loc_svc,
        rule_service=rule_svc,
        auth_service=auth_svc,
    )


class TestHandleLokalizacje:
    @pytest.mark.asyncio()
    async def test_lists_locations_in_polish(self) -> None:
        ctx = _make_ctx()
        ctx.location_service.list_locations.return_value = [
            _make_location(id=1, name="Warszawa"),
            _make_location(id=2, name="Kraków", latitude=50.0647, longitude=19.9450),
        ]
        result = await handle_lokalizacje(ctx)
        assert "📍" in result
        assert "Warszawa" in result
        assert "Kraków" in result

    @pytest.mark.asyncio()
    async def test_no_locations(self) -> None:
        ctx = _make_ctx()
        ctx.location_service.list_locations.return_value = []
        result = await handle_lokalizacje(ctx)
        assert "Brak zapisanych lokalizacji" in result

    @pytest.mark.asyncio()
    async def test_disabled_location_shown_with_x(self) -> None:
        ctx = _make_ctx()
        ctx.location_service.list_locations.return_value = [
            _make_location(id=1, name="Warszawa", enabled=False),
        ]
        result = await handle_lokalizacje(ctx)
        assert "❌" in result


class TestHandleReguly:
    @pytest.mark.asyncio()
    async def test_lists_rules_in_polish(self) -> None:
        ctx = _make_ctx()
        ctx.rule_service.list_rules.return_value = [
            _make_rule(short_id="R7K2", expression="temp > 30"),
            _make_rule(id=2, short_id="R3M5", expression="wind > 20", enabled=False),
        ]
        result = await handle_reguly(ctx)
        assert "📋" in result
        assert "#R7K2" in result
        assert "#R3M5" in result
        assert "✅" in result
        assert "❌" in result

    @pytest.mark.asyncio()
    async def test_no_rules(self) -> None:
        ctx = _make_ctx()
        ctx.rule_service.list_rules.return_value = []
        result = await handle_reguly(ctx)
        assert "Brak reguł" in result

    @pytest.mark.asyncio()
    async def test_dry_run_flag_shown(self) -> None:
        ctx = _make_ctx()
        ctx.rule_service.list_rules.return_value = [
            _make_rule(short_id="R7K2", dry_run=True),
        ]
        result = await handle_reguly(ctx)
        assert "DRY-RUN" in result


class TestHandleDodajLok:
    @pytest.mark.asyncio()
    async def test_add_location_success(self) -> None:
        ctx = _make_ctx()
        created = _make_location(id=5, name="Gdańsk", latitude=54.3520, longitude=18.6466)
        ctx.location_service.create_location.return_value = created
        result = await handle_dodaj_lok(ctx, "Gdańsk 54.3520 18.6466")
        assert "✅" in result
        assert "Gdańsk" in result

    @pytest.mark.asyncio()
    async def test_add_location_missing_args(self) -> None:
        ctx = _make_ctx()
        result = await handle_dodaj_lok(ctx, "Warszawa")
        assert "Użycie" in result

    @pytest.mark.asyncio()
    async def test_add_location_invalid_coords(self) -> None:
        ctx = _make_ctx()
        result = await handle_dodaj_lok(ctx, "Miasto abc def")
        assert "Błędne współrzędne" in result


class TestHandleUsunLok:
    @pytest.mark.asyncio()
    async def test_delete_location_success(self) -> None:
        ctx = _make_ctx()
        ctx.location_service.get_location.return_value = _make_location(id=3, name="Poznań")
        ctx.location_service.delete_location.return_value = True
        result = await handle_usun_lok(ctx, "3")
        assert "✅" in result
        assert "Poznań" in result

    @pytest.mark.asyncio()
    async def test_delete_location_not_found(self) -> None:
        ctx = _make_ctx()
        ctx.location_service.get_location.return_value = None
        result = await handle_usun_lok(ctx, "99")
        assert "nie istnieje" in result

    @pytest.mark.asyncio()
    async def test_delete_location_invalid_id(self) -> None:
        ctx = _make_ctx()
        result = await handle_usun_lok(ctx, "abc")
        assert "numeryczne" in result

    @pytest.mark.asyncio()
    async def test_delete_location_empty_args(self) -> None:
        ctx = _make_ctx()
        result = await handle_usun_lok(ctx, "")
        assert "Użycie" in result


class TestHandleWlacz:
    @pytest.mark.asyncio()
    async def test_enable_rule_success(self) -> None:
        ctx = _make_ctx()
        rule = _make_rule(short_id="R7K2", user_id=42)
        ctx.rule_service.get_rule.return_value = rule
        ctx.rule_service.enable_rule.return_value = _make_rule(short_id="R7K2", enabled=True)
        result = await handle_wlacz(ctx, "R7K2")
        assert "✅" in result
        assert "R7K2" in result

    @pytest.mark.asyncio()
    async def test_enable_rule_not_found(self) -> None:
        ctx = _make_ctx()
        ctx.rule_service.get_rule.return_value = None
        result = await handle_wlacz(ctx, "R9999")
        assert "nie istnieje" in result

    @pytest.mark.asyncio()
    async def test_enable_rule_unauthorized(self) -> None:
        ctx = _make_ctx(user_id=42)
        rule = _make_rule(short_id="R7K2", user_id=99)
        ctx.rule_service.get_rule.return_value = rule
        result = await handle_wlacz(ctx, "R7K2")
        assert "Brak uprawnień" in result

    @pytest.mark.asyncio()
    async def test_enable_rule_with_hash(self) -> None:
        ctx = _make_ctx()
        rule = _make_rule(short_id="R7K2", user_id=42)
        ctx.rule_service.get_rule.return_value = rule
        ctx.rule_service.enable_rule.return_value = _make_rule(short_id="R7K2", enabled=True)
        result = await handle_wlacz(ctx, "#R7K2")
        assert "✅" in result


class TestHandleWylacz:
    @pytest.mark.asyncio()
    async def test_disable_rule_success(self) -> None:
        ctx = _make_ctx()
        rule = _make_rule(short_id="R7K2", user_id=42)
        ctx.rule_service.get_rule.return_value = rule
        ctx.rule_service.disable_rule.return_value = _make_rule(short_id="R7K2", enabled=False)
        result = await handle_wylacz(ctx, "R7K2")
        assert "❌" in result
        assert "Wyłączono" in result

    @pytest.mark.asyncio()
    async def test_disable_rule_not_found(self) -> None:
        ctx = _make_ctx()
        ctx.rule_service.get_rule.return_value = None
        result = await handle_wylacz(ctx, "R9999")
        assert "nie istnieje" in result


class TestHandleUsun:
    @pytest.mark.asyncio()
    async def test_delete_rule_success(self) -> None:
        ctx = _make_ctx()
        rule = _make_rule(short_id="R7K2", user_id=42)
        ctx.rule_service.get_rule.return_value = rule
        ctx.rule_service.delete_rule.return_value = True
        result = await handle_usun(ctx, "R7K2")
        assert "🗑️" in result
        assert "R7K2" in result

    @pytest.mark.asyncio()
    async def test_delete_rule_not_found(self) -> None:
        ctx = _make_ctx()
        ctx.rule_service.get_rule.return_value = None
        result = await handle_usun(ctx, "R9999")
        assert "nie istnieje" in result

    @pytest.mark.asyncio()
    async def test_delete_rule_unauthorized(self) -> None:
        ctx = _make_ctx(user_id=42)
        rule = _make_rule(short_id="R7K2", user_id=99)
        ctx.rule_service.get_rule.return_value = rule
        result = await handle_usun(ctx, "R7K2")
        assert "Brak uprawnień" in result


class TestHandleDrystart:
    @pytest.mark.asyncio()
    async def test_toggle_dry_run_on(self) -> None:
        ctx = _make_ctx()
        rule = _make_rule(short_id="R7K2", user_id=42, dry_run=False)
        ctx.rule_service.get_rule.return_value = rule
        ctx.rule_service.set_dry_run.return_value = _make_rule(short_id="R7K2", dry_run=True)
        result = await handle_drystart(ctx, "R7K2")
        assert "WŁĄCZONY" in result

    @pytest.mark.asyncio()
    async def test_toggle_dry_run_off(self) -> None:
        ctx = _make_ctx()
        rule = _make_rule(short_id="R7K2", user_id=42, dry_run=True)
        ctx.rule_service.get_rule.return_value = rule
        ctx.rule_service.set_dry_run.return_value = _make_rule(short_id="R7K2", dry_run=False)
        result = await handle_drystart(ctx, "R7K2")
        assert "WYŁĄCZONY" in result

    @pytest.mark.asyncio()
    async def test_dry_run_rule_not_found(self) -> None:
        ctx = _make_ctx()
        ctx.rule_service.get_rule.return_value = None
        result = await handle_drystart(ctx, "R9999")
        assert "nie istnieje" in result

    @pytest.mark.asyncio()
    async def test_dry_run_empty_args(self) -> None:
        ctx = _make_ctx()
        result = await handle_drystart(ctx, "")
        assert "Podaj" in result or "identyfikator" in result


class TestHandleStatus:
    @pytest.mark.asyncio()
    async def test_status_shows_summary(self) -> None:
        ctx = _make_ctx()
        ctx.location_service.list_locations.return_value = [
            _make_location(id=1, name="Warszawa"),
        ]
        ctx.rule_service.list_rules.return_value = [
            _make_rule(short_id="R7K2", enabled=True),
            _make_rule(id=2, short_id="R3M5", enabled=False),
        ]
        result = await handle_status(ctx)
        assert "📊" in result
        assert "Lokalizacje: 1" in result
        assert "Reguły: 1/2" in result

    @pytest.mark.asyncio()
    async def test_status_shows_dry_run_count(self) -> None:
        ctx = _make_ctx()
        ctx.location_service.list_locations.return_value = []
        ctx.rule_service.list_rules.return_value = [
            _make_rule(short_id="R7K2", dry_run=True),
        ]
        result = await handle_status(ctx)
        assert "Dry-run: 1" in result

    @pytest.mark.asyncio()
    async def test_status_no_dry_run_rules(self) -> None:
        ctx = _make_ctx()
        ctx.location_service.list_locations.return_value = []
        ctx.rule_service.list_rules.return_value = [
            _make_rule(short_id="R7K2", dry_run=False),
        ]
        result = await handle_status(ctx)
        assert "Dry-run" not in result