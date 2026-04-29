from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from weather_agent.adapters.telegram.commands import (
    CommandContext,
    handle_status,
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
    if isinstance(rule_svc, MagicMock):
        rule_svc.list_rules = AsyncMock()
    return CommandContext(
        user_id=user_id,
        chat_id=100,
        message_thread_id=None,
        location_service=loc_svc,
        rule_service=rule_svc,
        auth_service=auth_svc,
    )


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
