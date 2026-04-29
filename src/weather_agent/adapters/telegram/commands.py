"""Telegram command helpers surfaced by the bot runtime.

Only ``/status`` is currently wired in :class:`weather_agent.adapters.telegram.bot.TelegramBot`.
The broader command handlers that previously lived here have been removed because
they were not wired to Telegram commands and are not part of the supported MVP
runtime surface.
"""

from __future__ import annotations

from datetime import datetime

from weather_agent import __version__
from weather_agent.domain.auth import AuthorizationService
from weather_agent.domain.locations import LocationService
from weather_agent.domain.rules import NotificationRuleService
from weather_agent.observability.langsmith_tracing import LangSmithTracing


class SystemStatus:
    def __init__(
        self,
        db_connected: bool = False,
        scheduler_status: str = "stopped",
        last_forecast_fetch: datetime | None = None,
        last_rule_evaluation: datetime | None = None,
        provider_status: dict[str, str] | None = None,
    ) -> None:
        self.db_connected = db_connected
        self.scheduler_status = scheduler_status
        self.last_forecast_fetch = last_forecast_fetch
        self.last_rule_evaluation = last_rule_evaluation
        self.provider_status = provider_status or {}


class CommandContext:
    def __init__(
        self,
        user_id: int,
        chat_id: int,
        message_thread_id: int | None,
        location_service: LocationService,
        rule_service: NotificationRuleService,
        auth_service: AuthorizationService,
        system_status: SystemStatus | None = None,
    ) -> None:
        self.user_id = user_id
        self.chat_id = chat_id
        self.message_thread_id = message_thread_id
        self.location_service = location_service
        self.rule_service = rule_service
        self.auth_service = auth_service
        self.system_status = system_status


async def handle_status(ctx: CommandContext) -> str:
    locations = await ctx.location_service.list_locations(ctx.user_id)
    rules = await ctx.rule_service.list_rules(ctx.user_id, include_disabled=True)
    enabled_rules = [r for r in rules if r.enabled]
    dry_run_rules = [r for r in rules if r.dry_run]

    lines = [
        "📊 Status bota:",
        f"  Lokalizacje: {len(locations)}",
        f"  Reguły: {len(enabled_rules)}/{len(rules)} aktywnych",
    ]
    if dry_run_rules:
        lines.append(f"  Dry-run: {len(dry_run_rules)} reguł")

    if ctx.system_status is not None:
        db_icon = "✅" if ctx.system_status.db_connected else "❌"
        lines.append(f"  Baza danych: {db_icon}")
        lines.append(f"  Scheduler: {ctx.system_status.scheduler_status}")
        if ctx.system_status.last_forecast_fetch is not None:
            fmt = ctx.system_status.last_forecast_fetch
            lines.append(f"  Ostatni pobór prognozy: {fmt:%Y-%m-%d %H:%M UTC}")
        else:
            lines.append("  Ostatni pobór prognozy: brak")
        if ctx.system_status.last_rule_evaluation is not None:
            fmt = ctx.system_status.last_rule_evaluation
            lines.append(f"  Ostatnia ewaluacja reguł: {fmt:%Y-%m-%d %H:%M UTC}")
        else:
            lines.append("  Ostatnia ewaluacja reguł: brak")
        if ctx.system_status.provider_status:
            for provider, status in ctx.system_status.provider_status.items():
                icon = "✅" if status == "ok" else "❌"
                lines.append(f"  {provider}: {icon} {status}")
        langsmith = "włączony" if LangSmithTracing.is_enabled() else "wyłączony"
        lines.append(f"  LangSmith: {langsmith}")
        lines.append(f"  Wersja: {__version__}")
        if ctx.system_status.db_connected:
            lines.append("  ✅ Bot działa poprawnie")
        else:
            lines.append("  ⚠️ Problemy z bazą danych")
    else:
        lines.append("  ✅ Bot działa poprawnie")
    return "\n".join(lines)
