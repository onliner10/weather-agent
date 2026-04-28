from __future__ import annotations

from datetime import datetime

from weather_agent import __version__
from weather_agent.domain.auth import AuthorizationService
from weather_agent.domain.locations import LocationService
from weather_agent.domain.rules import NotificationRuleService, strip_hash_prefix
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


async def handle_lokalizacje(ctx: CommandContext) -> str:
    locations = await ctx.location_service.list_locations(ctx.user_id)
    if not locations:
        return "Brak zapisanych lokalizacji. Użyj /dodaj_lok aby dodać."
    lines = ["📍 Twoje lokalizacje:"]
    for loc in locations:
        status = "✅" if loc.enabled else "❌"
        lines.append(f"  {status} {loc.name} ({loc.latitude:.4f}, {loc.longitude:.4f})")
    return "\n".join(lines)


async def handle_reguly(ctx: CommandContext) -> str:
    rules = await ctx.rule_service.list_rules(ctx.user_id, include_disabled=True)
    if not rules:
        return "Brak reguł powiadomień. Poproś bota o utworzenie reguły."
    lines = ["📋 Twoje reguły:"]
    for rule in rules:
        status = "✅" if rule.enabled else "❌"
        dry = " [DRY-RUN]" if rule.dry_run else ""
        desc = rule.description or rule.expression
        lines.append(f"  {status} #{rule.short_id}{dry} — {desc}")
    return "\n".join(lines)


async def handle_dodaj_lok(ctx: CommandContext, args: str) -> str:
    parts = args.strip().split()
    if len(parts) < 3:
        return "Użycie: /dodaj_lok <nazwa> <lat> <lon>"
    name = parts[0]
    try:
        lat = float(parts[1])
        lon = float(parts[2])
    except ValueError:
        return "Błędne współrzędne. Podaj liczby dla lat i lon."
    from weather_agent.domain.locations import LocationCreate

    data = LocationCreate(name=name, aliases=[], latitude=lat, longitude=lon)
    try:
        loc = await ctx.location_service.create_location(ctx.user_id, data)
    except Exception as exc:
        return f"Nie udało się dodać lokalizacji: {exc}"
    return f"✅ Dodano lokalizację: {loc.name} ({loc.latitude:.4f}, {loc.longitude:.4f})"


async def handle_usun_lok(ctx: CommandContext, args: str) -> str:
    raw = args.strip()
    if not raw:
        return "Użycie: /usun_lok <id lokalizacji>"
    try:
        loc_id = int(raw)
    except ValueError:
        return "Podaj numeryczne ID lokalizacji."
    location = await ctx.location_service.get_location(loc_id)
    if location is None:
        return f"Lokalizacja {loc_id} nie istnieje."
    deleted = await ctx.location_service.delete_location(loc_id)
    if deleted:
        return f"✅ Usunięto lokalizację: {location.name}"
    return "Nie udało się usunąć lokalizacji."


async def _find_rule_by_short_id(
    ctx: CommandContext, raw_id: str
) -> tuple[str, int | None]:
    short_id = strip_hash_prefix(raw_id.strip())
    if not short_id:
        return ("Podaj identyfikator reguły (np. R7K2).", None)
    rule = await ctx.rule_service.get_rule(short_id=short_id)
    if rule is None:
        return (f"Reguła #{short_id} nie istnieje.", None)
    if rule.user_id != ctx.user_id:
        return (f"Brak uprawnień do reguły #{short_id}.", None)
    return ("", rule.id)


async def handle_wlacz(ctx: CommandContext, args: str) -> str:
    error_msg, rule_id = await _find_rule_by_short_id(ctx, args)
    if rule_id is None:
        return error_msg
    short_id = strip_hash_prefix(args.strip())
    try:
        await ctx.rule_service.enable_rule(rule_id)
        return f"✅ Włączono regułę #{short_id}"
    except Exception as exc:
        return f"Błąd: {exc}"


async def handle_wylacz(ctx: CommandContext, args: str) -> str:
    error_msg, rule_id = await _find_rule_by_short_id(ctx, args)
    if rule_id is None:
        return error_msg
    short_id = strip_hash_prefix(args.strip())
    try:
        await ctx.rule_service.disable_rule(rule_id)
        return f"❌ Wyłączono regułę #{short_id}"
    except Exception as exc:
        return f"Błąd: {exc}"


async def handle_usun(ctx: CommandContext, args: str) -> str:
    error_msg, rule_id = await _find_rule_by_short_id(ctx, args)
    if rule_id is None:
        return error_msg
    short_id = strip_hash_prefix(args.strip())
    deleted = await ctx.rule_service.delete_rule(rule_id)
    if deleted:
        return f"🗑️ Usunięto regułę #{short_id}"
    return "Nie udało się usunąć reguły."


async def handle_drystart(ctx: CommandContext, args: str) -> str:
    error_msg, rule_id = await _find_rule_by_short_id(ctx, args)
    if rule_id is None:
        return error_msg
    short_id = strip_hash_prefix(args.strip())
    rule = await ctx.rule_service.get_rule(rule_id=rule_id)
    if rule is None:
        return f"Reguła #{short_id} nie istnieje."
    new_dry_run = not rule.dry_run
    try:
        await ctx.rule_service.set_dry_run(rule_id, new_dry_run)
        status = "WŁĄCZONY" if new_dry_run else "WYŁĄCZONY"
        return f"🔧 Dry-run dla #{short_id}: {status}"
    except Exception as exc:
        return f"Błąd: {exc}"


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