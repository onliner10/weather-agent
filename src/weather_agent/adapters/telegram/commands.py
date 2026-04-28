from __future__ import annotations

from weather_agent.domain.auth import AuthorizationService
from weather_agent.domain.locations import LocationService
from weather_agent.domain.rules import NotificationRuleService, strip_hash_prefix


class CommandContext:
    def __init__(
        self,
        user_id: int,
        chat_id: int,
        message_thread_id: int | None,
        location_service: LocationService,
        rule_service: NotificationRuleService,
        auth_service: AuthorizationService,
    ) -> None:
        self.user_id = user_id
        self.chat_id = chat_id
        self.message_thread_id = message_thread_id
        self.location_service = location_service
        self.rule_service = rule_service
        self.auth_service = auth_service


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
    lines.append("  ✅ Bot działa poprawnie")
    return "\n".join(lines)