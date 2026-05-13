from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal, cast

from croniter import croniter  # type: ignore[import-untyped]

from weather_agent.domain.time import WARSAW_TZ, ensure_aware, parse_datetime

_WARSAW = WARSAW_TZ

ScheduleKind = Literal["none", "once", "cron"]


class ScheduleParseResult:
    kind: ScheduleKind
    error: str | None = None
    raw: str = ""

    def __init__(self, kind: ScheduleKind, error: str | None = None, raw: str = "") -> None:
        self.kind = kind
        self.error = error
        self.raw = raw

    @property
    def valid(self) -> bool:
        return self.error is None


def parse_schedule(schedule: str | None) -> ScheduleParseResult:
    if not schedule:
        return ScheduleParseResult(kind="none")
    schedule = schedule.strip()
    if schedule.startswith("once:"):
        iso = schedule[5:]
        try:
            parse_datetime(iso, _WARSAW)
        except (ValueError, TypeError):
            err = f"Invalid once datetime: {iso}"
            return ScheduleParseResult(kind="once", error=err, raw=schedule)
        return ScheduleParseResult(kind="once", raw=schedule)
    if schedule.startswith("cron:"):
        expr = schedule[5:]
        if not croniter.is_valid(expr):
            err = f"Invalid cron expression: {expr}"
            return ScheduleParseResult(kind="cron", error=err, raw=schedule)
        return ScheduleParseResult(kind="cron", raw=schedule)
    err = f"Unknown schedule format: {schedule}"
    return ScheduleParseResult(kind="none", error=err, raw=schedule)


def is_rule_due(schedule: str | None, now: datetime | None = None) -> bool:
    parsed = parse_schedule(schedule)
    if not parsed.valid:
        return False
    if parsed.kind == "none":
        return True
    if parsed.kind == "once":
        if now is None:
            now = datetime.now(_WARSAW)
        else:
            now = ensure_aware(now, _WARSAW)
        dt = parse_datetime(parsed.raw[5:], _WARSAW)
        return now >= dt
    return True


def last_cron_slot(schedule: str, now: datetime | None = None) -> datetime | None:
    parsed = parse_schedule(schedule)
    if parsed.kind != "cron":
        return None
    expr = parsed.raw[5:]
    if now is None:
        now = datetime.now(_WARSAW)
    else:
        now = ensure_aware(now, _WARSAW)
    cron = croniter(expr, now + timedelta(microseconds=1))
    prev = cast("datetime", cron.get_prev(datetime))
    return ensure_aware(prev, _WARSAW)
