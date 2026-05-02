from __future__ import annotations

from datetime import UTC, datetime, tzinfo
from zoneinfo import ZoneInfo

WARSAW_TZ = ZoneInfo("Europe/Warsaw")


def ensure_aware(value: datetime, default_tz: ZoneInfo = WARSAW_TZ) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=default_tz)
    return value.astimezone(default_tz)


def ensure_utc(value: datetime, naive_tz: tzinfo = UTC) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=naive_tz).astimezone(UTC)
    return value.astimezone(UTC)


def parse_datetime(value: str, default_tz: ZoneInfo = WARSAW_TZ) -> datetime:
    parsed = datetime.fromisoformat(value)
    return ensure_aware(parsed, default_tz)


def parse_datetime_as_utc(value: str, naive_tz: tzinfo = UTC) -> datetime:
    parsed = datetime.fromisoformat(value)
    return ensure_utc(parsed, naive_tz=naive_tz)
