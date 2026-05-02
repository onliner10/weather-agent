from __future__ import annotations

from datetime import UTC, datetime

from weather_agent.domain.time import WARSAW_TZ, ensure_aware, ensure_utc, parse_datetime


def test_parse_naive_datetime_as_warsaw_local_time() -> None:
    parsed = parse_datetime("2026-07-01T12:00:00")

    assert parsed.tzinfo == WARSAW_TZ
    assert parsed.utcoffset() is not None


def test_aware_datetime_is_converted_to_warsaw() -> None:
    parsed = parse_datetime("2026-07-01T10:00:00+00:00")

    assert parsed.tzinfo == WARSAW_TZ
    assert parsed.hour == 12


def test_ensure_utc_converts_aware_instant() -> None:
    warsaw_time = datetime(2026, 7, 1, 12, 0, tzinfo=WARSAW_TZ)

    assert ensure_utc(warsaw_time) == datetime(2026, 7, 1, 10, 0, tzinfo=UTC)


def test_ensure_aware_attaches_default_only_for_naive_values() -> None:
    naive = datetime(2026, 1, 15, 8, 0)
    aware = ensure_aware(naive, WARSAW_TZ)

    assert aware.tzinfo == WARSAW_TZ
    assert aware.hour == 8
