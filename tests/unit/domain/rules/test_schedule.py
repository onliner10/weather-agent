from __future__ import annotations

from datetime import UTC, datetime

from freezegun import freeze_time

from weather_agent.domain.rules.schedule import (
    is_rule_due,
    last_cron_slot,
    parse_schedule,
)


class TestParseSchedule:
    def test_none_returns_none_kind(self) -> None:
        result = parse_schedule(None)
        assert result.kind == "none"
        assert result.valid is True

    def test_empty_returns_none_kind(self) -> None:
        result = parse_schedule("")
        assert result.kind == "none"
        assert result.valid is True

    def test_once_valid_iso(self) -> None:
        result = parse_schedule("once:2026-05-01T08:00:00+02:00")
        assert result.kind == "once"
        assert result.valid is True

    def test_once_valid_iso_naive(self) -> None:
        result = parse_schedule("once:2026-05-01T08:00:00")
        assert result.kind == "once"
        assert result.valid is True

    def test_once_invalid_iso(self) -> None:
        result = parse_schedule("once:not-a-date")
        assert result.kind == "once"
        assert result.valid is False
        assert "Invalid once datetime" in (result.error or "")

    def test_cron_valid(self) -> None:
        result = parse_schedule("cron:0 8 * * *")
        assert result.kind == "cron"
        assert result.valid is True

    def test_cron_invalid(self) -> None:
        result = parse_schedule("cron:70 8 * * *")
        assert result.kind == "cron"
        assert result.valid is False
        assert "Invalid cron expression" in (result.error or "")

    def test_unknown_format(self) -> None:
        result = parse_schedule("weekly:friday")
        assert result.kind == "none"
        assert result.valid is False
        assert "Unknown schedule format" in (result.error or "")


class TestIsRuleDue:
    def test_none_is_always_due(self) -> None:
        assert is_rule_due(None) is True
        assert is_rule_due("") is True

    def test_once_before_time_is_not_due(self) -> None:
        now = datetime(2026, 4, 29, 10, 0, 0, tzinfo=UTC)
        assert is_rule_due("once:2026-05-01T08:00:00+02:00", now) is False

    def test_once_after_time_is_due(self) -> None:
        now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
        assert is_rule_due("once:2026-05-01T08:00:00+02:00", now) is True

    def test_once_at_exact_time_is_due(self) -> None:
        now = datetime(2026, 5, 1, 6, 0, 0, tzinfo=UTC)
        assert is_rule_due("once:2026-05-01T08:00:00+02:00", now) is True

    def test_cron_during_active_hour_is_due(self) -> None:
        with freeze_time("2026-04-29 08:30:00", tz_offset=2):
            assert is_rule_due("cron:30 8 * * *") is True

    def test_cron_is_always_due_for_dedup_by_worker(self) -> None:
        with freeze_time("2026-04-29 07:00:00", tz_offset=2):
            assert is_rule_due("cron:0 8 * * *") is True
        with freeze_time("2026-04-29 08:30:00", tz_offset=2):
            assert is_rule_due("cron:0 8 * * *") is True

    def test_invalid_schedule_is_not_due(self) -> None:
        assert is_rule_due("unknown:bad") is False

    def test_unknown_cron_is_not_due(self) -> None:
        assert is_rule_due("cron:99 99 * * *") is False


class TestLastCronSlot:
    def test_returns_prev_cron_slot(self) -> None:
        now = datetime(2026, 4, 29, 10, 0, 0, tzinfo=UTC)
        slot = last_cron_slot("cron:0 8 * * *", now)
        assert slot is not None
        assert slot.hour == 8
        assert slot.minute == 0

    def test_cron_slot_uses_warsaw_timezone_during_dst(self) -> None:
        now = datetime(2026, 5, 13, 10, 1, 0, tzinfo=UTC)
        slot = last_cron_slot("cron:0 12 * * 1-5", now)
        assert slot is not None
        assert slot.year == 2026
        assert slot.month == 5
        assert slot.day == 13
        assert slot.hour == 12
        assert slot.minute == 0

    def test_cron_slot_before_local_noon_uses_previous_weekday(self) -> None:
        now = datetime(2026, 5, 13, 9, 59, 0, tzinfo=UTC)
        slot = last_cron_slot("cron:0 12 * * 1-5", now)
        assert slot is not None
        assert slot.year == 2026
        assert slot.month == 5
        assert slot.day == 12
        assert slot.hour == 12
        assert slot.minute == 0

    def test_returns_none_for_once(self) -> None:
        assert last_cron_slot("once:2026-05-01T08:00:00+02:00") is None

    def test_returns_none_for_unknown(self) -> None:
        assert last_cron_slot("unknown:bad") is None

    def test_weekly_returns_prev_friday(self) -> None:
        # 2026-04-29 is Wednesday
        now = datetime(2026, 4, 29, 10, 0, 0, tzinfo=UTC)
        slot = last_cron_slot("cron:0 8 * * 5", now)
        assert slot is not None
        # Previous Friday was 2026-04-24
        assert slot.weekday() == 4  # Friday
        assert slot.day == 24
