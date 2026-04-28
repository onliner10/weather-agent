"""Deterministic unit tests for DateResolver with frozen clock."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from freezegun import freeze_time
from pydantic import ValidationError

from weather_agent.domain.date_resolver import DateResolver, ResolvedTimeRange
from weather_agent.domain.holidays import Holiday

_WARSAW = ZoneInfo("Europe/Warsaw")


class FakeHolidayProvider:
    """In-memory holiday provider for deterministic tests."""

    def __init__(self, holidays: dict[int, list[Holiday]] | None = None) -> None:
        self._holidays = holidays or {}

    async def get_holidays(self, year: int, country_code: str) -> list[Holiday]:
        return self._holidays.get(year, [])


POLISH_HOLIDAYS_2025: list[Holiday] = [
    Holiday(date=date(2025, 5, 1), name="Labour Day", local_name="Święto Pracy"),
    Holiday(date=date(2025, 5, 3), name="Constitution Day", local_name="Święto Konstytucji 3 Maja"),
    Holiday(date=date(2025, 1, 1), name="New Year", local_name="Nowy Rok"),
]

POLISH_HOLIDAYS_2026: list[Holiday] = [
    Holiday(date=date(2026, 5, 1), name="Labour Day", local_name="Święto Pracy"),
    Holiday(date=date(2026, 5, 3), name="Constitution Day", local_name="Święto Konstytucji 3 Maja"),
]


def _dt(year: int, month: int, day: int, hour: int = 12, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=_WARSAW)


def _make_resolver(
    now: datetime,
    holidays: dict[int, list[Holiday]] | None = None,
) -> DateResolver:
    provider = FakeHolidayProvider(holidays)
    return DateResolver(holiday_provider=provider, now=now)


class TestDzis:
    @pytest.mark.asyncio
    async def test_dzis_basic(self) -> None:
        now = _dt(2025, 4, 28, 14, 30)
        resolver = _make_resolver(now)
        result = await resolver.resolve("dziś")
        assert result is not None
        assert result.start == now
        assert result.end == datetime(2025, 4, 28, 23, 59, tzinfo=_WARSAW)
        assert "Dziś" in result.explanation

    @pytest.mark.asyncio
    async def test_dzis_alternate_spelling(self) -> None:
        now = _dt(2025, 4, 28, 14, 30)
        resolver = _make_resolver(now)
        result = await resolver.resolve("dzisiaj")
        assert result is not None
        assert result.start == now

    @pytest.mark.asyncio
    async def test_dzis_wieczorem(self) -> None:
        now = _dt(2025, 4, 28, 14, 30)
        resolver = _make_resolver(now)
        result = await resolver.resolve("dziś wieczorem")
        assert result is not None
        assert result.start == datetime(2025, 4, 28, 18, 0, tzinfo=_WARSAW)
        assert result.end == datetime(2025, 4, 28, 23, 0, tzinfo=_WARSAW)


class TestJutro:
    @pytest.mark.asyncio
    async def test_jutro(self) -> None:
        now = _dt(2025, 4, 28, 14, 30)
        resolver = _make_resolver(now)
        result = await resolver.resolve("jutro")
        assert result is not None
        assert result.start == datetime(2025, 4, 29, 0, 0, tzinfo=_WARSAW)
        assert result.end == datetime(2025, 4, 29, 23, 59, tzinfo=_WARSAW)

    @pytest.mark.asyncio
    async def test_jutro_rano(self) -> None:
        now = _dt(2025, 4, 28, 14, 30)
        resolver = _make_resolver(now)
        result = await resolver.resolve("jutro rano")
        assert result is not None
        assert result.start == datetime(2025, 4, 29, 6, 0, tzinfo=_WARSAW)
        assert result.end == datetime(2025, 4, 29, 10, 0, tzinfo=_WARSAW)

    @pytest.mark.asyncio
    async def test_jutro_po_poludniu(self) -> None:
        now = _dt(2025, 4, 28, 14, 30)
        resolver = _make_resolver(now)
        result = await resolver.resolve("jutro po południu")
        assert result is not None
        assert result.start == datetime(2025, 4, 29, 12, 0, tzinfo=_WARSAW)
        assert result.end == datetime(2025, 4, 29, 18, 0, tzinfo=_WARSAW)

    @pytest.mark.asyncio
    async def test_jutro_wieczorem(self) -> None:
        now = _dt(2025, 4, 28, 14, 30)
        resolver = _make_resolver(now)
        result = await resolver.resolve("jutro wieczorem")
        assert result is not None
        assert result.start == datetime(2025, 4, 29, 18, 0, tzinfo=_WARSAW)
        assert result.end == datetime(2025, 4, 29, 23, 0, tzinfo=_WARSAW)


class TestNastepneDni:
    @pytest.mark.asyncio
    async def test_nastepne_3_dni(self) -> None:
        now = _dt(2025, 4, 28, 10, 0)
        resolver = _make_resolver(now)
        result = await resolver.resolve("następne 3 dni")
        assert result is not None
        assert result.start == now
        assert result.end == now + timedelta(hours=72)
        assert "3" in result.explanation

    @pytest.mark.asyncio
    async def test_nastepne_7_dni(self) -> None:
        now = _dt(2025, 4, 28, 10, 0)
        resolver = _make_resolver(now)
        result = await resolver.resolve("następne 7 dni")
        assert result is not None
        assert result.start == now
        assert result.end == now + timedelta(hours=168)

    @pytest.mark.asyncio
    async def test_ascii_fallback(self) -> None:
        now = _dt(2025, 4, 28, 10, 0)
        resolver = _make_resolver(now)
        result = await resolver.resolve("nastepne 3 dni")
        assert result is not None
        assert result.end == now + timedelta(hours=72)


class TestWeekend:
    @pytest.mark.asyncio
    async def test_weekend_from_monday(self) -> None:
        now = _dt(2025, 4, 28, 10, 0)
        resolver = _make_resolver(now)
        result = await resolver.resolve("weekend")
        assert result is not None
        assert result.start == datetime(2025, 5, 3, 0, 0, tzinfo=_WARSAW)
        assert result.end == datetime(2025, 5, 4, 23, 59, tzinfo=_WARSAW)

    @pytest.mark.asyncio
    async def test_ten_weekend_synonym(self) -> None:
        now = _dt(2025, 4, 28, 10, 0)
        resolver = _make_resolver(now)
        result = await resolver.resolve("ten weekend")
        assert result is not None
        assert result.start == datetime(2025, 5, 3, 0, 0, tzinfo=_WARSAW)

    @pytest.mark.asyncio
    async def test_weekend_on_friday(self) -> None:
        now = _dt(2025, 5, 2, 10, 0)
        resolver = _make_resolver(now)
        result = await resolver.resolve("weekend")
        assert result is not None
        assert result.start == datetime(2025, 5, 3, 0, 0, tzinfo=_WARSAW)
        assert result.end == datetime(2025, 5, 4, 23, 59, tzinfo=_WARSAW)

    @pytest.mark.asyncio
    async def test_weekend_on_saturday(self) -> None:
        now = _dt(2025, 5, 3, 10, 0)
        resolver = _make_resolver(now)
        result = await resolver.resolve("weekend")
        assert result is not None
        assert result.start == datetime(2025, 5, 3, 0, 0, tzinfo=_WARSAW)
        assert result.end == datetime(2025, 5, 4, 23, 59, tzinfo=_WARSAW)

    @pytest.mark.asyncio
    async def test_weekend_on_sunday(self) -> None:
        now = _dt(2025, 5, 4, 10, 0)
        resolver = _make_resolver(now)
        result = await resolver.resolve("weekend")
        assert result is not None
        assert result.start == datetime(2025, 5, 3, 0, 0, tzinfo=_WARSAW)
        assert result.end == datetime(2025, 5, 4, 23, 59, tzinfo=_WARSAW)


class TestNastepnyWeekend:
    @pytest.mark.asyncio
    async def test_next_weekend_from_monday(self) -> None:
        now = _dt(2025, 4, 28, 10, 0)
        resolver = _make_resolver(now)
        result = await resolver.resolve("następny weekend")
        assert result is not None
        saturday = result.start.date()
        assert saturday.weekday() == 5
        assert saturday > now.date()

    @pytest.mark.asyncio
    async def test_next_weekend_7_days_out(self) -> None:
        now = _dt(2025, 5, 3, 10, 0)
        resolver = _make_resolver(now)
        result = await resolver.resolve("następny weekend")
        assert result is not None
        assert result.start.date() == date(2025, 5, 10)
        assert result.end.date() == date(2025, 5, 11)

    @pytest.mark.asyncio
    async def test_next_weekend_on_sunday(self) -> None:
        now = _dt(2025, 5, 4, 10, 0)
        resolver = _make_resolver(now)
        result = await resolver.resolve("następny weekend")
        assert result is not None
        assert result.start.date() == date(2025, 5, 10)
        assert result.end.date() == date(2025, 5, 11)


class TestMajowka:
    @pytest.mark.asyncio
    async def test_majowka_2025(self) -> None:
        now = _dt(2025, 4, 15, 10, 0)
        holidays = {2025: POLISH_HOLIDAYS_2025}
        resolver = _make_resolver(now, holidays)
        result = await resolver.resolve("majówka")
        assert result is not None
        assert result.start.date() <= date(2025, 5, 1)
        assert result.end.date() >= date(2025, 5, 3)
        assert "Majówka" in result.explanation

    @pytest.mark.asyncio
    async def test_majowka_2026(self) -> None:
        now = _dt(2026, 3, 15, 10, 0)
        holidays = {2026: POLISH_HOLIDAYS_2026}
        resolver = _make_resolver(now, holidays)
        result = await resolver.resolve("majówka")
        assert result is not None
        assert result.start.month == 5
        assert result.end.month == 5

    @pytest.mark.asyncio
    async def test_majowka_2025_covers_3_days(self) -> None:
        now = _dt(2025, 4, 15, 10, 0)
        holidays = {2025: POLISH_HOLIDAYS_2025}
        resolver = _make_resolver(now, holidays)
        result = await resolver.resolve("majówka")
        assert result is not None
        coverage_days = (result.end.date() - result.start.date()).days + 1
        assert coverage_days >= 3

    @pytest.mark.asyncio
    async def test_majowka_after_may_wraps_to_next_year(self) -> None:
        now = _dt(2025, 6, 15, 10, 0)
        holidays = {2025: POLISH_HOLIDAYS_2025, 2026: POLISH_HOLIDAYS_2026}
        resolver = _make_resolver(now, holidays)
        result = await resolver.resolve("majówka")
        assert result is not None
        assert result.start.year == 2026
        assert result.start.month == 5

    @pytest.mark.asyncio
    async def test_majowka_ascii(self) -> None:
        now = _dt(2025, 4, 15, 10, 0)
        resolver = _make_resolver(now)
        result = await resolver.resolve("majowka")
        assert result is not None
        assert "Majówka" in result.explanation

    @pytest.mark.asyncio
    async def test_majowka_2026_saturday_bridge(self) -> None:
        now = _dt(2026, 4, 15, 10, 0)
        holidays = {2026: POLISH_HOLIDAYS_2026}
        resolver = _make_resolver(now, holidays)
        result = await resolver.resolve("majówka")
        assert result is not None
        assert result.start.year == 2026
        assert result.start.month == 5
        assert result.end.month == 5


class TestExplicitDateRange:
    @pytest.mark.asyncio
    async def test_single_iso_date(self) -> None:
        now = _dt(2025, 4, 28, 10, 0)
        resolver = _make_resolver(now)
        result = await resolver.resolve("2025-05-01")
        assert result is not None
        assert result.start == datetime(2025, 5, 1, 0, 0, tzinfo=_WARSAW)
        assert result.end == datetime(2025, 5, 1, 23, 59, tzinfo=_WARSAW)

    @pytest.mark.asyncio
    async def test_date_range_with_dash(self) -> None:
        now = _dt(2025, 4, 28, 10, 0)
        resolver = _make_resolver(now)
        result = await resolver.resolve("2025-05-01 - 2025-05-03")
        assert result is not None
        assert result.start == datetime(2025, 5, 1, 0, 0, tzinfo=_WARSAW)
        assert result.end == datetime(2025, 5, 3, 23, 59, tzinfo=_WARSAW)

    @pytest.mark.asyncio
    async def test_date_range_with_em_dash(self) -> None:
        now = _dt(2025, 4, 28, 10, 0)
        resolver = _make_resolver(now)
        result = await resolver.resolve("2025-05-01—2025-05-03")
        assert result is not None
        assert result.start == datetime(2025, 5, 1, 0, 0, tzinfo=_WARSAW)
        assert result.end == datetime(2025, 5, 3, 23, 59, tzinfo=_WARSAW)


class TestFallback:
    @pytest.mark.asyncio
    async def test_unknown_expression_returns_none(self) -> None:
        now = _dt(2025, 4, 28, 10, 0)
        resolver = _make_resolver(now)
        result = await resolver.resolve("next week")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_string(self) -> None:
        now = _dt(2025, 4, 28, 10, 0)
        resolver = _make_resolver(now)
        result = await resolver.resolve("")
        assert result is None


class TestDeterminism:
    @pytest.mark.asyncio
    async def test_same_input_same_output(self) -> None:
        now = _dt(2025, 4, 28, 14, 30)
        resolver = _make_resolver(now)
        r1 = await resolver.resolve("jutro")
        r2 = await resolver.resolve("jutro")
        assert r1 == r2

    @freeze_time("2025-04-28 14:30:00", tz_offset=2)
    @pytest.mark.asyncio
    async def test_with_freezegun(self) -> None:
        resolver = DateResolver()
        result = await resolver.resolve("dziś")
        assert result is not None
        assert result.start.date() == date(2025, 4, 28)


class TestWeekendEdgeCases:
    @pytest.mark.asyncio
    async def test_weekend_on_monday(self) -> None:
        now = _dt(2025, 4, 28, 10, 0)
        resolver = _make_resolver(now)
        result = await resolver.resolve("weekend")
        assert result is not None
        assert result.start.date() == date(2025, 5, 3)
        assert result.end.date() == date(2025, 5, 4)

    @pytest.mark.asyncio
    async def test_weekend_on_wednesday(self) -> None:
        now = _dt(2025, 4, 30, 10, 0)
        resolver = _make_resolver(now)
        result = await resolver.resolve("weekend")
        assert result is not None
        assert result.start.date() == date(2025, 5, 3)
        assert result.end.date() == date(2025, 5, 4)

    @pytest.mark.asyncio
    async def test_weekend_on_thursday(self) -> None:
        now = _dt(2025, 5, 1, 10, 0)
        resolver = _make_resolver(now)
        result = await resolver.resolve("weekend")
        assert result is not None
        assert result.start.date() == date(2025, 5, 3)
        assert result.end.date() == date(2025, 5, 4)

    @pytest.mark.asyncio
    async def test_next_weekend_from_tuesday(self) -> None:
        now = _dt(2025, 4, 29, 10, 0)
        resolver = _make_resolver(now)
        result = await resolver.resolve("następny weekend")
        assert result is not None
        assert result.start.date() == date(2025, 5, 10)
        assert result.end.date() == date(2025, 5, 11)

    @pytest.mark.asyncio
    async def test_weekend_from_monday_march(self) -> None:
        now = _dt(2025, 3, 3, 10, 0)
        resolver = _make_resolver(now)
        result = await resolver.resolve("weekend")
        assert result is not None
        assert result.start.date() == date(2025, 3, 8)
        assert result.end.date() == date(2025, 3, 9)


class TestResolvedTimeRangeModel:
    def test_frozen_model_rejects_mutation(self) -> None:
        r = ResolvedTimeRange(
            start=datetime(2025, 5, 1, 0, 0, tzinfo=_WARSAW),
            end=datetime(2025, 5, 3, 23, 59, tzinfo=_WARSAW),
            explanation="Majówka",
        )
        with pytest.raises(ValidationError):
            r.start = datetime(2025, 1, 1, tzinfo=_WARSAW)

    def test_timezone_default(self) -> None:
        r = ResolvedTimeRange(
            start=datetime(2025, 5, 1, tzinfo=_WARSAW),
            end=datetime(2025, 5, 3, 23, 59, tzinfo=_WARSAW),
            explanation="Test",
        )
        assert r.timezone == "Europe/Warsaw"