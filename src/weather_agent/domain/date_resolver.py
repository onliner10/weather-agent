"""Deterministic resolver for Polish natural time range expressions."""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from weather_agent.domain.holidays import HolidayProvider

_WARSAW = ZoneInfo("Europe/Warsaw")


class ResolvedTimeRange(BaseModel):
    """Resolved time range with explicit boundaries and human-readable explanation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: datetime
    end: datetime
    timezone: str = "Europe/Warsaw"
    explanation: str


def _warsaw_now(now: datetime | None = None) -> datetime:
    if now is None:
        now = datetime.now(tz=_WARSAW)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_WARSAW)
    return now.astimezone(_WARSAW)


def _start_of_day(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=_WARSAW)


def _end_of_day(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, 23, 59, tzinfo=_WARSAW)


def _nearest_upcoming_weekend(now: datetime, next_weekend: bool = False) -> tuple[date, date]:
    today = now.date()
    weekday = today.weekday()

    if next_weekend:
        if weekday == 5:
            saturday = today + timedelta(days=7)
        elif weekday == 6:
            saturday = today + timedelta(days=6)
        else:
            days_to_sat = 5 - weekday
            if days_to_sat <= 0:
                days_to_sat += 7
            saturday = today + timedelta(days=days_to_sat) + timedelta(days=7)
    else:
        if weekday == 6:
            saturday = today - timedelta(days=1)
        elif weekday == 5:
            saturday = today
        else:
            saturday = today + timedelta(days=5 - weekday)

    sunday = saturday + timedelta(days=1)
    return saturday, sunday


class DateResolver:
    """Resolves Polish natural-language time range expressions into concrete datetimes."""

    def __init__(
        self,
        holiday_provider: HolidayProvider | None = None,
        *,
        now: datetime | None = None,
    ) -> None:
        self._holiday_provider = holiday_provider
        self._now_override = now

    def _now(self) -> datetime:
        return _warsaw_now(self._now_override)

    async def resolve(self, text: str) -> ResolvedTimeRange | None:
        t = text.strip().lower()
        now = self._now()
        today = now.date()

        if t in ("dziś", "dzisiaj", "dzis"):
            return ResolvedTimeRange(
                start=now,
                end=_end_of_day(today),
                explanation=f"Dziś ({today.isoformat()})",
            )

        if t in ("dziś wieczorem", "dzis wieczorem", "dzisiaj wieczorem"):
            start = datetime.combine(today, time(18, 0), tzinfo=_WARSAW)
            end = datetime.combine(today, time(23, 0), tzinfo=_WARSAW)
            return ResolvedTimeRange(
                start=start,
                end=end,
                explanation=f"Dziś wieczorem ({today.isoformat()}, 18:00–23:00)",
            )

        if t == "jutro":
            tomorrow = today + timedelta(days=1)
            return ResolvedTimeRange(
                start=_start_of_day(tomorrow),
                end=_end_of_day(tomorrow),
                explanation=f"Jutro ({tomorrow.isoformat()})",
            )

        if t == "jutro rano":
            tomorrow = today + timedelta(days=1)
            start = datetime.combine(tomorrow, time(6, 0), tzinfo=_WARSAW)
            end = datetime.combine(tomorrow, time(10, 0), tzinfo=_WARSAW)
            return ResolvedTimeRange(
                start=start,
                end=end,
                explanation=f"Jutro rano ({tomorrow.isoformat()}, 06:00–10:00)",
            )

        if t in ("jutro po południu", "jutro po poludniu"):
            tomorrow = today + timedelta(days=1)
            start = datetime.combine(tomorrow, time(12, 0), tzinfo=_WARSAW)
            end = datetime.combine(tomorrow, time(18, 0), tzinfo=_WARSAW)
            return ResolvedTimeRange(
                start=start,
                end=end,
                explanation=f"Jutro po południu ({tomorrow.isoformat()}, 12:00–18:00)",
            )

        if t == "jutro wieczorem":
            tomorrow = today + timedelta(days=1)
            start = datetime.combine(tomorrow, time(18, 0), tzinfo=_WARSAW)
            end = datetime.combine(tomorrow, time(23, 0), tzinfo=_WARSAW)
            return ResolvedTimeRange(
                start=start,
                end=end,
                explanation=f"Jutro wieczorem ({tomorrow.isoformat()}, 18:00–23:00)",
            )

        match_n_dni = re.match(r"^następne\s+(\d+)\s+dni$", t) or re.match(
            r"^nastepne\s+(\d+)\s+dni$", t
        )
        if match_n_dni:
            n = int(match_n_dni.group(1))
            end = now + timedelta(hours=n * 24)
            return ResolvedTimeRange(
                start=now,
                end=end,
                explanation=f"Następne {n} dni (od {now.isoformat()} do {end.isoformat()})",
            )

        if t in ("weekend", "ten weekend"):
            sat, sun = _nearest_upcoming_weekend(now, next_weekend=False)
            start = _start_of_day(sat)
            end = _end_of_day(sun)
            return ResolvedTimeRange(
                start=start,
                end=end,
                explanation=f"Ten weekend ({sat.isoformat()} – {sun.isoformat()})",
            )

        if t in ("następny weekend", "nastepny weekend"):
            sat, sun = _nearest_upcoming_weekend(now, next_weekend=True)
            start = _start_of_day(sat)
            end = _end_of_day(sun)
            return ResolvedTimeRange(
                start=start,
                end=end,
                explanation=f"Następny weekend ({sat.isoformat()} – {sun.isoformat()})",
            )

        if t in ("majówka", "majowka"):
            return await self._resolve_majowka(now, today)

        date_range = await self._try_explicit_date_range(t)
        if date_range is not None:
            return date_range

        return None

    async def _resolve_majowka(self, now: datetime, today: date) -> ResolvedTimeRange:
        year = now.year
        if today.month > 5:
            year += 1

        may1 = date(year, 5, 1)
        may3 = date(year, 5, 3)

        holiday_dates: set[date] = {may1, may3}

        if self._holiday_provider is not None:
            try:
                holidays = await self._holiday_provider.get_holidays(year, "PL")
                for h in holidays:
                    if h.date.month == 5:
                        holiday_dates.add(h.date)
            except Exception:
                pass

        all_dates = set(holiday_dates)
        for hd in list(holiday_dates):
            dow = hd.weekday()
            if dow == 5:
                all_dates.add(hd + timedelta(days=1))
            elif dow == 6:
                all_dates.add(hd - timedelta(days=1))

        may_range = sorted(d for d in all_dates if d.month == 5 and d.year == year)
        if not may_range:
            may_range = [may1, may3]

        start_date = may_range[0]
        end_date = may_range[-1]

        start = _start_of_day(start_date)
        end = _end_of_day(end_date)

        return ResolvedTimeRange(
            start=start,
            end=end,
            explanation=f"Majówka ({start_date.isoformat()} – {end_date.isoformat()})",
        )

    async def _try_explicit_date_range(self, t: str) -> ResolvedTimeRange | None:
        m = re.match(r"^(\d{4}-\d{2}-\d{2})$", t)
        if m:
            try:
                d = date.fromisoformat(m.group(1))
                return ResolvedTimeRange(
                    start=_start_of_day(d),
                    end=_end_of_day(d),
                    explanation=f"Data ({d.isoformat()})",
                )
            except ValueError:
                return None

        m = re.match(r"^(\d{4}-\d{2}-\d{2})\s*[-–—]\s*(\d{4}-\d{2}-\d{2})$", t)
        if m:
            try:
                start_d = date.fromisoformat(m.group(1))
                end_d = date.fromisoformat(m.group(2))
                return ResolvedTimeRange(
                    start=_start_of_day(start_d),
                    end=_end_of_day(end_d),
                    explanation=f"Zakres dat ({start_d.isoformat()} – {end_d.isoformat()})",
                )
            except ValueError:
                return None

        return None
