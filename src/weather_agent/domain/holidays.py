"""Holiday provider abstraction and Nager.Date-backed cached implementation."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class Holiday(BaseModel):
    """A single public holiday."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    date: date
    name: str
    local_name: str


@runtime_checkable
class HolidayProvider(Protocol):
    """Async protocol for fetching public holidays for a country and year."""

    async def get_holidays(self, year: int, country_code: str) -> list[Holiday]: ...


class CachedHolidayProvider:
    """Wraps Nager.Date API with in-memory cache, falling back to cache on API failure."""

    def __init__(
        self,
        base_url: str = "https://date.nager.at/api/v3",
        timeout_seconds: int = 10,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._cache: dict[tuple[int, str], list[Holiday]] = {}
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def _fetch_from_api(self, year: int, country_code: str) -> list[Holiday]:
        client = self._get_client()
        url = f"{self._base_url}/PublicHolidays/{year}/{country_code}"
        response = await client.get(url)
        response.raise_for_status()
        data: list[dict[str, Any]] = response.json()
        holidays: list[Holiday] = []
        for entry in data:
            holidays.append(
                Holiday(
                    date=date.fromisoformat(entry["date"]),
                    name=entry.get("name", ""),
                    local_name=entry.get("localName", ""),
                )
            )
        return holidays

    async def get_holidays(self, year: int, country_code: str) -> list[Holiday]:
        key = (year, country_code)
        if key in self._cache:
            return self._cache[key]
        try:
            holidays = await self._fetch_from_api(year, country_code)
            self._cache[key] = holidays
            return holidays
        except Exception:
            logger.warning(
                "Nager.Date API unavailable for year=%s country=%s, returning cache if available",
                year,
                country_code,
            )
            return self._cache.get(key, [])

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None