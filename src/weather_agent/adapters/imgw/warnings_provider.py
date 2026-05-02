from __future__ import annotations

import asyncio
from datetime import datetime

import httpx

from weather_agent.domain.errors import (
    WeatherProviderResponseError,
    WeatherProviderTimeoutError,
    WeatherProviderUnavailableError,
)
from weather_agent.domain.providers import WarningProvider
from weather_agent.domain.time import WARSAW_TZ, ensure_utc
from weather_agent.domain.weather import (
    LocationRef,
    TimeRange,
    WarningCategory,
    WarningSeverity,
    WeatherWarning,
)
from weather_agent.settings import ImgwSettings

_METEO_PROVIDER_NAME = "imgw-meteo"
_HYDRO_PROVIDER_NAME = "imgw-hydro"
_MAX_RETRIES = 3
_BASE_BACKOFF_SECONDS = 0.5

_METEO_SEVERITY_MAP: dict[str, WarningSeverity] = {
    "1": WarningSeverity.low,
    "2": WarningSeverity.moderate,
    "3": WarningSeverity.high,
}

_HYDRO_SEVERITY_MAP: dict[str, WarningSeverity] = {
    "-1": WarningSeverity.low,
    "0": WarningSeverity.low,
    "1": WarningSeverity.moderate,
    "2": WarningSeverity.high,
}

_METEO_CATEGORY_MAP: dict[str, WarningCategory] = {
    "Upały": WarningCategory.meteo,
    "Chłd": WarningCategory.meteo,
    "Przymrozki": WarningCategory.meteo,
    "Silne opady deszczu": WarningCategory.meteo,
    "Ulewy": WarningCategory.meteo,
    "Burze": WarningCategory.meteo,
    "Burze z gradem": WarningCategory.meteo,
    "Silny wiatr": WarningCategory.meteo,
    "Trąby powietrzne": WarningCategory.meteo,
    "Zjawiska lodowe na drogach": WarningCategory.meteo,
    "Gołoledź": WarningCategory.meteo,
    "Oblodzenie": WarningCategory.meteo,
    "Śnieżyce": WarningCategory.meteo,
    "Zarazy": WarningCategory.meteo,
    "MG": WarningCategory.meteo,
    "Rozbryzg": WarningCategory.meteo,
    "Osuwiska": WarningCategory.meteo,
}

_HYDRO_CATEGORY = WarningCategory.hydro


def _parse_imgw_datetime(raw: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(raw, fmt)
            return ensure_utc(dt, naive_tz=WARSAW_TZ)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse IMGW datetime: {raw!r}")


def _build_meteo_external_id(item: dict[str, object]) -> str:
    raw_id = item.get("id", "")
    if raw_id:
        return f"imgw-meteo:{raw_id}"
    stopien = item.get("stopien", "?")
    nazwa = item.get("nazwa_zdarzenia", "unknown")
    obowiazuje_od = item.get("obowiazuje_od", "unknown")
    return f"imgw-meteo:{nazwa}:{stopien}:{obowiazuje_od}"


def _build_hydro_external_id(item: dict[str, object]) -> str:
    numer = item.get("numer", "")
    if numer:
        return f"imgw-hydro:{numer}"
    zdarzenie = item.get("zdarzenie", "unknown")
    data_od = item.get("data_od", "unknown")
    return f"imgw-hydro:{zdarzenie}:{data_od}"


def _parse_meteo_warnings(
    raw: list[dict[str, object]], location: LocationRef
) -> list[WeatherWarning]:
    warnings: list[WeatherWarning] = []
    for item in raw:
        nazwa = str(item.get("nazwa_zdarzenia", ""))
        stopien = str(item.get("stopien", ""))
        tresc = str(item.get("tresc", ""))
        obowiazuje_od = str(item.get("obowiazuje_od", ""))
        obowiazuje_do = str(item.get("obowiazuje_do", ""))

        try:
            valid_from = _parse_imgw_datetime(obowiazuje_od)
            valid_to = _parse_imgw_datetime(obowiazuje_do)
        except ValueError:
            continue

        severity = _METEO_SEVERITY_MAP.get(stopien)
        category = _METEO_CATEGORY_MAP.get(nazwa, WarningCategory.meteo)
        external_id = _build_meteo_external_id(item)

        warnings.append(
            WeatherWarning(
                provider=_METEO_PROVIDER_NAME,
                external_id=external_id,
                location_id=location.id,
                severity=severity,
                category=category,
                headline=nazwa,
                description=tresc,
                valid_from=valid_from,
                valid_to=valid_to,
                raw_payload=item,
            )
        )

    return warnings


def _parse_hydro_warnings(
    raw: list[dict[str, object]], location: LocationRef
) -> list[WeatherWarning]:
    warnings: list[WeatherWarning] = []
    for item in raw:
        zdarzenie = str(item.get("zdarzenie", ""))
        stopien = str(item.get("stopień", item.get("stopien", "")))
        przebieg = str(item.get("przebieg", ""))
        data_od = str(item.get("data_od", ""))
        data_do = str(item.get("data_do", ""))

        try:
            valid_from = _parse_imgw_datetime(data_od)
            valid_to = _parse_imgw_datetime(data_do)
        except ValueError:
            continue

        severity = _HYDRO_SEVERITY_MAP.get(stopien)
        external_id = _build_hydro_external_id(item)

        warnings.append(
            WeatherWarning(
                provider=_HYDRO_PROVIDER_NAME,
                external_id=external_id,
                location_id=location.id,
                severity=severity,
                category=_HYDRO_CATEGORY,
                headline=zdarzenie,
                description=przebieg,
                valid_from=valid_from,
                valid_to=valid_to,
                raw_payload=item,
            )
        )

    return warnings


class _BaseImgwWarningsProvider:
    def __init__(
        self,
        settings: ImgwSettings,
        base_url: str | None = None,
        provider_name: str = "imgw",
        httpx_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url or settings.warnings_base_url
        self._timeout_seconds = settings.timeout_seconds
        self._provider_name = provider_name
        self._client = httpx_client

    async def _request_with_retry(self) -> list[dict[str, object]]:
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return await self._make_request()
            except WeatherProviderTimeoutError as exc:
                last_error = exc
            except WeatherProviderUnavailableError as exc:
                last_error = exc
            except WeatherProviderResponseError:
                raise
            if attempt < _MAX_RETRIES - 1:
                backoff = _BASE_BACKOFF_SECONDS * (2**attempt)
                await asyncio.sleep(backoff)
        raise last_error  # type: ignore[misc]

    async def _make_request(self) -> list[dict[str, object]]:
        timeout = httpx.Timeout(self._timeout_seconds)
        try:
            if self._client is not None:
                response = await self._client.get(self._base_url, timeout=timeout)
            else:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(self._base_url)
        except httpx.TimeoutException:
            raise WeatherProviderTimeoutError(self._provider_name) from None
        except httpx.ConnectError:
            raise WeatherProviderUnavailableError(self._provider_name) from None
        except httpx.HTTPError as exc:
            raise WeatherProviderUnavailableError(
                self._provider_name, f"HTTP error: {exc}"
            ) from exc

        if response.status_code >= 500:
            raise WeatherProviderResponseError(
                self._provider_name, f"Server error: {response.status_code}"
            )
        if response.status_code >= 400:
            raise WeatherProviderResponseError(
                self._provider_name,
                f"Client error: {response.status_code} — {response.text[:200]}",
            )

        try:
            data = response.json()
        except Exception:
            raise WeatherProviderResponseError(
                self._provider_name, "Response is not valid JSON"
            ) from None

        if not isinstance(data, list):
            raise WeatherProviderResponseError(self._provider_name, "Response is not a JSON array")

        return data


class ImgwMeteoWarningsProvider(_BaseImgwWarningsProvider, WarningProvider):
    def __init__(
        self, settings: ImgwSettings, httpx_client: httpx.AsyncClient | None = None
    ) -> None:
        super().__init__(
            settings=settings,
            base_url=settings.warnings_base_url,
            provider_name=_METEO_PROVIDER_NAME,
            httpx_client=httpx_client,
        )

    async def get_warnings(
        self,
        location: LocationRef,
        time_range: TimeRange,
    ) -> list[WeatherWarning]:
        raw = await self._request_with_retry()
        return _parse_meteo_warnings(raw, location)


class ImgwHydroWarningsProvider(_BaseImgwWarningsProvider, WarningProvider):
    _HYDRO_BASE_URL = "https://danepubliczne.imgw.pl/api/data/warningshydro"

    def __init__(
        self, settings: ImgwSettings, httpx_client: httpx.AsyncClient | None = None
    ) -> None:
        super().__init__(
            settings=settings,
            base_url=self._HYDRO_BASE_URL,
            provider_name=_HYDRO_PROVIDER_NAME,
            httpx_client=httpx_client,
        )

    async def get_warnings(
        self,
        location: LocationRef,
        time_range: TimeRange,
    ) -> list[WeatherWarning]:
        raw = await self._request_with_retry()
        return _parse_hydro_warnings(raw, location)
