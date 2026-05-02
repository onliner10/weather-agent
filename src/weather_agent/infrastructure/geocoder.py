from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from typing import cast

import httpx

from weather_agent.domain.polish_utils import normalize_for_matching, normalize_polish
from weather_agent.domain.weather import LocationRef
from weather_agent.llm.contracts.geocoder import LocationGuess
from weather_agent.llm.model_factory import ModelFactory
from weather_agent.llm.prompts.geocoder_prompts import GEOCODING_TEMPLATE
from weather_agent.observability.logging import get_logger
from weather_agent.observability.metrics import (
    GEOCODE_DURATION_SECONDS,
    GEOCODE_REQUESTS_TOTAL,
)
from weather_agent.settings import GeocodingSettings

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({"w", "we", "na", "nad", "pod", "przy", "do", "z", "ze", "i"})
_PLACE_CLASSES = frozenset({"boundary", "place"})
_PLACE_TYPES = frozenset(
    {
        "administrative",
        "city",
        "city_district",
        "district",
        "hamlet",
        "locality",
        "neighbourhood",
        "neighborhood",
        "quarter",
        "suburb",
        "town",
        "village",
    }
)


class Geocoder:
    def __init__(
        self,
        settings: GeocodingSettings | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        model_factory: ModelFactory | None = None,
        httpx_client: httpx.AsyncClient | None = None,
        api_key: str | None = None,
    ) -> None:
        effective_settings = settings or GeocodingSettings()
        self._base_url = (base_url or effective_settings.base_url).rstrip("/")
        self._timeout = timeout_seconds or effective_settings.timeout_seconds
        self._api_key = api_key or (
            effective_settings.api_key.get_secret_value()
            if effective_settings.api_key is not None
            else None
        )
        self._countrycodes = effective_settings.countrycodes
        self._language = effective_settings.language
        self._result_limit = effective_settings.result_limit
        self._use_autocomplete_fallback = effective_settings.use_autocomplete_fallback
        self._model_factory = model_factory
        self._client = httpx_client

    async def geocode(self, name: str) -> LocationRef | None:
        start = time.perf_counter()
        try:
            result = await self._try_geocode(name)
            if result is not None:
                GEOCODE_REQUESTS_TOTAL.labels(outcome="success").inc()
                return result

            if self._model_factory is not None:
                guess = await self._ask_llm(name)
                if guess is not None:
                    query = guess.search_query or guess.display_name
                    result = await self._try_geocode(query)
                    if result is not None:
                        GEOCODE_REQUESTS_TOTAL.labels(outcome="success").inc()
                        return result
                    logger.info(
                        "geocoding_llm_suggestion_no_results",
                        query=query,
                        original_name=name,
                    )

            GEOCODE_REQUESTS_TOTAL.labels(outcome="not_found").inc()
            return None
        except Exception:
            GEOCODE_REQUESTS_TOTAL.labels(outcome="failure").inc()
            logger.exception(
                "geocode_failed",
                query_name=name,
            )
            return None
        finally:
            GEOCODE_DURATION_SECONDS.observe(time.perf_counter() - start)

    async def _ask_llm(self, name: str) -> LocationGuess | None:
        try:
            chat = self._model_factory.create_chat_model()  # type: ignore[union-attr]
            structured = chat.with_structured_output(LocationGuess)
            chain = GEOCODING_TEMPLATE | structured
            response = await chain.ainvoke({"location_name": name})
            if isinstance(response, LocationGuess):
                logger.info(
                    "llm_location_guess",
                    query_name=name,
                    display_name=response.display_name,
                    search_query=response.search_query,
                )
                return response
        except Exception:
            logger.warning(
                "llm_location_resolution_failed",
                query_name=name,
                exc_info=True,
            )
        return None

    async def _try_geocode(self, name: str) -> LocationRef | None:
        if self._api_key is None or not self._api_key.strip():
            logger.warning("geocoding_api_key_missing")
            return None

        query = name.strip()
        if not query:
            return None

        return await self._try_query(query)

    async def _try_query(self, query: str) -> LocationRef | None:
        candidates = await self._fetch_candidates("search", query)
        result = self._select_best_candidate(query, candidates)
        if result is not None:
            return result

        if self._use_autocomplete_fallback:
            candidates = await self._fetch_candidates("autocomplete", query)
            result = self._select_best_candidate(query, candidates)
            if result is not None:
                return result

        logger.info(
            "geocoding_no_results",
            query_name=query,
            normalized=normalize_polish(query),
        )
        return None

    async def _fetch_candidates(self, endpoint: str, query: str) -> list[Mapping[str, object]]:
        params: dict[str, str | int] = {
            "key": self._api_key or "",
            "q": query,
            "format": "json",
            "countrycodes": self._countrycodes,
            "accept-language": self._language,
            "addressdetails": 1,
            "normalizeaddress": 1,
            "normalizecity": 1,
            "limit": self._result_limit,
        }
        try:
            if self._client is not None:
                response = await self._client.get(
                    f"{self._base_url}/{endpoint}",
                    params=params,
                    timeout=self._timeout,
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.get(f"{self._base_url}/{endpoint}", params=params)
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException:
            logger.warning("geocoding_timeout", query_name=query, endpoint=endpoint)
            return []
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "geocoding_http_error",
                query_name=query,
                endpoint=endpoint,
                http_status=exc.response.status_code,
            )
            return []
        except ValueError:
            logger.warning("geocoding_malformed_json", query_name=query, endpoint=endpoint)
            return []

        if not isinstance(payload, list):
            logger.warning("geocoding_unexpected_payload", query_name=query, endpoint=endpoint)
            return []
        return [cast(Mapping[str, object], item) for item in payload if isinstance(item, Mapping)]

    def _select_best_candidate(
        self, query: str, candidates: Sequence[Mapping[str, object]]
    ) -> LocationRef | None:
        tokens = _query_tokens(query)
        scored = [
            (_candidate_score(candidate, tokens), index, candidate)
            for index, candidate in enumerate(candidates)
        ]
        usable = [(score, index, candidate) for score, index, candidate in scored if score > 0]
        if not usable:
            return None
        _score, _index, candidate = max(usable, key=lambda item: (item[0], -item[1]))
        return _candidate_to_location_ref(candidate)


def _query_tokens(query: str) -> frozenset[str]:
    normalized = normalize_for_matching(query)
    return frozenset(
        token
        for token in _TOKEN_RE.findall(normalized)
        if len(token) >= 2 and token not in _STOPWORDS
    )


def _candidate_score(candidate: Mapping[str, object], query_tokens: frozenset[str]) -> float:
    if not _is_polish_candidate(candidate):
        return 0.0

    lat = _float_field(candidate, "lat")
    lon = _float_field(candidate, "lon")
    if lat is None or lon is None:
        return 0.0

    text = _candidate_text(candidate)
    matched = query_tokens.intersection(_query_tokens(text))
    if query_tokens and not matched:
        return 0.0

    is_place = _is_place_candidate(candidate)
    if not is_place and query_tokens and matched != query_tokens:
        return 0.0

    name_text = _string_field(candidate, "name") or _first_display_part(candidate) or ""
    name_matches = query_tokens.intersection(_query_tokens(name_text))
    score = float(len(matched) * 10 + len(name_matches) * 3)
    if is_place:
        score += 5.0
    score += _float_field(candidate, "importance") or 0.0
    return score


def _is_polish_candidate(candidate: Mapping[str, object]) -> bool:
    address = _address(candidate)
    country_code = _string_field(address, "country_code") or _string_field(
        candidate, "country_code"
    )
    return country_code is None or country_code.lower() == "pl"


def _is_place_candidate(candidate: Mapping[str, object]) -> bool:
    category = _string_field(candidate, "class") or _string_field(candidate, "category")
    candidate_type = _string_field(candidate, "type")
    addresstype = _string_field(candidate, "addresstype")
    return (
        category in _PLACE_CLASSES or candidate_type in _PLACE_TYPES or addresstype in _PLACE_TYPES
    )


def _candidate_to_location_ref(candidate: Mapping[str, object]) -> LocationRef | None:
    lat = _float_field(candidate, "lat")
    lon = _float_field(candidate, "lon")
    if lat is None or lon is None:
        return None
    location_id = _candidate_id(candidate)
    location_name = _string_field(candidate, "name") or _first_display_part(candidate)
    if location_name is None:
        location_name = _string_field(candidate, "display_name") or "Lokalizacja"
    return LocationRef(
        id=location_id,
        name=location_name,
        latitude=lat,
        longitude=lon,
    )


def _candidate_id(candidate: Mapping[str, object]) -> str:
    place_id = candidate.get("place_id")
    if place_id is not None:
        return str(place_id)
    osm_type = _string_field(candidate, "osm_type")
    osm_id = candidate.get("osm_id")
    if osm_type is not None and osm_id is not None:
        return f"{osm_type}:{osm_id}"
    return str(hash(_candidate_text(candidate)))


def _candidate_text(candidate: Mapping[str, object]) -> str:
    parts: list[str] = []
    for key in ("name", "display_name", "type", "class", "category", "addresstype"):
        value = _string_field(candidate, key)
        if value:
            parts.append(value)
    address = _address(candidate)
    parts.extend(value for value in _string_values(address) if value)
    return " ".join(parts)


def _first_display_part(candidate: Mapping[str, object]) -> str | None:
    display_name = _string_field(candidate, "display_name")
    if display_name is None:
        return None
    first_part = display_name.split(",", maxsplit=1)[0].strip()
    return first_part or None


def _address(candidate: Mapping[str, object]) -> Mapping[str, object]:
    address = candidate.get("address")
    if isinstance(address, Mapping):
        return cast(Mapping[str, object], address)
    return {}


def _string_field(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _float_field(data: Mapping[str, object], key: str) -> float | None:
    value = data.get(key)
    try:
        if isinstance(value, str | int | float):
            return float(value)
    except ValueError:
        return None
    return None


def _string_values(data: Mapping[str, object]) -> list[str]:
    return [value.strip() for value in data.values() if isinstance(value, str) and value.strip()]
