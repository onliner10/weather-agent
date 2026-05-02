from __future__ import annotations

import time
from collections.abc import Mapping
from typing import cast

import httpx

from weather_agent.domain.polish_utils import normalize_polish
from weather_agent.domain.weather import LocationRef
from weather_agent.infrastructure.locationiq_candidates import select_best_candidate
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
        result = select_best_candidate(query, candidates)
        if result is not None:
            return result

        if self._use_autocomplete_fallback:
            candidates = await self._fetch_candidates("autocomplete", query)
            result = select_best_candidate(query, candidates)
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
