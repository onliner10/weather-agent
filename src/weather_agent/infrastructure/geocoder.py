from __future__ import annotations

import time

import httpx

from weather_agent.domain.polish_utils import normalize_polish
from weather_agent.domain.weather import LocationRef
from weather_agent.llm.contracts.geocoder import LocationGuess
from weather_agent.llm.model_factory import ModelFactory
from weather_agent.llm.prompts.geocoder_prompts import GEOCODING_TEMPLATE
from weather_agent.observability.logging import get_logger
from weather_agent.observability.metrics import (
    GEOCODE_DURATION_SECONDS,
    GEOCODE_REQUESTS_TOTAL,
)

logger = get_logger(__name__)


class Geocoder:
    def __init__(
        self,
        base_url: str = "https://geocoding-api.open-meteo.com/v1/search",
        timeout_seconds: float = 10.0,
        model_factory: ModelFactory | None = None,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout_seconds
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
        normalized = normalize_polish(name)
        try:
            if self._client is not None:
                r = await self._client.get(
                    self._base_url,
                    params={"name": normalized, "count": 1, "language": "pl"},
                    timeout=self._timeout,
                )
                r.raise_for_status()
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    r = await client.get(
                        self._base_url,
                        params={"name": normalized, "count": 1, "language": "pl"},
                    )
                    r.raise_for_status()
        except httpx.TimeoutException:
            logger.warning(
                "geocoding_timeout",
                query_name=name,
            )
            return None
        except httpx.HTTPStatusError:
            logger.warning(
                "geocoding_http_error",
                query_name=name,
                http_status=r.status_code,
            )
            return None

        data = r.json()
        results = data.get("results")
        if not results:
            logger.info(
                "geocoding_no_results",
                query_name=name,
                normalized=normalized,
            )
            return None

        first = results[0]
        return LocationRef(
            id=str(first["id"]),
            name=first["name"],
            latitude=first["latitude"],
            longitude=first["longitude"],
        )
