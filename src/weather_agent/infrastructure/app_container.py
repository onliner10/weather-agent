from __future__ import annotations

import sys
from types import TracebackType
from typing import TYPE_CHECKING, Self

import httpx
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from weather_agent.infrastructure.db.setup import create_engine, create_session_factory
from weather_agent.observability.logging import get_logger

if TYPE_CHECKING:
    from weather_agent.adapters.imgw.synop_provider import ImgwSynopProvider
    from weather_agent.adapters.open_meteo.forecast_provider import (
        OpenMeteoDwdIconProvider,
    )
    from weather_agent.domain.rule_expression.evaluator import RuleExpressionEvaluator
    from weather_agent.infrastructure.geocoder import Geocoder
    from weather_agent.llm.model_factory import ModelFactory
    from weather_agent.settings import AppSettings

logger = get_logger(__name__)


class AppContainer:
    """Async composition root with explicit lifecycle.

    All services are created eagerly during ``__aenter__`` and torn down
    during ``__aexit__`` / ``close()``.
    """

    settings: AppSettings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    httpx_client: httpx.AsyncClient
    rule_expression_evaluator: RuleExpressionEvaluator
    forecast_provider: OpenMeteoDwdIconProvider
    observation_provider: ImgwSynopProvider
    model_factory: ModelFactory
    geocoder: Geocoder

    def __init__(self) -> None:
        from weather_agent.observability.langsmith_tracing import configure_tracing
        from weather_agent.observability.logging import configure_logging
        from weather_agent.settings import load_settings

        try:
            settings = load_settings()
        except Exception as exc:
            print(
                f"Error loading configuration: {exc}\n"
                "Ensure all required environment variables are set "
                "(WEATHER_AGENT_DATABASE_URL, WEATHER_AGENT_TELEGRAM__BOT_TOKEN, etc.).",
                file=sys.stderr,
            )
            sys.exit(1)

        configure_logging()
        configure_tracing(settings.langsmith)

        self.settings = settings
        self.engine = create_engine(settings.database_url)
        self.session_factory = create_session_factory(self.engine)
        self.httpx_client = httpx.AsyncClient()

    async def __aenter__(self) -> Self:
        self._init_services()
        logger.info("Application services initialized")
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        await self.httpx_client.aclose()
        await self.engine.dispose()
        logger.info("Application services closed")

    def _init_services(self) -> None:
        from weather_agent.adapters.imgw.synop_provider import ImgwSynopProvider
        from weather_agent.adapters.open_meteo.forecast_provider import (
            OpenMeteoDwdIconProvider,
        )
        from weather_agent.domain.rule_expression.evaluator import RuleExpressionEvaluator
        from weather_agent.infrastructure.geocoder import Geocoder
        from weather_agent.llm.model_factory import ModelFactory

        self.rule_expression_evaluator = RuleExpressionEvaluator()
        self.forecast_provider = OpenMeteoDwdIconProvider(
            settings=self.settings.open_meteo,
            httpx_client=self.httpx_client,
        )
        self.observation_provider = ImgwSynopProvider(
            settings=self.settings.imgw,
            httpx_client=self.httpx_client,
        )
        self.model_factory = ModelFactory(settings=self.settings.model)
        self.geocoder = Geocoder(
            settings=self.settings.geocoding,
            model_factory=self.model_factory,
            httpx_client=self.httpx_client,
        )
