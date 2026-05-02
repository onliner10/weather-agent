from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from weather_agent.infrastructure.db.setup import create_engine, create_session_factory
from weather_agent.observability.logging import get_logger

if TYPE_CHECKING:
    from weather_agent.adapters.imgw.synop_provider import ImgwSynopProvider
    from weather_agent.adapters.open_meteo.forecast_provider import OpenMeteoDwdIconProvider
    from weather_agent.domain.cel.evaluator import CELEvaluator
    from weather_agent.infrastructure.geocoder import Geocoder
    from weather_agent.llm.model_factory import ModelFactory
    from weather_agent.settings import AppSettings

logger = get_logger(__name__)


class BotServices:
    settings: AppSettings
    cel_evaluator: CELEvaluator | None
    forecast_provider: OpenMeteoDwdIconProvider | None
    observation_provider: ImgwSynopProvider | None
    model_factory: ModelFactory | None
    geocoder: Geocoder | None

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

        self.cel_evaluator = None
        self.forecast_provider = None
        self.observation_provider = None
        self.model_factory = None
        self.geocoder = None

    def init_services(self) -> None:
        from weather_agent.adapters.imgw.synop_provider import ImgwSynopProvider
        from weather_agent.adapters.open_meteo.forecast_provider import (
            OpenMeteoDwdIconProvider,
        )
        from weather_agent.domain.cel.evaluator import CELEvaluator
        from weather_agent.infrastructure.geocoder import Geocoder
        from weather_agent.llm.model_factory import ModelFactory

        self.cel_evaluator = CELEvaluator()
        self.forecast_provider = OpenMeteoDwdIconProvider(settings=self.settings.open_meteo)
        self.observation_provider = ImgwSynopProvider(settings=self.settings.imgw)
        self.model_factory = ModelFactory(settings=self.settings.model)
        self.geocoder = Geocoder(
            settings=self.settings.geocoding,
            model_factory=self.model_factory,
        )
        logger.info("Application services initialized")
