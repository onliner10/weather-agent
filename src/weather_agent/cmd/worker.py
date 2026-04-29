from __future__ import annotations

import argparse
import asyncio
import os

from langsmith import trace

from weather_agent.api.health import create_health_app
from weather_agent.domain.cel.evaluator import CELEvaluator
from weather_agent.infrastructure.db.setup import run_migrations
from weather_agent.infrastructure.services import BotServices
from weather_agent.infrastructure.utils import acquire_lock
from weather_agent.observability.logging import (
    bound_worker_context,
    get_logger,
)
from weather_agent.observability.server import start_observability_server

logger = get_logger(__name__)


def cmd_worker(_args: argparse.Namespace) -> None:
    pid_file = acquire_lock("worker")
    services = BotServices()

    if services.settings.observability.enabled:
        health_app = create_health_app(session_factory=services.session_factory)
        start_observability_server(
            app=health_app,
            host="0.0.0.0",
            port=services.settings.observability.worker_port,
        )

    try:
        run_migrations()
    except Exception as exc:
        logger.warning(
            "migration_failed",
            error_class=type(exc).__name__,
            error_message=str(exc),
        )

    async def _run_worker() -> None:
        from weather_agent.domain.rules.service import NotificationRuleService
        from weather_agent.infrastructure.repositories.forecast_repository import (
            ForecastRepository,
        )
        from weather_agent.infrastructure.worker.rule_evaluator import RuleEvaluationWorker

        async with services.session_factory() as session:
            async with trace(
                "worker_startup",
                run_type="tool",
                metadata={"mode": "worker"},
            ):
                cel_evaluator = CELEvaluator()
                rule_service = NotificationRuleService(
                    session=session,
                    cel_evaluator=cel_evaluator,
                )
                forecast_repo = ForecastRepository(session=session)

                worker = RuleEvaluationWorker(
                    session=session,
                    forecast_repo=forecast_repo,
                    cel_evaluator=cel_evaluator,
                    rule_service=rule_service,
                    settings=services.settings.scheduler,
                )

            with bound_worker_context():
                logger.info("starting_rule_evaluation_worker")
                try:
                    await worker.run_loop()
                finally:
                    if os.path.exists(pid_file):
                        os.unlink(pid_file)

    asyncio.run(_run_worker())
