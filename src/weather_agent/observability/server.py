from __future__ import annotations

import threading

import uvicorn
from fastapi import FastAPI

from weather_agent.observability.logging import get_logger

logger = get_logger(__name__)


def start_observability_server(app: FastAPI, host: str, port: int) -> threading.Thread:
    """Start a lightweight observability HTTP server in a background thread.

    The returned thread is marked as daemon so it does not block process exit.
    """

    def _run() -> None:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    logger.info("Observability server started on %s:%d", host, port)
    return thread
