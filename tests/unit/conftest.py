from __future__ import annotations

import os

import pytest

from weather_agent.observability.logging import configure_logging

configure_logging(log_level="DEBUG")


@pytest.fixture(autouse=True)
def isolate_pid_dir(tmp_path: pytest.TempPathFactory) -> None:
    """Isolate PID files during tests to avoid interference with the host."""
    os.environ["WEATHER_AGENT_PID_DIR"] = str(tmp_path)
