from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import langsmith
import langsmith.run_helpers
import langsmith.utils
import pytest

from weather_agent.observability.logging import configure_logging

_TESTS_DIR = Path(__file__).resolve().parents[1]
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from support.sync_aiosqlite import install_sync_aiosqlite  # noqa: E402

_LANGSMITH_ENV_KEYS = (
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
    "LANGCHAIN_ENDPOINT",
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
)


class _NoopTrace:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> bool:
        return False

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_args: object) -> bool:
        return False


def _noop_trace(*args: object, **kwargs: object) -> _NoopTrace:
    return _NoopTrace(*args, **kwargs)


def _noop_traceable(
    wrapped: Callable[..., Any] | None = None,
    **_kwargs: object,
) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    return decorator if wrapped is None else wrapped


configure_logging(log_level="DEBUG")
install_sync_aiosqlite()
langsmith.trace = _noop_trace  # type: ignore[assignment]
langsmith.traceable = _noop_traceable  # type: ignore[assignment]
langsmith.run_helpers.trace = _noop_trace  # type: ignore[assignment]
langsmith.run_helpers.traceable = _noop_traceable  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def isolate_langsmith_env() -> None:
    for key in _LANGSMITH_ENV_KEYS:
        os.environ.pop(key, None)
    langsmith.utils.get_env_var.cache_clear()
    yield
    for key in _LANGSMITH_ENV_KEYS:
        os.environ.pop(key, None)
    langsmith.utils.get_env_var.cache_clear()


@pytest.fixture(autouse=True)
def isolate_pid_dir(tmp_path: pytest.TempPathFactory) -> None:
    """Isolate PID files during tests to avoid interference with the host."""
    os.environ["WEATHER_AGENT_PID_DIR"] = str(tmp_path)
