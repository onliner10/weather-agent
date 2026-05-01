"""Run weather grounding evaluation in LangSmith.

Usage:
    LANGSMITH_API_KEY=... uv run python scripts/eval/run_weather_grounding_eval.py
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal, cast

from langsmith import Client
from pydantic import SecretStr

from weather_agent.eval.evaluators import weather_functional_correctness
from weather_agent.eval.targets import build_weather_answer_async_target_from_factory
from weather_agent.llm.model_factory import ModelFactory
from weather_agent.observability.langsmith_tracing import configure_tracing
from weather_agent.settings import LangSmithSettings, ModelSettings

DATASET_NAME = "weather-agent-weather-functional-v5"
REPO_ROOT = Path(__file__).resolve().parents[2]
ROUTING_SORT_VALUES = {"price", "latency", "throughput"}
RoutingSort = Literal["price", "latency", "throughput"]


def _get_git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


async def _run() -> None:
    ls_settings = LangSmithSettings(
        enabled=True,
        api_key=(
            SecretStr(api_key)
            if (api_key := os.environ.get("LANGSMITH_API_KEY")) is not None
            else None
        ),
        project=os.environ.get("LANGSMITH_PROJECT", "weather-agent-dev"),
    )
    configure_tracing(ls_settings)

    if not ls_settings.api_key:
        print("Error: LANGSMITH_API_KEY is required", file=sys.stderr)
        sys.exit(1)

    routing_sort = os.environ.get("WEATHER_AGENT_MODEL__ROUTING_SORT")
    if routing_sort not in ROUTING_SORT_VALUES:
        routing_sort = None
    model_settings = ModelSettings(
        provider=os.environ.get("WEATHER_AGENT_MODEL__PROVIDER", "openai"),
        model_name=os.environ.get("WEATHER_AGENT_MODEL__MODEL_NAME", "gpt-4.1-mini"),
        api_key=(
            SecretStr(api_key)
            if (api_key := os.environ.get("WEATHER_AGENT_MODEL__API_KEY")) is not None
            else None
        ),
        base_url=os.environ.get("WEATHER_AGENT_MODEL__BASE_URL"),
        routing_sort=cast(RoutingSort | None, routing_sort),
        require_supported_parameters=os.environ.get(
            "WEATHER_AGENT_MODEL__REQUIRE_SUPPORTED_PARAMETERS", "true"
        ).lower()
        not in {"0", "false", "no"},
    )
    target = build_weather_answer_async_target_from_factory(
        lambda: ModelFactory(model_settings).create_chat_model()
    )

    client = Client()
    results = await client.aevaluate(
        target,
        data=DATASET_NAME,
        evaluators=[weather_functional_correctness],
        experiment_prefix=f"weather-functional-{model_settings.provider}-{model_settings.model_name}",
        metadata={
            "model_provider": model_settings.provider,
            "model_name": model_settings.model_name,
            "git_sha": _get_git_sha() or "unknown",
            "dataset_version": "v4",
            "prompt_version": "production-weather-agent-md",
            "metric": "weather_functional_correctness",
            "evaluator": "deterministic_functional_correctness",
        },
        max_concurrency=4,
    )
    async for _ in results:
        pass
    print(f"\nExperiment results: {results}")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
