"""Run weather presentation evaluation in LangSmith.

Usage:
    LANGSMITH_API_KEY=... WEATHER_AGENT_MODEL__API_KEY=... \
      uv run python scripts/eval/run_weather_presentation_eval.py
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

from weather_agent.eval.evaluators import weather_presentation_tool_use
from weather_agent.eval.langsmith_experiments import (
    build_langsmith_eval_experiment_prefix,
    build_langsmith_eval_metadata,
)
from weather_agent.eval.targets import build_weather_presentation_async_target_from_factory
from weather_agent.eval.weather_presentation_dataset import DATASET_NAME
from weather_agent.llm.model_factory import ModelFactory
from weather_agent.observability.langsmith_tracing import configure_tracing
from weather_agent.settings import LangSmithSettings, ModelSettings

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
    fallback_model_settings = ModelSettings()
    model_settings = ModelSettings(
        provider=os.environ.get("WEATHER_AGENT_MODEL__PROVIDER", fallback_model_settings.provider),
        model_name=os.environ.get(
            "WEATHER_AGENT_MODEL__MODEL_NAME",
            fallback_model_settings.model_name,
        ),
        temperature=float(
            os.environ.get(
                "WEATHER_AGENT_MODEL__TEMPERATURE",
                str(fallback_model_settings.temperature),
            )
        ),
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
    target = build_weather_presentation_async_target_from_factory(
        lambda: ModelFactory(model_settings).create_chat_model()
    )

    client = Client()
    git_sha = _get_git_sha() or "unknown"
    eval_suite = "weather-presentation"
    results = await client.aevaluate(
        target,
        data=DATASET_NAME,
        evaluators=[weather_presentation_tool_use],
        experiment_prefix=build_langsmith_eval_experiment_prefix(
            eval_suite=eval_suite,
            model_provider=model_settings.provider,
            model_name=model_settings.model_name,
        ),
        metadata={
            **build_langsmith_eval_metadata(
                eval_suite=eval_suite,
                dataset_name=DATASET_NAME,
                git_sha=git_sha,
            ),
            "model_provider": model_settings.provider,
            "model_name": model_settings.model_name,
            "models": [f"{model_settings.provider}:{model_settings.model_name}"],
            "dataset_version": "v2",
            "prompt_version": "production-weather-agent-md",
            "metric": "weather_presentation_tool_use",
            "evaluator": "deterministic_weather_presentation_tool_use",
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
