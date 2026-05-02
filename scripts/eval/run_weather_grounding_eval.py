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
from weather_agent.eval.judges import (
    WEATHER_GROUNDEDNESS_JUDGE_PROMPT_VERSION,
    build_weather_groundedness_judge,
)
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


def _model_settings_from_env(prefix: str, defaults: ModelSettings | None = None) -> ModelSettings:
    fallback = defaults or ModelSettings()
    routing_sort = os.environ.get(f"{prefix}ROUTING_SORT")
    if routing_sort not in ROUTING_SORT_VALUES:
        routing_sort = None
    return ModelSettings(
        provider=os.environ.get(f"{prefix}PROVIDER", fallback.provider),
        model_name=os.environ.get(f"{prefix}MODEL_NAME", fallback.model_name),
        temperature=float(os.environ.get(f"{prefix}TEMPERATURE", str(fallback.temperature))),
        api_key=(
            SecretStr(api_key)
            if (api_key := os.environ.get(f"{prefix}API_KEY")) is not None
            else fallback.api_key
        ),
        base_url=os.environ.get(f"{prefix}BASE_URL", fallback.base_url),
        timeout_seconds=float(
            os.environ.get(f"{prefix}TIMEOUT_SECONDS", str(fallback.timeout_seconds))
        ),
        routing_sort=cast(RoutingSort | None, routing_sort),
        require_supported_parameters=os.environ.get(
            f"{prefix}REQUIRE_SUPPORTED_PARAMETERS",
            str(fallback.require_supported_parameters).lower(),
        ).lower()
        not in {"0", "false", "no"},
    )


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

    model_settings = _model_settings_from_env(
        "WEATHER_AGENT_MODEL__",
        ModelSettings(),
    )
    judge_model_settings = _model_settings_from_env(
        "WEATHER_AGENT_JUDGE_MODEL__",
        model_settings.model_copy(update={"temperature": 0.0}),
    )
    target = build_weather_answer_async_target_from_factory(
        lambda: ModelFactory(model_settings).create_chat_model()
    )
    groundedness_judge = build_weather_groundedness_judge(
        lambda: ModelFactory(judge_model_settings).create_chat_model()
    )

    client = Client()
    results = await client.aevaluate(
        target,
        data=DATASET_NAME,
        evaluators=[weather_functional_correctness, groundedness_judge],
        experiment_prefix=f"weather-functional-{model_settings.provider}-{model_settings.model_name}",
        metadata={
            "model_provider": model_settings.provider,
            "model_name": model_settings.model_name,
            "judge_model_provider": judge_model_settings.provider,
            "judge_model_name": judge_model_settings.model_name,
            "git_sha": _get_git_sha() or "unknown",
            "dataset_version": "v5",
            "prompt_version": "production-weather-agent-md",
            "metrics": [
                "weather_functional_correctness",
                "weather_answer_groundedness_judge",
            ],
            "evaluators": [
                "deterministic_functional_correctness",
                "llm_groundedness_judge",
            ],
            "judge_prompt_version": WEATHER_GROUNDEDNESS_JUDGE_PROMPT_VERSION,
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
