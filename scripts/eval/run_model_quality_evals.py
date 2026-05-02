"""Run all LangSmith model-quality eval suites.

Usage:
    LANGSMITH_API_KEY=... WEATHER_AGENT_MODEL__API_KEY=... \
      uv run python scripts/eval/run_model_quality_evals.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_ENV_VARS = (
    "LANGSMITH_API_KEY",
    "WEATHER_AGENT_MODEL__API_KEY",
)

DEFAULT_MODEL_ENV = {
    "WEATHER_AGENT_MODEL__PROVIDER": "openrouter",
    "WEATHER_AGENT_MODEL__MODEL_NAME": "qwen/qwen3.5-flash-02-23",
    "WEATHER_AGENT_MODEL__BASE_URL": "https://openrouter.ai/api/v1",
    "WEATHER_AGENT_MODEL__ROUTING_SORT": "price",
    "WEATHER_AGENT_MODEL__REQUIRE_SUPPORTED_PARAMETERS": "true",
}

EVAL_STEPS = (
    ("Sync notification rule dataset", "scripts/eval/create_notification_rule_dataset.py"),
    ("Run notification rule eval", "scripts/eval/run_notification_rule_eval.py"),
    ("Sync location management dataset", "scripts/eval/create_location_management_dataset.py"),
    ("Run location management eval", "scripts/eval/run_location_management_eval.py"),
    ("Sync weather grounding dataset", "scripts/eval/create_weather_grounding_dataset.py"),
    ("Run weather grounding eval", "scripts/eval/run_weather_grounding_eval.py"),
)


def _missing_env_vars(names: Sequence[str]) -> list[str]:
    return [name for name in names if not os.environ.get(name)]


def _run_step(label: str, script_path: str, env: dict[str, str]) -> None:
    print(f"\n==> {label}", flush=True)
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=REPO_ROOT,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _build_child_env() -> dict[str, str]:
    env = dict(os.environ)
    for name, value in DEFAULT_MODEL_ENV.items():
        env.setdefault(name, value)
    return env


def main() -> None:
    missing = _missing_env_vars(REQUIRED_ENV_VARS)
    if missing:
        joined = ", ".join(missing)
        print(f"Error: missing required environment variables: {joined}", file=sys.stderr)
        raise SystemExit(1)

    child_env = _build_child_env()

    for label, script_path in EVAL_STEPS:
        _run_step(label, script_path, child_env)


if __name__ == "__main__":
    main()
