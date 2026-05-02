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


def _run_step(label: str, script_path: str) -> None:
    print(f"\n==> {label}", flush=True)
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    missing = _missing_env_vars(REQUIRED_ENV_VARS)
    if missing:
        joined = ", ".join(missing)
        print(f"Error: missing required environment variables: {joined}", file=sys.stderr)
        raise SystemExit(1)

    for label, script_path in EVAL_STEPS:
        _run_step(label, script_path)


if __name__ == "__main__":
    main()
