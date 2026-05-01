"""Create the weather-agent-weather-functional-v2 LangSmith dataset.

Usage:
    uv run python scripts/eval/create_weather_grounding_dataset.py
"""

from __future__ import annotations

import sys

from langsmith import Client

from weather_agent.eval.dataset_gen import generate_cases

DATASET_NAME = "weather-agent-weather-functional-v2"


def main() -> None:
    cases = generate_cases()
    client = Client()
    existing = None
    try:
        existing = client.read_dataset(dataset_name=DATASET_NAME)
    except Exception:
        pass

    if existing is not None:
        print(f"Dataset '{DATASET_NAME}' already exists (id={existing.id}).")
        print(
            "Delete it first if you want to recreate: "
            f"client.delete_dataset(dataset_id='{existing.id}')"
        )
        sys.exit(0)

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "Weather functional correctness benchmark. The real DeepAgent uses "
            "fixture-backed tools and must answer with required weather facts. "
            "Covers current conditions and forecast periods with explicit hours. "
            "Forecast fixtures include 24-hour hourly data with distinct values."
        ),
    )
    examples = [
        {
            "inputs": {
                "id": case["id"],
                "question": case["question"],
                "frozen_facts": case["frozen_facts"],
                "hourly_values": case.get("hourly_values"),
                "target_hour": case.get("target_hour"),
            },
            "outputs": {
                "expected_facts": case["frozen_facts"],
                "required_location": True,
                "requested_attributes": case["requested_attributes"],
            },
        }
        for case in cases
    ]
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"Created dataset '{DATASET_NAME}' with {len(examples)} examples (id={dataset.id})")


if __name__ == "__main__":
    main()
