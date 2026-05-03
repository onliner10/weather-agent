"""Create the weather presentation LangSmith dataset.

Usage:
    uv run python scripts/eval/create_weather_presentation_dataset.py
"""

from __future__ import annotations

import sys

from langsmith import Client

from weather_agent.eval.weather_presentation_dataset import (
    DATASET_NAME,
    generate_weather_presentation_cases,
)


def main() -> None:
    cases = generate_weather_presentation_cases()
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
            "Weather presentation benchmark. The real agent runtime uses fixture-backed "
            "weather tools and must choose render_forecast_chart for explicit chart "
            "requests across several weather variables while keeping point answers text-only."
        ),
    )
    examples = [
        {
            "inputs": {
                "id": case["id"],
                "question": case["question"],
                "current_time": case["current_time"],
                "frozen_facts": case["frozen_facts"],
                "hourly_values": case["hourly_values"],
                "expected_target_time": case["expected_target_time"],
            },
            "outputs": {
                "expect_chart": case["expect_chart"],
                "expected_chart_variables": case.get("expected_chart_variables", []),
                "expected_chart_start_date": case.get("expected_chart_start_date"),
                "expected_chart_end_date": case.get("expected_chart_end_date"),
            },
        }
        for case in cases
    ]
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"Created dataset '{DATASET_NAME}' with {len(examples)} examples (id={dataset.id})")


if __name__ == "__main__":
    main()
