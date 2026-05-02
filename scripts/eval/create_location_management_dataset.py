"""Create the weather-agent-location-management-v1 LangSmith dataset.

Usage:
    uv run python scripts/eval/create_location_management_dataset.py
"""

from __future__ import annotations

import sys

from langsmith import Client

from weather_agent.eval.location_management_dataset import (
    DATASET_NAME,
    generate_location_management_cases,
)


def main() -> None:
    cases = generate_location_management_cases()
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
            "Location management benchmark. The real DeepAgent uses fixture-backed "
            "location and rule tools and must add, edit, remove, and use default "
            "locations without inventing a location."
        ),
    )
    examples = [
        {
            "inputs": {
                "id": case.id,
                "question": case.question,
                "current_time": case.current_time.isoformat(),
                "seed_locations": [seed.model_dump(mode="json") for seed in case.seed_locations],
            },
            "outputs": {"expected": case.expected.model_dump(mode="json")},
        }
        for case in cases
    ]
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"Created dataset '{DATASET_NAME}' with {len(examples)} examples (id={dataset.id})")


if __name__ == "__main__":
    main()
