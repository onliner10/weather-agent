"""Create the weather-agent-notification-rule-proposal-v2 LangSmith dataset.

Usage:
    uv run python scripts/eval/create_notification_rule_dataset.py
"""

from __future__ import annotations

import sys

from langsmith import Client

from weather_agent.eval.notification_rule_dataset import (
    DATASET_NAME,
    generate_notification_rule_cases,
)


def main() -> None:
    cases = generate_notification_rule_cases()
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
            "Notification rule proposal fidelity benchmark. The real agent runtime uses "
            "fixture-backed rule tools and must propose safe pending notification rules "
            "or schedules without unauthorized confirmation."
        ),
    )
    examples = [
        {
            "inputs": {
                "id": case.id,
                "question": case.question,
                "current_time": case.current_time.isoformat(),
            },
            "outputs": {
                "expected": case.expected.model_dump(mode="json"),
                "current_time": case.current_time.isoformat(),
            },
        }
        for case in cases
    ]
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"Created dataset '{DATASET_NAME}' with {len(examples)} examples (id={dataset.id})")


if __name__ == "__main__":
    main()
