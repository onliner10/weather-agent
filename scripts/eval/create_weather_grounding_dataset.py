"""Create the weather-agent-weather-functional-v1 LangSmith dataset.

Usage:
    uv run python scripts/eval/create_weather_grounding_dataset.py
"""

from __future__ import annotations

import sys

from langsmith import Client

DATASET_NAME = "weather-agent-weather-functional-v1"

_CASES: list[dict[str, object]] = [
    {
        "id": "grounding-001",
        "question": "Podaj aktualną wartość temperatury w Chwarznie.",
        "frozen_facts": {
            "location": "Chwarzno",
            "period": "teraz",
            "temperature_c": 12.0,
        },
        "requested_attributes": ["temperature_c"],
    },
    {
        "id": "grounding-002",
        "question": "Podaj aktualną prędkość wiatru nad Jeziorakiem.",
        "frozen_facts": {
            "location": "Jeziorak",
            "period": "teraz",
            "wind_speed_ms": 8.5,
        },
        "requested_attributes": ["wind_speed_ms"],
    },
    {
        "id": "grounding-003",
        "question": "Podaj aktualną wartość wilgotności w Gdyni.",
        "frozen_facts": {
            "location": "Gdynia",
            "period": "teraz",
            "humidity_pct": 82.0,
        },
        "requested_attributes": ["humidity_pct"],
    },
    {
        "id": "grounding-004",
        "question": "Podaj aktualną sumę opadów w Warszawie.",
        "frozen_facts": {
            "location": "Warszawa",
            "period": "teraz",
            "precipitation_mm": 3.0,
        },
        "requested_attributes": ["precipitation_mm"],
    },
    {
        "id": "grounding-005",
        "question": "Podaj aktualną wartość ciśnienia w Chwarznie.",
        "frozen_facts": {
            "location": "Chwarzno",
            "period": "teraz",
            "pressure_hpa": 1012.0,
        },
        "requested_attributes": ["pressure_hpa"],
    },
    {
        "id": "grounding-006",
        "question": "Podaj aktualny kierunek wiatru w Chwarznie.",
        "frozen_facts": {
            "location": "Chwarzno",
            "period": "teraz",
            "wind_direction_deg": 270.0,
        },
        "requested_attributes": ["wind_direction_deg"],
    },
]


def main() -> None:
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
            "fixture-backed tools and must answer with required weather facts."
        ),
    )
    examples = [
        {
            "inputs": {
                "id": case["id"],
                "question": case["question"],
                "frozen_facts": case["frozen_facts"],
            },
            "outputs": {
                "expected_facts": case["frozen_facts"],
                "required_location": True,
                "requested_attributes": case["requested_attributes"],
            },
        }
        for case in _CASES
    ]
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"Created dataset '{DATASET_NAME}' with {len(examples)} examples (id={dataset.id})")


if __name__ == "__main__":
    main()
