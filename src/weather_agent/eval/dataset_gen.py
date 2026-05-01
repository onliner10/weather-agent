from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Literal, NotRequired, TypedDict
from zoneinfo import ZoneInfo

from weather_agent.eval.schemas import WeatherAttribute

LocationLabel = Literal["Warszawa"]
PeriodLabel = Literal[
    "teraz",
    "jutro",
    "za 2 dni",
    "3 maja",
    "w weekend",
    "w Święto Pracy",
    "za tydzień",
]


class GeneratedCase(TypedDict):
    id: str
    question: str
    current_time: str
    frozen_facts: dict[str, object]
    requested_attributes: list[WeatherAttribute]
    hourly_values: NotRequired[dict[int, dict[WeatherAttribute, float]]]
    target_hour: NotRequired[int]
    expected_target_time: NotRequired[str]
    note: NotRequired[str]


ATTRIBUTE_RANGES: dict[WeatherAttribute, tuple[float, float, float]] = {
    "temperature_c": (-5.0, 30.0, 0.5),
    "wind_speed_ms": (0.5, 25.0, 0.5),
    "wind_direction_deg": (0.0, 355.0, 5.0),
    "pressure_hpa": (990.0, 1035.0, 1.0),
    "humidity_pct": (30.0, 100.0, 1.0),
    "precipitation_mm": (0.0, 20.0, 0.5),
}
ATTRIBUTES: list[WeatherAttribute] = list(ATTRIBUTE_RANGES)
PERIODS: list[PeriodLabel] = [
    "teraz",
    "jutro",
    "za 2 dni",
    "3 maja",
    "w weekend",
    "w Święto Pracy",
    "za tydzień",
]
FORECAST_HOURS: list[int] = [9, 13, 18]
EVAL_TIMEZONE = ZoneInfo("Europe/Warsaw")
EVAL_CURRENT_TIME = datetime(2026, 5, 1, 12, 0, tzinfo=EVAL_TIMEZONE)

_LOCATION_PREP: dict[LocationLabel, str] = {"Warszawa": "w Warszawie"}
_CURRENT_FRAGMENTS: dict[WeatherAttribute, str] = {
    "temperature_c": "aktualną wartość temperatury",
    "wind_speed_ms": "aktualną prędkość wiatru",
    "wind_direction_deg": "aktualny kierunek wiatru",
    "pressure_hpa": "aktualną wartość ciśnienia",
    "humidity_pct": "aktualną wartość wilgotności",
    "precipitation_mm": "aktualną ilość opadów",
}
_FORECAST_FRAGMENTS: dict[WeatherAttribute, str] = {
    "temperature_c": "wartość temperatury",
    "wind_speed_ms": "prędkość wiatru",
    "wind_direction_deg": "kierunek wiatru",
    "pressure_hpa": "wartość ciśnienia",
    "humidity_pct": "wartość wilgotności",
    "precipitation_mm": "ilość opadów",
}
_PERIOD_SUFFIX: dict[PeriodLabel, str] = {
    "teraz": "",
    "jutro": " jutro",
    "za 2 dni": " za 2 dni",
    "3 maja": " 3 maja",
    "w weekend": " w weekend",
    "w Święto Pracy": " w Święto Pracy",
    "za tydzień": " za tydzień",
}
_PERIOD_TARGET_DAYS: dict[PeriodLabel, tuple[int, int, int]] = {
    "teraz": (2026, 5, 1),
    "jutro": (2026, 5, 2),
    "za 2 dni": (2026, 5, 3),
    "3 maja": (2026, 5, 3),
    "w weekend": (2026, 5, 2),
    "w Święto Pracy": (2026, 5, 1),
    "za tydzień": (2026, 5, 8),
}


def deterministic_value(attribute: WeatherAttribute, period: str, hour: int | None = None) -> float:
    lo, hi, step = ATTRIBUTE_RANGES[attribute]
    num_steps = int((hi - lo) / step) + 1
    digest = hashlib.sha256(f"{attribute}:{period}".encode()).digest()
    base_idx = int.from_bytes(digest[:4], "big") % num_steps
    idx = base_idx if hour is None else (base_idx + hour * 7) % num_steps
    return round(lo + idx * step, 1)


def build_question(
    attribute: WeatherAttribute,
    period: PeriodLabel,
    location: LocationLabel,
    hour: int | None = None,
) -> str:
    loc_phrase = _LOCATION_PREP[location]
    if period == "teraz":
        return f"Podaj {_CURRENT_FRAGMENTS[attribute]} {loc_phrase}."
    if hour is None:
        raise ValueError("Forecast eval questions must include an explicit hour.")
    return (
        f"Podaj {_FORECAST_FRAGMENTS[attribute]}{_PERIOD_SUFFIX[period]} "
        f"o {hour:02d}:00 {loc_phrase}."
    )


def _values(period: str, hour: int | None = None) -> dict[WeatherAttribute, float]:
    return {attr: deterministic_value(attr, period, hour) for attr in ATTRIBUTES}


def build_hourly_values(period: str) -> dict[int, dict[WeatherAttribute, float]]:
    return {hour: _values(period, hour) for hour in range(24)}


def _forecast_hour(attribute: WeatherAttribute, period: PeriodLabel) -> int:
    period_idx = PERIODS.index(period)
    attr_idx = ATTRIBUTES.index(attribute)
    return FORECAST_HOURS[(period_idx + attr_idx) % len(FORECAST_HOURS)]


def expected_target_time(period: PeriodLabel, hour: int) -> datetime:
    year, month, day = _PERIOD_TARGET_DAYS[period]
    return datetime(year, month, day, hour, 0, tzinfo=EVAL_TIMEZONE)


def _case(
    idx: int,
    attribute: WeatherAttribute,
    period: PeriodLabel,
    location: LocationLabel,
) -> GeneratedCase:
    if period == "teraz":
        hour = None
        hourly_values = None
        expected_values = _values(period)
    else:
        hour = _forecast_hour(attribute, period)
        hourly_values = build_hourly_values(period)
        expected_values = hourly_values[hour]
    case: GeneratedCase = {
        "id": f"grounding-{idx:03d}",
        "question": build_question(attribute, period, location, hour),
        "current_time": EVAL_CURRENT_TIME.isoformat(),
        "frozen_facts": {
            "location": location,
            "period": period,
            **{str(attr): value for attr, value in expected_values.items()},
        },
        "requested_attributes": [attribute],
    }
    if hourly_values is not None and hour is not None:
        case["hourly_values"] = hourly_values
        case["target_hour"] = hour
        case["expected_target_time"] = expected_target_time(period, hour).isoformat()
    return case


def generate_cases(location: LocationLabel = "Warszawa") -> list[GeneratedCase]:
    args = [(attr, period) for period in PERIODS for attr in ATTRIBUTES]
    return [_case(idx, attr, period, location) for idx, (attr, period) in enumerate(args, 1)]
