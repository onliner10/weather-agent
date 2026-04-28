from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from weather_agent.domain.weather import ForecastPoint, WeatherVariable


class SnapshotDelta(BaseModel):
    """Delta between two forecast points at the same target time."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    previous_value: float | None
    current_value: float | None
    delta: float | None
    abs_delta: float | None
    percent_change: float | None


class SnapshotComparison(BaseModel):
    """Comparison result between two forecast snapshots."""

    model_config = ConfigDict(extra="forbid")

    location_id: str
    previous_fetched_at: datetime
    current_fetched_at: datetime
    deltas: list[SnapshotDelta]


_NUMERIC_FIELDS: list[str] = [
    v.value for v in WeatherVariable if v != WeatherVariable.weather_code
]


def compute_delta(
    previous_value: float | None,
    current_value: float | None,
) -> tuple[float | None, float | None, float | None]:
    if previous_value is None or current_value is None:
        return None, None, None

    delta = current_value - previous_value
    abs_delta = abs(delta)

    if previous_value == 0:
        percent_change = None
    else:
        percent_change = (delta / previous_value) * 100.0

    return delta, abs_delta, percent_change


def compare_snapshots(
    current: list[ForecastPoint],
    previous: list[ForecastPoint],
) -> SnapshotComparison:
    previous_by_time: dict[datetime, ForecastPoint] = {
        p.target_time: p for p in previous
    }

    all_deltas: list[SnapshotDelta] = []

    for current_point in current:
        previous_point = previous_by_time.get(current_point.target_time)
        if previous_point is None:
            continue

        for metric in _NUMERIC_FIELDS:
            prev_val = getattr(previous_point, metric)
            curr_val = getattr(current_point, metric)

            delta, abs_delta, percent_change = compute_delta(prev_val, curr_val)

            all_deltas.append(
                SnapshotDelta(
                    metric=metric,
                    previous_value=prev_val,
                    current_value=curr_val,
                    delta=delta,
                    abs_delta=abs_delta,
                    percent_change=percent_change,
                )
            )

    current_fetched_at = current[0].fetched_at if current else datetime.now()
    previous_fetched_at = previous[0].fetched_at if previous else datetime.now()
    location_id = current[0].location_id if current else "unknown"

    return SnapshotComparison(
        location_id=location_id,
        previous_fetched_at=previous_fetched_at,
        current_fetched_at=current_fetched_at,
        deltas=all_deltas,
    )


def detect_significant_changes(
    comparison: SnapshotComparison,
    thresholds: dict[str, float],
) -> list[SnapshotDelta]:
    significant: list[SnapshotDelta] = []

    for delta in comparison.deltas:
        threshold = thresholds.get(delta.metric)
        if threshold is None:
            continue
        if delta.abs_delta is None:
            continue
        if delta.abs_delta > threshold:
            significant.append(delta)

    return significant