from __future__ import annotations

from datetime import timedelta
from typing import TypeGuard
from zoneinfo import ZoneInfo

from weather_agent.domain.weather import TimeRange

_CHART_TIMEZONE = ZoneInfo("Europe/Warsaw")
_MULTI_DAY_TIME_AXIS_LABEL_ANGLE = -35


def normalize_time_axes(spec: dict[str, object], time_range: TimeRange) -> None:
    axis_defaults = _time_axis_defaults(time_range)

    def visit(value: object) -> None:
        if isinstance(value, dict):
            encoding = value.get("encoding")
            if isinstance(encoding, dict):
                x = encoding.get("x")
                if _is_time_x_encoding(x):
                    axis = x.get("axis")
                    if isinstance(axis, dict):
                        for key, axis_value in axis_defaults.items():
                            axis.setdefault(key, axis_value)
                    elif axis is None:
                        x["axis"] = dict(axis_defaults)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(spec)


def _is_time_x_encoding(value: object) -> TypeGuard[dict[str, object]]:
    return (
        isinstance(value, dict) and value.get("field") == "time" and value.get("type") == "temporal"
    )


def _time_axis_defaults(time_range: TimeRange) -> dict[str, object]:
    interval_hours: int
    duration = time_range.end - time_range.start
    if duration <= timedelta(hours=12):
        interval_hours = 1
        label_format = "%H:%M"
        label_angle = 0
    elif duration <= timedelta(days=1):
        interval_hours = 2
        label_format = "%H:%M"
        label_angle = 0
    elif duration <= timedelta(days=2):
        interval_hours = 3
        label_format = "%d.%m %H:%M"
        label_angle = _MULTI_DAY_TIME_AXIS_LABEL_ANGLE
    elif duration <= timedelta(days=3):
        interval_hours = 6
        label_format = "%d.%m %H:%M"
        label_angle = _MULTI_DAY_TIME_AXIS_LABEL_ANGLE
    else:
        interval_hours = 12
        label_format = "%d.%m %H:%M"
        label_angle = _MULTI_DAY_TIME_AXIS_LABEL_ANGLE

    return {
        "format": label_format,
        "labelAngle": label_angle,
        "labelBound": True,
        "labelFlush": True,
        "labelOverlap": "greedy",
        "labelPadding": 4,
        "values": _time_axis_values(time_range, interval_hours=interval_hours),
    }


def _time_axis_values(time_range: TimeRange, *, interval_hours: int) -> list[str]:
    current = time_range.start.astimezone(_CHART_TIMEZONE).replace(
        minute=0,
        second=0,
        microsecond=0,
    )
    end = time_range.end.astimezone(_CHART_TIMEZONE)
    if current < time_range.start.astimezone(_CHART_TIMEZONE):
        current += timedelta(hours=1)

    values: list[str] = []
    while current <= end:
        values.append(current.replace(tzinfo=None).isoformat())
        current += timedelta(hours=interval_hours)
    return values
