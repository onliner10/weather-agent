from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from typing import Any, cast

import altair as alt
import vl_convert as vlc

from weather_agent.domain.weather import ForecastPoint, TimeRange, WeatherVariable

_MAX_DAYS = 7
_DEFAULT_WIDTH = 720
_DEFAULT_HEIGHT = 360
_MAX_WIDTH = 900
_MAX_HEIGHT = 500
_DATASET_NAME = "forecast"
_PNG_HEADER = b"\x89PNG\r\n\x1a\n"
_DISALLOWED_TOP_LEVEL_KEYS = frozenset(
    {"datasets", "facet", "concat", "hconcat", "vconcat", "repeat", "params"}
)
_DISALLOWED_ANYWHERE_KEYS = frozenset({"transform"})


class ForecastChartError(ValueError):
    pass


def forecast_points_to_records(points: list[ForecastPoint]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for point in points:
        record: dict[str, object] = {"time": point.target_time.isoformat()}
        for variable in WeatherVariable:
            value = getattr(point, variable.value)
            if value is not None:
                record[variable.value] = value
        records.append(record)
    return records


def render_forecast_chart_png(
    *,
    spec: dict[str, object],
    records: list[dict[str, object]],
    variables: list[WeatherVariable],
    time_range: TimeRange,
) -> bytes:
    prepared_spec = prepare_vega_lite_spec(
        spec=spec,
        records=records,
        variables=variables,
        time_range=time_range,
    )
    png = vlc.vegalite_to_png(prepared_spec, allowed_base_urls=[])
    if not isinstance(png, bytes) or not png.startswith(_PNG_HEADER):
        raise ForecastChartError("Nie udało się wyrenderować poprawnego obrazu PNG.")
    return png


def prepare_vega_lite_spec(
    *,
    spec: dict[str, object],
    records: list[dict[str, object]],
    variables: list[WeatherVariable],
    time_range: TimeRange,
) -> dict[str, object]:
    _validate_time_range(time_range)
    allowed_fields = {"time", *(variable.value for variable in variables)}
    prepared = deepcopy(spec)
    _validate_shape(prepared)
    _validate_data_sources(prepared)
    _validate_fields(prepared, allowed_fields)
    _normalize_dimensions(prepared)
    prepared["background"] = prepared.get("background", "white")
    prepared["data"] = {"name": _DATASET_NAME}
    prepared["datasets"] = {_DATASET_NAME: records}
    alt.TopLevelSpec.from_dict(prepared, validate=True)
    return prepared


def _validate_time_range(time_range: TimeRange) -> None:
    if time_range.end < time_range.start:
        raise ForecastChartError("start_date nie może być późniejsza niż end_date.")
    if time_range.end - time_range.start > timedelta(days=_MAX_DAYS):
        raise ForecastChartError("Wykres może obejmować maksymalnie 7 dni prognozy.")


def _validate_shape(spec: dict[str, object]) -> None:
    if not isinstance(spec, dict):
        raise ForecastChartError("Specyfikacja Vega-Lite musi być obiektem JSON.")
    disallowed = _DISALLOWED_TOP_LEVEL_KEYS.intersection(spec)
    if disallowed:
        raise ForecastChartError(
            "Nieobsługiwane elementy Vega-Lite: " + ", ".join(sorted(disallowed))
        )
    if "mark" not in spec and "layer" not in spec:
        raise ForecastChartError("Specyfikacja musi zawierać mark albo layer.")
    if "layer" in spec and not isinstance(spec["layer"], list):
        raise ForecastChartError("Pole layer musi być listą.")
    _reject_disallowed_keys(spec)


def _reject_disallowed_keys(value: object) -> None:
    if isinstance(value, dict):
        disallowed = _DISALLOWED_ANYWHERE_KEYS.intersection(value)
        if disallowed:
            raise ForecastChartError(
                "Nieobsługiwane elementy Vega-Lite: " + ", ".join(sorted(disallowed))
            )
        for child in value.values():
            _reject_disallowed_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_disallowed_keys(child)


def _validate_data_sources(spec: dict[str, object]) -> None:
    data_sources = _find_key_values(spec, "data")
    for path, data in data_sources:
        if path != ("data",):
            raise ForecastChartError("Dane mogą być wskazane tylko na top-level specyfikacji.")
        if data != {"name": _DATASET_NAME}:
            raise ForecastChartError('Specyfikacja musi używać data: {"name": "forecast"}.')
    for forbidden_key in ("url", "values", "datasets"):
        if _find_key_values(spec, forbidden_key):
            raise ForecastChartError(f"Specyfikacja nie może zawierać pola {forbidden_key}.")


def _validate_fields(spec: dict[str, object], allowed_fields: set[str]) -> None:
    for _path, field in _find_key_values(spec, "field"):
        if isinstance(field, str) and field not in allowed_fields:
            raise ForecastChartError(f"Nieznane pole danych na wykresie: {field}.")


def _normalize_dimensions(spec: dict[str, object]) -> None:
    spec["width"] = _bounded_dimension(spec.get("width"), _DEFAULT_WIDTH, _MAX_WIDTH)
    spec["height"] = _bounded_dimension(spec.get("height"), _DEFAULT_HEIGHT, _MAX_HEIGHT)


def _bounded_dimension(value: object, default: int, maximum: int) -> int:
    if isinstance(value, int | float):
        return max(1, min(int(value), maximum))
    return default


def _find_key_values(value: object, key: str) -> list[tuple[tuple[str, ...], object]]:
    found: list[tuple[tuple[str, ...], object]] = []

    def visit(current: object, path: tuple[str, ...]) -> None:
        if isinstance(current, dict):
            for child_key, child_value in current.items():
                child_path = (*path, str(child_key))
                if child_key == key:
                    found.append((child_path, child_value))
                visit(child_value, child_path)
        elif isinstance(current, list):
            for index, child in enumerate(current):
                visit(child, (*path, str(index)))

    visit(cast(dict[str, Any], value), ())
    return found
