from __future__ import annotations

import re
import unicodedata
from typing import Any, cast

from weather_agent.eval.schemas import (
    WEATHER_ATTRIBUTES,
    WeatherAttribute,
    WeatherFacts,
    WeatherPresentationOutput,
    WeatherToolCallRecord,
)

_TEMP_RE = re.compile(r"(?P<value>(?<!\d)-?\d+(?:[,.]\d+)?)\s*(?:°\s*)?c\b", re.IGNORECASE)
_TEMP_RANGE_RE = re.compile(
    r"(?P<low>-?\d+(?:[,.]\d+)?)\s*(?:-|–|do)\s*"
    r"(?P<high>-?\d+(?:[,.]\d+)?)\s*(?:°\s*)?c\b",
    re.IGNORECASE,
)
_MM_RE = re.compile(r"(?P<value>(?<!\d)\d+(?:[,.]\d+)?)\s*mm\b", re.IGNORECASE)
_PCT_RE = re.compile(r"(?P<value>(?<!\d)\d+(?:[,.]\d+)?)\s*%", re.IGNORECASE)
_MS_RE = re.compile(r"(?P<value>(?<!\d)\d+(?:[,.]\d+)?)\s*m/s\b", re.IGNORECASE)
_HPA_RE = re.compile(r"(?P<value>(?<!\d)\d+(?:[,.]\d+)?)\s*hpa\b", re.IGNORECASE)
_DEG_RE = re.compile(r"(?P<value>(?<!\d)\d+(?:[,.]\d+)?)\s*(?:°|deg\b)", re.IGNORECASE)


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _extract_numbers(pattern: re.Pattern[str], text: str) -> list[float]:
    values: list[float] = []
    for match in pattern.finditer(text):
        values.append(float(match.group("value").replace(",", ".")))
    return values


def _extract_temperatures(text: str) -> list[float]:
    values = _extract_numbers(_TEMP_RE, text)
    for match in _TEMP_RANGE_RE.finditer(text):
        values.append(float(match.group("low").replace(",", ".")))
        values.append(float(match.group("high").replace(",", ".")))
    # Preserve deterministic output while removing duplicates from overlapping regexes.
    return list(dict.fromkeys(values))


def _mentions_location(text: str, location: str) -> bool:
    normalized_location = _normalize(location)
    # Polish location names are often inflected in answers, e.g. Chwarzno ->
    # Chwarznie. A short stem is enough for the curated eval location names.
    stem_len = max(5, min(7, len(normalized_location) - 1))
    location_stem = normalized_location[:stem_len]
    return normalized_location in text or location_stem in text


def _attribute_values(attribute: WeatherAttribute, text: str) -> list[float]:
    match attribute:
        case "temperature_c":
            return _extract_temperatures(text)
        case "precipitation_mm":
            return _extract_numbers(_MM_RE, text)
        case "humidity_pct":
            return _extract_numbers(_PCT_RE, text)
        case "wind_speed_ms":
            return _extract_numbers(_MS_RE, text)
        case "pressure_hpa":
            return _extract_numbers(_HPA_RE, text)
        case "wind_direction_deg":
            return _extract_numbers(_DEG_RE, text)


_FLOAT_EPSILON = 1e-9


def _attribute_expected_value(
    attribute: WeatherAttribute,
    facts: WeatherFacts,
) -> float | None:
    match attribute:
        case "temperature_c":
            return facts.temperature_c
        case "precipitation_mm":
            return facts.precipitation_mm
        case "humidity_pct":
            return facts.humidity_pct
        case "wind_speed_ms":
            return facts.wind_speed_ms
        case "pressure_hpa":
            return facts.pressure_hpa
        case "wind_direction_deg":
            return facts.wind_direction_deg
    return None


def _attribute_value_matches_expected(
    attribute: WeatherAttribute,
    text: str,
    facts: WeatherFacts,
) -> bool:
    values = _attribute_values(attribute, text)
    expected = _attribute_expected_value(attribute, facts)
    if expected is None:
        return False
    return any(abs(value - expected) <= _FLOAT_EPSILON for value in values)


def weather_functional_correctness(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    """Deterministically check that an answer contains required weather facts.

    The first production-impact eval is intentionally narrow: it verifies that
    the real DeepAgent path mentions the expected location and explicitly
    requested ObservationPoint attribute values with matching units.
    """
    answer = str(outputs.get("answer", ""))
    text = _normalize(answer)
    facts = WeatherFacts.model_validate(reference_outputs["expected_facts"])
    required_location = bool(reference_outputs.get("required_location", True))
    failures: list[str] = []
    requested_attributes: list[WeatherAttribute] = []
    for raw_attribute in reference_outputs.get("requested_attributes", []):
        attribute = str(raw_attribute)
        if attribute not in WEATHER_ATTRIBUTES:
            failures.append(f"unknown_requested_attribute:{attribute}")
            continue
        requested_attributes.append(cast(WeatherAttribute, attribute))

    if required_location and not _mentions_location(text, facts.location):
        failures.append(f"missing_location:{facts.location}")

    for attribute in requested_attributes:
        values = _attribute_values(attribute, text)
        if not values:
            failures.append(f"missing_attribute_value:{attribute}")
            continue
        if not _attribute_value_matches_expected(attribute, text, facts):
            rendered_values = ",".join(f"{value:g}" for value in values)
            failures.append(f"attribute_value_mismatch:{attribute}:{rendered_values}")

    return {
        "key": "weather_functional_correctness",
        "score": 0.0 if failures else 1.0,
        "comment": "ok" if not failures else ";".join(failures),
    }


def weather_presentation_tool_use(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    run_output = WeatherPresentationOutput.model_validate(outputs)
    expect_chart = reference_outputs.get("expect_chart")
    expected_chart_variables = _string_set(reference_outputs.get("expected_chart_variables"))
    expected_chart_start_date = _optional_string(reference_outputs.get("expected_chart_start_date"))
    expected_chart_end_date = _optional_string(reference_outputs.get("expected_chart_end_date"))
    chart_calls = [call for call in run_output.tool_calls if call.name == "render_forecast_chart"]
    successful_chart_calls = [call for call in chart_calls if call.result_error is None]
    failures: list[str] = []

    if expect_chart is True:
        if not chart_calls:
            failures.append("missing_render_forecast_chart_call")
        if run_output.attachment_count < 1:
            failures.append("missing_chart_attachment")
    elif expect_chart is False:
        if chart_calls:
            failures.append("unexpected_render_forecast_chart_call")
    if chart_calls and run_output.attachment_count < 1:
        failures.append("chart_call_without_attachment")
    if len(successful_chart_calls) > 1:
        failures.append(
            f"repeated_successful_render_forecast_chart_calls:{len(successful_chart_calls)}"
        )
    if expected_chart_variables and successful_chart_calls:
        if not any(
            expected_chart_variables <= _chart_call_variables(call)
            for call in successful_chart_calls
        ):
            failures.append(
                "missing_expected_chart_variables:" + ",".join(sorted(expected_chart_variables))
            )
        successful_chart_calls_with_spec = [
            call
            for call in successful_chart_calls
            if isinstance(call.args.get("vega_lite_spec"), dict)
        ]
        if not any(
            expected_chart_variables <= _chart_call_spec_fields(call)
            for call in successful_chart_calls_with_spec
        ):
            if successful_chart_calls_with_spec:
                failures.append(
                    "missing_expected_chart_fields:" + ",".join(sorted(expected_chart_variables))
                )
    if expected_chart_start_date and successful_chart_calls:
        if not any(
            call.args.get("start_date") == expected_chart_start_date
            for call in successful_chart_calls
        ):
            failures.append(f"chart_start_date_mismatch:{expected_chart_start_date}")
    if expected_chart_end_date and successful_chart_calls:
        if not any(
            call.args.get("end_date") == expected_chart_end_date for call in successful_chart_calls
        ):
            failures.append(f"chart_end_date_mismatch:{expected_chart_end_date}")

    return {
        "key": "weather_presentation_tool_use",
        "score": 0.0 if failures else 1.0,
        "comment": "ok" if not failures else ";".join(failures),
    }


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _string_set(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str)}


def _chart_call_variables(call: WeatherToolCallRecord) -> set[str]:
    variables = call.args.get("variables")
    if not isinstance(variables, list):
        return set()
    return {variable for variable in variables if isinstance(variable, str)}


def _chart_call_spec_fields(
    call: WeatherToolCallRecord,
) -> set[str]:
    return _find_field_names(call.args.get("vega_lite_spec"))


def _find_field_names(value: object) -> set[str]:
    fields: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "field" and isinstance(child, str):
                fields.add(child)
            else:
                fields.update(_find_field_names(child))
    elif isinstance(value, list):
        for child in value:
            fields.update(_find_field_names(child))
    return fields
