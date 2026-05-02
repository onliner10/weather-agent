from __future__ import annotations

import unicodedata
from typing import Any

from weather_agent.eval.location_management_schemas import (
    ExpectedLocationAction,
    LocationManagementEvalOutput,
    LocationToolCallRecord,
)

_POLISH_RESPONSE_MARKERS = frozenset(
    {
        "deszcz",
        "lokalizac",
        "pada",
        "podaj",
        "pogod",
        "potwierd",
        "prognoz",
        "usun",
        "usunie",
        "usunieto",
        "zapamieta",
        "zaktualiz",
    }
)
_MANAGED_TOOLS = frozenset(
    {
        "save_location",
        "edit_location",
        "remove_location",
        "get_forecast",
        "get_observations",
        "propose_notification_rule",
        "schedule_notification",
        "delete_location",
    }
)


def _normalize(text: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(text).casefold())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).strip()


def _value_matches(candidate: object, expected: str) -> bool:
    candidate_norm = _normalize(candidate)
    expected_norm = _normalize(expected)
    return candidate_norm == expected_norm or expected_norm in candidate_norm


def _has_polish_location_response(answer: str) -> bool:
    text = _normalize(answer)
    return any(marker in text for marker in _POLISH_RESPONSE_MARKERS)


def _terminal_managed_call(calls: list[LocationToolCallRecord]) -> LocationToolCallRecord | None:
    managed = [call for call in calls if call.name in _MANAGED_TOOLS]
    return managed[-1] if managed else None


def _validate_location_arg(
    *,
    call: LocationToolCallRecord,
    expected: ExpectedLocationAction,
) -> list[str]:
    if expected.expected_location is None:
        return []

    location_name = call.args.get("location_name", "")
    if expected.expect_default_location and expected.allow_default_alias:
        allowed = {"", "dom", "home", expected.expected_location}
        if any(_value_matches(location_name, item) for item in allowed):
            return []

    if _value_matches(location_name, expected.expected_location):
        return []
    return [f"location_mismatch:expected={expected.expected_location}:actual={location_name}"]


def _arg_candidates(call: LocationToolCallRecord, field: str) -> list[object]:
    if field != "alias":
        return [call.args.get(field, "")]
    alias = call.args.get("alias", "")
    aliases = call.args.get("aliases", [])
    candidates: list[object] = [alias]
    if isinstance(aliases, list):
        candidates.extend(aliases)
    return candidates


def _validate_expected_arg(
    *,
    call: LocationToolCallRecord,
    expected_value: str | None,
    expected_attr: str,
    arg_name: str,
    failure_name: str,
) -> list[str]:
    if expected_value is None:
        return []
    candidates = _arg_candidates(call, arg_name)
    if any(_value_matches(candidate, expected_value) for candidate in candidates):
        return []
    actual = call.args.get(arg_name, call.args.get(expected_attr, ""))
    return [f"{failure_name}:expected={expected_value}:actual={actual}"]


def location_management_fidelity(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    run_output = LocationManagementEvalOutput.model_validate(outputs)
    expected = ExpectedLocationAction.model_validate(reference_outputs["expected"])
    failures: list[str] = []
    managed_calls = [call for call in run_output.tool_calls if call.name in _MANAGED_TOOLS]

    if any(call.name == "delete_location" for call in run_output.tool_calls):
        failures.append("destructive_delete_tool_call")
    for call in run_output.tool_calls:
        if call.name == "remove_location" and call.args.get("hard_delete") is True:
            failures.append("destructive_remove_requested")

    if expected.expect_no_tool_call:
        if managed_calls:
            failures.append(f"unexpected_tool_call:{managed_calls[-1].name}")
    else:
        terminal_call = _terminal_managed_call(run_output.tool_calls)
        if terminal_call is None:
            failures.append("missing_tool_call")
        elif terminal_call.name != expected.expected_tool:
            failures.append(
                f"terminal_tool_mismatch:expected={expected.expected_tool}:"
                f"actual={terminal_call.name}"
            )
        elif terminal_call.result_error:
            failures.append(f"terminal_tool_error:{terminal_call.result_error}")
        if terminal_call is not None and terminal_call.name == expected.expected_tool:
            failures.extend(_validate_location_arg(call=terminal_call, expected=expected))
            for expected_value, expected_attr, arg_name, failure_name in (
                (expected.expected_alias, "expected_alias", "alias", "alias_mismatch"),
                (
                    expected.expected_new_name,
                    "expected_new_name",
                    "new_name",
                    "new_name_mismatch",
                ),
            ):
                failures.extend(
                    _validate_expected_arg(
                        call=terminal_call,
                        expected_value=expected_value,
                        expected_attr=expected_attr,
                        arg_name=arg_name,
                        failure_name=failure_name,
                    )
                )

    if not _has_polish_location_response(run_output.answer):
        failures.append("missing_polish_location_response")

    return {
        "key": "location_management_fidelity",
        "score": 0.0 if failures else 1.0,
        "comment": "ok" if not failures else ";".join(failures),
    }
