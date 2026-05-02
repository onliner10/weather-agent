from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, tzinfo
from typing import Any
from unittest.mock import patch
from zoneinfo import ZoneInfo

import weather_agent.domain.rule_expression.evaluator as rule_expression_module
from weather_agent.domain.rule_expression.evaluator import RuleExpressionEvaluator
from weather_agent.eval.notification_rule_schemas import (
    ExpectedRuleProposal,
    RuleProposalEvalOutput,
    RuleToolCallRecord,
)

_WARSAW = ZoneInfo("Europe/Warsaw")
_CONFIRMATION_RE = re.compile(
    r"(potwierd|tak/nie|czy chcesz|zatwierd|akcept)",
    re.IGNORECASE,
)
_FORBIDDEN_TOOLS = frozenset({"confirm_pending_action", "cancel_pending_action"})


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).strip()


def _location_matches(candidate: object, expected: str) -> bool:
    candidate_text = _normalize_text(str(candidate))
    expected_text = _normalize_text(expected)
    return candidate_text == expected_text or expected_text in candidate_text


def _result_bool(result: object) -> bool | None:
    if isinstance(result, bool):
        return result
    return None


def _parse_when(raw: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_WARSAW)
    return parsed.astimezone(_WARSAW)


@contextmanager
def _frozen_rule_expression_now(now: datetime) -> Iterator[None]:
    effective_now = now.astimezone(_WARSAW)

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:  # type: ignore[override]
            if tz is None:
                return effective_now.replace(tzinfo=None)
            return effective_now.astimezone(tz)

    with patch.object(rule_expression_module, "datetime", FrozenDateTime):
        yield


def _evaluate_rule_expression(
    expression: str, points: list[dict[str, object]], now: datetime
) -> bool | None:
    with _frozen_rule_expression_now(now):
        result = RuleExpressionEvaluator({"points": points}).evaluate(expression)
    if not result.valid:
        return None
    return _result_bool(result.result)


def _validate_rule_expression(
    *,
    candidate_expression: str,
    expected: ExpectedRuleProposal,
    now: datetime,
) -> list[str]:
    failures: list[str] = []
    candidate_validation = RuleExpressionEvaluator().validate(candidate_expression)
    expected_validation = RuleExpressionEvaluator().validate(expected.expected_rule_expression)
    if not candidate_validation.valid:
        return [f"invalid_rule_expression:{candidate_validation.error}"]
    if not expected_validation.valid:
        return [f"invalid_reference_rule_expression:{expected_validation.error}"]

    candidate_metrics = set(candidate_validation.evaluated_metrics)
    expected_metrics = set(expected_validation.evaluated_metrics)
    if candidate_metrics != expected_metrics:
        failures.append(
            "metric_mismatch:"
            f"expected={','.join(sorted(expected_metrics))}:"
            f"actual={','.join(sorted(candidate_metrics))}"
        )

    for profile in expected.rule_expression_discriminators:
        actual = _evaluate_rule_expression(candidate_expression, profile.points, now)
        if actual is None:
            failures.append(f"rule_expression_profile_error:{profile.name}")
            continue
        if actual != profile.expected_result:
            failures.append(
                f"rule_expression_profile_mismatch:{profile.name}:"
                f"expected={profile.expected_result}:actual={actual}"
            )

    return failures


def _validate_schedule(
    *,
    terminal_call: RuleToolCallRecord,
    expected: ExpectedRuleProposal,
) -> list[str]:
    failures: list[str] = []
    args = terminal_call.args
    if args.get("schedule_type") != expected.expected_schedule_type:
        failures.append(
            "schedule_type_mismatch:"
            f"expected={expected.expected_schedule_type}:actual={args.get('schedule_type')}"
        )
        return failures

    actual_expr = str(args.get("schedule_expression", ""))
    expected_expr = expected.expected_schedule_expression
    if expected.expected_schedule_type == "cron":
        if actual_expr.strip() != expected_expr:
            failures.append(f"cron_mismatch:expected={expected_expr}:actual={actual_expr.strip()}")
    elif expected.expected_schedule_type == "once" and expected_expr is not None:
        actual_dt = _parse_when(actual_expr)
        expected_dt = _parse_when(expected_expr)
        if actual_dt is None or expected_dt is None or actual_dt != expected_dt:
            failures.append(f"once_mismatch:expected={expected_expr}:actual={actual_expr}")
    return failures


def notification_rule_proposal_fidelity(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    """Score safe notification rule proposals from recorded tool calls."""
    run_output = RuleProposalEvalOutput.model_validate(outputs)
    expected = ExpectedRuleProposal.model_validate(reference_outputs["expected"])
    now = datetime.fromisoformat(str(reference_outputs["current_time"]))
    failures: list[str] = []

    tool_calls = run_output.tool_calls
    if not tool_calls:
        failures.append("missing_tool_call")
    forbidden = [call.name for call in tool_calls if call.name in _FORBIDDEN_TOOLS]
    if forbidden:
        failures.append(f"forbidden_tool_call:{','.join(forbidden)}")

    terminal_call = tool_calls[-1] if tool_calls else None
    if terminal_call is not None:
        if terminal_call.name != expected.expected_tool:
            failures.append(
                f"terminal_tool_mismatch:expected={expected.expected_tool}:actual={terminal_call.name}"
            )
        if terminal_call.result_error:
            failures.append(f"terminal_tool_error:{terminal_call.result_error}")
        if terminal_call.result_pending is not True:
            failures.append(f"terminal_not_pending:{terminal_call.result_pending}")

    if terminal_call is not None and terminal_call.name == expected.expected_tool:
        args = terminal_call.args
        location_name = args.get("location_name", "")
        if not _location_matches(location_name, expected.expected_location):
            failures.append(
                f"location_mismatch:expected={expected.expected_location}:actual={location_name}"
            )

        candidate_rule_expression = str(args.get("rule_expression", ""))
        failures.extend(
            _validate_rule_expression(
                candidate_expression=candidate_rule_expression,
                expected=expected,
                now=now,
            )
        )
        if expected.expected_tool == "schedule_notification":
            failures.extend(_validate_schedule(terminal_call=terminal_call, expected=expected))

    if not _CONFIRMATION_RE.search(_normalize_text(run_output.answer)):
        failures.append("missing_confirmation_surface")

    return {
        "key": "notification_rule_proposal_fidelity",
        "score": 0.0 if failures else 1.0,
        "comment": "ok" if not failures else ";".join(failures),
    }
