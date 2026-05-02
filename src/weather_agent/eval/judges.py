from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from weather_agent.eval.schemas import WeatherFacts

WEATHER_GROUNDEDNESS_JUDGE_PROMPT_VERSION = "weather_groundedness_judge_v2"


class WeatherGroundednessJudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)


def _facts_for_prompt(facts: WeatherFacts) -> dict[str, object]:
    return facts.model_dump(exclude_none=True)


def _extract_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()

    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        decoded = json.loads(stripped[start : end + 1])

    if not isinstance(decoded, dict):
        raise ValueError("Judge response must be a JSON object.")
    return decoded


def _parse_groundedness_judge_result(content: str) -> WeatherGroundednessJudgeResult:
    return WeatherGroundednessJudgeResult.model_validate(_extract_json_object(content))


def _build_groundedness_judge_messages(
    *,
    answer: str,
    question: str,
    facts: WeatherFacts,
    current_time: object,
    expected_target_time: object,
    target_hour: object,
    required_location: bool,
    requested_attributes: list[object],
) -> list[SystemMessage | HumanMessage]:
    facts_json = json.dumps(_facts_for_prompt(facts), ensure_ascii=False, sort_keys=True)
    requested_json = json.dumps(requested_attributes, ensure_ascii=False)
    timing_context_json = json.dumps(
        {
            "current_time": current_time,
            "expected_target_time": expected_target_time,
            "target_hour": target_hour,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return [
        SystemMessage(
            content=(
                "You are an evaluator for a Polish Telegram weather assistant. "
                "Judge only whether the final answer is grounded in the provided frozen weather "
                "facts, user question, and eval timing context. Dates, times, weekdays, and "
                "Polish restatements of time are supported when they are derived from the user "
                "question, current_time, target_hour, or expected_target_time. Do not penalize "
                "normal Polish phrasing around a requested date or hour. Do penalize wrong or "
                "contradictory date/time claims, unsupported weather interpretation, invented "
                "weather values, wrong location, wrong period, or misleading wind-direction "
                "wording. "
                "Return only JSON with keys score and reason. Use score 1.0 for fully grounded, "
                "0.5 for mostly grounded with minor unsupported or incomplete claims, and 0.0 for "
                "invented, contradictory, wrong-location, wrong-period, or ungrounded answers."
            )
        ),
        HumanMessage(
            content=(
                f"Judge prompt version: {WEATHER_GROUNDEDNESS_JUDGE_PROMPT_VERSION}\n"
                f"User question: {question}\n"
                f"Frozen weather facts JSON: {facts_json}\n"
                f"Eval timing context JSON: {timing_context_json}\n"
                f"Required location mention: {required_location}\n"
                f"Requested attributes JSON: {requested_json}\n"
                f"Final assistant answer: {answer}\n\n"
                'Return JSON exactly like: {"score": 1.0, "reason": "short reason"}'
            )
        ),
    ]


def build_weather_groundedness_judge(
    model_factory: Callable[[], Any],
) -> Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]:
    async def weather_groundedness_judge(
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: dict[str, Any],
    ) -> dict[str, Any]:
        facts = WeatherFacts.model_validate(reference_outputs["expected_facts"])
        answer = str(outputs.get("answer", ""))
        question = str(inputs.get("question", ""))
        messages = _build_groundedness_judge_messages(
            answer=answer,
            question=question,
            facts=facts,
            current_time=inputs.get("current_time"),
            expected_target_time=inputs.get("expected_target_time"),
            target_hour=inputs.get("target_hour"),
            required_location=bool(reference_outputs.get("required_location", True)),
            requested_attributes=list(reference_outputs.get("requested_attributes", [])),
        )

        try:
            response = await model_factory().ainvoke(messages)
            result = _parse_groundedness_judge_result(str(response.content))
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            return {
                "key": "weather_answer_groundedness_judge",
                "score": 0.0,
                "comment": f"judge_parse_error:{exc}",
                "metadata": {"judge_prompt_version": WEATHER_GROUNDEDNESS_JUDGE_PROMPT_VERSION},
            }

        return {
            "key": "weather_answer_groundedness_judge",
            "score": result.score,
            "comment": result.reason,
            "metadata": {"judge_prompt_version": WEATHER_GROUNDEDNESS_JUDGE_PROMPT_VERSION},
        }

    return weather_groundedness_judge
