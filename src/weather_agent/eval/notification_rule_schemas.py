from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

ExpectedRuleTool = Literal["propose_notification_rule", "schedule_notification"]
ExpectedScheduleType = Literal["once", "cron"]


class RuleExpressionDiscriminatorProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    points: list[dict[str, object]]
    expected_result: bool


class ExpectedRuleProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_tool: ExpectedRuleTool
    expected_rule_expression: str
    expected_location: str
    expected_schedule_type: ExpectedScheduleType | None = None
    expected_schedule_expression: str | None = None
    rule_expression_discriminators: list[RuleExpressionDiscriminatorProfile]


class NotificationRuleEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    current_time: datetime
    expected: ExpectedRuleProposal
    note: str | None = None


class RuleToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    args: dict[str, object]
    result_pending: bool | None = None
    result_error: str | None = None


class RuleProposalEvalOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    answer: str
    tool_calls: list[RuleToolCallRecord]
