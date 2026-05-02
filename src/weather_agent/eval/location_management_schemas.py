from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

ExpectedLocationTool = Literal[
    "save_location",
    "edit_location",
    "remove_location",
    "get_forecast",
    "get_observations",
    "propose_notification_rule",
    "ask_for_location",
]


class ExpectedLocationAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_tool: ExpectedLocationTool
    expected_location: str | None = None
    expected_alias: str | None = None
    expected_new_name: str | None = None
    allow_default_alias: bool = False
    expect_default_location: bool = False
    expect_no_tool_call: bool = False


class SeedLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    aliases: list[str]
    latitude: float
    longitude: float
    enabled: bool = True


class LocationManagementEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    current_time: datetime
    expected: ExpectedLocationAction
    seed_locations: list[SeedLocation]
    note: str | None = None


class LocationToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    args: dict[str, object]
    result_error: str | None = None
    result_success: str | None = None


class LocationManagementEvalOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    answer: str
    tool_calls: list[LocationToolCallRecord]
