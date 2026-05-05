from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool

from weather_agent.agent_factory import build_current_time_prompt_suffix, create_weather_agent
from weather_agent.domain.rule_expression.allowlist import get_allowlist_for_prompt
from weather_agent.domain.rule_expression.evaluator import RuleExpressionEvaluator
from weather_agent.domain.rules.schedule import parse_schedule
from weather_agent.eval.notification_rule_schemas import (
    RuleProposalEvalOutput,
    RuleToolCallRecord,
)
from weather_agent.llm.tools.rules_tools import (
    CRON_FIXED_DATE_RANGE_ERROR,
    CancelPendingActionArgs,
    ConfirmPendingActionArgs,
    GetRuleExpressionCapabilitiesArgs,
    ListNotificationRulesArgs,
    ProposeNotificationRuleArgs,
    ScheduleNotificationArgs,
    ToolResult,
    cron_schedule_uses_fixed_date_range,
)
from weather_agent.observability.logging import get_logger

logger = get_logger(__name__)


class RecordingRulesToolbox:
    def __init__(self) -> None:
        self.tool_calls: list[RuleToolCallRecord] = []
        self._rule_expression_evaluator = RuleExpressionEvaluator()

    def _record(
        self,
        *,
        name: str,
        args: dict[str, object],
        result: object,
        pending: bool | None = None,
        error: str | None = None,
    ) -> object:
        self.tool_calls.append(
            RuleToolCallRecord(
                name=name,
                args=args,
                result_pending=pending,
                result_error=error,
            )
        )
        return result

    async def list_notification_rules(self, include_disabled: bool = False) -> ToolResult:
        result: ToolResult = {"rules": [], "count": 0}
        return cast(
            ToolResult,
            self._record(
                name="list_notification_rules",
                args={"include_disabled": include_disabled},
                result=result,
                error=cast(str | None, result.get("error")),
            ),
        )

    async def get_rule_expression_capabilities(self) -> ToolResult:
        allowlist = get_allowlist_for_prompt()
        result: ToolResult = {
            "functions": cast(dict[str, list[str]], allowlist["functions"]),
            "metrics": cast(list[str], allowlist["metrics"]),
            "signatures": cast(dict[str, str], allowlist["signatures"]),
            "rules": cast(list[str], allowlist["rules"]),
            "examples": cast(list[str], allowlist["examples"]),
        }
        return cast(
            ToolResult,
            self._record(
                name="get_rule_expression_capabilities",
                args={},
                result=result,
                error=cast(str | None, result.get("error")),
            ),
        )

    async def propose_notification_rule(
        self,
        rule_expression: str,
        explanation: str,
        location_name: str = "",
        edit_short_id: str = "",
    ) -> ToolResult:
        validation = self._rule_expression_evaluator.validate(rule_expression)
        if not validation.valid:
            result: ToolResult = {
                "rule_expression": rule_expression,
                "explanation": explanation,
                "error": f"Nieprawidłowe wyrażenie reguły: {validation.error}",
            }
        else:
            result = {
                "proposal": (
                    "Propozycja nowej reguły:\n\n"
                    f"Wyrażenie reguły: `{validation.expression}`\n"
                    f"Opis: {explanation}\n\n"
                    "Czy chcesz potwierdzić? (tak/nie)"
                ),
                "rule_expression": validation.expression,
                "explanation": explanation,
                "validated": True,
                "pending": True,
            }
        return cast(
            ToolResult,
            self._record(
                name="propose_notification_rule",
                args={
                    "rule_expression": rule_expression,
                    "explanation": explanation,
                    "location_name": location_name,
                    "edit_short_id": edit_short_id,
                },
                result=result,
                pending=cast(bool | None, result.get("pending")),
                error=cast(str | None, result.get("error")),
            ),
        )

    async def confirm_pending_action(self) -> ToolResult:
        result: ToolResult = {
            "error": "Eval fixture: confirm_pending_action is not allowed during proposal scoring."
        }
        return cast(
            ToolResult,
            self._record(
                name="confirm_pending_action",
                args={},
                result=result,
                error=cast(str | None, result.get("error")),
            ),
        )

    async def cancel_pending_action(self) -> ToolResult:
        result: ToolResult = {
            "error": "Eval fixture: cancel_pending_action is not allowed during proposal scoring."
        }
        return cast(
            ToolResult,
            self._record(
                name="cancel_pending_action",
                args={},
                result=result,
                error=cast(str | None, result.get("error")),
            ),
        )

    async def schedule_notification(
        self,
        schedule_type: str,
        schedule_expression: str,
        explanation: str,
        location_name: str = "",
        rule_expression: str = "true",
    ) -> ToolResult:
        validation = self._rule_expression_evaluator.validate(rule_expression)
        if not validation.valid:
            result: ToolResult = {"error": f"Nieprawidłowe wyrażenie reguły: {validation.error}"}
        else:
            parsed = parse_schedule(f"{schedule_type}:{schedule_expression}")
            if not parsed.valid:
                result = {"error": f"Nieprawidłowy harmonogram: {parsed.error}"}
            elif cron_schedule_uses_fixed_date_range(schedule_type, validation.expression):
                result = {"error": CRON_FIXED_DATE_RANGE_ERROR}
            else:
                result = {
                    "proposal": (
                        "Propozycja zaplanowanego powiadomienia:\n\n"
                        f"Harmonogram: {schedule_type}:{schedule_expression}\n"
                        f"Wyrażenie reguły: `{validation.expression}`\n"
                        f"Opis: {explanation}\n\n"
                        "Czy chcesz potwierdzić? (tak/nie)"
                    ),
                    "pending": True,
                }

        return cast(
            ToolResult,
            self._record(
                name="schedule_notification",
                args={
                    "schedule_type": schedule_type,
                    "schedule_expression": schedule_expression,
                    "explanation": explanation,
                    "location_name": location_name,
                    "rule_expression": rule_expression,
                },
                result=result,
                pending=cast(bool | None, result.get("pending")),
                error=cast(str | None, result.get("error")),
            ),
        )

    def to_langchain_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                coroutine=self.list_notification_rules,
                name="list_notification_rules",
                description="Wyświetl reguły powiadomień użytkownika.",
                args_schema=ListNotificationRulesArgs,
            ),
            StructuredTool.from_function(
                coroutine=self.get_rule_expression_capabilities,
                name="get_rule_expression_capabilities",
                description=(
                    "Pobierz listę dostępnych funkcji wyrażenie reguły i metryk pogodowych. "
                    "Użyj przed tworzeniem wyrażenia reguły dla reguły powiadomienia."
                ),
                args_schema=GetRuleExpressionCapabilitiesArgs,
            ),
            StructuredTool.from_function(
                coroutine=self.propose_notification_rule,
                name="propose_notification_rule",
                description=(
                    "Zaproponuj regułę powiadomienia na podstawie wyrażenia reguły i opisu. "
                    "Narzędzie waliduje wyrażenie reguły i zapisuje propozycję do potwierdzenia."
                ),
                args_schema=ProposeNotificationRuleArgs,
            ),
            StructuredTool.from_function(
                coroutine=self.confirm_pending_action,
                name="confirm_pending_action",
                description="Potwierdź oczekującą akcję, gdy użytkownik odpowiada twierdząco.",
                args_schema=ConfirmPendingActionArgs,
            ),
            StructuredTool.from_function(
                coroutine=self.cancel_pending_action,
                name="cancel_pending_action",
                description="Anuluj oczekującą akcję, gdy użytkownik ją odrzuca.",
                args_schema=CancelPendingActionArgs,
            ),
            StructuredTool.from_function(
                coroutine=self.schedule_notification,
                name="schedule_notification",
                description=(
                    "Zaplanuj powiadomienie z opcjonalnym warunkiem wyrażenie reguły. "
                    "Użyj cron dla powtarzalnych próśb typu pon-pt, codziennie, "
                    "co godzinę lub w każdy czwartek. Waliduje harmonogram i zapisuje "
                    "propozycję do potwierdzenia."
                ),
                args_schema=ScheduleNotificationArgs,
            ),
        ]


def build_notification_rule_async_target_from_factory(
    model_factory: Callable[[], BaseChatModel],
) -> Callable[[dict[str, object]], Awaitable[dict[str, Any]]]:
    async def notification_rule_target(inputs: dict[str, object]) -> dict[str, Any]:
        example_id = str(inputs["id"])
        question = str(inputs["question"])
        current_time = datetime.fromisoformat(str(inputs["current_time"]))
        logger.debug("notification_rule_eval_run", id=example_id, question=question[:80])
        toolbox = RecordingRulesToolbox()
        agent = create_weather_agent(
            model=model_factory(),
            tools=toolbox.to_langchain_tools(),
            system_prompt_suffix=build_current_time_prompt_suffix(current_time),
        )
        result = cast(
            dict[str, Any],
            await agent.ainvoke(
                {"messages": [HumanMessage(content=question)]},
                config={"configurable": {"thread_id": example_id}},
            ),
        )
        final = result["messages"][-1]
        answer = final.content if hasattr(final, "content") else str(final)
        return RuleProposalEvalOutput(
            example_id=example_id,
            answer=str(answer),
            tool_calls=toolbox.tool_calls,
        ).model_dump()

    return notification_rule_target
