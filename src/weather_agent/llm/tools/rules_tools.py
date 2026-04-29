from __future__ import annotations

import time as _time
from datetime import UTC, datetime
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from langsmith import traceable
from pydantic import BaseModel, Field

from weather_agent.application.conversation_models import PendingConfirmation
from weather_agent.domain.cel.allowlist import get_allowlist_for_prompt
from weather_agent.domain.cel.evaluator import CELEvaluator
from weather_agent.domain.locations import LocationService
from weather_agent.domain.rules.models import RuleCreate
from weather_agent.domain.rules.schedule import parse_schedule
from weather_agent.domain.rules.service import NotificationRuleService
from weather_agent.domain.rules.short_id_generator import strip_hash_prefix
from weather_agent.infrastructure.geocoder import Geocoder
from weather_agent.infrastructure.memory.thread_memory import ThreadMemoryService
from weather_agent.observability.logging import get_logger
from weather_agent.observability.metrics import (
    TOOL_CALL_DURATION_SECONDS,
    TOOL_CALLS_TOTAL,
)

logger = get_logger(__name__)


class ListNotificationRulesArgs(BaseModel):
    include_disabled: bool = Field(
        default=False,
        description="Czy uwzględnić wyłączone reguły (domyślnie nie)",
    )


class GetCELCapabilitiesArgs(BaseModel):
    pass


class ProposeNotificationRuleArgs(BaseModel):
    cel_expression: str = Field(
        description=(
            "Wyrażenie CEL definiujące warunek reguły powiadomienia. "
            "Używaj tylko dozwolonych funkcji i metryk (sprawdź get_cel_capabilities)."
        ),
    )
    explanation: str = Field(
        description="Opis reguły po polsku, np. 'Powiadom mnie gdy spadnie śnieg'",
    )
    location_name: str = Field(
        default="",
        description="Nazwa lokalizacji dla reguły (np. 'Warszawa', 'dom'). "
        "Jeśli pusta, użyje domyślnej lokalizacji użytkownika.",
    )
    edit_short_id: str = Field(
        default="",
        description="ID reguły do edycji (np. 'R1A2B3'). Puste oznacza nową regułę.",
    )


class ConfirmPendingActionArgs(BaseModel):
    pass


class CancelPendingActionArgs(BaseModel):
    pass


class ListRulesToolResult(BaseModel):
    rules: list[dict[str, Any]] | None = None
    count: int = 0
    error: str | None = None


class CELCapabilitiesToolResult(BaseModel):
    functions: dict[str, list[str]] | None = None
    metrics: list[str] | None = None
    error: str | None = None


class ProposeRuleToolResult(BaseModel):
    proposal: str | None = None
    cel_expression: str | None = None
    explanation: str | None = None
    validated: bool = False
    pending: bool = False
    error: str | None = None


class ConfirmActionToolResult(BaseModel):
    answer: str | None = None
    short_id: str | None = None
    error: str | None = None


class CancelActionToolResult(BaseModel):
    answer: str | None = None
    error: str | None = None


class ScheduleNotificationArgs(BaseModel):
    schedule_type: Literal["once", "cron"] = Field(
        description=(
            "Typ harmonogramu: 'once' dla jednorazowego powiadomienia, 'cron' dla cyklicznego"
        ),
    )
    schedule_expression: str = Field(
        description=(
            "Wyrażenie harmonogramu: ISO datetime dla 'once', 5-polowe wyrażenie cron dla 'cron'"
        ),
    )
    explanation: str = Field(
        description="Opis powiadomienia po polsku",
    )
    location_name: str = Field(
        default="",
        description=(
            "Nazwa lokalizacji (np. 'Warszawa', 'dom'). Pusta = domyślna lokalizacja użytkownika."
        ),
    )
    cel_expression: str = Field(
        default="True",
        description=(
            "Opcjonalne wyrażenie CEL warunku. Domyślnie 'True' (natychmiastowe przypomnienie)."
        ),
    )


class ScheduleRuleToolResult(BaseModel):
    proposal: str | None = None
    pending: bool = False
    error: str | None = None


class RulesToolbox:
    def __init__(
        self,
        rule_service: NotificationRuleService,
        location_service: LocationService,
        cel_evaluator: CELEvaluator,
        geocoder: Geocoder,
        memory_service: ThreadMemoryService,
        context_key: str,
        user_id: int,
        chat_id: int,
        message_thread_id: int | None,
    ) -> None:
        self.rule_service = rule_service
        self.location_service = location_service
        self.cel_evaluator = cel_evaluator
        self.geocoder = geocoder
        self.memory_service = memory_service
        self.context_key = context_key
        self.user_id = user_id
        self.chat_id = chat_id
        self.message_thread_id = message_thread_id

    async def _resolve_location_id(self, location_name: str) -> int | None:
        if not location_name.strip():
            default = await self.location_service.get_default_location(self.user_id)
            if default is not None:
                try:
                    return int(default.id)
                except (ValueError, TypeError):
                    return None
            return None

        resolved = await self.location_service.resolve_location(location_name, self.user_id)
        if resolved is not None:
            try:
                return int(resolved.id)
            except (ValueError, TypeError):
                return None

        geo = await self.geocoder.geocode(location_name)
        if geo is not None:
            return None

        return None

    @traceable(run_type="tool")
    async def list_notification_rules(
        self,
        include_disabled: bool = False,
    ) -> ListRulesToolResult:
        TOOL_CALLS_TOTAL.labels(tool="list_notification_rules").inc()
        start = _time.perf_counter()
        try:
            rules = await self.rule_service.list_rules(
                self.user_id,
                include_disabled=include_disabled,
            )
            rules_data: list[dict[str, Any]] = []
            for r in rules:
                rules_data.append(
                    {
                        "short_id": f"#{r.short_id}",
                        "expression": r.expression,
                        "description": r.description or "",
                        "enabled": r.enabled,
                        "location_id": r.location_id,
                    }
                )
            return ListRulesToolResult(rules=rules_data, count=len(rules_data))
        except Exception as exc:
            logger.exception("list_notification_rules_failed", user_id=self.user_id)
            return ListRulesToolResult(error=f"Błąd podczas pobierania reguł: {exc}")
        finally:
            TOOL_CALL_DURATION_SECONDS.labels(tool="list_notification_rules").observe(
                _time.perf_counter() - start,
            )

    @traceable(run_type="tool")
    async def get_cel_capabilities(self) -> CELCapabilitiesToolResult:
        TOOL_CALLS_TOTAL.labels(tool="get_cel_capabilities").inc()
        start = _time.perf_counter()
        try:
            allowlist = get_allowlist_for_prompt()
            functions: dict[str, list[str]] = allowlist["functions"]  # type: ignore[assignment]
            metrics: list[str] = allowlist["metrics"]  # type: ignore[assignment]
            return CELCapabilitiesToolResult(
                functions=functions,
                metrics=metrics,
            )
        finally:
            TOOL_CALL_DURATION_SECONDS.labels(tool="get_cel_capabilities").observe(
                _time.perf_counter() - start,
            )

    @traceable(run_type="tool")
    async def propose_notification_rule(
        self,
        cel_expression: str,
        explanation: str,
        location_name: str = "",
        edit_short_id: str = "",
    ) -> ProposeRuleToolResult:
        TOOL_CALLS_TOTAL.labels(tool="propose_notification_rule").inc()
        start = _time.perf_counter()
        try:
            return await self._execute_propose(
                cel_expression,
                explanation,
                location_name,
                edit_short_id,
            )
        finally:
            TOOL_CALL_DURATION_SECONDS.labels(tool="propose_notification_rule").observe(
                _time.perf_counter() - start,
            )

    async def _execute_propose(
        self,
        cel_expression: str,
        explanation: str,
        location_name: str,
        edit_short_id: str,
    ) -> ProposeRuleToolResult:
        validation = self.cel_evaluator.validate(cel_expression)
        if not validation.valid:
            return ProposeRuleToolResult(
                error=f"Nieprawidłowe wyrażenie CEL: {validation.error}",
                cel_expression=cel_expression,
                explanation=explanation,
            )

        location_id = await self._resolve_location_id(location_name)
        if location_id is None and location_name.strip():
            return ProposeRuleToolResult(
                error=f"Nie znaleziono lokalizacji: {location_name}",
                cel_expression=cel_expression,
                explanation=explanation,
            )

        if edit_short_id:
            edit_short_id = strip_hash_prefix(edit_short_id.upper().replace("#", ""))

        action = "edit_rule" if edit_short_id else "create_rule"

        pending = PendingConfirmation(
            action=action,
            cel_expression=validation.expression,
            explanation=explanation,
            validated=True,
            location_id=location_id,
            chat_id=self.chat_id,
            message_thread_id=self.message_thread_id,
            stored_at=datetime.now(UTC).isoformat(),
            edit_short_id=edit_short_id or None,
        )

        await self.memory_service.store_pending_confirmation(
            self.context_key,
            pending.to_dict(),
        )

        header = "Propozycja edycji reguły" if action == "edit_rule" else "Propozycja nowej reguły"
        if edit_short_id:
            header += f" #{edit_short_id}"
        proposal_text = (
            f"{header}:\n\n"
            f"\U0001f4dd Wyrażenie CEL: `{validation.expression}`\n"
            f"\U0001f4d6 Opis: {explanation}\n\n"
            "Czy chcesz potwierdzić? (tak/nie)"
        )

        return ProposeRuleToolResult(
            proposal=proposal_text,
            cel_expression=validation.expression,
            explanation=explanation,
            validated=True,
            pending=True,
        )

    @traceable(run_type="tool")
    async def confirm_pending_action(self) -> ConfirmActionToolResult:
        TOOL_CALLS_TOTAL.labels(tool="confirm_pending_action").inc()
        start = _time.perf_counter()
        try:
            return await self._execute_confirm()
        finally:
            TOOL_CALL_DURATION_SECONDS.labels(tool="confirm_pending_action").observe(
                _time.perf_counter() - start,
            )

    async def _execute_confirm(self) -> ConfirmActionToolResult:
        pending_dict = await self.memory_service.get_pending_confirmation(self.context_key)
        if pending_dict is None:
            return ConfirmActionToolResult(error="Brak oczekującej akcji do potwierdzenia.")

        pending = PendingConfirmation.from_dict(pending_dict)

        if pending.cel_expression == "" and pending.action == "create_rule":
            return ConfirmActionToolResult(error="Brak oczekującej reguły do potwierdzenia.")

        location_id = pending.location_id
        if location_id is None:
            default = await self.location_service.get_default_location(self.user_id)
            if default is not None:
                try:
                    location_id = int(default.id)
                except (ValueError, TypeError):
                    pass

        if location_id is None:
            return ConfirmActionToolResult(
                error="Nie udało się rozpoznać lokalizacji dla reguły.",
            )

        effective_chat_id = pending.chat_id if pending.chat_id is not None else self.chat_id
        effective_thread_id = (
            pending.message_thread_id
            if pending.message_thread_id is not None
            else self.message_thread_id
        )
        cel_expression = pending.cel_expression
        explanation = pending.explanation

        try:
            if pending.action == "schedule_notification":
                rule = await self.rule_service.create_rule(
                    self.user_id,
                    RuleCreate(
                        telegram_chat_id=effective_chat_id,
                        telegram_message_thread_id=effective_thread_id,
                        location_id=location_id,
                        expression=cel_expression,
                        schedule=pending.schedule,
                        lead_time_minutes=pending.lead_time_minutes,
                        description=explanation,
                    ),
                )
                schedule_info = f"harmonogram: {pending.schedule}" if pending.schedule else ""
                answer = (
                    f"Nowe zaplanowane powiadomienie #{rule.short_id} zostało zapisane.\n"
                    f"\U0001f4dd CEL: `{rule.expression}`\n"
                    f"\U0001f4c5 {schedule_info}"
                )
                short_id = rule.short_id
            elif pending.action == "edit_rule" and pending.edit_short_id:
                existing = await self.rule_service.get_rule(short_id=pending.edit_short_id)
                if existing is None:
                    return ConfirmActionToolResult(
                        error=f"Nie znaleziono reguły #{pending.edit_short_id}",
                    )
                from weather_agent.domain.rules.models import RuleUpdate

                rule = await self.rule_service.update_rule(
                    existing.id,
                    RuleUpdate(expression=cel_expression, description=explanation),
                )
                answer = (
                    f"Reguła #{rule.short_id} została zaktualizowana.\n"
                    f"\U0001f4dd CEL: `{rule.expression}`"
                )
                short_id = rule.short_id
            else:
                rule = await self.rule_service.create_rule(
                    self.user_id,
                    RuleCreate(
                        telegram_chat_id=effective_chat_id,
                        telegram_message_thread_id=effective_thread_id,
                        location_id=location_id,
                        expression=cel_expression,
                        description=explanation,
                    ),
                )
                answer = (
                    f"Nowa reguła #{rule.short_id} została zapisana.\n"
                    f"\U0001f4dd CEL: `{rule.expression}`"
                )
                short_id = rule.short_id

            await self.memory_service.clear_pending_confirmation(self.context_key)
            return ConfirmActionToolResult(answer=answer, short_id=short_id)

        except Exception as exc:
            logger.exception("confirm_pending_action_failed", user_id=self.user_id)
            return ConfirmActionToolResult(error=f"Błąd podczas potwierdzania: {exc}")

    @traceable(run_type="tool")
    async def cancel_pending_action(self) -> CancelActionToolResult:
        TOOL_CALLS_TOTAL.labels(tool="cancel_pending_action").inc()
        start = _time.perf_counter()
        try:
            return await self._execute_cancel()
        finally:
            TOOL_CALL_DURATION_SECONDS.labels(tool="cancel_pending_action").observe(
                _time.perf_counter() - start,
            )

    async def _execute_cancel(self) -> CancelActionToolResult:
        pending_dict = await self.memory_service.get_pending_confirmation(self.context_key)
        if pending_dict is None:
            return CancelActionToolResult(error="Brak oczekującej akcji do anulowania.")

        pending = PendingConfirmation.from_dict(pending_dict)
        action = pending.action

        await self.memory_service.clear_pending_confirmation(self.context_key)

        if action == "edit_rule" and pending.edit_short_id:
            return CancelActionToolResult(
                answer=f"Edycja reguły #{pending.edit_short_id} została anulowana.",
            )
        return CancelActionToolResult(answer="Reguła została anulowana.")

    @traceable(run_type="tool")
    async def schedule_notification(
        self,
        schedule_type: str,
        schedule_expression: str,
        explanation: str,
        location_name: str = "",
        cel_expression: str = "True",
    ) -> ScheduleRuleToolResult:
        TOOL_CALLS_TOTAL.labels(tool="schedule_notification").inc()
        start = _time.perf_counter()
        try:
            validation = self.cel_evaluator.validate(cel_expression)
            if not validation.valid:
                return ScheduleRuleToolResult(
                    error=f"Nieprawidłowe wyrażenie CEL: {validation.error}",
                )

            schedule_str = f"{schedule_type}:{schedule_expression}"
            parsed = parse_schedule(schedule_str)
            if not parsed.valid:
                return ScheduleRuleToolResult(
                    error=f"Nieprawidłowy harmonogram: {parsed.error}",
                )

            location_id = await self._resolve_location_id(location_name)
            if location_id is None and location_name.strip():
                return ScheduleRuleToolResult(
                    error=f"Nie znaleziono lokalizacji: {location_name}",
                )

            pending = PendingConfirmation(
                action="schedule_notification",
                cel_expression=validation.expression,
                explanation=explanation,
                validated=True,
                location_id=location_id,
                chat_id=self.chat_id,
                message_thread_id=self.message_thread_id,
                stored_at=datetime.now(UTC).isoformat(),
                schedule=schedule_str,
            )

            await self.memory_service.store_pending_confirmation(
                self.context_key,
                pending.to_dict(),
            )

            schedule_desc = (
                f"jednorazowo {schedule_expression}"
                if schedule_type == "once"
                else f"cyklicznie ({schedule_expression})"
            )
            proposal_text = (
                f"Propozycja zaplanowanego powiadomienia:\n\n"
                f"\U0001f4c5 Harmonogram: {schedule_desc}\n"
                f"\U0001f4dd Wyrażenie CEL: `{validation.expression}`\n"
                f"\U0001f4d6 Opis: {explanation}\n\n"
                "Czy chcesz potwierdzić? (tak/nie)"
            )

            return ScheduleRuleToolResult(proposal=proposal_text, pending=True)
        finally:
            TOOL_CALL_DURATION_SECONDS.labels(tool="schedule_notification").observe(
                _time.perf_counter() - start,
            )

    def to_langchain_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                coroutine=self.list_notification_rules,
                name="list_notification_rules",
                description=(
                    "Wyświetl reguły powiadomień użytkownika. "
                    "Domyślnie pokazuje tylko aktywne reguły."
                ),
                args_schema=ListNotificationRulesArgs,
            ),
            StructuredTool.from_function(
                coroutine=self.get_cel_capabilities,
                name="get_cel_capabilities",
                description=(
                    "Pobierz listę dostępnych funkcji CEL i metryk pogodowych. "
                    "Użyj przed tworzeniem wyrażenia CEL dla reguły powiadomienia."
                ),
                args_schema=GetCELCapabilitiesArgs,
            ),
            StructuredTool.from_function(
                coroutine=self.propose_notification_rule,
                name="propose_notification_rule",
                description=(
                    "Zaproponuj regułę powiadomienia na podstawie wyrażenia CEL i opisu. "
                    "Najpierw użyj get_cel_capabilities aby poznać dostępne funkcje i metryki. "
                    "Narzędzie waliduje wyrażenie CEL deterministycznie i zapisuje propozycję "
                    "do potwierdzenia przez użytkownika. NIE tworzy reguły natychmiast — "
                    "użytkownik musi potwierdzić za pomocą confirm_pending_action."
                ),
                args_schema=ProposeNotificationRuleArgs,
            ),
            StructuredTool.from_function(
                coroutine=self.confirm_pending_action,
                name="confirm_pending_action",
                description=(
                    "Potwierdź oczekującą akcję (np. utworzenie reguły powiadomienia). "
                    "Użyj gdy użytkownik potwierdza propozycję reguły (np. odpowiada 'tak'). "
                    "Narzędzie tworzy regułę w bazie danych i usuwa oczekującą akcję."
                ),
                args_schema=ConfirmPendingActionArgs,
            ),
            StructuredTool.from_function(
                coroutine=self.cancel_pending_action,
                name="cancel_pending_action",
                description=(
                    "Anuluj oczekującą akcję (np. propozycję reguły powiadomienia). "
                    "Użyj gdy użytkownik odrzuca propozycję "
                    "(np. odpowiada 'nie')."
                ),
                args_schema=CancelPendingActionArgs,
            ),
            StructuredTool.from_function(
                coroutine=self.schedule_notification,
                name="schedule_notification",
                description=(
                    "Zaplanuj powiadomienie z opcjonalnym warunkiem CEL. "
                    "Przyjmuje typ harmonogramu (once/cron), wyrażenie harmonogramu "
                    "(ISO datetime lub 5-polowe cron), opis, lokalizację "
                    "i opcjonalne wyrażenie CEL. Waliduje harmonogram i CEL, "
                    "a następnie zapisuje propozycję do potwierdzenia przez użytkownika."
                ),
                args_schema=ScheduleNotificationArgs,
            ),
        ]
