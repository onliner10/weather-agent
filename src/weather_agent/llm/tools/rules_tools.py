from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from langsmith import traceable
from pydantic import BaseModel, Field

from weather_agent.application.conversation_models import PendingConfirmation
from weather_agent.domain.locations import LocationCreate, LocationService
from weather_agent.domain.rule_expression.allowlist import get_allowlist_for_prompt
from weather_agent.domain.rule_expression.evaluator import RuleExpressionEvaluator
from weather_agent.domain.rules.models import (
    RuleCreate,
    ScheduledNotificationContext,
    ScheduledNotificationTurn,
)
from weather_agent.domain.rules.schedule import parse_schedule
from weather_agent.domain.rules.service import NotificationRuleService
from weather_agent.domain.rules.short_id_generator import strip_hash_prefix
from weather_agent.infrastructure.geocoder import Geocoder
from weather_agent.infrastructure.memory.thread_memory import ThreadMemoryService
from weather_agent.observability.logging import get_logger
from weather_agent.observability.metrics import observe_tool_call

logger = get_logger(__name__)
_SCHEDULE_CONTEXT_TURNS = 6
_DATE_RANGE_FUNCTION_RE = re.compile(r"\bdate_range\s*\(")
CRON_FIXED_DATE_RANGE_ERROR = (
    "Dla cyklicznych harmonogramów cron nie używaj stałego date_range(...). "
    'Użyj zakresu względnego do każdego uruchomienia, np. between(today(), "1800", "2100") '
    "albo next_hours(1)."
)


def cron_schedule_uses_fixed_date_range(schedule_type: str, rule_expression: str) -> bool:
    return schedule_type == "cron" and _DATE_RANGE_FUNCTION_RE.search(rule_expression) is not None


class ListNotificationRulesArgs(BaseModel):
    include_disabled: bool = Field(
        default=False,
        description="Czy uwzględnić wyłączone reguły (domyślnie nie)",
    )


class GetRuleExpressionCapabilitiesArgs(BaseModel):
    pass


class ProposeNotificationRuleArgs(BaseModel):
    rule_expression: str = Field(
        description=(
            "Wyrażenie reguły definiujące warunek reguły powiadomienia. "
            "Używaj tylko dozwolonych funkcji i metryk (sprawdź get_rule_expression_capabilities)."
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


ToolResult = dict[str, Any]


class ScheduleNotificationArgs(BaseModel):
    schedule_type: Literal["once", "cron"] = Field(
        description=(
            "Typ harmonogramu: 'once' dla jednorazowego powiadomienia, 'cron' dla cyklicznego"
        ),
    )
    schedule_expression: str = Field(
        description=(
            "Wyrażenie harmonogramu bez prefiksu typu: ISO datetime dla 'once', "
            "5-polowe wyrażenie cron dla 'cron'. Cron obsługuje zakresy i dni tygodnia, "
            "np. '0 8-18 * * 1-5' dla godzinnych sprawdzeń pon-pt w dzień albo "
            "'0 8 * * 4' dla czwartku o 08:00."
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
    rule_expression: str = Field(
        default="true",
        description=(
            "Opcjonalne wyrażenie CEL warunku. Domyślnie 'true' (natychmiastowe przypomnienie). "
            "Dla cyklicznych alertów pogodowych harmonogram cron opisuje kiedy sprawdzać, "
            "a warunek powinien używać zakresu prognozy względnego do sprawdzenia, np. "
            'next_hours(1), today(), tomorrow() albo between(today(), "1800", "2100"), '
            "nie stałego date_range dla bieżącego tygodnia lub dnia."
        ),
    )


class RulesToolbox:
    def __init__(
        self,
        rule_service: NotificationRuleService,
        location_service: LocationService,
        rule_expression_evaluator: RuleExpressionEvaluator,
        geocoder: Geocoder,
        memory_service: ThreadMemoryService,
        context_key: str,
        user_id: int,
        chat_id: int,
        message_thread_id: int | None,
        current_user_message: str = "",
        session_lock: asyncio.Lock | None = None,
    ) -> None:
        self.rule_service = rule_service
        self.location_service = location_service
        self.rule_expression_evaluator = rule_expression_evaluator
        self.geocoder = geocoder
        self.memory_service = memory_service
        self.context_key = context_key
        self.user_id = user_id
        self.chat_id = chat_id
        self.message_thread_id = message_thread_id
        self.current_user_message = current_user_message
        self._lock = session_lock or asyncio.Lock()

    async def _resolve_location(self, location_name: str) -> tuple[int | None, str | None]:
        if not location_name.strip():
            default = await self.location_service.get_default_location(self.user_id)
            if default is not None:
                try:
                    return int(default.id), str(default.name)
                except (ValueError, TypeError):
                    return None, None
            return None, None

        resolved = await self.location_service.resolve_location(location_name, self.user_id)
        if resolved is not None:
            try:
                return int(resolved.id), str(resolved.name)
            except (ValueError, TypeError):
                return None, None

        geo = await self.geocoder.geocode(location_name)
        if geo is not None:
            try:
                created = await self.location_service.create_location(
                    self.user_id,
                    LocationCreate(
                        name=geo.name,
                        aliases=(
                            [location_name] if location_name.lower() != geo.name.lower() else []
                        ),
                        latitude=geo.latitude,
                        longitude=geo.longitude,
                    ),
                )
                return created.id, created.name
            except Exception:
                logger.exception("auto_save_location_failed", location_name=location_name)
                return None, None

        return None, None

    async def _resolve_location_id(self, location_name: str) -> int | None:
        location_id, _location_name = await self._resolve_location(location_name)
        return location_id

    async def _build_scheduled_notification_context(
        self,
        *,
        schedule: str,
        explanation: str,
        location_id: int | None,
        location_name: str | None,
    ) -> ScheduledNotificationContext:
        raw_turns = await self.memory_service.load_turns(self.context_key)
        prior_turns: list[ScheduledNotificationTurn] = []
        for turn in raw_turns[-_SCHEDULE_CONTEXT_TURNS:]:
            role = turn.get("role")
            text = turn.get("text") or turn.get("answer_summary")
            if isinstance(role, str) and isinstance(text, str) and text.strip():
                prior_turns.append(ScheduledNotificationTurn(role=role, text=text.strip()[:500]))

        scheduling_message = self.current_user_message.strip() or explanation
        return ScheduledNotificationContext(
            scheduling_message=scheduling_message[:1000],
            human_request=explanation.strip()[:1000],
            schedule=schedule,
            location_id=location_id,
            location_name=location_name,
            prior_turns=tuple(prior_turns),
        )

    @traceable(run_type="tool")
    async def list_notification_rules(
        self,
        include_disabled: bool = False,
    ) -> ToolResult:
        async with self._lock:
            with observe_tool_call("list_notification_rules"):
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
                    return {"rules": rules_data, "count": len(rules_data)}
                except Exception as exc:
                    logger.exception("list_notification_rules_failed", user_id=self.user_id)
                    return {"error": f"Błąd podczas pobierania reguł: {exc}"}

    @traceable(run_type="tool")
    async def get_rule_expression_capabilities(self) -> ToolResult:
        async with self._lock:
            with observe_tool_call("get_rule_expression_capabilities"):
                allowlist = get_allowlist_for_prompt()
                functions: dict[str, list[str]] = allowlist["functions"]  # type: ignore[assignment]
                metrics: list[str] = allowlist["metrics"]  # type: ignore[assignment]
                signatures: dict[str, str] = allowlist["signatures"]  # type: ignore[assignment]
                rules: list[str] = allowlist["rules"]  # type: ignore[assignment]
                examples: list[str] = allowlist["examples"]  # type: ignore[assignment]
                return {
                    "functions": functions,
                    "metrics": metrics,
                    "signatures": signatures,
                    "rules": rules,
                    "examples": examples,
                }

    @traceable(run_type="tool")
    async def propose_notification_rule(
        self,
        rule_expression: str,
        explanation: str,
        location_name: str = "",
        edit_short_id: str = "",
    ) -> ToolResult:
        async with self._lock:
            with observe_tool_call("propose_notification_rule"):
                return await self._execute_propose(
                    rule_expression,
                    explanation,
                    location_name,
                    edit_short_id,
                )

    async def _execute_propose(
        self,
        rule_expression: str,
        explanation: str,
        location_name: str,
        edit_short_id: str,
    ) -> ToolResult:
        validation = self.rule_expression_evaluator.validate(rule_expression)
        if not validation.valid:
            return {
                "error": f"Nieprawidłowe wyrażenie reguły: {validation.error}",
                "rule_expression": rule_expression,
                "explanation": explanation,
            }

        location_id = await self._resolve_location_id(location_name)
        if location_id is None and location_name.strip():
            return {
                "error": f"Nie znaleziono lokalizacji: {location_name}",
                "rule_expression": rule_expression,
                "explanation": explanation,
            }

        if edit_short_id:
            edit_short_id = strip_hash_prefix(edit_short_id.upper().replace("#", ""))

        action: Literal["create_rule", "edit_rule", "schedule_notification"] = (
            "edit_rule" if edit_short_id else "create_rule"
        )

        pending = PendingConfirmation(
            action=action,
            rule_expression=validation.expression,
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
            f"\U0001f4dd Wyrażenie reguły: `{validation.expression}`\n"
            f"\U0001f4d6 Opis: {explanation}\n\n"
            "Czy chcesz potwierdzić? (tak/nie)"
        )

        return {
            "proposal": proposal_text,
            "rule_expression": validation.expression,
            "explanation": explanation,
            "validated": True,
            "pending": True,
        }

    @traceable(run_type="tool")
    async def confirm_pending_action(self) -> ToolResult:
        async with self._lock:
            with observe_tool_call("confirm_pending_action"):
                return await self._execute_confirm()

    async def _execute_confirm(self) -> ToolResult:
        pending_dict = await self.memory_service.get_pending_confirmation(self.context_key)
        if pending_dict is None:
            return {"error": "Brak oczekującej akcji do potwierdzenia."}

        try:
            pending = PendingConfirmation.from_dict(pending_dict)
        except Exception:
            await self.memory_service.clear_pending_confirmation(self.context_key)
            return {"error": "Nie udało się odczytać oczekującej akcji. Utwórz ją ponownie."}

        if pending.rule_expression == "" and pending.action == "create_rule":
            return {"error": "Brak oczekującej reguły do potwierdzenia."}

        location_id = pending.location_id
        if location_id is None:
            default = await self.location_service.get_default_location(self.user_id)
            if default is not None:
                try:
                    location_id = int(default.id)
                except (ValueError, TypeError):
                    pass

        if location_id is None:
            return {"error": "Nie udało się rozpoznać lokalizacji dla reguły."}

        effective_chat_id = pending.chat_id if pending.chat_id is not None else self.chat_id
        effective_thread_id = (
            pending.message_thread_id
            if pending.message_thread_id is not None
            else self.message_thread_id
        )
        rule_expression = pending.rule_expression
        explanation = pending.explanation

        try:
            if pending.action == "schedule_notification":
                if pending.schedule is None:
                    return {"error": "Brak harmonogramu dla zaplanowanego powiadomienia."}
                existing_rule = await self.rule_service.find_matching_scheduled_rule(
                    user_id=self.user_id,
                    telegram_chat_id=effective_chat_id,
                    telegram_message_thread_id=effective_thread_id,
                    location_id=location_id,
                    expression=rule_expression,
                    schedule=pending.schedule,
                    notification_context=pending.notification_context,
                )
                if existing_rule is None:
                    rule = await self.rule_service.create_rule(
                        self.user_id,
                        RuleCreate(
                            telegram_chat_id=effective_chat_id,
                            telegram_message_thread_id=effective_thread_id,
                            location_id=location_id,
                            expression=rule_expression,
                            schedule=pending.schedule,
                            lead_time_minutes=pending.lead_time_minutes,
                            description=explanation,
                            notification_context=pending.notification_context,
                        ),
                    )
                    answer_prefix = "Nowe zaplanowane powiadomienie"
                else:
                    rule = existing_rule
                    answer_prefix = "Takie zaplanowane powiadomienie jest już zapisane"
                schedule_info = f"harmonogram: {pending.schedule}" if pending.schedule else ""
                answer = (
                    f"{answer_prefix} #{rule.short_id}.\n"
                    f"\U0001f4dd wyrażenie reguły: `{rule.expression}`\n"
                    f"\U0001f4c5 {schedule_info}"
                )
                short_id = rule.short_id
            elif pending.action == "edit_rule" and pending.edit_short_id:
                existing = await self.rule_service.get_rule_for_user(
                    self.user_id,
                    short_id=pending.edit_short_id,
                )
                if existing is None:
                    return {"error": f"Nie znaleziono reguły #{pending.edit_short_id}"}
                from weather_agent.domain.rules.models import RuleUpdate

                rule = await self.rule_service.update_rule(
                    existing.id,
                    RuleUpdate(expression=rule_expression, description=explanation),
                )
                answer = (
                    f"Reguła #{rule.short_id} została zaktualizowana.\n"
                    f"\U0001f4dd wyrażenie reguły: `{rule.expression}`"
                )
                short_id = rule.short_id
            else:
                rule = await self.rule_service.create_rule(
                    self.user_id,
                    RuleCreate(
                        telegram_chat_id=effective_chat_id,
                        telegram_message_thread_id=effective_thread_id,
                        location_id=location_id,
                        expression=rule_expression,
                        description=explanation,
                    ),
                )
                answer = (
                    f"Nowa reguła #{rule.short_id} została zapisana.\n"
                    f"\U0001f4dd wyrażenie reguły: `{rule.expression}`"
                )
                short_id = rule.short_id

            await self.memory_service.clear_pending_confirmation(self.context_key)
            return {"answer": answer, "short_id": short_id}

        except Exception as exc:
            logger.exception("confirm_pending_action_failed", user_id=self.user_id)
            return {"error": f"Błąd podczas potwierdzania: {exc}"}

    @traceable(run_type="tool")
    async def cancel_pending_action(self) -> ToolResult:
        async with self._lock:
            with observe_tool_call("cancel_pending_action"):
                return await self._execute_cancel()

    async def _execute_cancel(self) -> ToolResult:
        pending_dict = await self.memory_service.get_pending_confirmation(self.context_key)
        if pending_dict is None:
            return {"error": "Brak oczekującej akcji do anulowania."}

        try:
            pending = PendingConfirmation.from_dict(pending_dict)
        except Exception:
            await self.memory_service.clear_pending_confirmation(self.context_key)
            return {"answer": "Nieprawidłowa oczekująca akcja została usunięta."}

        action = pending.action

        await self.memory_service.clear_pending_confirmation(self.context_key)

        if action == "edit_rule" and pending.edit_short_id:
            return {"answer": f"Edycja reguły #{pending.edit_short_id} została anulowana."}
        return {"answer": "Reguła została anulowana."}

    @traceable(run_type="tool")
    async def schedule_notification(
        self,
        schedule_type: str,
        schedule_expression: str,
        explanation: str,
        location_name: str = "",
        rule_expression: str = "true",
    ) -> ToolResult:
        async with self._lock:
            with observe_tool_call("schedule_notification"):
                validation = self.rule_expression_evaluator.validate(rule_expression)
                if not validation.valid:
                    return {"error": f"Nieprawidłowe wyrażenie reguły: {validation.error}"}

                schedule_str = f"{schedule_type}:{schedule_expression}"
                parsed = parse_schedule(schedule_str)
                if not parsed.valid:
                    return {"error": f"Nieprawidłowy harmonogram: {parsed.error}"}
                if cron_schedule_uses_fixed_date_range(schedule_type, validation.expression):
                    return {"error": CRON_FIXED_DATE_RANGE_ERROR}

                location_id, resolved_location_name = await self._resolve_location(location_name)
                if location_id is None and location_name.strip():
                    return {"error": f"Nie znaleziono lokalizacji: {location_name}"}

                pending = PendingConfirmation(
                    action="schedule_notification",
                    rule_expression=validation.expression,
                    explanation=explanation,
                    validated=True,
                    location_id=location_id,
                    chat_id=self.chat_id,
                    message_thread_id=self.message_thread_id,
                    stored_at=datetime.now(UTC).isoformat(),
                    schedule=schedule_str,
                    notification_context=await self._build_scheduled_notification_context(
                        schedule=schedule_str,
                        explanation=explanation,
                        location_id=location_id,
                        location_name=resolved_location_name or location_name.strip() or None,
                    ),
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
                    f"\U0001f4dd Wyrażenie reguły: `{validation.expression}`\n"
                    f"\U0001f4d6 Opis: {explanation}\n\n"
                    "Czy chcesz potwierdzić? (tak/nie)"
                )

                return {"proposal": proposal_text, "pending": True}

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
                    "Najpierw użyj get_rule_expression_capabilities, aby poznać "
                    "dostępne funkcje i metryki. "
                    "Narzędzie waliduje wyrażenie reguły deterministycznie i zapisuje propozycję "
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
                    "Zaplanuj powiadomienie z opcjonalnym warunkiem wyrażenie reguły. "
                    "Przyjmuje typ harmonogramu (once/cron), wyrażenie harmonogramu "
                    "(ISO datetime lub 5-polowe cron bez prefiksu cron:), opis, lokalizację "
                    "i opcjonalne wyrażenie reguły. Użyj cron dla powtarzalnych próśb typu "
                    "'pon-pt', 'w każdy czwartek', 'codziennie' albo 'co godzinę'. "
                    "Waliduje harmonogram i wyrażenie reguły, "
                    "a następnie zapisuje propozycję do potwierdzenia przez użytkownika."
                ),
                args_schema=ScheduleNotificationArgs,
            ),
        ]
