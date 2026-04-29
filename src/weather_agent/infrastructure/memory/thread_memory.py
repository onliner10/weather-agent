from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

from weather_agent.adapters.telegram.context import TelegramContextService

MAX_TURNS = 20


class ThreadMemoryService:
    def __init__(
        self,
        context_service: TelegramContextService,
        default_ttl_days: int = 14,
    ) -> None:
        self._context_service = context_service
        self._default_ttl_days = default_ttl_days

    def _is_expired(self, created_at: datetime, ttl_days: int) -> bool:
        return datetime.now(UTC) - created_at > timedelta(days=ttl_days)

    async def store_pending_confirmation(
        self, context_key: str, confirmation: dict[str, Any]
    ) -> None:
        ctx = await self._context_service.get_or_create_context(*_parse_context_key(context_key))
        metadata = dict(ctx.metadata)
        metadata["pending_confirmation"] = confirmation
        metadata["pending_confirmation_stored_at"] = datetime.now(UTC).isoformat()
        await self._context_service.update_context(context_key, metadata)

    async def get_pending_confirmation(
        self, context_key: str, ttl_days: int | None = None
    ) -> dict[str, Any] | None:
        ctx = await self._context_service.get_or_create_context(*_parse_context_key(context_key))
        confirmation = ctx.metadata.get("pending_confirmation")
        if confirmation is None:
            return None
        stored_at_raw = ctx.metadata.get("pending_confirmation_stored_at")
        if stored_at_raw is not None and isinstance(stored_at_raw, str):
            stored_at = datetime.fromisoformat(stored_at_raw)
            effective_ttl = ttl_days if ttl_days is not None else self._default_ttl_days
            if self._is_expired(stored_at, effective_ttl):
                await self.clear_pending_confirmation(context_key)
                return None
        return confirmation if isinstance(confirmation, dict) else None

    async def store_last_forecast(
        self, context_key: str, forecast_context: dict[str, Any]
    ) -> None:
        ctx = await self._context_service.get_or_create_context(*_parse_context_key(context_key))
        metadata = dict(ctx.metadata)
        metadata["last_forecast"] = forecast_context
        metadata["last_forecast_stored_at"] = datetime.now(UTC).isoformat()
        await self._context_service.update_context(context_key, metadata)

    async def load_last_forecast(
        self, context_key: str, ttl_hours: int = 24
    ) -> dict[str, Any] | None:
        ctx = await self._context_service.get_or_create_context(*_parse_context_key(context_key))
        forecast = ctx.metadata.get("last_forecast")
        if forecast is None:
            return None
        stored_at_raw = ctx.metadata.get("last_forecast_stored_at")
        if stored_at_raw is not None and isinstance(stored_at_raw, str):
            stored_at = datetime.fromisoformat(stored_at_raw)
            if datetime.now(UTC) - stored_at > timedelta(hours=ttl_hours):
                metadata = dict(ctx.metadata)
                metadata.pop("last_forecast", None)
                metadata.pop("last_forecast_stored_at", None)
                await self._context_service.update_context(context_key, metadata)
                return None
        return forecast if isinstance(forecast, dict) else None

    async def clear_pending_confirmation(self, context_key: str) -> None:
        ctx = await self._context_service.get_or_create_context(*_parse_context_key(context_key))
        metadata = dict(ctx.metadata)
        metadata.pop("pending_confirmation", None)
        metadata.pop("pending_confirmation_stored_at", None)
        await self._context_service.update_context(context_key, metadata)

    async def save_turn(
        self,
        context_key: str,
        turn: dict[str, Any],
        ttl_days: int | None = None,
    ) -> None:
        effective_ttl = ttl_days if ttl_days is not None else self._default_ttl_days
        ctx = await self._context_service.get_or_create_context(*_parse_context_key(context_key))
        metadata = dict(ctx.metadata)
        existing_turns: list[dict[str, Any]] = list(
            cast(list[dict[str, Any]], metadata.get("turns") or [])
        )
        if "timestamp" not in turn or turn.get("timestamp") is None:
            turn["timestamp"] = datetime.now(UTC).isoformat()
        existing_turns.append(turn)
        existing_turns = existing_turns[-MAX_TURNS:]
        metadata["turns"] = existing_turns
        metadata["turns_stored_at"] = datetime.now(UTC).isoformat()
        metadata["turns_ttl_days"] = effective_ttl
        await self._context_service.update_context(context_key, metadata)

    async def load_turns(
        self,
        context_key: str,
        ttl_days: int | None = None,
    ) -> list[dict[str, Any]]:
        ctx = await self._context_service.get_or_create_context(*_parse_context_key(context_key))
        turns_raw = ctx.metadata.get("turns")
        if turns_raw is None or not isinstance(turns_raw, list):
            return []
        stored_at_raw = ctx.metadata.get("turns_stored_at")
        if stored_at_raw is not None and isinstance(stored_at_raw, str):
            stored_at = datetime.fromisoformat(stored_at_raw)
            turns_ttl_raw = ctx.metadata.get("turns_ttl_days", self._default_ttl_days)
            context_ttl: int = (
                int(turns_ttl_raw)
                if isinstance(turns_ttl_raw, (int, float))
                else self._default_ttl_days
            )
            effective_ttl = ttl_days if ttl_days is not None else context_ttl
            if self._is_expired(stored_at, effective_ttl):
                metadata = dict(ctx.metadata)
                metadata.pop("turns", None)
                metadata.pop("turns_stored_at", None)
                metadata.pop("turns_ttl_days", None)
                await self._context_service.update_context(context_key, metadata)
                return []
        return turns_raw[-MAX_TURNS:]

    async def find_turn_by_message_id(
        self,
        context_key: str,
        message_id: int,
    ) -> dict[str, Any] | None:
        turns = await self.load_turns(context_key)
        for turn in reversed(turns):
            if turn.get("message_id") == message_id:
                return turn
        return None

    async def update_last_bot_turn_message_id(
        self,
        context_key: str,
        bot_message_id: int,
    ) -> None:
        ctx = await self._context_service.get_or_create_context(*_parse_context_key(context_key))
        metadata = dict(ctx.metadata)
        turns: list[dict[str, Any]] = list(cast(list[dict[str, Any]], metadata.get("turns") or []))
        for i in range(len(turns) - 1, -1, -1):
            if turns[i].get("role") == "bot":
                turns[i]["message_id"] = bot_message_id
                break
        metadata["turns"] = turns
        await self._context_service.update_context(context_key, metadata)


def _parse_context_key(context_key: str) -> tuple[int, int | None]:
    if ":" in context_key:
        parts = context_key.split(":", 1)
        return int(parts[0]), int(parts[1])
    return int(context_key), None
