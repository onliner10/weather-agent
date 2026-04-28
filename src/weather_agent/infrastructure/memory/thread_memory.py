from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from weather_agent.adapters.telegram.context import TelegramContextService


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
        ctx = await self._context_service.get_or_create_context(
            *_parse_context_key(context_key)
        )
        metadata = dict(ctx.metadata)
        metadata["pending_confirmation"] = confirmation
        metadata["pending_confirmation_stored_at"] = datetime.now(UTC).isoformat()
        await self._context_service.update_context(context_key, metadata)

    async def get_pending_confirmation(
        self, context_key: str, ttl_days: int | None = None
    ) -> dict[str, Any] | None:
        ctx = await self._context_service.get_or_create_context(
            *_parse_context_key(context_key)
        )
        confirmation = ctx.metadata.get("pending_confirmation")
        if confirmation is None:
            return None
        stored_at_raw = ctx.metadata.get("pending_confirmation_stored_at")
        if stored_at_raw is not None and isinstance(stored_at_raw, str):
            stored_at = datetime.fromisoformat(stored_at_raw)
            effective_ttl = (
                ttl_days if ttl_days is not None else self._default_ttl_days
            )
            if self._is_expired(stored_at, effective_ttl):
                await self.clear_pending_confirmation(context_key)
                return None
        return confirmation if isinstance(confirmation, dict) else None

    async def clear_pending_confirmation(self, context_key: str) -> None:
        ctx = await self._context_service.get_or_create_context(
            *_parse_context_key(context_key)
        )
        metadata = dict(ctx.metadata)
        metadata.pop("pending_confirmation", None)
        metadata.pop("pending_confirmation_stored_at", None)
        await self._context_service.update_context(context_key, metadata)

    async def store_recent_context(
        self,
        context_key: str,
        context: dict[str, Any],
        ttl_days: int | None = None,
    ) -> None:
        effective_ttl = (
            ttl_days if ttl_days is not None else self._default_ttl_days
        )
        ctx = await self._context_service.get_or_create_context(
            *_parse_context_key(context_key)
        )
        metadata = dict(ctx.metadata)
        metadata["recent_context"] = context
        metadata["recent_context_stored_at"] = datetime.now(UTC).isoformat()
        metadata["recent_context_ttl_days"] = effective_ttl
        await self._context_service.update_context(context_key, metadata)

    async def get_recent_context(
        self, context_key: str, ttl_days: int | None = None
    ) -> dict[str, Any] | None:
        ctx = await self._context_service.get_or_create_context(
            *_parse_context_key(context_key)
        )
        recent = ctx.metadata.get("recent_context")
        if recent is None:
            return None
        stored_at_raw = ctx.metadata.get("recent_context_stored_at")
        context_ttl_raw = ctx.metadata.get(
            "recent_context_ttl_days", self._default_ttl_days
        )
        if stored_at_raw is not None and isinstance(stored_at_raw, str):
            stored_at = datetime.fromisoformat(stored_at_raw)
            context_ttl: int = (
                int(context_ttl_raw)
                if isinstance(context_ttl_raw, (int, float))
                else self._default_ttl_days
            )
            effective_ttl = ttl_days if ttl_days is not None else context_ttl
            if self._is_expired(stored_at, effective_ttl):
                metadata = dict(ctx.metadata)
                metadata.pop("recent_context", None)
                metadata.pop("recent_context_stored_at", None)
                metadata.pop("recent_context_ttl_days", None)
                await self._context_service.update_context(context_key, metadata)
                return None
        return recent if isinstance(recent, dict) else None


def _parse_context_key(context_key: str) -> tuple[int, int | None]:
    if ":" in context_key:
        parts = context_key.split(":", 1)
        return int(parts[0]), int(parts[1])
    return int(context_key), None