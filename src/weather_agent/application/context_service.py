from __future__ import annotations

import json
from typing import Any

from weather_agent.application.conversation_models import (
    LoadedContext,
    PendingConfirmation,
    TurnRecord,
)
from weather_agent.domain.date_resolver import ResolvedTimeRange
from weather_agent.domain.weather import LocationRef
from weather_agent.infrastructure.memory.thread_memory import ThreadMemoryService
from weather_agent.observability.logging import get_logger

logger = get_logger(__name__)


async def load_thread_context(
    memory_service: ThreadMemoryService | None,
    context_key: str,
    reply_to_message_id: int | None = None,
) -> LoadedContext:
    if memory_service is None:
        return LoadedContext()

    updates = LoadedContext()
    try:
        pending = await memory_service.get_pending_confirmation(context_key)
        if pending is not None:
            updates = LoadedContext(
                pending_confirmation=pending,
                resolved_location=updates.resolved_location,
                resolved_time_range=updates.resolved_time_range,
                user_focus=updates.user_focus,
                reply_context_turns=updates.reply_context_turns,
            )

        if reply_to_message_id is not None:
            turns = await memory_service.load_turns(context_key)
            anchor = await memory_service.find_turn_by_message_id(context_key, reply_to_message_id)
            if anchor is not None:
                loc = anchor.get("resolved_location")
                resolved_loc = LocationRef(**loc) if loc and isinstance(loc, dict) else None
                tr = anchor.get("resolved_time_range")
                resolved_tr = ResolvedTimeRange(**tr) if tr and isinstance(tr, dict) else None
                focus = anchor.get("user_focus")

                ctx_turns = [anchor]
                if turns:
                    aid = anchor.get("message_id")
                    anchor_idx = -1
                    for i, t in enumerate(turns):
                        if t.get("message_id") is not None and t.get("message_id") == aid:
                            anchor_idx = i
                            break
                    if anchor_idx > 0:
                        prev = turns[anchor_idx - 1]
                        if prev.get("role") == "user":
                            ctx_turns.insert(0, prev)

                updates = LoadedContext(
                    pending_confirmation=updates.pending_confirmation,
                    resolved_location=resolved_loc,
                    resolved_time_range=resolved_tr,
                    user_focus=focus,
                    reply_context_turns=ctx_turns,
                )
    except Exception:
        logger.warning("thread_context_load_failed", context_key=context_key, exc_info=True)

    return updates


def _serialize_location(location: LocationRef | None) -> dict[str, Any] | None:
    if location is None:
        return None
    return {
        "id": location.id,
        "name": location.name,
        "latitude": location.latitude,
        "longitude": location.longitude,
    }


def _json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    return json.loads(json.dumps(obj, default=str))


async def save_thread_context(
    memory_service: ThreadMemoryService | None,
    context_key: str,
    user_message: str,
    answer: str,
    user_message_id: int | None,
    resolved_location: LocationRef | None,
    resolved_time_range: ResolvedTimeRange | None,
    user_focus: str | None,
    pending_confirmation: PendingConfirmation | None,
) -> None:
    if memory_service is None:
        return

    try:
        if pending_confirmation is not None:
            await memory_service.store_pending_confirmation(
                context_key, pending_confirmation.to_dict()
            )
        else:
            await memory_service.clear_pending_confirmation(context_key)

        user_turn: TurnRecord = {
            "message_id": user_message_id,
            "role": "user",
            "text": user_message,
            "resolved_location": _serialize_location(resolved_location),
            "resolved_time_range": (
                _json_safe(resolved_time_range.model_dump())
                if resolved_time_range is not None
                else None
            ),
            "user_focus": user_focus,
            "timestamp": None,
        }
        await memory_service.save_turn(context_key, dict(user_turn))

        if answer:
            summary = answer[:200] if len(answer) > 200 else answer
            bot_turn: TurnRecord = {
                "message_id": None,
                "role": "bot",
                "text": None,
                "answer_summary": summary,
                "resolved_location": _serialize_location(resolved_location),
                "resolved_time_range": (
                    _json_safe(resolved_time_range.model_dump())
                    if resolved_time_range is not None
                    else None
                ),
                "user_focus": user_focus,
                "timestamp": None,
            }
            await memory_service.save_turn(context_key, dict(bot_turn))
    except Exception:
        logger.warning("thread_context_save_failed", context_key=context_key, exc_info=True)