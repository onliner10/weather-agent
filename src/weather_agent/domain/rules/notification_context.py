from __future__ import annotations

import json

from pydantic import ValidationError

from weather_agent.domain.rules.models import ScheduledNotificationContext


def notification_context_from_mapping(
    value: dict[str, object] | None,
) -> ScheduledNotificationContext | None:
    if value is None:
        return None
    try:
        return ScheduledNotificationContext.model_validate(value)
    except ValidationError:
        return None


def notification_context_fingerprint(context: ScheduledNotificationContext | None) -> str:
    if context is None:
        return ""
    stable = {
        "human_request": " ".join(context.human_request.lower().split()),
        "schedule": context.schedule,
        "location_id": context.location_id,
        "location_name": (
            " ".join(context.location_name.lower().split())
            if context.location_name is not None
            else None
        ),
    }
    return json.dumps(stable, sort_keys=True, ensure_ascii=False)
