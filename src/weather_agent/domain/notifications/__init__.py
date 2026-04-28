from __future__ import annotations

from weather_agent.domain.notifications.deduplication import (
    DedupeKey,
    NotificationCandidate,
    NotificationDeduplicator,
    compute_dedupe_key,
    compute_payload_hash,
    has_significant_change,
)
from weather_agent.domain.notifications.events import (
    EventNotFoundError,
    ExplanationService,
    NotificationEventService,
)

__all__ = [
    "DedupeKey",
    "EventNotFoundError",
    "ExplanationService",
    "NotificationCandidate",
    "NotificationDeduplicator",
    "NotificationEventService",
    "compute_dedupe_key",
    "compute_payload_hash",
    "has_significant_change",
]