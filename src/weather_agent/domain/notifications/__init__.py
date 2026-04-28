from __future__ import annotations

from weather_agent.domain.notifications.deduplication import (
    DedupeKey,
    NotificationCandidate,
    NotificationDeduplicator,
    compute_dedupe_key,
    compute_payload_hash,
    has_significant_change,
)

__all__ = [
    "DedupeKey",
    "NotificationCandidate",
    "NotificationDeduplicator",
    "compute_dedupe_key",
    "compute_payload_hash",
    "has_significant_change",
]